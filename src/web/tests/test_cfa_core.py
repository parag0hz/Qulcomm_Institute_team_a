import unittest
import json
from pathlib import Path
import struct
import time
from types import SimpleNamespace
from unittest.mock import patch

from web.cfa_service.copilot import ask_copilot, copilot_status
from web.cfa_service.predictor import (
    DEFAULT_MODEL_PATH,
    analyze_parameters,
    load_dataset_stats,
    load_parameter_schema,
    maybe_predict_parameters,
    optimize_parameters,
    predict_from_stl_points,
)
from web.cfa_service.providers import ProviderResult
from web.cfa_service.stl import parse_stl_bytes
from web.models.train_parametric_baseline import select_model


ASCII_STL = b"""solid cube
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
facet normal 0 0 1
outer loop
vertex 1 1 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
endsolid cube
"""


class DeterministicTestRouter:
    """Artifact-free provider for prediction, sensitivity, and optimizer tests."""

    def __init__(self, schema):
        numeric_columns = [item["name"] for item in schema["parameters"]]
        self._parameters = schema["parameters"]
        self.local = SimpleNamespace(
            model_path=DEFAULT_MODEL_PATH,
            artifact={
                "model_name": "DeterministicTestModel",
                "feature_columns": [*numeric_columns, "CarRear", "Wheels"],
                "numeric_columns": numeric_columns,
                "categorical_columns": ["CarRear", "Wheels"],
                "metrics": {"mae": 0.004},
            },
        )

    def predict(self, rows):
        values = []
        for row in rows:
            cd = 0.22
            for index, parameter in enumerate(self._parameters, start=1):
                span = max(float(parameter["max"]) - float(parameter["min"]), 1e-9)
                normalized = (float(row[parameter["name"]]) - float(parameter["min"])) / span
                cd += normalized * index * 0.0002
            values.append(cd)
        return ProviderResult(values, "Deterministic test provider", [])


def design_and_router():
    schema = load_parameter_schema()
    design = {parameter["name"]: parameter["default"] for parameter in schema["parameters"]}
    design.update(CarRear="Fastback", Wheels="Closed smooth")
    return design, DeterministicTestRouter(schema)


class CFACoreTest(unittest.TestCase):
    def test_reference_glb_is_valid_and_optimized(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "cfa_service"
            / "static"
            / "models"
            / "drivaer_reference.glb"
        )
        data = path.read_bytes()
        magic, version, total_length = struct.unpack_from("<III", data, 0)
        json_length, json_type = struct.unpack_from("<I4s", data, 12)
        document = json.loads(data[20 : 20 + json_length].decode("utf-8"))

        self.assertEqual(magic, 0x46546C67)
        self.assertEqual(version, 2)
        self.assertEqual(total_length, len(data))
        self.assertEqual(json_type, b"JSON")
        self.assertEqual(document["extras"]["simplifier"], "fast-simplification QEM")
        self.assertEqual(document["extras"]["component_count"], 5)
        self.assertEqual(document["extras"]["face_count"], 112_000)
        self.assertEqual(document["extras"]["boundary_edges"], 0)
        self.assertLessEqual(document["extras"]["nonmanifold_edges"], 33)
        self.assertEqual(
            {node["name"] for node in document["nodes"]},
            {"Body", "Wheel_FL", "Wheel_FR", "Wheel_RL", "Wheel_RR"},
        )
        self.assertEqual(len(document["extras"]["wheel_specs"]), 4)
        self.assertEqual(document["extras"]["axis_convention"], "X length, Y width, Z height")

    def test_parameter_schema_exposes_designer_inputs(self):
        schema = load_parameter_schema()

        self.assertEqual(len(schema["parameters"]), 23)
        self.assertEqual(
            schema["categories"]["CarRear"],
            ["Fastback", "Estateback", "Notchback"],
        )
        self.assertEqual(
            sum(parameter["high_impact"] for parameter in schema["parameters"]),
            3,
        )
        self.assertEqual(len(schema["presets"]), 4)

    def test_designer_analysis_and_optimization_without_model_artifact(self):
        design, router = design_and_router()
        with patch("web.cfa_service.predictor._PROVIDER_ROUTER", router):
            prediction = maybe_predict_parameters(design)
            self.assertEqual(prediction["provider"], "Deterministic test provider")
            self.assertIn(prediction["domain_status"], {"inside", "edge", "outside"})
            self.assertIn("estimate", prediction["uncertainty"])

            analysis = analyze_parameters(design)
            self.assertEqual(len(analysis["drivers"]), 23)
            self.assertGreaterEqual(analysis["drivers"][0]["impact"], analysis["drivers"][-1]["impact"])

            width = design["A_Car_Width"]
            optimized = optimize_parameters(design, prediction["cd"] - 0.002, ["A_Car_Width"])
            self.assertEqual(len(optimized["recommendations"]), 3)
            self.assertTrue(all(item["parameters"]["A_Car_Width"] == width for item in optimized["recommendations"]))

    def test_ascii_stl_prediction_flow(self):
        cloud = parse_stl_bytes(ASCII_STL)
        stats = load_dataset_stats()
        result = predict_from_stl_points(
            cloud.points, cloud.triangle_count, cloud.source_format, stats
        )

        self.assertEqual(cloud.point_count, 6)
        self.assertGreaterEqual(result["cd"], stats.cd_min)
        self.assertLessEqual(result["cd"], stats.cd_max)
        self.assertEqual(result["mesh"]["source_format"], "ascii")

    def test_ascii_stl_rejects_oversized_coordinate_without_regex_backtracking(self):
        payload = b"solid attack\nvertex " + (b"1" * 16_000) + b"x\nendsolid attack\n"
        started = time.perf_counter()
        with self.assertRaisesRegex(ValueError, "Could not find STL vertices"):
            parse_stl_bytes(payload)
        self.assertLess(time.perf_counter() - started, 1.0)

    def test_default_training_model_is_serving_compatible_random_forest(self):
        model_name, estimator = select_model(random_state=42)
        self.assertEqual(model_name, "RandomForest")
        self.assertEqual(type(estimator).__name__, "RandomForestRegressor")

    def test_copilot_works_without_external_api_key_or_model_artifact(self):
        design, router = design_and_router()
        with (
            patch("web.cfa_service.predictor._PROVIDER_ROUTER", router),
            patch.dict("os.environ", {}, clear=True),
        ):
            status = copilot_status()
            result = ask_copilot("현재 Cd가 높은 이유를 알려줘", design)
        self.assertFalse(status["configured"])
        self.assertEqual(result["provider"], "Grounded local explainer")
        self.assertIn("예측 Cd", result["answer"])
        self.assertIn("prediction", result["evidence"])


if __name__ == "__main__":
    unittest.main()
