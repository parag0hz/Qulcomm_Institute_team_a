import unittest
import importlib.util
import json
from pathlib import Path
import struct
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
from web.cfa_service.stl import parse_stl_bytes


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

MODEL_RUNTIME_AVAILABLE = (
    DEFAULT_MODEL_PATH.exists() and importlib.util.find_spec("sklearn") is not None
)


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

    @unittest.skipUnless(
        MODEL_RUNTIME_AVAILABLE,
        "Local model artifact and scikit-learn runtime are required for this integration test.",
    )
    def test_designer_analysis_and_optimization(self):
        schema = load_parameter_schema()
        design = {parameter["name"]: parameter["default"] for parameter in schema["parameters"]}
        design.update(CarRear="Fastback", Wheels="Closed smooth")
        prediction = maybe_predict_parameters(design)
        self.assertEqual(prediction["provider"], "Local RandomForest")
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

    @unittest.skipUnless(
        MODEL_RUNTIME_AVAILABLE,
        "Local model artifact and scikit-learn runtime are required for this integration test.",
    )
    def test_copilot_works_without_external_api_key(self):
        schema = load_parameter_schema()
        design = {parameter["name"]: parameter["default"] for parameter in schema["parameters"]}
        design.update(CarRear="Fastback", Wheels="Closed smooth")
        with patch.dict("os.environ", {}, clear=True):
            status = copilot_status()
            result = ask_copilot("현재 Cd가 높은 이유를 알려줘", design)
        self.assertFalse(status["configured"])
        self.assertEqual(result["provider"], "Grounded local explainer")
        self.assertIn("예측 Cd", result["answer"])
        self.assertIn("prediction", result["evidence"])


if __name__ == "__main__":
    unittest.main()
