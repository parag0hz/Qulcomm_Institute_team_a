import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.cfa_service.app import app
from backend.cfa_service.predictor import load_parameter_schema


EXPECTED_API_PATHS = {
    "/api/status",
    "/api/parameters",
    "/api/predict",
    "/api/predict/parameters",
    "/api/analyze/parameters",
    "/api/optimize/parameters",
    "/api/copilot",
    "/api/providers/vertex/test",
}


class ParagonApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._client_context = TestClient(app, raise_server_exceptions=False)
        cls.client = cls._client_context.__enter__()
        schema = load_parameter_schema()
        cls.design = {item["name"]: item["default"] for item in schema["parameters"]}
        cls.design.update(CarRear="Fastback", Wheels="Closed smooth")

    @classmethod
    def tearDownClass(cls):
        cls._client_context.__exit__(None, None, None)

    def assert_error(self, response, status_code, code):
        self.assertEqual(response.status_code, status_code)
        payload = response.json()
        self.assertEqual(payload["error"]["code"], code)
        self.assertIsInstance(payload["error"]["message"], str)

    def test_status_parameters_and_openapi_contract(self):
        status = self.client.get("/api/status")
        self.assertEqual(status.status_code, 200)
        status_payload = status.json()
        self.assertEqual(status_payload["product"], "paragon")
        self.assertEqual(len(status_payload["input_schema"]["numeric_features"]), 23)
        self.assertEqual(
            status_payload["input_schema"]["categorical_features"],
            ["CarRear", "Wheels"],
        )

        parameters = self.client.get("/api/parameters")
        self.assertEqual(parameters.status_code, 200)
        parameter_payload = parameters.json()
        self.assertEqual(len(parameter_payload["parameters"]), 23)
        self.assertIn("presets", parameter_payload)
        median = next(item for item in parameter_payload["presets"] if item["id"] == "median")
        valid_pairs = {
            (item["CarRear"], item["Wheels"])
            for item in parameter_payload["valid_combinations"]
        }
        self.assertIn(
            (median["design"]["CarRear"], median["design"]["Wheels"]),
            valid_pairs,
        )

        openapi = self.client.get("/openapi.json")
        self.assertEqual(openapi.status_code, 200)
        self.assertTrue(EXPECTED_API_PATHS.issubset(openapi.json()["paths"]))
        self.assertEqual(self.client.get("/docs").status_code, 200)
        self.assertEqual(self.client.get("/redoc").status_code, 200)

    def test_parameter_prediction_contract(self):
        fixture = {
            "cd": 0.244,
            "provider": "Local RandomForest",
            "domain_status": "inside",
            "nearest_sample_distance": 0.08,
            "uncertainty": {"estimate": 0.01, "lower": 0.224, "upper": 0.264},
            "warnings": [],
        }
        with patch("backend.cfa_service.app.maybe_predict_parameters", return_value=fixture):
            response = self.client.post("/api/predict/parameters", json=self.design)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        for key in ("cd", "provider", "domain_status", "uncertainty", "warnings"):
            self.assertIn(key, result)
        self.assertIn(result["domain_status"], {"inside", "edge", "outside"})

    def test_analysis_optimization_and_copilot_contracts(self):
        analysis_fixture = {
            "base_cd": 0.244,
            "provider": "Local RandomForest",
            "drivers": [
                {
                    "name": name,
                    "label": name,
                    "minus_delta": -0.001,
                    "plus_delta": 0.001,
                    "impact": 0.001,
                }
                for name in list(self.design)[:23]
            ],
            "warnings": [],
        }
        optimization_fixture = {
            "current_cd": 0.244,
            "target_cd": 0.24,
            "locked": ["A_Car_Width", "CarRear", "Wheels"],
            "recommendations": [
                {"cd": 0.24 + index * 0.001, "parameters": self.design, "changes": []}
                for index in range(3)
            ],
        }
        copilot_fixture = {
            "answer": "Grounded design explanation.",
            "provider": "grounded_local",
            "model": None,
            "evidence": {"prediction": {"cd": 0.244}, "top_drivers": []},
            "disclaimer": "Validate with CFD.",
        }
        with (
            patch("backend.cfa_service.app.analyze_parameters", return_value=analysis_fixture),
            patch("backend.cfa_service.app.optimize_parameters", return_value=optimization_fixture),
            patch("backend.cfa_service.app.ask_copilot", return_value=copilot_fixture),
        ):
            analysis = self.client.post("/api/analyze/parameters", json=self.design)
            optimization = self.client.post(
                "/api/optimize/parameters",
                json={"parameters": self.design, "target_cd": 0.24, "locked": ["A_Car_Width"]},
            )
            copilot = self.client.post(
                "/api/copilot",
                json={"message": "현재 설계를 설명해줘", "parameters": self.design, "history": []},
            )
        self.assertEqual(analysis.status_code, 200)
        self.assertEqual(len(analysis.json()["drivers"]), 23)
        self.assertEqual(optimization.status_code, 200)
        self.assertEqual(len(optimization.json()["recommendations"]), 3)
        self.assertEqual(copilot.status_code, 200)
        self.assertIn("evidence", copilot.json())

    @patch("backend.cfa_service.app.test_vertex_provider")
    def test_vertex_test_success_contract(self, vertex_provider):
        vertex_provider.return_value = {
            "provider": "Vertex AI",
            "prediction": 0.2412,
            "latency_ms": 18,
        }
        response = self.client.post(
            "/api/providers/vertex/test",
            json={"parameters": self.design},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider"], "Vertex AI")
        vertex_provider.assert_called_once()

    @patch("backend.cfa_service.app.predict_from_stl_points")
    @patch("backend.cfa_service.app.parse_stl_bytes")
    def test_stl_prediction_contract(self, parse_stl, predict_points):
        parse_stl.return_value = SimpleNamespace(
            points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangle_count=1,
            source_format="ascii",
        )
        predict_points.return_value = {"cd": 0.25, "preview_points": []}
        response = self.client.post(
            "/api/predict",
            files={"file": ("sample.STL", b"solid sample", "model/stl")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["file"]["name"], "sample.STL")
        self.assertEqual(response.json()["file"]["size_bytes"], 12)

    def test_validation_is_field_specific_and_forbids_extra_fields(self):
        invalid = dict(self.design)
        invalid.pop("A_Car_Length")
        invalid["unknown_feature"] = 1
        response = self.client.post("/api/predict/parameters", json=invalid)
        self.assertEqual(response.status_code, 422)
        locations = [item["loc"] for item in response.json()["detail"]]
        self.assertTrue(any("A_Car_Length" in location for location in locations))
        self.assertTrue(any("unknown_feature" in location for location in locations))

    def test_validation_rejects_categories_nonfinite_values_and_bad_locks(self):
        invalid_category = dict(self.design, CarRear="Coupe")
        response = self.client.post("/api/predict/parameters", json=invalid_category)
        self.assertEqual(response.status_code, 422)

        nonfinite = dict(self.design, A_Car_Length="NaN")
        response = self.client.post("/api/predict/parameters", json=nonfinite)
        self.assertEqual(response.status_code, 422)

        duplicate_locks = {
            "parameters": self.design,
            "target_cd": 0.24,
            "locked": ["A_Car_Width", "A_Car_Width"],
        }
        response = self.client.post("/api/optimize/parameters", json=duplicate_locks)
        self.assertEqual(response.status_code, 422)
        self.assertTrue(any("locked" in item["loc"] for item in response.json()["detail"]))

        unknown_lock = dict(duplicate_locks, locked=["not_a_parameter"])
        response = self.client.post("/api/optimize/parameters", json=unknown_lock)
        self.assertEqual(response.status_code, 422)

    def test_validation_rejects_bad_targets_and_blank_copilot_messages(self):
        response = self.client.post(
            "/api/optimize/parameters",
            json={"parameters": self.design, "target_cd": 0.5, "locked": []},
        )
        self.assertEqual(response.status_code, 422)

        response = self.client.post(
            "/api/copilot",
            json={"message": "   ", "parameters": self.design, "history": []},
        )
        self.assertEqual(response.status_code, 422)

    def test_stl_upload_limits_extension_and_parse_errors(self):
        wrong_extension = self.client.post(
            "/api/predict",
            files={"file": ("sample.txt", b"not stl", "text/plain")},
        )
        self.assert_error(wrong_extension, 400, "invalid_file_type")

        with patch("backend.cfa_service.app.MAX_UPLOAD_BYTES", 8):
            oversized = self.client.post(
                "/api/predict",
                files={"file": ("sample.stl", b"123456789", "model/stl")},
            )
        self.assert_error(oversized, 413, "payload_too_large")

        invalid_stl = self.client.post(
            "/api/predict",
            files={"file": ("sample.stl", b"not an stl", "model/stl")},
        )
        self.assert_error(invalid_stl, 400, "invalid_stl")

    def test_json_request_size_limit(self):
        with patch("backend.cfa_service.app.MAX_JSON_BYTES", 32):
            response = self.client.post(
                "/api/predict/parameters",
                content=json.dumps(self.design),
                headers={"content-type": "application/json"},
            )
        self.assert_error(response, 413, "payload_too_large")

    def test_streamed_json_request_size_limit(self):
        def chunks():
            yield b'{"payload":"'
            yield b"x" * 64
            yield b'"}'

        with patch("backend.cfa_service.app.MAX_JSON_BYTES", 32):
            response = self.client.post(
                "/api/predict/parameters",
                content=chunks(),
                headers={"content-type": "application/json"},
            )
        self.assert_error(response, 413, "payload_too_large")

    @patch(
        "backend.cfa_service.app.test_vertex_provider",
        side_effect=RuntimeError("ADC unavailable"),
    )
    def test_vertex_failure_uses_common_error_contract(self, _vertex_provider):
        response = self.client.post(
            "/api/providers/vertex/test",
            json={"parameters": self.design},
        )
        self.assert_error(response, 503, "vertex_unavailable")
        self.assertEqual(response.json()["error"]["fallback_provider"], "Local RandomForest")
        self.assertNotIn("detail", response.json())

    def test_unknown_api_route_uses_json_error_instead_of_spa(self):
        response = self.client.get("/api/does-not-exist")
        self.assert_error(response, 404, "not_found")

    @patch("backend.cfa_service.app.LOGGER.exception")
    @patch(
        "backend.cfa_service.app.load_parameter_schema",
        side_effect=RuntimeError("sensitive detail"),
    )
    def test_unhandled_errors_do_not_leak_internal_details(self, _load_schema, log_exception):
        response = self.client.get("/api/parameters")
        self.assert_error(response, 500, "internal_error")
        self.assertNotIn("sensitive detail", response.text)
        log_exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
