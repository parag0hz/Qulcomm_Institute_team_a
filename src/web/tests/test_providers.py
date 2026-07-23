import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import httpx

from web.cfa_service.providers import ProviderResult, ProviderRouter, VertexProvider


class FakeLocalProvider:
    name = "Local RandomForest"

    def __init__(self):
        self.artifact = {"numeric_columns": ["A_Car_Length"]}
        self.calls = 0

    def predict(self, rows):
        self.calls += 1
        return ProviderResult([0.251 for _ in rows], self.name, [])


class FakeVertexProvider:
    name = "Vertex AI"
    enabled = True

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def predict(self, rows, numeric_columns):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result or ProviderResult([0.241 for _ in rows], self.name, [])


class ProviderRouterTest(unittest.TestCase):
    def make_router(self, vertex):
        router = ProviderRouter(Path("missing-model.pkl"))
        router.local = FakeLocalProvider()
        router.vertex = vertex
        return router

    def test_vertex_success_does_not_use_local_fallback(self):
        vertex = FakeVertexProvider()
        router = self.make_router(vertex)

        result = router.predict([{"A_Car_Length": 20.0}])

        self.assertEqual(result.provider, "Vertex AI")
        self.assertEqual(result.values, [0.241])
        self.assertEqual(vertex.calls, 1)
        self.assertEqual(router.local.calls, 0)

    def test_vertex_error_falls_back_and_opens_circuit_for_60_seconds(self):
        vertex = FakeVertexProvider(error=RuntimeError("endpoint timed out"))
        router = self.make_router(vertex)

        with patch("web.cfa_service.providers.time.monotonic", side_effect=[10.0, 10.0, 11.0]):
            first = router.predict([{"A_Car_Length": 20.0}])
            second = router.predict([{"A_Car_Length": 20.0}])

        self.assertEqual(first.provider, "Local RandomForest")
        self.assertIn("Vertex AI unavailable", first.warnings[0])
        self.assertIn("temporarily paused", second.warnings[0])
        self.assertEqual(vertex.calls, 1)
        self.assertEqual(router.local.calls, 2)
        self.assertEqual(router._vertex_retry_after, 70.0)


class VertexProviderTest(unittest.TestCase):
    def make_provider(self):
        environment = {
            "PARAGON_PROVIDER": "vertex",
            "VERTEX_PROJECT_ID": "test-project",
            "VERTEX_LOCATION": "us-central1",
            "VERTEX_ENDPOINT_ID": "123456",
        }
        with patch.dict(os.environ, environment, clear=True):
            return VertexProvider()

    def response_mock(self, payload):
        response = MagicMock()
        response.json.return_value = payload
        return response

    def test_rest_request_uses_numeric_schema_and_parses_prediction(self):
        provider = self.make_provider()
        response = self.response_mock({"predictions": [{"value": 0.2395}]})

        with (
            patch.object(provider, "_access_token", return_value="test-token"),
            patch("web.cfa_service.providers.httpx.post", return_value=response) as post,
        ):
            result = provider.predict(
                [{"A_Car_Length": "21.5", "CarRear": "Fastback"}],
                ["A_Car_Length"],
            )

        self.assertEqual(result.values, [0.2395])
        self.assertEqual(
            post.call_args.kwargs["json"],
            {"instances": [{"A_Car_Length": "21.5"}]},
        )
        self.assertEqual(post.call_args.kwargs["timeout"], 8)
        response.raise_for_status.assert_called_once()

    def test_nested_automl_predictions_preserve_row_order(self):
        provider = self.make_provider()
        response = self.response_mock(
            {
                "predictions": [
                    [{"value": 0.2395}],
                    [{"value": 0.241}],
                ]
            }
        )

        with (
            patch.object(provider, "_access_token", return_value="test-token"),
            patch("web.cfa_service.providers.httpx.post", return_value=response),
        ):
            result = provider.predict(
                [
                    {"A_Car_Length": 21.5},
                    {"A_Car_Length": 22.0},
                ],
                ["A_Car_Length"],
            )

        self.assertEqual(result.values, [0.2395, 0.241])

    def test_timeout_is_reported_as_vertex_request_failure(self):
        provider = self.make_provider()

        with (
            patch.object(provider, "_access_token", return_value="test-token"),
            patch(
                "web.cfa_service.providers.httpx.post",
                side_effect=TimeoutError("timed out"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Vertex AI request failed"):
                provider.predict([{"A_Car_Length": 21.5}], ["A_Car_Length"])

    def test_http_error_includes_vertex_message(self):
        provider = self.make_provider()
        response = httpx.Response(
            400,
            json={"error": {"message": "Required feature A_Car_Length was missing."}},
            request=httpx.Request("POST", "https://example.test/predict"),
        )

        with (
            patch.object(provider, "_access_token", return_value="test-token"),
            patch("web.cfa_service.providers.httpx.post", return_value=response),
        ):
            with self.assertRaisesRegex(RuntimeError, "Required feature A_Car_Length"):
                provider.predict([{"A_Car_Length": 21.5}], ["A_Car_Length"])

    def test_invalid_response_shape_is_rejected(self):
        provider = self.make_provider()

        for payload in (
            {"predictions": []},
            {"predictions": [{"unexpected": "value"}]},
            {"predictions": [[]]},
            {"predictions": [[{"value": 0.2}, {"value": 0.3}]]},
            {"predictions": [{"value": None}]},
            {"predictions": [{"value": True}]},
            {"predictions": [{"value": "NaN"}]},
        ):
            with self.subTest(payload=payload):
                response = self.response_mock(payload)
                with (
                    patch.object(provider, "_access_token", return_value="test-token"),
                    patch("web.cfa_service.providers.httpx.post", return_value=response),
                ):
                    with self.assertRaisesRegex(RuntimeError, "response schema"):
                        provider.predict([{"A_Car_Length": 21.5}], ["A_Car_Length"])


if __name__ == "__main__":
    unittest.main()
