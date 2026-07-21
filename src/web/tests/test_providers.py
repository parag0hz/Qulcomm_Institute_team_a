import json
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

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

    def response_context(self, payload):
        response = MagicMock()
        response.read.return_value = json.dumps(payload).encode("utf-8")
        context = MagicMock()
        context.__enter__.return_value = response
        return context

    def test_rest_request_uses_numeric_schema_and_parses_prediction(self):
        provider = self.make_provider()
        context = self.response_context({"predictions": [{"value": 0.2395}]})

        with (
            patch.object(provider, "_access_token", return_value="test-token"),
            patch("web.cfa_service.providers.request.urlopen", return_value=context) as urlopen,
        ):
            result = provider.predict(
                [{"A_Car_Length": "21.5", "CarRear": "Fastback"}],
                ["A_Car_Length"],
            )

        self.assertEqual(result.values, [0.2395])
        request_object = urlopen.call_args.args[0]
        self.assertEqual(
            json.loads(request_object.data.decode("utf-8")),
            {"instances": [{"A_Car_Length": 21.5}]},
        )
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 8)

    def test_timeout_is_reported_as_vertex_request_failure(self):
        provider = self.make_provider()

        with (
            patch.object(provider, "_access_token", return_value="test-token"),
            patch(
                "web.cfa_service.providers.request.urlopen",
                side_effect=TimeoutError("timed out"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Vertex AI request failed"):
                provider.predict([{"A_Car_Length": 21.5}], ["A_Car_Length"])

    def test_invalid_response_shape_is_rejected(self):
        provider = self.make_provider()

        for payload in (
            {"predictions": []},
            {"predictions": [{"unexpected": "value"}]},
        ):
            with self.subTest(payload=payload):
                context = self.response_context(payload)
                with (
                    patch.object(provider, "_access_token", return_value="test-token"),
                    patch("web.cfa_service.providers.request.urlopen", return_value=context),
                ):
                    with self.assertRaisesRegex(RuntimeError, "response schema"):
                        provider.predict([{"A_Car_Length": 21.5}], ["A_Car_Length"])


if __name__ == "__main__":
    unittest.main()
