"""Prediction providers for Paragon's parametric workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import subprocess
import time
from typing import Mapping, Sequence
from urllib import request


@dataclass
class ProviderResult:
    values: list[float]
    provider: str
    warnings: list[str]


class LocalProvider:
    name = "Local RandomForest"

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._artifact = None

    @property
    def artifact(self):
        if self._artifact is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    "No trained parametric model found. Run web/models/train_parametric_baseline.py first."
                )
            with self.model_path.open("rb") as handle:
                self._artifact = pickle.load(handle)
        return self._artifact

    def predict(self, rows: Sequence[Mapping[str, object]]) -> ProviderResult:
        artifact = self.artifact
        feature_columns = artifact["feature_columns"]
        try:
            import pandas as pd

            model_input = pd.DataFrame(rows, columns=feature_columns)
        except ImportError:
            model_input = [[row[column] for column in feature_columns] for row in rows]
        values = [float(value) for value in artifact["model"].predict(model_input)]
        return ProviderResult(values, self.name, [])

    def status(self) -> dict:
        artifact = self.artifact
        return {
            "available": True,
            "name": self.name,
            "model_name": artifact.get("model_name", "parametric baseline"),
            "metrics": artifact.get("metrics", {}),
            "feature_columns": artifact.get("feature_columns", []),
            "numeric_columns": artifact.get("numeric_columns", []),
            "categorical_columns": artifact.get("categorical_columns", []),
        }


class VertexProvider:
    """Small REST provider. It is disabled until endpoint env vars are configured."""

    name = "Vertex AI"

    def __init__(self):
        self.project = os.getenv("VERTEX_PROJECT_ID", "")
        self.location = os.getenv("VERTEX_LOCATION", "us-central1")
        self.endpoint = os.getenv("VERTEX_ENDPOINT_ID", "")
        self.enabled = os.getenv("PARAGON_PROVIDER", "local").lower() == "vertex"

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.project and self.endpoint)

    def predict(self, rows: Sequence[Mapping[str, object]], numeric_columns: Sequence[str]) -> ProviderResult:
        if not self.available:
            raise RuntimeError("Vertex AI endpoint is not configured.")
        token = self._access_token()
        url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project}"
            f"/locations/{self.location}/endpoints/{self.endpoint}:predict"
        )
        instances = [{column: float(row[column]) for column in numeric_columns} for row in rows]
        payload = json.dumps({"instances": instances}).encode("utf-8")
        req = request.Request(
            url,
            data=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=8) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Vertex AI request failed: {exc}") from exc
        predictions = result.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(rows):
            raise RuntimeError("Vertex AI response schema did not match the request.")
        values = []
        try:
            for item in predictions:
                if isinstance(item, dict):
                    item = item.get("value", item.get("prediction"))
                if isinstance(item, list):
                    item = item[0]
                values.append(float(item))
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("Vertex AI response schema did not match the request.") from exc
        return ProviderResult(values, self.name, [])

    def test(self, row: Mapping[str, object], numeric_columns: Sequence[str]) -> dict[str, object]:
        started = time.monotonic()
        result = self.predict([row], numeric_columns)
        return {
            "connected": True,
            "provider": self.name,
            "prediction": round(result.values[0], 5),
            "latency_ms": round((time.monotonic() - started) * 1000),
            "endpoint_id": self.endpoint,
            "location": self.location,
        }

    def _access_token(self) -> str:
        try:
            import google.auth
            from google.auth.transport.requests import Request

            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            credentials.refresh(Request())
            return str(credentials.token)
        except ImportError:
            try:
                configured_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                adc_path = Path(configured_path) if configured_path else Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
                if not adc_path.exists():
                    raise RuntimeError("Application Default Credentials file was not found.")
                completed = subprocess.run(
                    ["gcloud", "auth", "application-default", "print-access-token"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                token = completed.stdout.strip()
                if not token:
                    raise RuntimeError("gcloud returned an empty access token.")
                return token
            except Exception as exc:
                raise RuntimeError(
                    "Vertex authentication unavailable. Install google-auth or run gcloud auth application-default login."
                ) from exc

    def status(self) -> dict:
        return {
            "available": self.available,
            "enabled": self.enabled,
            "name": self.name,
            "project": self.project or None,
            "location": self.location,
            "endpoint_id": self.endpoint or None,
            "schema": "23 numeric features (AutoML CSV)",
        }


class ProviderRouter:
    def __init__(self, model_path: Path):
        self.local = LocalProvider(model_path)
        self.vertex = VertexProvider()
        self._vertex_retry_after = 0.0
        self._vertex_last_error = ""

    def predict(self, rows: Sequence[Mapping[str, object]]) -> ProviderResult:
        artifact = self.local.artifact
        if self.vertex.enabled:
            if time.monotonic() < self._vertex_retry_after:
                fallback = self.local.predict(rows)
                fallback.warnings.append(
                    f"Vertex AI temporarily paused after a connection error; local fallback used. {self._vertex_last_error}"
                )
                return fallback
            try:
                return self.vertex.predict(rows, artifact.get("numeric_columns", []))
            except Exception as exc:
                self._vertex_last_error = str(exc)
                self._vertex_retry_after = time.monotonic() + 60
                fallback = self.local.predict(rows)
                fallback.warnings.append(f"Vertex AI unavailable; local fallback used. {exc}")
                return fallback
        return self.local.predict(rows)

    def status(self) -> dict:
        return {"active": self.vertex.name if self.vertex.enabled else self.local.name,
                "local": self.local.status(), "vertex": self.vertex.status()}
