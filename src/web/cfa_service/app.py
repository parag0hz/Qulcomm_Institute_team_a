"""FastAPI application for the Paragon vehicle design workspace."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import partial
import logging
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.concurrency import run_in_threadpool

from .config import load_env_file

load_env_file()

from .copilot import ask_copilot, copilot_status
from .predictor import (
    analyze_parameters,
    load_dataset_stats,
    load_parameter_schema,
    maybe_predict_parameters,
    model_status,
    optimize_parameters,
    predict_from_stl_points,
    provider_status,
    test_vertex_provider,
)
from .pointnet import demo_predictions, pointnet_status
from .schemas import CopilotRequest, DesignParameters, OptimizeRequest, VertexTestRequest
from .stl import parse_stl_bytes


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
FRONTEND_DIST = REPO_ROOT / "web" / "frontend" / "dist"
MODEL_ASSETS = APP_ROOT / "static" / "models"
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
LOGGER = logging.getLogger(__name__)

# macOS does not consistently register the binary glTF MIME type.
mimetypes.add_type("model/gltf-binary", ".glb")


def _error_payload(code: str, message: str, **details: object) -> dict[str, object]:
    return {"error": {"code": code, "message": message, **details}}


def _error_response(
    status_code: int,
    code: str,
    message: str,
    **details: object,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=_error_payload(code, message, **details),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load schemas eagerly; the model itself stays lazy so /docs remains fast.
    load_parameter_schema()
    yield


app = FastAPI(
    title="Paragon Vehicle Design API",
    version="2.0.0",
    description="Parametric aerodynamic prediction, design analysis, optimization, and copilot API.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    """Reject clearly oversized API bodies before parsing them into memory or disk."""

    if request.url.path.startswith("/api/"):
        raw_length = request.headers.get("content-length")
        if raw_length:
            try:
                content_length = int(raw_length)
            except ValueError:
                return _error_response(
                    400,
                    "invalid_content_length",
                    "Content-Length must be an integer.",
                )
            if content_length < 0:
                return _error_response(
                    400,
                    "invalid_content_length",
                    "Content-Length cannot be negative.",
                )

            content_type = request.headers.get("content-type", "").lower()
            if content_type.startswith("multipart/form-data"):
                limit = MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
            else:
                limit = MAX_JSON_BYTES
            if content_length > limit:
                return _error_response(
                    413,
                    "payload_too_large",
                    "Upload is too large. Limit is 32 MB."
                    if content_type.startswith("multipart/form-data")
                    else "JSON request is too large. Limit is 1 MB.",
                )
        elif request.headers.get("content-type", "").lower().startswith("application/json"):
            # Transfer-encoded JSON has no Content-Length. Read only through the
            # configured boundary and let Starlette replay the cached body.
            buffered = bytearray()
            async for chunk in request.stream():
                buffered.extend(chunk)
                if len(buffered) > MAX_JSON_BYTES:
                    return _error_response(
                        413,
                        "payload_too_large",
                        "JSON request is too large. Limit is 1 MB.",
                    )
            request._body = bytes(buffered)  # Starlette's cached-body contract.

    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        content = exc.detail
    else:
        codes = {
            400: "bad_request",
            404: "not_found",
            405: "method_not_allowed",
            413: "payload_too_large",
            503: "service_unavailable",
        }
        content = _error_payload(
            codes.get(exc.status_code, "request_error"),
            str(exc.detail),
        )
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled Paragon API error", exc_info=exc)
    return _error_response(500, "internal_error", "Internal server error.")


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    stats = load_dataset_stats()
    status = model_status()
    return {
        "product": "paragon",
        "name": "Paragon Vehicle Design Studio",
        "model_status": status["name"],
        "trained_model_connected": status["connected"],
        "model_metrics": status["metrics"],
        "providers": provider_status(),
        "pointnet": pointnet_status(),
        "copilot": copilot_status(),
        "input_schema": {
            "numeric_features": list(stats.feature_columns),
            "categorical_features": ["CarRear", "Wheels"],
        },
        "dataset": stats.public_dict(),
    }


@app.get("/api/demo/pointnet")
async def get_pointnet_demo() -> dict[str, Any]:
    """홀드아웃 차량에 대한 라이브 PointNet 추론.

    사전 계산된 값을 되돌려주는 게 아니라 요청마다 실제로 모델을 돌린다.
    추론은 CPU 바운드라 이벤트 루프를 막지 않도록 스레드풀로 넘긴다.
    """

    return await run_in_threadpool(demo_predictions)


@app.get("/api/parameters")
def get_parameters() -> dict[str, Any]:
    return load_parameter_schema()


@app.post("/api/predict/parameters")
def predict_parameters(payload: DesignParameters) -> dict[str, Any]:
    return maybe_predict_parameters(payload.model_dump())


@app.post("/api/analyze/parameters")
def analyze_design(payload: DesignParameters) -> dict[str, Any]:
    return analyze_parameters(payload.model_dump())


@app.post("/api/optimize/parameters")
def optimize_design(payload: OptimizeRequest) -> dict[str, Any]:
    return optimize_parameters(
        payload.parameters.model_dump(),
        payload.target_cd,
        payload.locked,
    )


@app.post("/api/copilot")
def copilot(payload: CopilotRequest) -> dict[str, Any]:
    return ask_copilot(
        payload.message,
        payload.parameters.model_dump(),
        [item.model_dump() for item in payload.history],
    )


@app.post("/api/providers/vertex/test")
def vertex_test(payload: VertexTestRequest) -> dict[str, Any]:
    try:
        parameters = payload.parameters.model_dump() if payload.parameters else None
        return test_vertex_provider(parameters)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=_error_payload(
                "vertex_unavailable",
                str(exc),
                fallback_provider="Local RandomForest",
            ),
        ) from exc


@app.post("/api/predict")
async def predict_stl(file: UploadFile = File(...)) -> dict[str, Any]:
    file_name = Path(file.filename or "").name
    if Path(file_name).suffix.lower() != ".stl":
        raise HTTPException(
            status_code=400,
            detail=_error_payload("invalid_file_type", "Only .stl uploads are supported."),
        )
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=_error_payload(
                "payload_too_large",
                "Upload is too large. Limit is 32 MB.",
            ),
        )
    try:
        cloud = await run_in_threadpool(parse_stl_bytes, payload)
        result = await run_in_threadpool(
            partial(
                predict_from_stl_points,
                cloud.points,
                triangle_count=cloud.triangle_count,
                source_format=cloud.source_format,
            )
        )
        result["file"] = {"name": file_name, "size_bytes": len(payload)}
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=_error_payload("invalid_stl", str(exc)),
        ) from exc


if MODEL_ASSETS.exists():
    app.mount("/static/models", StaticFiles(directory=MODEL_ASSETS), name="models")


@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa(full_path: str) -> FileResponse:
    if full_path == "api" or full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found.")
    if not FRONTEND_DIST.exists():
        raise HTTPException(
            status_code=503,
            detail="React frontend is not built. Run npm run build:web.",
        )
    requested = (FRONTEND_DIST / full_path).resolve()
    if full_path and FRONTEND_DIST.resolve() in requested.parents and requested.is_file():
        return FileResponse(requested)
    index_file = FRONTEND_DIST / "index.html"
    if not index_file.is_file():
        raise HTTPException(
            status_code=503,
            detail="React frontend is not built. Run npm run build:web.",
        )
    return FileResponse(index_file)


def main() -> None:
    import uvicorn

    uvicorn.run("web.cfa_service.app:app", host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
