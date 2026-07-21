"""PointNet(포인트클라우드) Cd 예측 서빙.

파라메트릭 대체모델과 달리 이 경로는 형상 자체를 입력으로 받는다. 학습 때
사용한 센터링 상수와 타깃 표준화 해제가 ONNX 그래프에 내장되어 있으므로,
서빙 코드는 **미터 스케일 원본 좌표**를 그대로 넣기만 하면 된다.

⚠️ 좌표를 밖에서 센터링하거나 unit-sphere로 정규화하면 예측이 망가진다.
   절대 치수(특히 높이)가 이 데이터셋의 최강 신호다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import json
import os
import threading

import numpy as np

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]

# 학습에 쓴 점 개수. ONNX는 동적 축이라 다른 값도 실행되지만 정확도는 이 값 기준이다.
EXPECTED_POINTS = 2048

# 학습 타깃 분포(ONNX에 내장된 상수): 평균 0.2559, 표준편차 0.02286.
# ±4σ 밖은 학습에서 사실상 관측되지 않은 영역이라 수치를 신뢰할 수 없다.
PLAUSIBLE_CD_RANGE: Tuple[float, float] = (0.16, 0.36)


def _first_existing(*candidates: Path) -> Path | None:
    for path in candidates:
        if path and path.is_file():
            return path
    return None


def onnx_path() -> Path | None:
    """서빙용 ONNX 위치. Docker 이미지와 로컬 체크아웃 양쪽을 지원한다."""

    override = os.environ.get("PARAGON_POINTNET_ONNX")
    return _first_existing(
        Path(override) if override else None,
        APP_ROOT / "models" / "pointnet_serving.onnx",          # 컨테이너 배치 위치
        REPO_ROOT.parent / "ml" / "models" / "pointnet_serving.onnx",  # 로컬 저장소
    )


def demo_clouds_path() -> Path | None:
    """데모 홀드아웃 점군 캐시 위치."""

    override = os.environ.get("PARAGON_DEMO_CLOUDS")
    return _first_existing(
        Path(override) if override else None,
        APP_ROOT / "models" / "demo_clouds.npz",
        REPO_ROOT.parent / "ml" / "models" / "demo_clouds.npz",
    )


@dataclass(frozen=True)
class CloudPrediction:
    cd: float
    trusted: bool
    warnings: Tuple[str, ...]

    def public_dict(self) -> Dict[str, object]:
        return {
            "cd": round(self.cd, 4) if self.trusted else None,
            "raw_cd": round(self.cd, 4),
            "trusted": self.trusted,
            "warnings": list(self.warnings),
        }


class PointNetRunner:
    """ONNX 세션을 지연 로드해 재사용한다. 스레드풀 라우트에서 동시 호출될 수 있다."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = model_path
        self._session = None
        self._lock = threading.Lock()

    @property
    def model_path(self) -> Path | None:
        return self._model_path or onnx_path()

    @property
    def available(self) -> bool:
        path = self.model_path
        if path is None:
            return False
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def session(self):
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is None:
                path = self.model_path
                if path is None:
                    raise FileNotFoundError(
                        "No PointNet ONNX found. Expected ml/models/pointnet_serving.onnx."
                    )
                import onnxruntime as ort

                self._session = ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
        return self._session

    def predict(self, clouds: np.ndarray) -> np.ndarray:
        """clouds: (B, N, 3) 미터 스케일 원본 좌표 → (B,) Cd."""

        batch = np.ascontiguousarray(clouds, dtype=np.float32)
        if batch.ndim == 2:
            batch = batch[None]
        if batch.ndim != 3 or batch.shape[-1] != 3:
            raise ValueError(f"Expected point clouds shaped (B, N, 3); received {batch.shape}.")
        return np.asarray(self.session().run(["cd"], {"points": batch})[0], dtype=np.float64)


_RUNNER = PointNetRunner()


def runner() -> PointNetRunner:
    return _RUNNER


def _guard(cd: float, n_points: int) -> CloudPrediction:
    warnings: List[str] = []
    low, high = PLAUSIBLE_CD_RANGE
    trusted = low <= cd <= high
    if not trusted:
        warnings.append(
            "Predicted Cd falls outside the range observed during training; "
            "the shape is likely out of distribution and the value is not reported."
        )
    if n_points != EXPECTED_POINTS:
        warnings.append(
            f"Model was trained on {EXPECTED_POINTS}-point clouds; received {n_points}."
        )
    return CloudPrediction(cd=float(cd), trusted=trusted, warnings=tuple(warnings))


def predict_cloud(points: Sequence[Sequence[float]] | np.ndarray) -> CloudPrediction:
    """단일 점군 → Cd. 분포 밖으로 판단되면 수치를 감춘다."""

    cloud = np.asarray(points, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[-1] != 3:
        raise ValueError(f"Expected a single cloud shaped (N, 3); received {cloud.shape}.")
    value = float(runner().predict(cloud[None])[0])
    return _guard(value, cloud.shape[0])


def load_demo_clouds() -> Tuple[np.ndarray, List[Dict[str, object]]] | None:
    """데모 홀드아웃 점군과 메타데이터. 파일이 없으면 None."""

    path = demo_clouds_path()
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as bundle:
        clouds = np.asarray(bundle["pts"], dtype=np.float32)
        meta = json.loads(str(bundle["meta"]))
    return clouds, meta


def demo_predictions() -> Dict[str, object]:
    """학습에서 영구 제외된 홀드아웃 차량에 대해 라이브 추론을 돌린다.

    사전 계산된 숫자가 아니라 매 호출마다 실제로 모델을 실행한다 — 이 경로가
    시연에서 정직성을 보증하는 부분이다.
    """

    bundle = load_demo_clouds()
    if bundle is None or not runner().available:
        return {
            "available": False,
            "reason": (
                "PointNet demo assets are not installed "
                "(need ml/models/pointnet_serving.onnx and demo_clouds.npz)."
            ),
            "items": [],
        }

    clouds, meta = bundle
    values = runner().predict(clouds)

    items: List[Dict[str, object]] = []
    errors: List[float] = []
    for index, entry in enumerate(meta):
        predicted = float(values[index])
        guarded = _guard(predicted, clouds.shape[1])
        true_cd = entry.get("true_cd")
        record: Dict[str, object] = {
            "id": entry.get("id"),
            "body_type": entry.get("body_type"),
            "true_cd": round(float(true_cd), 5) if true_cd is not None else None,
            **guarded.public_dict(),
        }
        if true_cd is not None:
            error_counts = abs(predicted - float(true_cd)) * 1000.0
            record["error_counts"] = round(error_counts, 2)
            errors.append(error_counts)
        items.append(record)

    return {
        "available": True,
        "point_count": int(clouds.shape[1]),
        "mean_error_counts": round(float(np.mean(errors)), 2) if errors else None,
        "items": items,
        "note": (
            "These designs were permanently excluded from training and validation. "
            "Predictions run live on every request."
        ),
    }


def pointnet_status() -> Dict[str, object]:
    active = runner()
    path = active.model_path
    clouds = demo_clouds_path()
    return {
        "available": active.available,
        "model": "PointNet (point cloud)" if path else None,
        "expected_points": EXPECTED_POINTS,
        "plausible_cd_range": list(PLAUSIBLE_CD_RANGE),
        "demo_clouds": bool(clouds),
    }
