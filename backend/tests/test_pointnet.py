import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.cfa_service import pointnet
from backend.cfa_service.pointnet import (
    EXPECTED_POINTS,
    PLAUSIBLE_CD_RANGE,
    _guard,
    demo_predictions,
    pointnet_status,
    predict_cloud,
)


def car_like_cloud(n: int = EXPECTED_POINTS, scale: float = 1.0) -> np.ndarray:
    """차량 크기 상자 표면. 좌표 프레임만 그럴듯하면 되는 배선 검증용."""

    rng = np.random.default_rng(0)
    p = rng.random((n, 3))
    pts = np.stack([p[:, 0] * 4.8 - 0.93, (p[:, 1] - 0.5) * 2.04, p[:, 2] * 1.37], 1)
    return (pts * scale).astype(np.float32)


class GuardTest(unittest.TestCase):
    """범위 가드는 ONNX 없이도 검증 가능한 순수 로직이다."""

    def test_in_range_prediction_is_trusted_and_reported(self):
        low, high = PLAUSIBLE_CD_RANGE
        middle = (low + high) / 2
        guarded = _guard(middle, EXPECTED_POINTS)
        self.assertTrue(guarded.trusted)
        self.assertEqual(guarded.warnings, ())
        self.assertAlmostEqual(guarded.public_dict()["cd"], round(middle, 4))

    def test_out_of_range_prediction_hides_the_number(self):
        low, high = PLAUSIBLE_CD_RANGE
        for value in (low - 0.05, high + 0.05, -0.14, 50.0):
            with self.subTest(value=value):
                guarded = _guard(value, EXPECTED_POINTS)
                self.assertFalse(guarded.trusted)
                payload = guarded.public_dict()
                self.assertIsNone(payload["cd"], "분포 밖 예측은 수치를 노출하면 안 된다")
                self.assertEqual(payload["raw_cd"], round(value, 4))
                self.assertTrue(payload["warnings"])

    def test_unexpected_point_count_is_warned(self):
        low, high = PLAUSIBLE_CD_RANGE
        guarded = _guard((low + high) / 2, 512)
        self.assertTrue(guarded.trusted, "점 개수는 신뢰 여부가 아니라 경고 사유다")
        self.assertTrue(any("512" in w for w in guarded.warnings))


class RunnerTest(unittest.TestCase):
    def test_status_reports_asset_availability(self):
        status = pointnet_status()
        self.assertEqual(status["expected_points"], EXPECTED_POINTS)
        self.assertEqual(status["plausible_cd_range"], list(PLAUSIBLE_CD_RANGE))
        self.assertIn("available", status)

    def test_rejects_malformed_cloud_shapes(self):
        for bad in (np.zeros((10, 2), np.float32), np.zeros((4, 8, 3), np.float32)):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError):
                    predict_cloud(bad)

    def test_millimetre_input_is_caught_by_the_guard(self):
        """단위를 mm로 잘못 넣으면 스케일 의존 모델이 완전히 틀린 값을 낸다."""

        if not pointnet.runner().available:
            self.skipTest("serving ONNX not installed")
        guarded = predict_cloud(car_like_cloud(scale=1000.0))
        self.assertFalse(guarded.trusted)
        self.assertIsNone(guarded.public_dict()["cd"])


class DemoEndpointTest(unittest.TestCase):
    def test_reports_unavailable_without_assets(self):
        with patch.object(pointnet, "demo_clouds_path", return_value=None):
            payload = demo_predictions()
        self.assertFalse(payload["available"])
        self.assertEqual(payload["items"], [])
        self.assertIn("reason", payload)

    def test_runs_live_inference_over_the_holdout_bundle(self):
        if not pointnet.runner().available:
            self.skipTest("serving ONNX not installed")

        meta = [
            {"id": "F_S_WWS_WM_587", "body_type": "Fastback", "true_cd": 0.22847},
            {"id": "E_S_WWC_WM_210", "body_type": "Estate", "true_cd": 0.28923},
        ]
        clouds = np.stack([car_like_cloud() for _ in meta])

        with tempfile.TemporaryDirectory() as work:
            bundle = Path(work) / "demo_clouds.npz"
            np.savez_compressed(bundle, pts=clouds, meta=json.dumps(meta))
            with patch.object(pointnet, "demo_clouds_path", return_value=bundle):
                payload = demo_predictions()

        self.assertTrue(payload["available"])
        self.assertEqual(payload["point_count"], EXPECTED_POINTS)
        self.assertEqual(len(payload["items"]), len(meta))
        for item, source in zip(payload["items"], meta):
            self.assertEqual(item["id"], source["id"])
            self.assertEqual(item["true_cd"], round(source["true_cd"], 5))
            self.assertIn("error_counts", item)
            self.assertIn("raw_cd", item)
            # 신뢰 못 할 예측이면 cd는 감춰져야 한다.
            if not item["trusted"]:
                self.assertIsNone(item["cd"])


if __name__ == "__main__":
    unittest.main()
