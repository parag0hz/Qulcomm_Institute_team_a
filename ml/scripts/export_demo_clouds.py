#!/usr/bin/env python3
"""데모 홀드아웃 5대의 FPS-2048 점군을 웹 서빙용으로 추출한다.

  python ml/scripts/export_demo_clouds.py

기본 경로는 fps2048.npz 캐시다. 그 캐시는 학습·평가가 실제로 먹은 배열이므로,
거기서 꺼내면 웹 데모의 입력이 학습 입력과 **바이트 단위로 동일**함이 보장된다.
캐시가 없으면 원본 .paddle_tensor에서 동일한 FPS로 다시 만든다.

출력: ml/models/demo_clouds.npz  (5 × 2048 × 3 float32 ≈ 120 KB)
  pts  : (5, 2048, 3) float32 — 미터 스케일 원본 좌표 (센터링하지 않음)
  meta : JSON 문자열 — id / body_type / true_cd / original_split
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = os.environ.get("QI_FPS_CACHE", "/home/kwy00/qi/data/fps2048.npz")
DEFAULT_HOLDOUT = REPO / "ml" / "data" / "demo_holdout.json"
DEFAULT_OUT = REPO / "ml" / "models" / "demo_clouds.npz"
K = 2048


def fps_numpy(points: np.ndarray, k: int) -> np.ndarray:
    """precompute_fps.fps_gpu와 동일한 알고리즘. 시작점 인덱스 0으로 결정적."""

    n = len(points)
    idx = np.zeros(k, dtype=np.int64)
    dist = np.full(n, 1e10, dtype=np.float64)
    far = 0
    for i in range(k):
        idx[i] = far
        dist = np.minimum(dist, ((points - points[far]) ** 2).sum(-1))
        far = int(dist.argmax())
    return points[idx]


def from_cache(cache: Path, wanted: list[str]) -> dict[str, np.ndarray]:
    """fps2048.npz에서 해당 설계의 점군을 꺼낸다 (학습이 먹은 그 배열)."""

    with np.load(cache, allow_pickle=False) as bundle:
        keys = [str(k) for k in bundle["keys"]]
        pts = bundle["pts"]
        index = {k: i for i, k in enumerate(keys)}
        found = {}
        for design in wanted:
            if design in index:
                found[design] = np.asarray(pts[index[design]], dtype=np.float32)
    return found


def from_raw(wanted: list[str]) -> dict[str, np.ndarray]:
    """캐시가 없을 때 원본 .paddle_tensor에서 FPS를 다시 계산한다."""

    sys.path.insert(0, str(REPO / "ml"))
    import cd_common as C  # noqa: E402

    files = C.file_index()
    found = {}
    for design in wanted:
        key = C.norm_id(design)
        path = files.get(key)
        if path is None:
            continue
        raw = C.safe_load_ndarray(path)
        found[design] = fps_numpy(raw.astype(np.float64), K).astype(np.float32)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(DEFAULT_CACHE))
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    items = holdout["items"]
    wanted = [item["id"] for item in items]

    if args.cache.is_file():
        clouds = from_cache(args.cache, wanted)
        source = f"cache {args.cache}"
    else:
        print(f"캐시 없음({args.cache}) → 원본에서 FPS 재계산", flush=True)
        clouds = from_raw(wanted)
        source = "raw .paddle_tensor"

    missing = [design for design in wanted if design not in clouds]
    if missing:
        raise SystemExit(f"점군을 찾지 못했다: {missing}")

    pts = np.stack([clouds[design] for design in wanted]).astype(np.float32)
    if pts.shape[1:] != (K, 3):
        raise SystemExit(f"예상 형태 (n,{K},3), 실제 {pts.shape}")

    meta = [
        {
            "id": item["id"],
            "body_type": item["body_type"],
            "true_cd": item["true_cd"],
            "original_split": item.get("original_split"),
        }
        for item in items
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, pts=pts, meta=json.dumps(meta, ensure_ascii=False))

    size_kb = args.out.stat().st_size / 1024
    print(f"저장 {args.out}  ({size_kb:.0f} KB, 출처: {source})")
    print(f"  형태 {pts.shape}")
    print(
        "  좌표 범위 "
        f"x[{pts[..., 0].min():.2f},{pts[..., 0].max():.2f}] "
        f"y[{pts[..., 1].min():.2f},{pts[..., 1].max():.2f}] "
        f"z[{pts[..., 2].min():.2f},{pts[..., 2].max():.2f}]  (미터 유지 확인)"
    )
    for item in meta:
        print(f"  {item['id']:20} {item['body_type']:10} true_cd={item['true_cd']:.5f}")


if __name__ == "__main__":
    main()
