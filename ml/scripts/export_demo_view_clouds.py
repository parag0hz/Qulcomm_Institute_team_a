#!/usr/bin/env python3
"""데모 홀드아웃 5대의 **FPS 순서를 보존한** 점군을 뽑는다.

  python ml/scripts/export_demo_view_clouds.py               # 기본 4096점
  python ml/scripts/export_demo_view_clouds.py --points 8192

FPS(farthest point sampling)는 탐욕 알고리즘이라, 선택된 순서를 그대로 두면
**앞에서 K개를 자른 것이 그 자체로 유효한 FPS-K 샘플**이 된다. 그래서 배열
하나만 있으면 1 → 512 → 1024 → 2048 → 4096 을 슬라이싱으로 전부 만들 수 있고,
데모 페이지에서 점을 점점 늘려가며 형상이 드러나는 장면을 보여줄 수 있다.

추론도 같은 배열의 앞 N개로 돌린다. 학습 조건은 2048점이므로 그 지점이
기준선이고, 나머지는 "점이 몇 개면 항력을 알 수 있는가"를 보여주는 실험이다.

출력: ml/models/demo_clouds_view.npz
  pts  : (5, N, 3) float32 — 미터 스케일 원본 좌표, FPS 선택 순서 유지
  meta : JSON 문자열 — id / body_type / true_cd
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_HOLDOUT = REPO / "ml" / "data" / "demo_holdout.json"
DEFAULT_OUT = REPO / "ml" / "models" / "demo_clouds_view.npz"


def fps_torch(points: np.ndarray, k: int) -> np.ndarray | None:
    """GPU가 있으면 훨씬 빠르다. precompute_fps.fps_gpu와 동일한 알고리즘."""

    try:
        import torch
    except ImportError:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return None  # CPU라면 numpy 쪽이 오버헤드가 적다

    with torch.no_grad():
        x = torch.from_numpy(points).to(device)
        n = x.shape[0]
        idx = torch.zeros(k, dtype=torch.long, device=device)
        dist = torch.full((n,), 1e10, device=device)
        far = torch.zeros((), dtype=torch.long, device=device)
        for i in range(k):
            idx[i] = far
            dist = torch.minimum(dist, ((x - x[far]) ** 2).sum(-1))
            far = dist.argmax()
        return x[idx].cpu().numpy().astype(np.float32)


def fps_numpy(points: np.ndarray, k: int) -> np.ndarray:
    """시작점 인덱스 0으로 고정 — 매번 같은 결과가 나온다."""

    n = len(points)
    idx = np.zeros(k, dtype=np.int64)
    dist = np.full(n, np.inf)
    far = 0
    for i in range(k):
        idx[i] = far
        diff = points - points[far]
        dist = np.minimum(dist, np.einsum("ij,ij->i", diff, diff))
        far = int(dist.argmax())
    return points[idx].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=4096, help="차량당 최대 점 개수")
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/home/kwy00/qi/data/fps4096.npz"),
        help="이미 만들어둔 FPS 캐시가 있으면 그대로 쓴다(학습 입력과 동일 보장)",
    )
    args = parser.parse_args()

    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    items = holdout["items"]
    wanted = [item["id"] for item in items]

    clouds: dict[str, np.ndarray] = {}
    source = ""

    if args.cache.is_file():
        with np.load(args.cache, allow_pickle=False) as bundle:
            keys = [str(k) for k in bundle["keys"]]
            cached = bundle["pts"]
            index = {k: i for i, k in enumerate(keys)}
            for design in wanted:
                if design in index:
                    clouds[design] = np.asarray(cached[index[design]], dtype=np.float32)
        if clouds:
            source = f"cache {args.cache}"

    missing = [d for d in wanted if d not in clouds]
    if missing:
        print(f"캐시에 없는 {len(missing)}대는 원본에서 FPS를 계산한다 (시간이 걸린다)", flush=True)
        sys.path.insert(0, str(REPO / "ml"))
        import cd_common as C  # noqa: E402

        files = C.file_index()
        for design in missing:
            path = files.get(C.norm_id(design))
            if path is None:
                raise SystemExit(f"점군 파일을 찾지 못했다: {design}")
            raw = C.safe_load_ndarray(path).astype(np.float64)
            started = time.time()
            picked = fps_torch(raw.astype(np.float32), args.points)
            if picked is None:
                picked = fps_numpy(raw, args.points)
            clouds[design] = picked
            print(f"  {design:20} {len(raw):>7,} → {len(picked):>5,}점  ({time.time()-started:.0f}s)", flush=True)
        source = source or "raw .paddle_tensor"

    smallest = min(len(clouds[d]) for d in wanted)
    if smallest < args.points:
        print(f"주의: 확보된 최대 점 개수는 {smallest:,}이다 (요청 {args.points:,})")
    pts = np.stack([clouds[d][:smallest] for d in wanted]).astype(np.float32)

    meta = [
        {"id": i["id"], "body_type": i["body_type"], "true_cd": i["true_cd"]}
        for i in items
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, pts=pts, meta=json.dumps(meta, ensure_ascii=False))

    size_kb = args.out.stat().st_size / 1024
    print(f"\n저장 {args.out}  ({size_kb:,.0f} KB, 출처: {source})")
    print(f"  형태 {pts.shape}  — 앞에서 K개를 자르면 FPS-K 샘플이 된다")
    print(
        "  좌표 범위 "
        f"x[{pts[..., 0].min():.2f},{pts[..., 0].max():.2f}] "
        f"y[{pts[..., 1].min():.2f},{pts[..., 1].max():.2f}] "
        f"z[{pts[..., 2].min():.2f},{pts[..., 2].max():.2f}]  (미터 유지 확인)"
    )


if __name__ == "__main__":
    main()
