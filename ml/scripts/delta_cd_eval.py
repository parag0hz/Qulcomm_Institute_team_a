"""ΔCd 정확도 검증 — Phase 3(반사실 변형) 진입 전 필수 게이트.

반사실 탐색은 '닮은 두 형상의 작은 Cd 차이'를 모델이 분해할 수 있어야 성립한다.
절대 Cd의 R²가 높아도 ΔCd가 노이즈면 변형 제안은 무의미하다.

  python scripts/delta_cd_eval.py outputs/pointnet_n2048_dims0_meter_pred.npz
  python scripts/delta_cd_eval.py outputs/holdout_family_F_S_WWC_WM_pred.npz

방법: 예측 npz의 설계들 안에서 복셀 점유 코사인 유사도(check_leakage.py와 동일한
격자 48x24x20, FPS-2048 기반) 기준 1-NN 쌍을 만들고, 쌍마다
실제 ΔCd(CSV) vs 예측 ΔCd(모델)를 비교한다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/kwy00/qi")

GRID = (48, 24, 20)
BINS = [(0, 5), (5, 15), (15, np.inf)]   # |ΔCd| drag counts


def voxelize(pts):
    """check_leakage.py와 동일: (M,N,3) -> (M, prod(GRID)) 점유 격자, 스케일 유지."""
    lo = pts.reshape(-1, 3).min(0)
    hi = pts.reshape(-1, 3).max(0)
    idx = ((pts - lo) / (hi - lo + 1e-9) * (np.array(GRID) - 1)).astype(np.int32)
    flat = idx[..., 0] * GRID[1] * GRID[2] + idx[..., 1] * GRID[2] + idx[..., 2]
    out = np.zeros((len(pts), int(np.prod(GRID))), dtype=np.float32)
    for i in range(len(pts)):
        out[i, flat[i]] = 1.0
    return out


def r2(a, b):
    return 1 - float(((a - b) ** 2).sum()) / float(((b - b.mean()) ** 2).sum())


def main(a):
    p = np.load(a.pred, allow_pickle=True)
    keys_p = np.array([str(k) for k in p["keys"]])
    yh, y = p["yh"].astype(np.float64), p["y"].astype(np.float64)

    d = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
    pos = {str(k): i for i, k in enumerate(d["keys"])}
    pts = d["pts"][[pos[k] for k in keys_p]]

    # --- 기하 최근접 쌍 (자기 제외, top-k, 대칭 중복 제거) ----------------------
    V = torch.from_numpy(voxelize(pts))
    Vn = V / V.norm(dim=1, keepdim=True)
    sim = Vn @ Vn.T
    sim.fill_diagonal_(-1)
    top = sim.topk(a.knn, dim=1)
    pairs, sims = [], []
    seen = set()
    for i in range(len(keys_p)):
        for r in range(a.knn):
            j = int(top.indices[i, r])
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            sims.append(float(top.values[i, r]))
    pairs = np.array(pairs)
    sims = np.array(sims)
    i, j = pairs[:, 0], pairs[:, 1]

    dt = (y[i] - y[j]) * 1000                    # 실제 ΔCd (drag counts)
    dp = (yh[i] - yh[j]) * 1000                  # 예측 ΔCd
    nz = dt != 0

    tag = os.path.basename(a.pred).replace("_pred.npz", "")
    name = f"delta_cd_{tag}"
    print(f"=== {name}  (knn={a.knn})")
    print(f"쌍 {len(pairs)}개 | 복셀 유사도 중앙값 {np.median(sims):.4f} | "
          f"|ΔCd| 중앙값 {np.median(np.abs(dt)):.1f} counts")

    res = {
        "n_pairs": len(pairs),
        "sim_median": float(np.median(sims)),
        "delta_r2": r2(dp, dt),
        "delta_mae_counts": float(np.abs(dp - dt).mean()),
        "sign_acc": float((np.sign(dp[nz]) == np.sign(dt[nz])).mean() * 100),
    }
    print(f"\nΔCd R² = {res['delta_r2']:+.3f}   Δ MAE = {res['delta_mae_counts']:.1f} counts   "
          f"부호 정확도 = {res['sign_acc']:.1f}%")

    print(f"\n{'|ΔCd| 구간':<16}{'쌍':>6}{'부호 정확도':>12}{'Δ MAE':>10}{'ΔCd R²':>10}")
    print("-" * 56)
    for lo, hi in BINS:
        m = (np.abs(dt) >= lo) & (np.abs(dt) < hi) & nz
        label = f"{lo}~{hi if np.isfinite(hi) else ''}+ counts".replace("~+", "+ ")
        if m.sum() < 2:
            print(f"{label:<16}{m.sum():>6}{'—':>12}")
            continue
        sa = float((np.sign(dp[m]) == np.sign(dt[m])).mean() * 100)
        mae = float(np.abs(dp[m] - dt[m]).mean())
        rr = r2(dp[m], dt[m])
        res[f"bin_{lo}_{'inf' if not np.isfinite(hi) else int(hi)}"] = {
            "n": int(m.sum()), "sign_acc": sa, "delta_mae": mae}
        print(f"{label:<16}{m.sum():>6}{sa:>11.1f}%{mae:>10.1f}{rr:>+10.3f}")

    if a.wandb:
        try:
            import wandb
            run = wandb.init(project="cfa", name=name, tags=["delta-cd"],
                             config={"pred": a.pred, "knn": a.knn, "grid": GRID})
            flat = {k: v for k, v in res.items() if not isinstance(v, dict)}
            for bk, bv in ((k, v) for k, v in res.items() if isinstance(v, dict)):
                flat |= {f"{bk}/{kk}": vv for kk, vv in bv.items()}
            run.summary.update(flat)
            run.finish()
        except Exception as e:
            print(f"  (wandb 비활성: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--knn", type=int, default=1)
    ap.add_argument("--wandb", type=int, default=1)
    main(ap.parse_args())
