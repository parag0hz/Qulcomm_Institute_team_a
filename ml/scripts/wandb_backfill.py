"""기존 실험 결과 -> wandb 일괄 백필. **한 번만 실행할 것** (재실행 시 중복 런 생성).

  wandb login   # 선행 필요 (또는 WANDB_MODE=offline 후 나중에 wandb sync)
  python scripts/wandb_backfill.py

올리는 것:
  1. outputs/<tag>.json (R2 계열 학습 런들) — config는 tag에서 파싱,
     짝이 되는 <tag>_pred.npz가 있으면 차종별 R²/MAE counts/순위 정확도를 재계산해 summary에 추가
  2. outputs/holdout_*.json (일반화 홀드아웃 3종)
  3. R0(치수 회귀) 기준선 — data/cd_model.npz로 test 예측을 재계산해 같은 지표로 기록
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import wandb

sys.path.insert(0, "/home/kwy00/qi")
import cd_common as C

OUT = "/home/kwy00/qi/outputs"
CLASSES = ["Fastback", "Estate", "Notchback"]


def r2(a, b):
    return 1 - float(((a - b) ** 2).sum()) / float(((b - b.mean()) ** 2).sum())


def rank_acc(a, b, n=200_000, seed=0):
    g = np.random.default_rng(seed)
    i, j = g.integers(0, len(b), (2, n))
    m = b[i] != b[j]
    return float(((a[i] < a[j]) == (b[i] < b[j]))[m].mean() * 100)


def full_metrics(yh, y, cls) -> dict:
    """전체 + 차종별 3지표."""
    out = {}
    for name, s in [("All", np.ones(len(y), bool))] + [(c, cls == c) for c in CLASSES]:
        a, b = yh[s], y[s]
        out[f"test/{name}_r2"] = r2(a, b)
        out[f"test/{name}_mae_counts"] = float(np.abs(a - b).mean() * 1000)
        out[f"test/{name}_mae_pct"] = float((np.abs(a - b) / b).mean() * 100)
        out[f"test/{name}_rank_acc"] = rank_acc(a, b)
    return out


def push(name: str, config: dict, summary: dict, tags: list[str]) -> None:
    run = wandb.init(project="cfa", name=name, config=config,
                     tags=["backfill"] + tags, reinit=True)
    run.summary.update(summary)
    run.finish()
    print(f"  ↑ {name}: {len(summary)}개 지표")


def main() -> None:
    # --- 1. R2 계열 학습 런 (tag.json + tag_pred.npz) --------------------------
    for jf in sorted(glob.glob(f"{OUT}/*.json")):
        tag = os.path.basename(jf)[:-5]
        if tag.startswith("holdout_"):
            continue
        r = json.load(open(jf))
        # tag 형식: <backbone>_n<npoints>_dims<0|1>_<scale>
        bk, np_, dm, sc = tag.split("_")
        config = {"backbone": bk, "npoints": int(np_[1:]), "dims": int(dm[4:]),
                  "scale": sc, "params": r["params"]}
        summary = {"val_r2_best": r["val_r2"]}
        pf = f"{OUT}/{tag}_pred.npz"
        if os.path.exists(pf):                      # 예측 보존된 런: 3지표 전부 재계산
            p = np.load(pf, allow_pickle=True)
            summary |= full_metrics(p["yh"].astype(np.float64),
                                    p["y"].astype(np.float64), p["cls"])
        else:                                        # json에 남은 R²/MAE%만
            summary |= {f"test/{c}_r2": v[0] for c, v in r["test"].items()}
            summary |= {f"test/{c}_mae_pct": v[1] for c, v in r["test"].items()}
        push(tag, config, summary, ["r2-ladder"])

    # --- 2. 홀드아웃 3종 -------------------------------------------------------
    for jf in sorted(glob.glob(f"{OUT}/holdout_*.json")):
        r = json.load(open(jf))
        mode = "body" if "_body_" in r["tag"] else "family"
        push(r["tag"],
             {"holdout": r["tag"].split("_", 2)[2], "mode": mode, "n_test": r["n_test"]},
             {k: v for k, v in r.items() if k not in ("tag",)}, ["holdout"])

    # --- 3. R0 기준선 (치수 6개 -> 선형+2차) ------------------------------------
    from train_r0 import design
    d = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
    m = np.load("/home/kwy00/qi/data/cd_model.npz")
    te = d["split"] == "test"
    yh = design(d["dims"][te].astype(np.float64), m["mu"], m["sd"]) @ m["coef"] + m["cmean"]
    push("r0_dims_quadratic", {"backbone": "r0", "params": int(m["coef"].shape[0])},
         full_metrics(yh, d["cd"][te].astype(np.float64), d["cls"][te]), ["r0-baseline"])

    print("\n백필 완료 — https://wandb.ai 의 'cfa' 프로젝트 확인")


if __name__ == "__main__":
    main()
