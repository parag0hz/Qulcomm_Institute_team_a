"""저장된 test 예측을 세 가지 렌즈로 평가.

  R2            평균 찍기보다 나은가        (부분집합 자신의 평균이 기준선)
  MAE counts    물리 단위로 얼마나 틀리나   (1 count = 0.001 Cd)
  순위 정확도    두 설계 중 저항 낮은 쪽 맞히기 (= 제품이 실제로 하는 일)

  python eval_metrics.py outputs/pointnet_n2048_dims0_meter_pred.npz
  python eval_metrics.py --r0                 # 치수 모델을 같은 렌즈로
"""
from __future__ import annotations
import sys
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

sys.path.insert(0, "/home/kwy00/qi")
import cd_common as C

CLASSES = ["Fastback", "Estate", "Notchback"]


def r2(a, b):
    return 1 - float(((a - b) ** 2).sum()) / float(((b - b.mean()) ** 2).sum())


def rank_acc(a, b, n=200_000, seed=0):
    """무작위 두 설계 쌍에서 저항이 낮은 쪽을 맞히는 비율. 동전던지기 = 50%."""
    if len(b) < 2:
        return float("nan")
    g = np.random.default_rng(seed)
    i, j = g.integers(0, len(b), (2, n))
    m = i != j
    return float(np.mean((a[i[m]] < a[j[m]]) == (b[i[m]] < b[j[m]])) * 100)


def twin_mask(keys):
    """test 설계마다: 같은 차종letter + 같은 인덱스인 형제가 train에 있는가."""
    files, sp = C.file_index(), C.splits()
    tr = set(sp["train"])
    by = defaultdict(list)
    for (p, i) in files:
        by[(p.split("_")[0], i)].append(p)
    out = []
    for k in keys:
        p, i = C.norm_id(str(k))
        sibs = [q for q in by[(p.split("_")[0], i)] if q != p and (q, i) in tr]
        out.append(bool(sibs))
    return np.array(out)


def table(name, yh, y, cls, keys):
    tw = twin_mask(keys)
    print(f"\n{'='*92}\n{name}\n{'='*92}")
    print(f"{'':<24}{'n':>6}{'R2':>9}{'MAE%':>8}{'MAE(counts)':>13}{'Spearman':>10}{'순위 정확도':>13}")
    print("-" * 92)
    groups = [("전체", np.ones(len(y), bool))] + [(c, cls == c) for c in CLASSES]
    groups += [("--- 쌍둥이 있음", tw), ("--- 쌍둥이 없음(고립)", ~tw)]
    for nm, s in groups:
        if s.sum() < 2:
            continue
        a, b = yh[s], y[s]
        print(f"{nm:<24}{s.sum():>6}{r2(a,b):>+9.3f}{np.mean(np.abs(a-b)/b)*100:>8.2f}"
              f"{np.mean(np.abs(a-b))*1000:>13.1f}{spearmanr(a,b).statistic:>+10.3f}"
              f"{rank_acc(a,b):>12.1f}%")


if __name__ == "__main__":
    if "--r0" in sys.argv:
        from train_r0 import design
        d = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
        m = np.load("/home/kwy00/qi/data/cd_model.npz")
        te = d["split"] == "test"
        yh = design(d["dims"][te].astype(np.float64), m["mu"], m["sd"]) @ m["coef"] + m["cmean"]
        table("R0 — 로버스트 치수 6개 -> 선형+2차항", yh, d["cd"][te].astype(np.float64),
              d["cls"][te], d["keys"][te])
    else:
        p = np.load(sys.argv[1], allow_pickle=True)
        table(sys.argv[1].split("/")[-1].replace("_pred.npz", ""),
              p["yh"].astype(np.float64), p["y"].astype(np.float64), p["cls"], p["keys"])
