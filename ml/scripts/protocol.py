"""공통 평가 프로토콜 — ML/DL이 반드시 이 모듈을 거쳐 데이터·분할·지표를 얻는다.

피드백 반영:
  #1  ML/DL 동일 데이터  : CSV(설계 파라미터) ∩ 포인트클라우드 교집합만 사용
  #2  K=5 rotating fold : 학습 3 / 검증 1 / 테스트 1 을 5세트로 회전
  #4  지표             : R², MAE, MSE  (+ drag counts MAE, 순위 정확도)
  #6  데모 홀드아웃      : data/demo_holdout.json 의 5대는 항상 제외

회전 방식 (fold 1~5):
  세트1  train 1,2,3 | val 4 | test 5
  세트2  train 2,3,4 | val 5 | test 1
  세트3  train 3,4,5 | val 1 | test 2
  세트4  train 4,5,1 | val 2 | test 3
  세트5  train 5,1,2 | val 3 | test 4
→ 각 fold가 테스트 1회·검증 1회. 학습 60% / 검증 20% / 테스트 20%.

  python scripts/protocol.py      # 자체 검증 (분할 크기·차종 균형 출력)
"""
from __future__ import annotations

import csv as _csv
import json
import sys

import numpy as np

sys.path.insert(0, "/home/kwy00/qi")
import cd_common as C

CSV = "/home/kwy00/qi/data/DrivAerNet_ParametricData.csv"
FPS = "/home/kwy00/qi/data/fps2048.npz"
DEMO = "/home/kwy00/qi/data/demo_holdout.json"
AERO_COLS = {"Average Cd", "Std Cd", "Average Cl", "Std Cl",
             "Average Cl_f", "Std Cl_f", "Average Cl_r", "Std Cl_r"}
K = 5
SEED = 42
CLASSES = ["Fastback", "Estate", "Notchback"]


# ============================== 데이터 ==============================

def load_dataset(npoints: int = 2048, cache: str = FPS) -> dict:
    """교집합 - 데모홀드아웃 데이터셋. ML은 X를, DL은 pts를 쓴다 (같은 순서/같은 차)."""
    d = np.load(cache, allow_pickle=True)
    keys_all = np.array([str(k) for k in d["keys"]])
    pos = {k: i for i, k in enumerate(keys_all)}

    # 설계 파라미터 23개 (공력 컬럼 = 누수, 전부 제외)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rdr = _csv.DictReader(f)
        feat_names = [c for c in rdr.fieldnames if c != "Experiment" and c not in AERO_COLS]
        rows = []
        for r in rdr:
            p, i = C.norm_id(r["Experiment"])
            rows.append((f"{p}_{i}", [float(r[c]) for c in feat_names]))

    demo = set(json.load(open(DEMO))["ids"])

    sel_key, sel_pos, X = [], [], []
    for k, feats in rows:
        if k in pos and k not in demo:          # 교집합 ∧ 데모 제외
            sel_key.append(k); sel_pos.append(pos[k]); X.append(feats)

    sel_pos = np.array(sel_pos)
    order = np.argsort(sel_key)                  # ID 정렬 → 결정적 순서
    sel_pos, X = sel_pos[order], np.array(X, dtype=np.float32)[order]
    keys = np.array(sel_key)[order]

    return {
        "keys": keys,
        "X": X,                                            # (N, 23) 설계 파라미터
        "feat_names": feat_names,
        "pts": d["pts"][sel_pos][:, :npoints],             # (N, npoints, 3) 포인트클라우드
        "cd": d["cd"][sel_pos].astype(np.float64),         # (N,) 타깃
        "cls": d["cls"][sel_pos],                          # (N,) 차종
        "n_demo_excluded": len(demo),
    }


# ============================== 분할 ==============================

def make_folds(cls: np.ndarray, k: int = K, seed: int = SEED) -> np.ndarray:
    """차종 층화(stratified) k-fold 배정. 각 fold의 차종 비율이 전체와 같도록."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(cls), dtype=int)
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        idx = idx[rng.permutation(len(idx))]      # 고정 시드 셔플
        fold[idx] = np.arange(len(idx)) % k       # 라운드로빈 → 균등 배분
    return fold


def rotating_sets(k: int = K) -> list[dict]:
    """세트 s: train {s,s+1,s+2}, val {s+3}, test {s+4} (mod k)."""
    return [{"set": s + 1,
             "train": [(s + j) % k for j in range(k - 2)],
             "val": (s + k - 2) % k,
             "test": (s + k - 1) % k} for s in range(k)]


def split_indices(fold: np.ndarray, k: int = K) -> list[dict]:
    """각 세트의 (train/val/test) 인덱스 배열."""
    out = []
    for r in rotating_sets(k):
        out.append({
            "set": r["set"],
            "train": np.where(np.isin(fold, r["train"]))[0],
            "val": np.where(fold == r["val"])[0],
            "test": np.where(fold == r["test"])[0],
            "folds": r,
        })
    return out


# ============================== 지표 ==============================

def rank_acc(yh, y, n=200_000, seed=0) -> float:
    """무작위 두 설계 쌍에서 저항 낮은 쪽을 맞히는 비율(%). 50% = 동전던지기."""
    if len(y) < 2:
        return float("nan")
    g = np.random.default_rng(seed)
    i, j = g.integers(0, len(y), (2, n))
    m = i != j
    return float(np.mean((yh[i[m]] < yh[j[m]]) == (y[i[m]] < y[j[m]])) * 100)


def metrics(yh, y) -> dict:
    """#4 요구 지표: R², MAE, MSE (+ 도메인 단위 MAE, 순위 정확도)."""
    yh, y = np.asarray(yh, float), np.asarray(y, float)
    err = yh - y
    return {
        "R2": 1 - float((err ** 2).sum()) / float(((y - y.mean()) ** 2).sum()),
        "MAE": float(np.abs(err).mean()),
        "MSE": float((err ** 2).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAE_counts": float(np.abs(err).mean() * 1000),   # 1 count = 0.001 Cd
        "MAPE": float(np.mean(np.abs(err) / np.abs(y)) * 100),  # 평균 절대 백분율 오차(%)
        "rank_acc": rank_acc(yh, y),
        "n": int(len(y)),
    }


def evaluate(yh, y, cls) -> dict:
    """전체 + 차종별 지표."""
    res = {"All": metrics(yh, y)}
    for c in CLASSES:
        m = cls == c
        if m.sum() >= 2:
            res[c] = metrics(yh[m], y[m])
    return res


def aggregate(per_set: list[dict]) -> dict:
    """5세트 결과를 평균 ± 표준편차로 집계."""
    out = {}
    for grp in per_set[0]:
        out[grp] = {}
        for k in per_set[0][grp]:
            v = [s[grp][k] for s in per_set if grp in s]
            out[grp][k] = {"mean": float(np.mean(v)), "std": float(np.std(v))} if k != "n" \
                else int(np.mean(v))
    return out


# ============================== 자체 검증 ==============================

if __name__ == "__main__":
    ds = load_dataset()
    n = len(ds["keys"])
    print(f"=== 데이터셋 (교집합 − 데모 {ds['n_demo_excluded']}대) ===")
    print(f"  총 {n:,}대 · 파라미터 {ds['X'].shape[1]}개 · 포인트클라우드 {ds['pts'].shape[1:]}")
    u, c = np.unique(ds["cls"], return_counts=True)
    print("  차종:", "  ".join(f"{a} {b}" for a, b in zip(u, c)))
    print(f"  Cd  : {ds['cd'].mean():.4f} ± {ds['cd'].std():.4f}  [{ds['cd'].min():.3f}, {ds['cd'].max():.3f}]")

    fold = make_folds(ds["cls"])
    print(f"\n=== K={K} 층화 fold 배정 ===")
    print(f"{'fold':<6}{'전체':>7}" + "".join(f"{c:>11}" for c in CLASSES))
    for f in range(K):
        m = fold == f
        print(f"{f+1:<6}{m.sum():>7}" + "".join(f"{(m & (ds['cls']==c)).sum():>11}" for c in CLASSES))

    print(f"\n=== 회전 분할 5세트 ===")
    print(f"{'세트':<6}{'학습 fold':<14}{'검증':<6}{'테스트':<7}{'train':>7}{'val':>7}{'test':>7}")
    for s in split_indices(fold):
        r = s["folds"]
        tr = "".join(str(x + 1) for x in r["train"])
        print(f"{s['set']:<6}{tr:<14}{r['val']+1:<6}{r['test']+1:<7}"
              f"{len(s['train']):>7}{len(s['val']):>7}{len(s['test']):>7}")

    # 무결성: 각 fold가 테스트 정확히 1회, 검증 1회
    from collections import Counter
    te = Counter(s["folds"]["test"] for s in split_indices(fold))
    va = Counter(s["folds"]["val"] for s in split_indices(fold))
    assert all(v == 1 for v in te.values()) and len(te) == K, "테스트 배정 오류"
    assert all(v == 1 for v in va.values()) and len(va) == K, "검증 배정 오류"
    # 무결성: train/val/test 겹침 없음
    for s in split_indices(fold):
        assert len(set(s["train"]) & set(s["val"])) == 0
        assert len(set(s["train"]) & set(s["test"])) == 0
        assert len(set(s["val"]) & set(s["test"])) == 0
    print("\n✅ 무결성 검증 통과 (각 fold 테스트 1회·검증 1회, 세트 내 겹침 없음)")
