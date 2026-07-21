"""R0 재현 + 평가 하네스.

핸드오프 §6/§9: 로버스트 치수 6개 -> 선형+2차항 회귀. 클린 test MAE 4.4%, R2 0.82 목표.
평가는 [열화조건] x [차종] 격자로. 전역 R2는 차종 분리 효과를 감추므로 반드시 쪼개 본다.

  python train_r0.py
"""
from __future__ import annotations

import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/kwy00/qi")
import cd_common as C

CONDS = ["clean", "noise", "sparse", "glass", "one-side",
         "phone", "phone+mirror", "phone+mirror(est)"]
CLASSES = ["Fastback", "Estate", "Notchback"]


def make_variants(p, rng):
    """클린 클라우드 -> 조건별 열화 클라우드."""
    phone = C.degrade_phone(p, rng)
    return {
        "clean": p,
        "noise": C.degrade_noise(p, rng),
        "sparse": C.degrade_sparse(p, rng, 4000),
        "glass": C.degrade_glass(p, rng),
        "one-side": C.degrade_oneside(p),
        "phone": phone,
        "phone+mirror": C.mirror_complete(phone, 0.0),                       # 정답 평면
        "phone+mirror(est)": C.mirror_complete(phone, C.estimate_symmetry_plane(phone)),
    }


def _train_row(args):
    i, key, path = args
    p = C.safe_load_ndarray(path).astype(np.float64)
    return C.raw_feats(p), C.robust_feats(p)


def _test_row(args):
    i, key, path = args
    p = C.safe_load_ndarray(path).astype(np.float64)
    rng = np.random.default_rng(1000 + i)
    v = make_variants(p, rng)
    return ({c: C.raw_feats(v[c]) for c in CONDS},
            {c: C.robust_feats(v[c]) for c in CONDS})


# ----------------------------------------------------------------------------
def fit(X, y):
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0)
    sd[sd < 1e-9] = 1.0
    cmean = float(y.mean())
    D = design(X, mu, sd)
    coef, *_ = np.linalg.lstsq(D, y - cmean, rcond=None)
    return {"coef": coef, "mu": mu, "sd": sd, "cmean": cmean}


def design(X, mu, sd):
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)   # 결측 -> 학습 평균
    return np.column_stack([np.ones(len(Z)), Z, Z ** 2])    # 선형 + 2차항


def predict(m, X):
    return design(X, m["mu"], m["sd"]) @ m["coef"] + m["cmean"]


def mae_pct(yh, y):
    return float(np.mean(np.abs(yh - y) / y) * 100)


def r2(yh, y):
    """부분집합의 자체 평균을 기준선으로 하는 R2."""
    sse = float(np.sum((yh - y) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1 - sse / sst if sst > 0 else float("nan")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    C.check_integrity()
    files, cd, sp = C.file_index(), C.drag_table(), C.splits()

    tr, te = sp["train"], sp["test"]
    y_tr = np.array([cd[k] for k in tr])
    y_te = np.array([cd[k] for k in te])
    cls_te = np.array([C.body_type(k) for k in te])
    cls_tr = np.array([C.body_type(k) for k in tr])

    print(f"train {len(tr)} / test {len(te)}   피처 계산 중...", flush=True)
    with ProcessPoolExecutor(max_workers=16) as ex:
        TR = list(ex.map(_train_row, [(i, k, files[k]) for i, k in enumerate(tr)], chunksize=32))
        TE = list(ex.map(_test_row, [(i, k, files[k]) for i, k in enumerate(te)], chunksize=8))

    Xtr = {"v1 raw": np.vstack([a for a, _ in TR]), "v2 robust": np.vstack([b for _, b in TR])}
    Xte = {
        "v1 raw": {c: np.vstack([a[c] for a, _ in TE]) for c in CONDS},
        "v2 robust": {c: np.vstack([b[c] for _, b in TE]) for c in CONDS},
    }

    # ---- 기준선: 차종 평균만으로 예측 (형상 정보 0) -------------------------
    cmeans = {c: y_tr[cls_tr == c].mean() for c in CLASSES}
    yh_cls = np.array([cmeans[c] for c in cls_te])
    print("\n" + "=" * 88)
    print("[기준선] 형상을 전혀 안 보고 '차종 평균'만 예측했을 때 (test)")
    print("=" * 88)
    print(f"  전역 R2 = {r2(yh_cls, y_te):.4f}    MAE = {mae_pct(yh_cls, y_te):.2f}%")
    print(f"  -> R0의 전역 R2 중 이만큼은 '차종을 맞히는 것'만으로 얻어진다.")

    models = {}
    for name, X in Xtr.items():
        models[name] = fit(X, y_tr)

    for name in ("v1 raw", "v2 robust"):
        m = models[name]
        print("\n" + "=" * 88)
        print(f"[R0 / {name}] 열화조건 x 차종 격자  —  R2 (MAE%)")
        print("=" * 88)
        hdr = f"{'조건':<20}" + "".join(f"{c:>17}" for c in ["전체(All)"] + CLASSES)
        print(hdr)
        print("-" * 88)
        for cond in CONDS:
            yh = predict(m, Xte[name][cond])
            cells = [f"{r2(yh, y_te):+.3f} ({mae_pct(yh, y_te):.1f}%)"]
            for c in CLASSES:
                sel = cls_te == c
                cells.append(f"{r2(yh[sel], y_te[sel]):+.3f} ({mae_pct(yh[sel], y_te[sel]):.1f}%)")
            print(f"{cond:<20}" + "".join(f"{s:>17}" for s in cells))

    # 핸드오프 §6 표와 대조
    print("\n" + "=" * 88)
    print("[대조] 핸드오프 §6 표 (전역 R2)")
    print("=" * 88)
    ref = {"clean": (0.82, 0.81), "noise": (0.40, 0.77), "sparse": (0.79, 0.79),
           "glass": (0.82, 0.81), "one-side": (-5.5, 0.08), "phone": (-3.9, 0.33),
           "phone+mirror": (0.51, 0.80)}
    print(f"{'조건':<20}{'v1 재현':>12}{'v1 §6':>10}{'v2 재현':>14}{'v2 §6':>10}")
    print("-" * 88)
    for cond, (r1, r2ref) in ref.items():
        a = r2(predict(models["v1 raw"], Xte["v1 raw"][cond]), y_te)
        b = r2(predict(models["v2 robust"], Xte["v2 robust"][cond]), y_te)
        print(f"{cond:<20}{a:>12.3f}{r1:>10.2f}{b:>14.3f}{r2ref:>10.2f}")

    np.savez("/home/kwy00/qi/data/cd_model.npz", **models["v2 robust"])
    print("\n저장: data/cd_model.npz (coef, mu, sd, cmean) — v2 robust")
