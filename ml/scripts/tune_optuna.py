"""Optuna 하이퍼파라미터 최적화 (피드백 #4) — 탐색 과정을 눈에 보이게.

누수 방지 설계:
  · 탐색은 **세트1의 val fold로만** 목적함수를 계산한다. test는 절대 보지 않는다.
  · 최적 파라미터를 확정한 뒤, 그 값으로 **5개 fold 전부 재평가**해 test 성능을 보고한다.
  · 따라서 test는 최종 1회만 사용된다.

  python scripts/tune_optuna.py --models xgb lgbm pointnet
  python scripts/tune_optuna.py --models xgb --n-trials 100
결과: outputs/optuna_results.json, outputs/optuna_*.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/kwy00/qi")
sys.path.insert(0, "/home/kwy00/qi/scripts")
from protocol import load_dataset, make_folds, split_indices, evaluate, aggregate

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(_REPO, "outputs", "optuna_results.json")


def r2(yh, y):
    return 1 - float(((yh - y) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


# ============================== ML ==============================

def space_xgb(t):
    return dict(
        n_estimators=t.suggest_int("n_estimators", 100, 1200, step=50),
        max_depth=t.suggest_int("max_depth", 3, 12),
        learning_rate=t.suggest_float("learning_rate", 0.005, 0.3, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=t.suggest_int("min_child_weight", 1, 20),
        reg_alpha=t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


def space_lgbm(t):
    return dict(
        n_estimators=t.suggest_int("n_estimators", 100, 1200, step=50),
        num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
        learning_rate=t.suggest_float("learning_rate", 0.005, 0.3, log=True),
        feature_fraction=t.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=t.suggest_float("bagging_fraction", 0.5, 1.0),
        min_child_samples=t.suggest_int("min_child_samples", 5, 60),
        reg_alpha=t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


def make_ml(name, params):
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(objective="reg:squarederror", random_state=42,
                                n_jobs=-1, verbosity=0, **params)
    import lightgbm as lgb
    return lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **params)


def tune_ml(name, ds, sets, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = ds["X"], ds["cd"]
    s = sets[0]                                   # 세트1의 train으로 학습, val로 평가
    space = space_xgb if name == "xgb" else space_lgbm

    def obj(t):
        m = make_ml(name, space(t)).fit(X[s["train"]], y[s["train"]])
        return r2(m.predict(X[s["val"]]), y[s["val"]])

    st = optuna.create_study(direction="maximize", study_name=f"{name}",
                             sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"  [{name}] {n_trials}회 탐색 완료 ({time.time()-t0:.0f}s) "
          f"best val R² {st.best_value:+.4f}", flush=True)
    return st


# ============================== DL ==============================

def train_pointnet(ds, s, p, epochs, patience, eval_test=False):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from models_pc import PointNet

    torch.manual_seed(0); np.random.seed(0)
    pts, y, cls = ds["pts"], ds["cd"], ds["cls"]
    tr, va, te = s["train"], s["val"], s["test"]
    center = pts[tr].reshape(-1, 3).mean(0)
    P = (pts - center).astype(np.float32)
    ymu, ysd = y[tr].mean(), y[tr].std()
    yz = ((y - ymu) / ysd).astype(np.float32)

    net = PointNet(n_dims=0, emb=p["emb"]).cuda()
    for m in net.head.modules():                       # 드롭아웃만 교체 (모델 코드 불변)
        if isinstance(m, nn.Dropout):
            m.p = p["dropout"]
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    lossf = nn.SmoothL1Loss(beta=1.0)
    dl = DataLoader(TensorDataset(torch.from_numpy(P[tr]), torch.from_numpy(yz[tr])),
                    batch_size=p["bs"], shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    @torch.no_grad()
    def infer(idx):
        net.eval()
        o = [net(torch.from_numpy(P[idx[i:i+128]]).cuda()).cpu() for i in range(0, len(idx), 128)]
        return torch.cat(o).numpy() * ysd + ymu

    best, best_sd, bad = -9e9, None, 0
    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            lossf(net(xb.cuda(non_blocking=True)), yb.cuda(non_blocking=True)).backward()
            opt.step()
        sch.step()
        vr = r2(infer(va), y[va])
        if vr > best:
            best, bad = vr, 0
            best_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if not eval_test:
        return best
    net.load_state_dict(best_sd)
    return evaluate(infer(te), y[te], cls[te])


def tune_dl(ds, sets, n_trials, epochs, patience):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = sets[0]

    def obj(t):
        p = dict(
            lr=t.suggest_float("lr", 1e-4, 5e-3, log=True),
            weight_decay=t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            bs=t.suggest_categorical("bs", [16, 32, 64]),
            dropout=t.suggest_float("dropout", 0.0, 0.5, step=0.1),
            emb=t.suggest_categorical("emb", [512, 1024]),
        )
        return train_pointnet(ds, s, p, epochs, patience)

    st = optuna.create_study(direction="maximize", study_name="pointnet",
                             sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"  [pointnet] {n_trials}회 탐색 완료 ({time.time()-t0:.0f}s) "
          f"best val R² {st.best_value:+.4f}", flush=True)
    return st


# ============================== 최종 5-fold 평가 ==============================

def final_ml(name, ds, sets, params):
    X, y, cls = ds["X"], ds["cd"], ds["cls"]
    per = []
    for s in sets:
        m = make_ml(name, params).fit(X[s["train"]], y[s["train"]])
        per.append(evaluate(m.predict(X[s["test"]]), y[s["test"]], cls[s["test"]]))
    return aggregate(per)


def final_dl(ds, sets, params, epochs=120, patience=30):
    per = []
    for s in sets:
        per.append(train_pointnet(ds, s, params, epochs, patience, eval_test=True))
        print(f"    세트{s['set']} test R² {per[-1]['All']['R2']:+.4f}", flush=True)
    return aggregate(per)


# ============================== 시각화 ==============================

def plot_study(st, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import optuna

    BG, FG, SUB, ACC, CARD = "#141414", "#E4E1DB", "#B9B6B0", "#D98A3D", "#1C1C1C"
    plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB,
        "axes.edgecolor": "#3a3a3a"})

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    vals = [t.value for t in st.trials if t.value is not None]
    best = np.maximum.accumulate(vals)
    ax[0].plot(range(1, len(vals) + 1), vals, "o", ms=4, color=SUB, alpha=0.6, label="각 trial")
    ax[0].plot(range(1, len(best) + 1), best, "-", lw=2, color=ACC, label="누적 최고")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("val R²")
    ax[0].set_title(f"{name} — 최적화 히스토리", fontsize=12, color=FG, pad=10)
    ax[0].legend(facecolor=CARD, edgecolor="#3a3a3a", labelcolor=FG, fontsize=9)
    ax[0].spines[["top", "right"]].set_visible(False)

    try:
        imp = optuna.importance.get_param_importances(st)
        k = list(imp)[::-1]; v = [imp[x] for x in k]
        ax[1].barh(range(len(k)), v, color=ACC, height=0.6)
        ax[1].set_yticks(range(len(k))); ax[1].set_yticklabels(k, fontsize=9.5)
        ax[1].set_xlabel("중요도")
        for i, vv in enumerate(v):
            ax[1].text(vv + max(v) * 0.02, i, f"{vv:.2f}", va="center", fontsize=9, color=FG)
        ax[1].set_xlim(0, max(v) * 1.2)
    except Exception as e:
        ax[1].text(0.5, 0.5, f"중요도 계산 불가\n{e}", ha="center", color=SUB)
    ax[1].set_title(f"{name} — 파라미터 중요도", fontsize=12, color=FG, pad=10)
    ax[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    p = os.path.join(_REPO, "outputs", f"optuna_{name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    그림 저장: {p}")


# ============================== main ==============================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["xgb", "lgbm", "pointnet"])
    ap.add_argument("--n-trials", type=int, default=80)
    ap.add_argument("--dl-trials", type=int, default=25)
    ap.add_argument("--dl-search-epochs", type=int, default=60)
    a = ap.parse_args()

    ds = load_dataset()
    sets = split_indices(make_folds(ds["cls"]))
    print(f"데이터 {len(ds['keys']):,}대 · 탐색은 세트1의 val({len(sets[0]['val'])}대)로만, test 미사용\n")

    res = {}
    for name in a.models:
        print(f"=== {name} 탐색 ===")
        if name in ("xgb", "lgbm"):
            st = tune_ml(name, ds, sets, a.n_trials)
            agg = final_ml(name, ds, sets, st.best_params)
        else:
            st = tune_dl(ds, sets, a.dl_trials, a.dl_search_epochs, patience=15)
            print("  최적 파라미터로 5-fold 최종 학습:")
            agg = final_dl(ds, sets, st.best_params)
        plot_study(st, name)
        try:
            import optuna as _o
            imp = dict(_o.importance.get_param_importances(st))
        except Exception:
            imp = {}
        res[name] = {
            "best_params": st.best_params,
            "best_val_r2": st.best_value,
            "n_trials": len(st.trials),
            "tuned_5fold": agg,
            "history": [t.value for t in st.trials],
            "param_importance": imp,
            "trial_params": [t.params for t in st.trials],
        }
        A = agg["All"]
        print(f"  → 튜닝 후 5-fold test R² {A['R2']['mean']:+.4f}±{A['R2']['std']:.4f}  "
              f"MAE {A['MAE']['mean']:.5f}  MAPE {A.get('MAPE',{}).get('mean',float('nan')):.2f}%\n", flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"저장: {OUT_JSON}")
