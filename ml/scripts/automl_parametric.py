"""논문 §5.1.2 재현 — 파라메트릭(테이블) 데이터 AutoML 벤치마크.

업스트림 AutoML_parametric.py(저자 코드)의 프로토콜을 따른다:
  80/20 고정 분할(seed 42) → 학습분율 [0.2,0.4,0.6,0.8,0.95] × n_splits회 재추출
  모델: AutoGluon(기본 fit) + XGBoost/LightGBM/RandomForest/GradientBoosting(기본값)
  데이터셋: Fastback_F(F_*만) / Combined_All(전체)

⚠ 저자 코드의 피처는 기하 파라미터 23개 + **양력계수(Average/Std Cl, Cl_f, Cl_r)**다.
  Cl은 Cd와 같은 CFD 해석의 산출물로, 설계 시점에는 알 수 없는 값(누수성 피처).
  --features paper     : 저자 코드 그대로 (수치 검증용)
  --features geoparams : 기하 파라미터 23개만 (설계 시점에 정직한 설정)

  python scripts/automl_parametric.py --n-splits 5 --ag-time-limit 120
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

CSV = "/home/kwy00/qi/data/DrivAerNet_ParametricData.csv"
SIZES = [0.2, 0.4, 0.6, 0.8, 0.95]          # 저자 코드: 0.95가 "100%"로 표기됨
DROP_ALWAYS = ["Experiment", "Average Cd", "Std Cd", "Design_Category"]
CL_COLS = ["Average Cl", "Std Cl", "Average Cl_f", "Std Cl_f", "Average Cl_r", "Std Cl_r"]

# 논문 Fig.5의 100% 지점 눈대중 목표치 (검증 기준선, ±0.05 허용)
PAPER_REF = {
    ("Fastback_F", "AutoGluon"): 0.83, ("Fastback_F", "LightGBM"): 0.79,
    ("Combined_All", "LightGBM"): 0.60, ("Combined_All", "AutoGluon"): 0.58,
    ("Combined_All", "XGBoost"): 0.55,
}


def sk_models():
    import lightgbm as lgb
    import xgboost as xgb
    return {
        "XGBoost": lambda: xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
        "LightGBM": lambda: lgb.LGBMRegressor(objective="regression", random_state=42, verbose=-1),
        "RandomForest": lambda: RandomForestRegressor(random_state=42),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=42),
    }


def fit_autogluon(X_tr, y_tr, X_te, y_te, time_limit):
    from autogluon.tabular import TabularDataset, TabularPredictor
    tr = TabularDataset(pd.DataFrame(X_tr).assign(Average_Cd=y_tr.values))
    te = TabularDataset(pd.DataFrame(X_te).assign(Average_Cd=y_te.values))
    pred = TabularPredictor(label="Average_Cd", problem_type="regression",
                            eval_metric="r2", verbosity=0).fit(
        tr, time_limit=time_limit if time_limit and time_limit > 0 else None)
    return r2_score(y_te, pred.predict(te))


def main(a):
    data = pd.read_csv(CSV)
    data["Design_Category"] = data["Experiment"].apply(lambda x: x.split("_")[0])
    print(f"CSV {data.shape}  차종: {dict(data['Design_Category'].value_counts())}")

    datasets = {"Fastback_F": data[data["Design_Category"] == "F"], "Combined_All": data}
    drop = DROP_ALWAYS + (CL_COLS if a.features == "geoparams" else [])

    models = dict(sk_models())
    if a.autogluon:
        models = {"AutoGluon": None, **models}

    all_results = {}
    for ds_name, ds in datasets.items():
        X = ds.drop(columns=drop)
        y = ds["Average Cd"]
        X_full, X_te, y_full, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"\n=== {ds_name}: train_full={len(X_full)} test={len(X_te)} "
              f"features={X.shape[1]} ({a.features})")

        ds_res = {}
        for mname, mfn in models.items():
            per_size = {}
            run = None
            if a.wandb:
                try:
                    import wandb
                    suffix = f"_ns{a.n_splits}" if a.n_splits != 5 else ""
                    run = wandb.init(project="cfa", name=f"automl_{ds_name}_{mname}_{a.features}{suffix}",
                                     tags=["automl", "5.1.2", a.features],
                                     config={"dataset": ds_name, "model": mname,
                                             "features": a.features, "n_splits": a.n_splits,
                                             "n_features": X.shape[1]}, reinit=True)
                except Exception as e:
                    print(f"  (wandb 비활성: {e})")
            for size in SIZES:
                scores = []
                for split in range(a.n_splits):
                    X_tr, _, y_tr, _ = train_test_split(
                        X_full, y_full, train_size=min(size, 0.95), random_state=split)
                    if mname == "AutoGluon":
                        r2 = fit_autogluon(X_tr, y_tr, X_te, y_te, a.ag_time_limit)
                    else:
                        m = mfn()
                        m.fit(X_tr, y_tr)
                        r2 = r2_score(y_te, m.predict(X_te))
                    scores.append(r2)
                mean = float(np.mean(scores))
                ci = float(t_dist.ppf(0.975, a.n_splits - 1) * np.std(scores) / np.sqrt(a.n_splits))
                per_size[str(size)] = {"mean_r2": mean, "std_r2": float(np.std(scores)), "ci": ci}
                if run:
                    run.log({"train_frac_pct": int(size * 100) if size < 0.95 else 100,
                             "mean_r2": mean, "ci": ci})
                print(f"  {mname:<18} {int(size*100 if size<0.95 else 100):>3}%  "
                      f"R²={mean:+.4f} ±{ci:.4f}", flush=True)
            ds_res[mname] = per_size
            if run:
                ref = PAPER_REF.get((ds_name, mname))
                run.summary.update({"final_mean_r2": per_size["0.95"]["mean_r2"],
                                    "final_ci": per_size["0.95"]["ci"],
                                    "paper_ref": ref if ref is not None else float("nan")})
                run.finish()
        all_results[ds_name] = ds_res

    out = f"/home/kwy00/qi/outputs/automl_parametric_{a.features}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\n저장: {out}")

    print(f"\n{'':<16}{'구성':<14}{'우리(100%)':>12}{'논문 Fig.5':>12}")
    for (ds, m), ref in PAPER_REF.items():
        if ds in all_results and m in all_results[ds]:
            ours = all_results[ds][m]["0.95"]["mean_r2"]
            print(f"{ds:<16}{m:<14}{ours:>+12.3f}{ref:>12.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-splits", type=int, default=5)        # 저자 코드는 20 (시간 절약 기본 5)
    p.add_argument("--ag-time-limit", type=int, default=120)  # AutoGluon fit당 초
    p.add_argument("--features", default="paper", choices=["paper", "geoparams"])
    p.add_argument("--autogluon", type=int, default=1)
    p.add_argument("--wandb", type=int, default=1)
    main(p.parse_args())
