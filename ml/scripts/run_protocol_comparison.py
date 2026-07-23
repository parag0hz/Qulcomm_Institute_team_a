"""#1+#2+#4 통합 재실행 — 동일 데이터·동일 fold로 ML vs DL 정량 비교.

protocol.py가 주는 것만 쓴다:
  데이터  교집합 3,709 − 데모 5 = 3,704대 (ML은 파라미터 23개, DL은 포인트클라우드)
  분할    K=5 rotating (학습3/검증1/테스트1) × 5세트
  지표    R², MAE, MSE (+ RMSE, MAE drag counts, 순위 정확도) 전체 + 차종별

  python scripts/run_protocol_comparison.py                 # 전체
  python scripts/run_protocol_comparison.py --only ml       # ML만
  python scripts/run_protocol_comparison.py --only dl
결과: outputs/protocol_comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/kwy00/qi")
sys.path.insert(0, "/home/kwy00/qi/scripts")
from protocol import load_dataset, make_folds, split_indices, evaluate, aggregate, CLASSES

OUT = "/home/kwy00/qi/outputs/protocol_comparison.json"


# ============================== ML ==============================

def run_ml(ds, sets, use_autogluon=True):
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    import xgboost as xgb

    X, y, cls, feats = ds["X"], ds["cd"], ds["cls"], ds["feat_names"]
    models = {
        "XGBoost": lambda: xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
        "RandomForest": lambda: RandomForestRegressor(random_state=42, n_jobs=-1),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=42),
    }
    try:
        import lightgbm as lgb
        models["LightGBM"] = lambda: lgb.LGBMRegressor(random_state=42, verbose=-1)
    except ImportError:
        print("  (lightgbm 없음 — 건너뜀)")

    res = {}
    for name, mk in models.items():
        per_set, t0 = [], time.time()
        for s in sets:
            m = mk().fit(X[s["train"]], y[s["train"]])
            per_set.append(evaluate(m.predict(X[s["test"]]), y[s["test"]], cls[s["test"]]))
        res[name] = {"per_set": per_set, "agg": aggregate(per_set)}
        a = res[name]["agg"]["All"]
        print(f"  {name:<18} R² {a['R2']['mean']:+.4f}±{a['R2']['std']:.4f}  "
              f"MAE {a['MAE']['mean']:.5f}  MSE {a['MSE']['mean']:.2e}  ({time.time()-t0:.0f}s)", flush=True)

    if use_autogluon:
        from autogluon.tabular import TabularDataset, TabularPredictor
        per_set, t0 = [], time.time()
        for s in sets:
            tr = pd.DataFrame(X[s["train"]], columns=feats).assign(Average_Cd=y[s["train"]])
            va = pd.DataFrame(X[s["val"]], columns=feats).assign(Average_Cd=y[s["val"]])
            te = pd.DataFrame(X[s["test"]], columns=feats)
            p = TabularPredictor(label="Average_Cd", problem_type="regression",
                                 eval_metric="r2", verbosity=0).fit(
                TabularDataset(tr), tuning_data=TabularDataset(va))
            per_set.append(evaluate(p.predict(TabularDataset(te)).values, y[s["test"]], cls[s["test"]]))
        res["AutoGluon"] = {"per_set": per_set, "agg": aggregate(per_set)}
        a = res["AutoGluon"]["agg"]["All"]
        print(f"  {'AutoGluon':<18} R² {a['R2']['mean']:+.4f}±{a['R2']['std']:.4f}  "
              f"MAE {a['MAE']['mean']:.5f}  MSE {a['MSE']['mean']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
    return res


# ============================== DL ==============================

BS_DEFAULT = {"pointnet": 32, "dgcnn": 16, "regdgcnn": 8, "triplane": 32, "mlp": 32}


def run_dl(ds, sets, backbone="pointnet", epochs=120, bs=None, lr=1e-3, patience=30):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from models_pc import BACKBONES

    bs = bs or BS_DEFAULT.get(backbone, 16)
    pts, y, cls = ds["pts"], ds["cd"], ds["cls"]
    per_set, t0 = [], time.time()
    for s in sets:
        torch.manual_seed(0); np.random.seed(0)
        tr, va, te = s["train"], s["val"], s["test"]

        center = pts[tr].reshape(-1, 3).mean(0)             # 미터 스케일: 학습 fold 기준 평행이동만
        P = (pts - center).astype(np.float32)
        ymu, ysd = y[tr].mean(), y[tr].std()
        yz = ((y - ymu) / ysd).astype(np.float32)

        net = BACKBONES[backbone](n_dims=0).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
        lossf = nn.SmoothL1Loss(beta=1.0)
        dl = DataLoader(TensorDataset(torch.from_numpy(P[tr]), torch.from_numpy(yz[tr])),
                        batch_size=bs, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

        @torch.no_grad()
        def infer(idx):
            net.eval()
            cs = max(bs, 8)
            o = [net(torch.from_numpy(P[idx[i:i+cs]]).cuda()).cpu() for i in range(0, len(idx), cs)]
            return torch.cat(o).numpy() * ysd + ymu

        best, best_sd, bad = -9e9, None, 0
        for ep in range(epochs):
            net.train()
            for xb, yb in dl:
                opt.zero_grad(set_to_none=True)
                lossf(net(xb.cuda(non_blocking=True)), yb.cuda(non_blocking=True)).backward()
                opt.step()
            sch.step()
            vr = 1 - ((infer(va) - y[va]) ** 2).sum() / ((y[va] - y[va].mean()) ** 2).sum()
            if vr > best:
                best, bad = vr, 0
                best_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            if bad >= patience:
                break
        net.load_state_dict(best_sd)
        per_set.append(evaluate(infer(te), y[te], cls[te]))
        a = per_set[-1]["All"]
        print(f"  [{backbone}] 세트{s['set']}  test R² {a['R2']:+.4f}  MAE {a['MAE']:.5f}  "
              f"(ep{ep}, {time.time()-t0:.0f}s)", flush=True)
    agg = aggregate(per_set)
    nm = {"pointnet": "PointNet", "dgcnn": "DGCNN", "regdgcnn": "RegDGCNN",
          "triplane": "Triplane", "mlp": "MLP"}.get(backbone, backbone)
    print(f"  {nm:<18} R² {agg['All']['R2']['mean']:+.4f}±{agg['All']['R2']['std']:.4f}  "
          f"MAE {agg['All']['MAE']['mean']:.5f}  MSE {agg['All']['MSE']['mean']:.2e}  "
          f"MAPE {agg['All'].get('MAPE',{}).get('mean',float('nan')):.2f}%", flush=True)
    return {nm: {"per_set": per_set, "agg": agg}}


# ============================== main ==============================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ml", "dl"], default=None)
    ap.add_argument("--no-autogluon", action="store_true")
    ap.add_argument("--backbones", nargs="+", default=["pointnet"])
    ap.add_argument("--npoints", type=int, default=2048)
    ap.add_argument("--cache", default=None,
                    help="FPS 캐시 경로. npoints>2048이면 반드시 fps4096.npz 등을 지정")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    cache = a.cache or ("/home/kwy00/qi/data/fps4096.npz" if a.npoints > 2048
                        else "/home/kwy00/qi/data/fps2048.npz")
    ds = load_dataset(npoints=a.npoints, cache=cache)
    if ds["pts"].shape[1] < a.npoints:      # 조용한 절단 방지
        raise SystemExit(f"캐시 점수 부족: {cache}는 {ds['pts'].shape[1]}점뿐인데 "
                         f"--npoints {a.npoints} 요청. 더 큰 FPS 캐시를 지정하세요 "
                         f"(scripts/precompute_fps_k.py --k {a.npoints} --out ...)")
    fold = make_folds(ds["cls"])
    sets = split_indices(fold)
    print(f"데이터 {len(ds['keys']):,}대 (교집합−데모) · K=5 rotating · "
          f"train~{len(sets[0]['train'])} val~{len(sets[0]['val'])} test~{len(sets[0]['test'])}\n")

    out = {}
    if a.only != "dl":
        print("=== ML (설계 파라미터 23개) ===")
        out.update(run_ml(ds, sets, use_autogluon=not a.no_autogluon))
    def save(o):
        """백본 하나 끝날 때마다 저장 — 뒤에서 크래시해도 앞 결과를 잃지 않는다."""
        with open(a.out, "w") as f:
            json.dump({"n": len(ds["keys"]), "K": 5, "npoints": a.npoints,
                       "protocol": "rotating 3/1/1, stratified by body type",
                       "models": {k: v["agg"] for k, v in o.items()},
                       "per_set": {k: v["per_set"] for k, v in o.items()}}, f,
                      indent=1, ensure_ascii=False)

    if out:
        save(out)
    if a.only != "ml":
        print(f"\n=== DL (포인트클라우드 {a.npoints}점) ===")
        for bb in a.backbones:
            try:
                out.update(run_dl(ds, sets, backbone=bb))
                save(out)                      # ← 즉시 저장
                print(f"  (중간 저장 완료: {a.out})", flush=True)
            except Exception as e:
                import traceback
                print(f"  ⚠ [{bb}] 실패 — 건너뜀: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                continue

    payload = {"n": len(ds["keys"]), "K": 5, "npoints": a.npoints,
               "protocol": "rotating 3/1/1, stratified by body type",
               "models": {k: v["agg"] for k, v in out.items()},
               "per_set": {k: v["per_set"] for k, v in out.items()}}
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print(f"\n저장: {a.out}")
