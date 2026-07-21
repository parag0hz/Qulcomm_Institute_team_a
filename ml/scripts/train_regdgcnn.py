"""RegDGCNN (저자 구현 그대로) — 논문 Table 4 검증 + 정규화 절제 실험.

모델은 external/DrivAerNet/DeepSurrogates/DeepSurrogate_models.py 의 RegDGCNN을
그대로 import 한다. 학습 레시피도 저자 v1 트레이너와 동일:
  Adam lr 1e-3 wd 1e-4, MSE(raw Cd), ReduceLROnPlateau(patience 20, factor 0.1),
  100 epochs, dropout 0.4, emb_dims 512, k 40.

변인은 입력 정규화 하나뿐:
  --norm minmax : 저자 DrivAerNetDataset.min_max_normalize와 동일 (클라우드별·축별 [0,1])
                  → 절대 스케일 + 종횡비 소거. 논문 Table 4 조건.
  --norm meter  : 미터 유지, train 평균으로 상수 평행이동만 (이 저장소 불변식 1)

  python scripts/train_regdgcnn.py --norm minmax --npoints 2048 --bs 16
  python scripts/train_regdgcnn.py --norm meter  --npoints 2048 --bs 16

주의: 논문은 5,000점 (fps2048 캐시는 최대 2048점 — 편차로 기록).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/home/kwy00/qi/external/DrivAerNet/DeepSurrogates")
from DeepSurrogate_models import RegDGCNN

CLASSES = ["Fastback", "Estate", "Notchback"]


def r2(yh, y):
    return 1 - float(((yh - y) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    d = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
    pts, cd = d["pts"][:, :a.npoints].copy(), d["cd"].astype(np.float32)
    split, cls = d["split"], d["cls"]
    tr, va, te = (split == "train"), (split == "val"), (split == "test")

    if a.norm == "minmax":                    # 저자 코드: 클라우드별·축별 [0,1]
        lo = pts.min(axis=1, keepdims=True)
        hi = pts.max(axis=1, keepdims=True)
        pts = (pts - lo) / (hi - lo + 1e-9)
    else:                                     # meter: 상수 평행이동만 (train에서 계산)
        pts = pts - pts[tr].reshape(-1, 3).mean(0)

    tag = f"regdgcnn_{a.norm}_n{a.npoints}" + (f"_seed{a.seed}" if a.seed else "")
    run = None
    if a.wandb:
        try:
            import wandb
            run = wandb.init(project="cfa", name=tag, tags=["regdgcnn"],
                             config={**vars(a), "arch": "RegDGCNN(paper)",
                                     "recipe": "adam1e-3_mse_plateau_100ep"})
        except Exception as e:
            print(f"  (wandb 비활성: {e})")

    T = lambda m: (torch.from_numpy(pts[m]), torch.from_numpy(cd[m]))
    dl = DataLoader(TensorDataset(*T(tr)), batch_size=a.bs, shuffle=True,
                    drop_last=True, num_workers=2, pin_memory=True)

    net = RegDGCNN({"k": a.k, "emb_dims": a.emb_dims, "dropout": a.dropout}).cuda()
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=20, factor=0.1)
    lossf = nn.MSELoss()

    @torch.no_grad()
    def infer(mask):
        net.eval()
        P, _ = T(mask)
        out = []
        for i in range(0, len(P), a.bs):
            x = P[i:i + a.bs].cuda().transpose(2, 1)      # (B,3,N) — 저자 forward 규약
            out.append(net(x).squeeze(-1).float().cpu())
        return torch.cat(out).numpy()

    print(f"=== {tag}  params={nparam/1e6:.2f}M  train={tr.sum()} val={va.sum()} test={te.sum()}", flush=True)
    best, best_state, t0 = 9e9, None, time.time()
    for ep in range(a.epochs):
        net.train()
        tot = 0.0
        for p, y in dl:
            p, y = p.cuda(non_blocking=True).transpose(2, 1), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(p).squeeze(-1), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y)
        yv = infer(va)
        vmse = float(((yv - cd[va]) ** 2).mean())
        vr = r2(yv, cd[va])
        sched.step(vmse)
        if vmse < best:
            best = vmse
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if run:
            run.log({"epoch": ep, "train_mse": tot / tr.sum(), "val_mse": vmse,
                     "val_r2": vr, "lr": opt.param_groups[0]["lr"]})
        print(f"  ep{ep:3d} trainMSE={tot/tr.sum():.6f} valMSE={vmse:.6f} valR2={vr:+.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    net.load_state_dict(best_state)
    yh, y = infer(te), cd[te]
    res = {"tag": tag, "params": nparam,
           "test": {"All": [r2(yh, y), float(np.mean(np.abs(yh - y) / y) * 100)]}}
    for c in CLASSES:
        s = cls[te] == c
        res["test"][c] = [r2(yh[s], y[s]), float(np.mean(np.abs(yh[s] - y[s]) / y[s]) * 100)]
    print(f"\n  {'':<12}{'R2':>10}{'MAE%':>9}")
    for c in ["All"] + CLASSES:
        print(f"  {c:<12}{res['test'][c][0]:>+10.3f}{res['test'][c][1]:>9.2f}")
    print(f"  (논문 Table 4 RegDGCNN: R² 0.641 / MAE 9.31e-3 — 8k 설계, 5k점, min-max)")

    with open(f"/home/kwy00/qi/outputs/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    np.savez(f"/home/kwy00/qi/outputs/{tag}_pred.npz",
             yh=yh, y=y, keys=d["keys"][te], cls=cls[te])
    if run:
        run.summary.update({f"test/{c}_r2": v[0] for c, v in res["test"].items()}
                           | {f"test/{c}_mae_pct": v[1] for c, v in res["test"].items()}
                           | {"paper_ref_r2": 0.641})
        run.finish()
    print(f"  -> outputs/{tag}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--norm", default="minmax", choices=["minmax", "meter"])
    p.add_argument("--npoints", type=int, default=2048)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--emb_dims", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", type=int, default=1)
    main(p.parse_args())
