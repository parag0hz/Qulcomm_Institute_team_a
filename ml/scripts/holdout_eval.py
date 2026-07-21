"""일반화 검증: 홀드아웃 재분할 학습 (train_r2.py 레시피 복제).

  python scripts/holdout_eval.py --test-body Estate            # 차종 홀드아웃
  python scripts/holdout_eval.py --test-prefix F_S_WWC_WM      # 계열 홀드아웃

train = 공식 train ∖ 홀드아웃, val = 공식 val ∖ 홀드아웃(조기종료용),
test = 홀드아웃 전체. 레시피는 train_r2.py와 동일 (PointNet 2048, meter,
AdamW lr1e-3, SmoothL1, 120ep, patience 30, seed 0).
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, "/home/kwy00/qi")
from models_pc import BACKBONES


def r2(yh, y):
    return 1 - float(((yh - y) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def rank_acc(yh, y, n=200_000, seed=0):
    g = np.random.default_rng(seed)
    i, j = g.integers(0, len(y), n), g.integers(0, len(y), n)
    m = y[i] != y[j]
    return float(((yh[i] < yh[j]) == (y[i] < y[j]))[m].mean())


def main(a):
    torch.manual_seed(0); np.random.seed(0)
    d = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
    pts, cd = d["pts"][:, :2048].copy(), d["cd"]
    split, cls, keys = d["split"], d["cls"], d["keys"]
    prefix = np.array(["_".join(k.split("_")[:-1]) for k in keys])

    if a.test_body:
        te = cls == a.test_body
        tag = f"holdout_body_{a.test_body}"
    else:
        te = prefix == a.test_prefix
        tag = f"holdout_family_{a.test_prefix}"
    tr = (split == "train") & ~te
    va = (split == "val") & ~te

    run = None
    if a.wandb:
        try:
            import wandb
            run = wandb.init(project="cfa", name=tag, tags=["holdout"],
                             config={"holdout": a.test_body or a.test_prefix,
                                     "mode": "body" if a.test_body else "family",
                                     "n_train": int(tr.sum()), "n_test": int(te.sum())})
        except Exception as e:
            print(f"  (wandb 비활성 — 학습은 계속: {e})")
    print(f"=== {tag}: train={tr.sum()} val={va.sum()} test(holdout)={te.sum()}")
    print(f"    test 차종 구성: {dict(zip(*np.unique(cls[te], return_counts=True)))}")
    print(f"    test Cd: {cd[te].mean():.4f} ± {cd[te].std():.4f}  "
          f"(train Cd: {cd[tr].mean():.4f} ± {cd[tr].std():.4f})")

    pts = pts - pts[tr].reshape(-1, 3).mean(0)          # 원점: train에서만
    ymu, ysd = cd[tr].mean(), cd[tr].std()
    yz = (cd - ymu) / ysd

    T = lambda m: (torch.from_numpy(pts[m]), torch.from_numpy(yz[m]))
    dl = DataLoader(TensorDataset(*T(tr)), batch_size=32, shuffle=True,
                    drop_last=True, num_workers=2, pin_memory=True)
    net = BACKBONES["pointnet"](n_dims=0).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 120, eta_min=1e-5)
    lossf = nn.SmoothL1Loss(beta=1.0)

    @torch.no_grad()
    def infer(mask):
        net.eval()
        P, _ = T(mask)
        out = [net(P[i:i+128].cuda(), None).cpu() for i in range(0, len(P), 128)]
        return torch.cat(out).numpy() * ysd + ymu

    best, best_state, bad = -9e9, None, 0
    t0 = time.time()
    for ep in range(120):
        net.train()
        for p, y in dl:
            p, y = p.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(p, None), y)
            loss.backward(); opt.step()
        sched.step()
        vr = r2(infer(va), cd[va])
        if vr > best:
            best, bad = vr, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if run:
            run.log({"epoch": ep, "val_r2": vr, "best_val_r2": best})
        if ep % 20 == 0:
            print(f"  ep{ep:3d} valR2={vr:+.4f} best={best:+.4f} ({time.time()-t0:.0f}s)", flush=True)
        if bad >= 30:
            print(f"  early stop @ep{ep}"); break

    net.load_state_dict(best_state)
    yh, y = infer(te), cd[te]
    res = {
        "tag": tag, "n_test": int(te.sum()), "val_r2": best,
        "R2": r2(yh, y),
        "MAE_counts": float(np.abs(yh - y).mean() * 1000),
        "MAE_pct": float((np.abs(yh - y) / y).mean() * 100),
        "rank_acc": rank_acc(yh, y),
        "bias_counts": float((yh - y).mean() * 1000),   # 체계적 편향 (외삽 진단)
    }
    print(f"\n>>> {tag}")
    for k, v in res.items():
        print(f"    {k:<12}: {v if isinstance(v,(int,str)) else round(v,4)}")
    with open(f"/home/kwy00/qi/outputs/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    np.savez(f"/home/kwy00/qi/outputs/{tag}_pred.npz",
             yh=yh, y=y, keys=keys[te], cls=cls[te])
    if run:
        run.summary.update(res)
        run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test-body", default=None)
    p.add_argument("--test-prefix", default=None)
    p.add_argument("--wandb", type=int, default=1)   # 0 = 로깅 끔
    a = p.parse_args()
    assert (a.test_body is None) != (a.test_prefix is None), "둘 중 하나만"
    main(a)
