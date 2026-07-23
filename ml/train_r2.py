"""R2: 포인트클라우드 -> Cd 회귀 베이스라인 (증강 없음, 클린).

R0 대비 무엇이 나아지는지 본다. 전역 R2가 아니라 **차종별 R2**가 판정 기준이다.
R0(v2 robust, clean test): All +0.814 / Fastback +0.881 / Estate -0.519 / Notchback -0.151

  python train_r2.py --backbone pointnet --dims 1
  python train_r2.py --backbone pointnet --scale unit     # 스케일 제거 ablation
  python train_r2.py --backbone dgcnn --npoints 1024
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, "/home/kwy00/qi")
from models_pc import BACKBONES

CLASSES = ["Fastback", "Estate", "Notchback"]
R0 = {"All": 0.814, "Fastback": 0.881, "Estate": -0.519, "Notchback": -0.151}


def r2(yh, y):
    return 1 - float(((yh - y) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def mae_pct(yh, y):
    return float(np.mean(np.abs(yh - y) / y) * 100)


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    d = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
    # FPS는 coarse-to-fine 순서 -> 앞 n개가 그 자체로 FPS n점 부분집합
    pts, dims, cd = d["pts"][:, :a.npoints], d["dims"], d["cd"]
    split, cls = d["split"], d["cls"]

    tr, va, te = (split == "train"), (split == "val"), (split == "test")

    if a.scale == "unit":                       # ⚠ ablation: 스케일 신호 제거 (클라우드별)
        c = pts.mean(1, keepdims=True)
        pts = pts - c
        pts = pts / np.linalg.norm(pts, axis=2).max(axis=1)[:, None, None]
    else:                                       # 미터 유지, 상수 평행이동만. 원점은 train에서만.
        pts = pts - pts[tr].reshape(-1, 3).mean(0)
    dmu, dsd = dims[tr].mean(0), dims[tr].std(0) + 1e-8
    dims = np.nan_to_num((dims - dmu) / dsd)
    ymu, ysd = cd[tr].mean(), cd[tr].std()
    yz = (cd - ymu) / ysd

    T = lambda m: (torch.from_numpy(pts[m]), torch.from_numpy(dims[m]), torch.from_numpy(yz[m]))
    dl = DataLoader(TensorDataset(*T(tr)), batch_size=a.bs, shuffle=True, drop_last=True,
                    num_workers=2, pin_memory=True)

    n_dims = 6 if a.dims else 0
    kw = {"n_dims": n_dims}
    if a.attn != "none":
        assert a.backbone == "pointnet", "--attn은 pointnet 백본만 지원"
        kw["attn"] = a.attn
    net = BACKBONES[a.backbone](**kw).cuda()
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-5)
    lossf = nn.SmoothL1Loss(beta=1.0)

    @torch.no_grad()
    def infer(mask):
        net.eval()
        P, D, _ = T(mask)
        cs = max(a.bs, 8)              # 추론 청크 = 학습 배치 (VRAM 큰 백본의 eval OOM 방지)
        out = []
        for i in range(0, len(P), cs):
            out.append(net(P[i:i+cs].cuda(), D[i:i+cs].cuda() if a.dims else None).cpu())
        return torch.cat(out).numpy() * ysd + ymu

    tag = (f"{a.backbone}_n{a.npoints}_dims{a.dims}_{a.scale}"
           + (f"_attn-{a.attn}" if a.attn != "none" else "")
           + (f"_seed{a.seed}" if a.seed else ""))
    run = None
    if a.wandb:
        try:
            import wandb
            run = wandb.init(project="cfa", name=tag, config={**vars(a), "params": nparam})
        except Exception as e:
            print(f"  (wandb 비활성 — 학습은 계속: {e})")
    print(f"=== {tag}  params={nparam/1e6:.2f}M  train={tr.sum()} val={va.sum()} test={te.sum()}")
    best, best_state, bad = -9e9, None, 0
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        tot = 0.0
        for p, dm, y in dl:
            p, dm, y = p.cuda(non_blocking=True), dm.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(p, dm if a.dims else None), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(p)
        sched.step()
        vr = r2(infer(va), cd[va])
        if vr > best:
            best, bad = vr, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if run:
            run.log({"epoch": ep, "train_loss": tot / tr.sum(), "val_r2": vr,
                     "best_val_r2": best, "lr": sched.get_last_lr()[0]})
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  ep{ep:3d} loss={tot/tr.sum():.4f} valR2={vr:+.4f} best={best:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if bad >= a.patience:
            print(f"  early stop @ep{ep}"); break

    net.load_state_dict(best_state)
    yh, y = infer(te), cd[te]
    res = {"tag": tag, "params": nparam, "val_r2": best,
           "test": {"All": [r2(yh, y), mae_pct(yh, y)]}}
    for c in CLASSES:
        s = cls[te] == c
        res["test"][c] = [r2(yh[s], y[s]), mae_pct(yh[s], y[s])]

    print(f"\n  {'':<12}{'R2':>10}{'MAE%':>9}{'R0 R2':>10}{'Δ vs R0':>10}")
    for c in ["All"] + CLASSES:
        r, m = res["test"][c]
        print(f"  {c:<12}{r:>+10.3f}{m:>9.2f}{R0[c]:>+10.3f}{r-R0[c]:>+10.3f}")
    with open(f"/home/kwy00/qi/outputs/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    # test 예측값 보존 — 쌍둥이/고립 부분집합 분석에 필요
    np.savez(f"/home/kwy00/qi/outputs/{tag}_pred.npz",
             yh=yh, y=y, keys=d["keys"][te], cls=cls[te])
    print(f"\n  -> outputs/{tag}.json   ({time.time()-t0:.0f}s)")
    if run:
        run.summary.update({"val_r2_best": best}
                           | {f"test/{c}_r2": v[0] for c, v in res["test"].items()}
                           | {f"test/{c}_mae_pct": v[1] for c, v in res["test"].items()})
        run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="pointnet", choices=list(BACKBONES))
    p.add_argument("--npoints", type=int, default=2048)
    p.add_argument("--dims", type=int, default=0)
    p.add_argument("--scale", default="meter", choices=["meter", "unit"])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn", default="none", choices=["none", "se", "cbam", "pool", "sa"])
    p.add_argument("--wandb", type=int, default=1)   # 0 = 로깅 끔
    main(p.parse_args())
