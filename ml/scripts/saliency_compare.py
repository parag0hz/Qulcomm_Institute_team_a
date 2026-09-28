"""백본별 입력 saliency 비교 — 각 모델이 차체의 어디를 보고 Cd를 판단하는가.

왜 Grad-CAM이 아니라 입력 saliency인가:
  Grad-CAM은 아키텍처마다 hook 지점(특징 텐서)이 달라 맵을 **직접 비교할 수 없다**.
  입력 기울기 |∂Cd/∂x_n| 는 **모든 모델에 동일한 정의**라 나란히 놓고 비교할 수 있다.
  (PointNet은 Grad-CAM도 함께 제공 — scripts/gradcam_pointcloud.py)

정량 지표 (그림만으로 끝내지 않기 위해):
  집중도   상위 5% 점이 전체 중요도의 몇 %를 차지하나 (높을수록 국소적)
  유효점수 중요도 분포의 유효 개수 exp(엔트로피) (높을수록 전역적)
  공간분산 중요도 가중 좌표의 표준편차 (넓게 볼수록 큼)

  python scripts/saliency_compare.py --train           # 가중치 없으면 학습부터
  python scripts/saliency_compare.py                   # 저장된 가중치로 비교만
결과: outputs/saliency_<ID>.png, outputs/saliency_metrics.json
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/kwy00/qi")
sys.path.insert(0, "/home/kwy00/qi/scripts")
from models_pc import BACKBONES
from protocol import load_dataset, make_folds, split_indices

OUT = "/home/kwy00/qi/outputs"
BG, FG, SUB, ACC = "#141414", "#E4E1DB", "#B9B6B0", "#D98A3D"
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB})
BS = {"pointnet": 32, "dgcnn": 16, "regdgcnn": 8, "triplane": 32}
NAME = {"pointnet": "PointNet", "dgcnn": "DGCNN", "regdgcnn": "RegDGCNN", "triplane": "Triplane"}


# ============================== 학습 & 저장 ==============================

def train_save(bb, ds, s, npoints, epochs=120):
    """fold1 기준으로 학습하고 가중치 저장 (saliency 비교용)."""
    torch.manual_seed(0); np.random.seed(0)
    pts, y = ds["pts"], ds["cd"]
    tr, va = s["train"], s["val"]
    center = pts[tr].reshape(-1, 3).mean(0)
    ymu, ysd = float(y[tr].mean()), float(y[tr].std())
    P = pts.astype(np.float32)
    yz = ((y - ymu) / ysd).astype(np.float32)
    ctr = torch.tensor(center, dtype=torch.float32).view(1, 1, 3).cuda()

    net = BACKBONES[bb](n_dims=0).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    lossf = nn.SmoothL1Loss(beta=1.0)
    dl = DataLoader(TensorDataset(torch.from_numpy(P[tr]), torch.from_numpy(yz[tr])),
                    batch_size=BS[bb], shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    @torch.no_grad()
    def vr_():
        net.eval()
        cs = max(BS[bb], 8)
        o = [net(torch.from_numpy(P[va[i:i+cs]]).cuda() - ctr).cpu() for i in range(0, len(va), cs)]
        yh = torch.cat(o).numpy() * ysd + ymu
        return 1 - ((yh - y[va]) ** 2).sum() / ((y[va] - y[va].mean()) ** 2).sum()

    best, best_sd, t0 = -9e9, None, time.time()
    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            lossf(net(xb.cuda(non_blocking=True) - ctr), yb.cuda(non_blocking=True)).backward()
            opt.step()
        sch.step()
        v = vr_()
        if v > best:
            best, best_sd = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
        if ep % 30 == 0 or ep == epochs - 1:
            print(f"    ep{ep:3d} valR2={v:+.4f} best={best:+.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"state_dict": best_sd, "center": center, "ymu": ymu, "ysd": ysd,
                "npoints": npoints, "backbone": bb, "val_r2": float(best)},
               f"{OUT}/backbone_{bb}_{npoints}.pt")
    print(f"    저장: backbone_{bb}_{npoints}.pt (val R² {best:+.4f}, {time.time()-t0:.0f}s)")
    return best


# ============================== saliency ==============================

def saliency(bb, ck, pts_np):
    """|∂Cd/∂x_n| — 모든 백본 동일 정의."""
    kw = {"n_dims": 0}
    if bb == "pointnet" and "config" in ck:          # 튜닝본은 emb가 다를 수 있다
        kw["emb"] = ck["config"]["emb"]
    net = BACKBONES[bb](**kw)
    net.load_state_dict(ck["state_dict"])
    net.eval().cuda()
    ctr = torch.tensor(ck["center"], dtype=torch.float32).view(1, 1, 3).cuda()
    x = torch.from_numpy(pts_np).cuda().requires_grad_(True)
    out = net(x - ctr)
    net.zero_grad()
    out.sum().backward()
    g = x.grad.detach()[0].norm(dim=1)              # (N,) 점별 기울기 크기
    pred = float(out.detach()) * ck["ysd"] + ck["ymu"]
    s = (g - g.min()) / (g.max() - g.min() + 1e-12)
    return s.cpu().numpy(), pred


def stats(sal, pts):
    """전역성 정량화."""
    w = sal / (sal.sum() + 1e-12)
    top5 = np.sort(w)[::-1][:max(1, len(w) // 20)].sum()      # 상위 5% 점유율
    ent = -(w * np.log(w + 1e-12)).sum()
    eff = float(np.exp(ent)) / len(w)                          # 유효 점 비율
    mu = (pts * w[:, None]).sum(0)
    spread = float(np.sqrt((((pts - mu) ** 2).sum(1) * w).sum()))
    return {"top5_share": float(top5), "effective_frac": eff, "spatial_spread_m": spread}


# ============================== main ==============================

def main(a):
    ds = load_dataset(npoints=a.npoints)
    sets = split_indices(make_folds(ds["cls"]))
    s = sets[0]

    for bb in a.backbones:
        p = f"{OUT}/backbone_{bb}_{a.npoints}.pt"
        import os
        if not os.path.exists(p):
            if not a.train:
                print(f"⚠ {bb}: 가중치 없음 — --train 으로 학습하거나 건너뜀"); continue
            print(f"=== {NAME[bb]} 학습 ===")
            train_save(bb, ds, s, a.npoints, epochs=a.epochs)

    avail = [bb for bb in a.backbones
             if __import__("os").path.exists(f"{OUT}/backbone_{bb}_{a.npoints}.pt")]
    if not avail:
        print("비교할 가중치가 없다."); return

    demo = json.load(open("/home/kwy00/qi/data/demo_holdout.json"))["items"][:a.n]
    d = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
    keys = np.array([str(k) for k in d["keys"]])
    allm = {}

    for it in demo:
        i = int(np.where(keys == it["id"])[0][0])
        pts = d["pts"][i:i+1, :a.npoints].astype(np.float32)
        fig, ax = plt.subplots(len(avail), 3, figsize=(12, 2.7 * len(avail)), squeeze=False)
        for r, bb in enumerate(avail):
            ck = torch.load(f"{OUT}/backbone_{bb}_{a.npoints}.pt", weights_only=False)
            sal, pred = saliency(bb, ck, pts)
            st = stats(sal, pts[0])
            allm.setdefault(bb, []).append(st)
            for c, (nm, i1, i2) in enumerate([("옆면", 0, 2), ("위", 0, 1), ("뒤", 1, 2)]):
                o = np.argsort(sal)
                ax[r][c].scatter(pts[0][o, i1], pts[0][o, i2], c=sal[o], s=2.5,
                                 cmap="inferno", vmin=0, vmax=1, linewidths=0)
                ax[r][c].set_aspect("equal"); ax[r][c].set_xticks([]); ax[r][c].set_yticks([])
                for sp in ax[r][c].spines.values():
                    sp.set_color("#3a3a3a")
                if r == 0:
                    ax[r][c].set_title(nm, fontsize=11, color=FG, pad=6)
            ax[r][0].set_ylabel(f"{NAME[bb]}\n예측 {pred:.4f}", fontsize=10, color=FG)
        fig.suptitle(f"입력 saliency 비교 — {it['id']} ({it['body_type']}) · 실제 Cd {it['true_cd']:.4f}",
                     fontsize=12.5, color=FG, y=1.0)
        plt.tight_layout()
        plt.savefig(f"{OUT}/saliency_{it['id']}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"저장: saliency_{it['id']}.png")

    print(f"\n=== 전역성 지표 (데모 {len(demo)}대 평균) ===")
    print(f"{'모델':<12}{'상위5%점유':>12}{'유효점비율':>12}{'공간분산(m)':>13}   해석")
    agg = {}
    for bb, v in allm.items():
        m = {k: float(np.mean([x[k] for x in v])) for k in v[0]}
        agg[NAME[bb]] = m
        tag = "국소적" if m["top5_share"] > 0.25 else "전역적"
        print(f"{NAME[bb]:<12}{m['top5_share']:>11.1%}{m['effective_frac']:>12.1%}"
              f"{m['spatial_spread_m']:>13.3f}   {tag}")
    with open(f"{OUT}/saliency_metrics.json", "w") as f:
        json.dump(agg, f, indent=1, ensure_ascii=False)
    print(f"\n저장: {OUT}/saliency_metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbones", nargs="+", default=["pointnet", "dgcnn", "triplane", "regdgcnn"])
    p.add_argument("--npoints", type=int, default=2048)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--train", action="store_true")
    main(p.parse_args())
