"""포인트클라우드용 Grad-CAM — 모델이 Cd를 예측할 때 차체의 어디를 보는가.

이미지 Grad-CAM을 점군으로 옮긴 것:
  · 이미지: 마지막 conv의 (C, H, W) 특징맵 → 채널 가중합 → 히트맵
  · 점군  : max pool 직전의 (C, N) 점별 특징 → 채널 가중합 → **점별 중요도**
  회귀이므로 class logit 대신 **Cd 출력**을 미분한다.

같이 그리는 것:
  · critical points — max pooling에서 살아남아 전역 특징에 실제로 기여한 점
    (PointNet 고유의 해석: 전역 특징은 이 점들만으로 결정된다)

  python scripts/gradcam_pointcloud.py                    # 데모 5대
  python scripts/gradcam_pointcloud.py --n 3
결과: outputs/gradcam_<ID>.png
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/kwy00/qi")
from models_pc import PointNet

BG, FG, SUB, ACC = "#141414", "#E4E1DB", "#B9B6B0", "#D98A3D"
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB})


def gradcam(net, x, center, ymu=0.0, ysd=1.0):
    """x: (1,N,3) 미터 원본. → (importance(N,), critical_mask(N,), pred_cd)"""
    feats = {}

    def hook(_m, _i, o):
        feats["A"] = o                       # (1, C, N)  max pool 직전
        o.retain_grad()

    h = net.mlp.register_forward_hook(hook)
    xin = (x - center).clone().requires_grad_(True)
    f = net.mlp(xin.transpose(1, 2))         # (1,C,N)
    g = f.amax(-1)                           # (1,C)  max pool
    out = net.head(g).squeeze(-1)            # (1,)
    net.zero_grad()
    out.backward()
    h.remove()

    A = f.detach()[0]                        # (C,N)
    G = f.grad.detach()[0] if f.grad is not None else feats["A"].grad.detach()[0]
    alpha = G.mean(dim=1, keepdim=True)      # (C,1)  채널 가중치 = 점축 평균 기울기
    cam = torch.relu((alpha * A).sum(0))     # (N,)   Grad-CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)

    crit = torch.zeros(A.shape[1], dtype=torch.bool)
    crit[A.argmax(dim=1).unique()] = True    # max를 차지한 점 = critical point
    return cam.cpu().numpy(), crit.cpu().numpy(), float(out.detach()) * ysd + ymu


def draw(pts, cam, crit, title, sub, path):
    """옆/위/뒤 3면 + critical points."""
    fig = plt.figure(figsize=(15, 4.2))
    views = [("옆면 (x-z)", 0, 2, (0, 1)), ("위 (x-y)", 0, 1, (0, 1)), ("뒤 (y-z)", 1, 2, (0, 1))]
    for k, (name, ax_i, ax_j, _) in enumerate(views):
        ax = fig.add_subplot(1, 4, k + 1)
        o = np.argsort(cam)                                  # 중요한 점을 위에 그림
        s = ax.scatter(pts[o, ax_i], pts[o, ax_j], c=cam[o], s=3,
                       cmap="inferno", vmin=0, vmax=1, linewidths=0)
        ax.set_title(name, fontsize=11, color=FG, pad=6)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#3a3a3a")
        if k == 2:
            cb = fig.colorbar(s, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("중요도", color=SUB, fontsize=9)
            cb.ax.tick_params(colors=SUB, labelsize=8)
            cb.outline.set_edgecolor("#3a3a3a")
    ax = fig.add_subplot(1, 4, 4)
    ax.scatter(pts[~crit, 0], pts[~crit, 2], c="#33312e", s=2, linewidths=0, label="그 외")
    ax.scatter(pts[crit, 0], pts[crit, 2], c=ACC, s=6, linewidths=0,
               label=f"critical ({crit.sum()}점)")
    ax.set_title("critical points (옆면)", fontsize=11, color=FG, pad=6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(facecolor="#1C1C1C", edgecolor="#3a3a3a", labelcolor=FG, fontsize=8, loc="upper right")
    for sp in ax.spines.values():
        sp.set_color("#3a3a3a")
    fig.suptitle(f"{title}    {sub}", fontsize=12.5, color=FG, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main(a):
    ck = torch.load("/home/kwy00/qi/outputs/pointnet_serving.pt", weights_only=False)
    net = PointNet(n_dims=0, emb=ck["config"]["emb"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    center = torch.tensor(ck["center"], dtype=torch.float32).view(1, 1, 3)
    npt = ck["npoints"]

    demo = json.load(open("/home/kwy00/qi/data/demo_holdout.json"))["items"][:a.n]
    d = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
    keys = np.array([str(k) for k in d["keys"]])

    print(f"PointNet Grad-CAM · {npt}점 · 데모 {len(demo)}대\n")
    print(f"{'ID':<22}{'차종':<12}{'실제Cd':>9}{'예측Cd':>9}{'critical':>10}{'상위10%중요영역':>18}")
    for it in demo:
        i = int(np.where(keys == it["id"])[0][0])
        pts = d["pts"][i:i+1, :npt].astype(np.float32)
        cam, crit, pred = gradcam(net, torch.from_numpy(pts), center, ck['ymu'], ck['ysd'])
        p = pts[0]
        top = p[cam >= np.quantile(cam, 0.9)]
        # 상위 중요 영역이 차체의 어디인지 (x: 앞0→뒤1 정규화)
        xr = (top[:, 0] - p[:, 0].min()) / np.ptp(p[:, 0])
        zone = f"x {xr.mean():.2f}·z {(top[:,2].mean()-p[:,2].min())/np.ptp(p[:,2]):.2f}"
        out = f"/home/kwy00/qi/outputs/gradcam_{it['id']}.png"
        draw(p, cam, crit, f"{it['id']} ({it['body_type']})",
             f"실제 {it['true_cd']:.4f} / 예측 {pred:.4f}", out)
        print(f"{it['id']:<22}{it['body_type']:<12}{it['true_cd']:>9.4f}{pred:>9.4f}"
              f"{crit.sum():>10}{zone:>18}")
    print(f"\n저장: outputs/gradcam_*.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    main(p.parse_args())
