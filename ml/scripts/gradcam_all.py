"""백본별 Grad-CAM을 같은 샘플에 나란히 시각화.

아키텍처마다 "점별 특징"이 있는 위치가 달라 hook 지점을 각각 잡는다:
  PointNet  net.mlp 출력 (B,C,N)      — max pool 직전
  DGCNN     net.c5  출력 (B,C,N)      — max/mean pool 직전
  Triplane  net.cnn[3] 출력 (B,C,4,4) — 2D 특징맵. 점을 격자셀로 되돌려 점별 값으로 환산
            (평면 3개의 CAM을 각 점이 속한 셀에서 읽어 합산)

Grad-CAM 정의는 동일: α_c = mean_n ∂Cd/∂A[c,n],  CAM_n = ReLU(Σ_c α_c A[c,n])
회귀이므로 class logit 대신 Cd 출력을 미분한다.

⚠ 맵끼리 직접 비교하면 안 된다 — 특징공간·해상도가 다르다(Triplane은 평면당 4×4로 매우 거칠다).
  "각 모델이 어디를 강조하는가"의 정성 비교로만 읽을 것. 정량 비교는 가림 실험을 쓴다.

  python scripts/gradcam_all.py --n 3
결과: outputs/gradcam_all_<ID>.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/kwy00/qi")
from models_pc import BACKBONES

OUT = "/home/kwy00/qi/outputs"
BG, CARD, FG, SUB, ACC = "#141414", "#1C1C1C", "#E4E1DB", "#B9B6B0", "#D98A3D"
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB})
NAME = {"pointnet": "PointNet", "dgcnn": "DGCNN", "regdgcnn": "RegDGCNN", "triplane": "Triplane"}


def _cam_from(A, G):
    """A,G: (C,N) 또는 (C,H,W) → 정규화된 CAM."""
    alpha = G.mean(dim=tuple(range(1, G.dim())), keepdim=True)
    cam = torch.relu((alpha * A).sum(0))
    return cam


def gradcam(bb, ck, x):
    """x: (1,N,3) 미터 원본 → (cam(N,), pred_cd)"""
    kw = {"n_dims": 0}
    if bb == "pointnet" and "config" in ck:
        kw["emb"] = ck["config"]["emb"]
    net = BACKBONES[bb](**kw)
    net.load_state_dict(ck["state_dict"])
    net.eval().cuda()
    ctr = torch.tensor(ck["center"], dtype=torch.float32).view(1, 1, 3).cuda()
    xin = x.cuda() - ctr

    store = []
    target = {"pointnet": lambda: net.mlp, "dgcnn": lambda: net.c5,
              "regdgcnn": lambda: net.net.conv5,      # 저자 코드: conv5 → (B,emb,N)
              "triplane": lambda: net.cnn[3]}[bb]()   # 지연 평가 (백본마다 속성이 다름)

    def hook(_m, _i, o):
        o.retain_grad(); store.append(o)

    h = target.register_forward_hook(hook)
    out = net(xin)
    net.zero_grad()
    out.sum().backward()
    h.remove()
    pred = float(out.detach()) * ck["ysd"] + ck["ymu"]

    if bb in ("pointnet", "dgcnn"):
        A, G = store[0].detach()[0], store[0].grad.detach()[0]      # (C,N)
        cam = _cam_from(A, G)
    else:
        # Triplane: 평면 3개의 2D CAM을 각 점이 속한 셀에서 읽어 합산
        R, Rf = net.R, store[0].shape[-1]                            # 64, 4
        step = R // Rf
        t = ((x.cuda() - net.lo) / (net.hi - net.lo)).clamp(0, 1 - 1e-6)[0]   # (N,3)
        g = (t * R).long()
        cam = torch.zeros(x.shape[1], device=x.device if x.is_cuda else "cuda")
        for k, (a, b, _c) in enumerate(((0, 1, 2), (0, 2, 1), (1, 2, 0))):
            f = store[k]
            m2d = _cam_from(f.detach()[0], f.grad.detach()[0])        # (Rf,Rf)
            u = (g[:, a] // step).clamp(0, Rf - 1)
            v = (g[:, b] // step).clamp(0, Rf - 1)
            cam += m2d[u, v]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
    del net; torch.cuda.empty_cache()
    return cam.detach().cpu().numpy(), pred


def main(a):
    avail = [b for b in a.backbones if os.path.exists(f"{OUT}/backbone_{b}_{a.npoints}.pt")]
    miss = [b for b in a.backbones if b not in avail]
    if miss:
        print(f"⚠ 가중치 없어 제외: {', '.join(NAME[m] for m in miss)}")
    if not avail:
        print("가중치가 없다."); return

    demo = json.load(open("/home/kwy00/qi/data/demo_holdout.json"))["items"][:a.n]
    d = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
    keys = np.array([str(k) for k in d["keys"]])
    VIEWS = [("옆면", 0, 2), ("위", 0, 1), ("뒤", 1, 2)]

    for it in demo:
        i = int(np.where(keys == it["id"])[0][0])
        pts = torch.from_numpy(d["pts"][i:i+1, :a.npoints].astype(np.float32))
        p = pts[0].numpy()
        fig, ax = plt.subplots(len(avail), 3, figsize=(11.5, 2.7 * len(avail)), squeeze=False)
        for r, bb in enumerate(avail):
            ck = torch.load(f"{OUT}/backbone_{bb}_{a.npoints}.pt", weights_only=False)
            cam, pred = gradcam(bb, ck, pts)
            o = np.argsort(cam)
            for c, (nm, i1, i2) in enumerate(VIEWS):
                ax[r][c].scatter(p[o, i1], p[o, i2], c=cam[o], s=2.5, cmap="inferno",
                                 vmin=0, vmax=1, linewidths=0)
                ax[r][c].set_aspect("equal"); ax[r][c].set_xticks([]); ax[r][c].set_yticks([])
                for sp in ax[r][c].spines.values():
                    sp.set_color("#3a3a3a")
                if r == 0:
                    ax[r][c].set_title(nm, fontsize=11, color=FG, pad=6)
            ax[r][0].set_ylabel(f"{NAME[bb]}\n예측 {pred:.4f}", fontsize=10, color=FG)
            # 상위 10% 중요영역의 위치 (앞0→뒤1, 하0→상1)
            top = p[cam >= np.quantile(cam, 0.9)]
            xr = ((top[:, 0] - p[:, 0].min()) / np.ptp(p[:, 0])).mean()
            zr = ((top[:, 2] - p[:, 2].min()) / np.ptp(p[:, 2])).mean()
            print(f"  {NAME[bb]:<10} 예측 {pred:.4f}  상위10% 위치: 앞뒤 {xr:.2f} · 높이 {zr:.2f}")
        fig.suptitle(f"백본별 Grad-CAM — {it['id']} ({it['body_type']}) · 실제 Cd {it['true_cd']:.4f}",
                     fontsize=12.5, color=FG, y=1.0)
        plt.tight_layout()
        f = f"{OUT}/gradcam_all_{it['id']}.png"
        plt.savefig(f, dpi=150, bbox_inches="tight"); plt.close()
        print(f"저장: {os.path.basename(f)}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbones", nargs="+", default=["pointnet", "dgcnn", "regdgcnn"])
    p.add_argument("--npoints", type=int, default=1024)
    p.add_argument("--n", type=int, default=3)
    main(p.parse_args())
