"""가림 실험(occlusion) — 모델이 차체의 어느 부위에 실제로 의존하는가.

saliency/Grad-CAM은 "어디를 보는가"의 상관 관찰이고, 아키텍처 부산물에 오염된다
(예: PointNet의 max-pool은 기울기를 소수 점에만 흘려 희소해 보인다).
가림 실험은 **인과적**이다: 그 부위를 없앴을 때 예측이 실제로 얼마나 변하는가.
게다가 정의가 모델과 무관해 **백본 간 직접 비교**가 된다.

방법:
  구역의 점을 제거하고, 남은 점에서 재표집해 **점 개수 N을 고정**한다
  (N을 줄이면 밀도 변화 자체가 교란이 되므로).
  ΔCd = |가린 예측 − 원본 예측|  (drag counts)

구역:
  세로 5분할  앞범퍼 / 앞유리·보닛 / 지붕중앙 / 뒷유리·트렁크 / 후면
  높이 3분할  하부(언더바디) / 중간 / 상부(지붕)

⚠ 한계: 가린 입력은 학습 분포 밖(OOD)이라, ΔCd에는 "중요도"와 "OOD 정도"가 섞인다.
   따라서 절대값보다 **구역 간·모델 간 상대 비교**로 읽어야 한다.

  python scripts/occlusion_test.py --backbones pointnet triplane
결과: outputs/occlusion.png, outputs/occlusion.json
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
sys.path.insert(0, "/home/kwy00/qi/scripts")
from models_pc import BACKBONES

OUT = "/home/kwy00/qi/outputs"
BG, CARD, FG, SUB, ACC = "#141414", "#1C1C1C", "#E4E1DB", "#B9B6B0", "#D98A3D"
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB,
    "axes.edgecolor": "#3a3a3a"})
NAME = {"pointnet": "PointNet", "dgcnn": "DGCNN", "regdgcnn": "RegDGCNN", "triplane": "Triplane"}

ZONES = [  # (라벨, 축, 하한비율, 상한비율)
    ("앞범퍼",        0, 0.0, 0.2),
    ("보닛·앞유리",   0, 0.2, 0.4),
    ("지붕 중앙",     0, 0.4, 0.6),
    ("뒷유리·트렁크", 0, 0.6, 0.8),
    ("후면",          0, 0.8, 1.0),
    ("하부",          2, 0.0, 0.33),
    ("중간 높이",     2, 0.33, 0.66),
    ("상부(지붕)",    2, 0.66, 1.0),
]


def zone_mask(p, axis, lo, hi):
    v = p[:, axis]
    t = (v - v.min()) / (np.ptp(v) + 1e-12)
    return (t >= lo) & (t < hi if hi < 1.0 else t <= 1.0)


def occlude(p, m, rng):
    """구역 점 제거 후 남은 점에서 재표집해 N 고정."""
    keep = np.where(~m)[0]
    if len(keep) < 32:
        return None
    return p[rng.choice(keep, len(p), replace=True)]


def load_net(bb, npoints=1024):
    f = f"{OUT}/backbone_{bb}_{npoints}.pt"
    if not os.path.exists(f):
        return None, None
    ck = torch.load(f, weights_only=False)
    kw = {"n_dims": 0}
    if bb == "pointnet" and "config" in ck:
        kw["emb"] = ck["config"]["emb"]
    net = BACKBONES[bb](**kw)
    net.load_state_dict(ck["state_dict"])
    return net.eval().cuda(), ck


@torch.no_grad()
def predict(net, ck, p):
    ctr = torch.tensor(ck["center"], dtype=torch.float32).view(1, 1, 3).cuda()
    x = torch.from_numpy(p[None].astype(np.float32)).cuda()
    return float(net(x - ctr)) * ck["ysd"] + ck["ymu"]


def main(a):
    demo = json.load(open("/home/kwy00/qi/data/demo_holdout.json"))["items"][:a.n]
    d = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
    keys = np.array([str(k) for k in d["keys"]])
    rng = np.random.default_rng(0)

    res, avail = {}, []
    for bb in a.backbones:
        net, ck = load_net(bb, a.npoints)
        if net is None:
            print(f"⚠ {bb}: 가중치 없음 — 건너뜀"); continue
        avail.append(bb)
        per_zone = {z[0]: [] for z in ZONES}; per_zone["무작위(대조)"] = []
        for it in demo:
            i = int(np.where(keys == it["id"])[0][0])
            p = d["pts"][i, :a.npoints].astype(np.float32)
            base = predict(net, ck, p)
            for lab, ax_, lo, hi in ZONES:
                m = zone_mask(p, ax_, lo, hi)
                q = occlude(p, m, rng)
                if q is None:
                    continue
                per_zone[lab].append(abs(predict(net, ck, q) - base) * 1000)
                # 대조군: 같은 개수를 무작위로 제거 (위치와 무관한 OOD 효과)
                rm = np.zeros(len(p), bool)
                rm[rng.choice(len(p), int(m.sum()), replace=False)] = True
                qr = occlude(p, rm, rng)
                if qr is not None:
                    per_zone["무작위(대조)"].append(abs(predict(net, ck, qr) - base) * 1000)
        res[bb] = {k: float(np.mean(v)) for k, v in per_zone.items() if v}
        del net
        torch.cuda.empty_cache()
        print(f"{NAME[bb]:<10} 완료")

    if not avail:
        print("비교할 모델이 없다."); return

    # ---------- 출력 ----------
    labs = [z[0] for z in ZONES] + ["무작위(대조)"]
    print(f"\n=== 가림 실험: 구역 제거 시 |ΔCd| (drag counts, 데모 {len(demo)}대 평균) ===")
    print(f"{'구역':<16}" + "".join(f"{NAME[b]:>12}" for b in avail))
    print("-" * (16 + 12 * len(avail)))
    for L in labs:
        print(f"{L:<16}" + "".join(f"{res[b].get(L, float('nan')):>12.1f}" for b in avail))
    print()
    for b in avail:
        v = res[b]
        top = max(v, key=v.get)
        spread = max(v.values()) / (min(v.values()) + 1e-9)
        print(f"  {NAME[b]:<10} 최대의존 '{top}' ({v[top]:.1f} counts) · "
              f"최대/최소 비 {spread:.1f}배  → {'특정 부위 편중' if spread > 3 else '고르게 분산'}")

    # ---------- 그림 ----------
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6),
                           gridspec_kw={"width_ratios": [1, 1.5]})
    # (좌) 구역 도식
    i = int(np.where(keys == demo[0]["id"])[0][0])
    p = d["pts"][i, :a.npoints]
    cols = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, 5))
    for k, (lab, ax_, lo, hi) in enumerate(ZONES[:5]):
        m = zone_mask(p, ax_, lo, hi)
        ax[0].scatter(p[m, 0], p[m, 2], s=3, color=cols[k], linewidths=0, label=lab)
    ax[0].set_aspect("equal"); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("세로 5분할 구역", fontsize=11.5, color=FG, pad=8)
    ax[0].legend(facecolor=CARD, edgecolor="#3a3a3a", labelcolor=FG, fontsize=8,
                 loc="upper center", ncols=2)
    # (우) 구역별 ΔCd
    x = np.arange(len(labs)); w = 0.8 / len(avail)
    palette = [ACC, "#4FA3AB", "#8a5f2e", "#6b7280"]
    for k, b in enumerate(avail):
        ax[1].bar(x + (k - (len(avail) - 1) / 2) * w,
                  [res[b].get(L, 0) for L in labs], w, label=NAME[b], color=palette[k % 4])
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs, rotation=20, ha="right", fontsize=9.5)
    ax[1].set_ylabel("제거 시 |ΔCd| (drag counts)")
    ax[1].set_title("구역을 가렸을 때 예측 변화 — 클수록 그 부위에 의존", fontsize=11.5, color=FG, pad=8)
    ax[1].legend(facecolor=CARD, edgecolor="#3a3a3a", labelcolor=FG, fontsize=9)
    ax[1].spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/occlusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    with open(f"{OUT}/occlusion.json", "w") as f:
        json.dump({NAME[b]: res[b] for b in avail}, f, indent=1, ensure_ascii=False)
    print(f"\n저장: {OUT}/occlusion.png, occlusion.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbones", nargs="+", default=["pointnet", "dgcnn", "regdgcnn"])
    p.add_argument("--npoints", type=int, default=1024)
    p.add_argument("--n", type=int, default=5)
    main(p.parse_args())
