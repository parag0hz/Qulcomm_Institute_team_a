"""크로스모달 정합 실증 (Future Work 검증) — CLIP식 파라미터↔형상 대조학습.

페어: 파라메트릭 CSV(기하 23변수) ∩ 포인트클라우드 미러 = 3,709대.
공식 split 준수 (train 페어로만 학습, val 페어로 조기종료, test 페어로 평가).

판정:
  1. 검색 top-1/top-5 — test 페어 N개 중 형상 잠재로 올바른 파라미터 행 찾기 (무작위 1/N)
  2. 역추정 프로브 — z_shape -> 23변수 Ridge 회귀 R² (inverse design 실현성)
  3. Cd 선형 프로브 — z_shape -> Cd Ridge R² (사전학습 특징의 품질)

  python scripts/train_crossmodal.py
"""
from __future__ import annotations

import argparse
import csv as _csv
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/home/kwy00/qi")
import cd_common as C

CSV = "/home/kwy00/qi/data/DrivAerNet_ParametricData.csv"
AERO_COLS = {"Average Cd", "Std Cd", "Average Cl", "Std Cl",
             "Average Cl_f", "Std Cl_f", "Average Cl_r", "Std Cl_r"}


class ShapeEncoder(nn.Module):
    """PointNet 트렁크 (models_pc와 동일 구조) -> 128차원 잠재."""

    def __init__(self, emb: int = 1024, z: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, emb, 1), nn.BatchNorm1d(emb), nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(emb, z)

    def forward(self, x):                      # (B,N,3)
        return self.proj(self.mlp(x.transpose(1, 2)).amax(-1))


class ParamEncoder(nn.Module):
    def __init__(self, d_in: int, z: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(inplace=True),
                                 nn.Linear(256, 256), nn.ReLU(inplace=True),
                                 nn.Linear(256, z))

    def forward(self, x):
        return self.net(x)


def clip_loss(zs, zp, logit_scale):
    zs, zp = F.normalize(zs, dim=1), F.normalize(zp, dim=1)
    logits = logit_scale.exp() * zs @ zp.t()
    tgt = torch.arange(len(zs), device=zs.device)
    return (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt)) / 2


@torch.no_grad()
def retrieval(zs, zp, ks=(1, 5)):
    zs, zp = F.normalize(zs, dim=1), F.normalize(zp, dim=1)
    sim = zs @ zp.t()
    rank = sim.argsort(dim=1, descending=True)
    tgt = torch.arange(len(zs), device=zs.device).unsqueeze(1)
    return {f"top{k}": float((rank[:, :k] == tgt).any(1).float().mean()) for k in ks}


def main(a):
    torch.manual_seed(0); np.random.seed(0)
    # ---- 페어 구성 ----
    d = np.load(a.cache, allow_pickle=True)
    pos = {C.norm_id(str(k)): i for i, k in enumerate(d["keys"])}
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rdr = _csv.DictReader(f)
        pcols = [c for c in rdr.fieldnames if c != "Experiment" and c not in AERO_COLS]
        rows = [(C.norm_id(r["Experiment"]), [float(r[c]) for c in pcols]) for r in rdr]
    pairs = [(pos[k], v) for k, v in rows if k in pos]
    idx = np.array([i for i, _ in pairs])
    P = np.array([v for _, v in pairs], dtype=np.float32)     # (M, 23)
    pts = d["pts"][idx, :a.npoints]
    split, cd = d["split"][idx], d["cd"][idx].astype(np.float64)
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    print(f"페어 {len(idx)}대 (train {tr.sum()} / val {va.sum()} / test {te.sum()}) · 변수 {P.shape[1]}개")

    pts = pts - pts[tr].reshape(-1, 3).mean(0)                # meter, train 기준 센터링
    pmu, psd = P[tr].mean(0), P[tr].std(0) + 1e-8
    P = (P - pmu) / psd

    enc_s = ShapeEncoder().cuda()
    enc_p = ParamEncoder(P.shape[1]).cuda()
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device="cuda"))
    opt = torch.optim.AdamW(list(enc_s.parameters()) + list(enc_p.parameters()) + [logit_scale],
                            lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-5)
    dl = DataLoader(TensorDataset(torch.from_numpy(pts[tr]), torch.from_numpy(P[tr])),
                    batch_size=a.bs, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    run = None
    if a.wandb:
        try:
            import wandb
            run = wandb.init(project="cfa", name=f"crossmodal_clip_n{a.npoints}",
                             tags=["crossmodal", "future-work"],
                             config={**vars(a), "n_pairs": len(idx), "n_params": P.shape[1]})
        except Exception as e:
            print(f"  (wandb 비활성: {e})")

    @torch.no_grad()
    def embed(mask):
        enc_s.eval(); enc_p.eval()
        zs, zp = [], []
        Pt, Xt = torch.from_numpy(P[mask]), torch.from_numpy(pts[mask])
        for i in range(0, mask.sum(), 256):
            zs.append(enc_s(Xt[i:i+256].cuda()))
            zp.append(enc_p(Pt[i:i+256].cuda()))
        return torch.cat(zs), torch.cat(zp)

    best, best_state, t0 = 0.0, None, time.time()
    for ep in range(a.epochs):
        enc_s.train(); enc_p.train()
        tot = 0.0
        for x, p in dl:
            opt.zero_grad(set_to_none=True)
            loss = clip_loss(enc_s(x.cuda(non_blocking=True)), enc_p(p.cuda(non_blocking=True)), logit_scale)
            loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        r = retrieval(*embed(va))
        if r["top1"] > best:
            best = r["top1"]
            best_state = ({k: v.clone() for k, v in enc_s.state_dict().items()},
                          {k: v.clone() for k, v in enc_p.state_dict().items()})
        if run:
            run.log({"epoch": ep, "loss": tot / tr.sum(), "val_top1": r["top1"], "val_top5": r["top5"]})
        if ep % 10 == 0:
            print(f"  ep{ep:3d} loss={tot/tr.sum():.3f} val_top1={r['top1']*100:.1f}% "
                  f"top5={r['top5']*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    enc_s.load_state_dict(best_state[0]); enc_p.load_state_dict(best_state[1])

    # ---- 판정 1: test 검색 ----
    zs_te, zp_te = embed(te)
    r = retrieval(zs_te, zp_te)
    n_te = int(te.sum())
    print(f"\n[1] 검색 (test {n_te}쌍, 무작위 top-1 = {100/n_te:.2f}%)")
    print(f"    형상→파라미터  top-1 {r['top1']*100:.1f}%  top-5 {r['top5']*100:.1f}%")

    # ---- 판정 2/3: 프로브 ----
    from sklearn.linear_model import Ridge
    zs_tr, _ = embed(tr)
    Ztr, Zte = zs_tr.cpu().numpy(), zs_te.cpu().numpy()
    ridge = Ridge(alpha=1.0).fit(Ztr, P[tr])
    r2p = 1 - ((ridge.predict(Zte) - P[te]) ** 2).sum(0) / ((P[te] - P[te].mean(0)) ** 2).sum(0)
    print(f"[2] 역추정 z→변수23: 평균 R² {r2p.mean():.3f} (최고 {r2p.max():.2f} / 최저 {r2p.min():.2f})")
    ridge_cd = Ridge(alpha=1.0).fit(Ztr, cd[tr])
    yh = ridge_cd.predict(Zte)
    r2cd = 1 - ((yh - cd[te]) ** 2).sum() / ((cd[te] - cd[te].mean()) ** 2).sum()
    print(f"[3] Cd 선형 프로브: R² {r2cd:.3f} (파라미터 없이, 대조학습 특징만으로)")

    if run:
        run.summary.update({"test_top1": r["top1"], "test_top5": r["top5"],
                            "inverse_r2_mean": float(r2p.mean()), "cd_probe_r2": float(r2cd)})
        run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npoints", type=int, default=2048)
    p.add_argument("--cache", default="/home/kwy00/qi/data/fps2048.npz")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--wandb", type=int, default=1)
    main(p.parse_args())
