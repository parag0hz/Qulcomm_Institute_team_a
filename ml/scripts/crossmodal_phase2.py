"""크로스모달 2단계 — 가능성 입증 실험 3종.

  1) 사전학습 재실행 + 인코더 저장 (CLIP식, train 페어)
  2) 역추정 MLP 디코더: z_shape -> 변수23, 변수별 R²
  3) 루프 폐쇄: 형상 -> 예측 파라미터 -> XGBoost(트랙A 대역) -> Cd, 실제 Cd와 비교
  4) 저라벨 파인튜닝: {5,10,25}% 라벨 × {pretrained, scratch}, 공식 test R²

  python scripts/crossmodal_phase2.py
결과: outputs/crossmodal_phase2.json + wandb(tags: crossmodal)
"""
from __future__ import annotations

import csv as _csv
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/home/kwy00/qi")
import cd_common as C
from train_crossmodal import ShapeEncoder, ParamEncoder, clip_loss, retrieval, CSV, AERO_COLS

OUT = "/home/kwy00/qi/outputs"
res: dict = {}


def r2(a, b):
    return 1 - float(((a - b) ** 2).sum()) / float(((b - b.mean()) ** 2).sum())


# ============================== 데이터 ==============================
torch.manual_seed(0); np.random.seed(0)
d = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
pos = {C.norm_id(str(k)): i for i, k in enumerate(d["keys"])}
with open(CSV, encoding="utf-8-sig", newline="") as f:
    rdr = _csv.DictReader(f)
    pcols = [c for c in rdr.fieldnames if c != "Experiment" and c not in AERO_COLS]
    rows = [(C.norm_id(r["Experiment"]), [float(r[c]) for c in pcols]) for r in rdr]
pairs = [(pos[k], v) for k, v in rows if k in pos]
pidx = np.array([i for i, _ in pairs])
P = np.array([v for _, v in pairs], dtype=np.float32)

ALL_pts = d["pts"][:, :2048]
ALL_split, ALL_cd, ALL_cls = d["split"], d["cd"].astype(np.float64), d["cls"]
ALL_tr, ALL_te = ALL_split == "train", ALL_split == "test"
center = ALL_pts[ALL_tr].reshape(-1, 3).mean(0)
ALL_pts = ALL_pts - center

pts = ALL_pts[pidx]
split, cd = ALL_split[pidx], ALL_cd[pidx]
tr, va, te = (split == "train"), (split == "val"), (split == "test")
pmu, psd = P[tr].mean(0), P[tr].std(0) + 1e-8
Pz = (P - pmu) / psd

# ====================== 1) 사전학습 (저장 포함) ======================
enc_s, enc_p = ShapeEncoder().cuda(), ParamEncoder(Pz.shape[1]).cuda()
logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device="cuda"))
opt = torch.optim.AdamW(list(enc_s.parameters()) + list(enc_p.parameters()) + [logit_scale], lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100, eta_min=1e-5)
dl = DataLoader(TensorDataset(torch.from_numpy(pts[tr]), torch.from_numpy(Pz[tr])),
                batch_size=64, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

@torch.no_grad()
def embed(mask):
    enc_s.eval()
    zs = [enc_s(torch.from_numpy(pts[mask][i:i+256]).cuda()) for i in range(0, mask.sum(), 256)]
    return torch.cat(zs)

best, best_sd = 0.0, None
t0 = time.time()
for ep in range(100):
    enc_s.train(); enc_p.train()
    for x, p_ in dl:
        opt.zero_grad(set_to_none=True)
        clip_loss(enc_s(x.cuda(non_blocking=True)), enc_p(p_.cuda(non_blocking=True)), logit_scale).backward()
        opt.step()
    sched.step()
    enc_p.eval()
    with torch.no_grad():
        zp = torch.cat([enc_p(torch.from_numpy(Pz[va][i:i+256]).cuda()) for i in range(0, va.sum(), 256)])
    r = retrieval(embed(va), zp)
    if r["top1"] > best:
        best, best_sd = r["top1"], {k: v.clone() for k, v in enc_s.state_dict().items()}
enc_s.load_state_dict(best_sd)
torch.save(best_sd, f"{OUT}/crossmodal_shape_enc.pt")
print(f"[1] 사전학습 완료 ({time.time()-t0:.0f}s, val top1 {best*100:.1f}%)", flush=True)

Ztr, Zva, Zte = embed(tr).cpu(), embed(va).cpu(), embed(te).cpu()

# ==================== 2) 역추정 MLP 디코더 ====================
dec = nn.Sequential(nn.Linear(128, 256), nn.ReLU(True), nn.Linear(256, 256), nn.ReLU(True),
                    nn.Linear(256, Pz.shape[1])).cuda()
optd = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-4)
dld = DataLoader(TensorDataset(Ztr, torch.from_numpy(Pz[tr])), batch_size=128, shuffle=True)
best_v, best_dsd = 9e9, None
for ep in range(200):
    dec.train()
    for z, p_ in dld:
        optd.zero_grad(set_to_none=True)
        F.mse_loss(dec(z.cuda()), p_.cuda()).backward()
        optd.step()
    dec.eval()
    with torch.no_grad():
        v = float(F.mse_loss(dec(Zva.cuda()), torch.from_numpy(Pz[va]).cuda()))
    if v < best_v:
        best_v, best_dsd = v, {k: t.clone() for k, t in dec.state_dict().items()}
dec.load_state_dict(best_dsd); dec.eval()
with torch.no_grad():
    Pd_te = dec(Zte.cuda()).cpu().numpy()
r2_each = 1 - ((Pd_te - Pz[te]) ** 2).sum(0) / ((Pz[te] - Pz[te].mean(0)) ** 2).sum(0)
order = np.argsort(-r2_each)
res["inverse_mlp_mean_r2"] = float(r2_each.mean())
res["inverse_per_param"] = {pcols[i]: float(r2_each[i]) for i in order}
print(f"[2] 역추정 MLP: 평균 R² {r2_each.mean():.3f}", flush=True)
for i in order[:5]: print(f"      {pcols[i]:<28}{r2_each[i]:+.2f}")
for i in order[-3:]: print(f"      {pcols[i]:<28}{r2_each[i]:+.2f}")

# ==================== 3) 루프 폐쇄 ====================
import xgboost as xgb
P_pred_te = Pd_te * psd + pmu                                   # 역표준화된 예측 파라미터
tabA = xgb.XGBRegressor(objective="reg:squarederror", random_state=42).fit(P[tr], cd[tr])
r2_true = r2(tabA.predict(P[te]), cd[te])                       # 진짜 파라미터로
r2_chain = r2(tabA.predict(P_pred_te.astype(np.float32)), cd[te])  # 형상→예측 파라미터로
res["loop_tabA_true_params_r2"] = r2_true
res["loop_tabA_pred_params_r2"] = r2_chain
print(f"[3] 루프 폐쇄: 트랙A(XGB) — 진짜 파라미터 R² {r2_true:.3f} vs 형상→역추정 파라미터 R² {r2_chain:.3f}", flush=True)

# ==================== 4) 저라벨 파인튜닝 ====================
def head():
    return nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.3),
                         nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.3),
                         nn.Linear(256, 1))

class Reg(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        e = ShapeEncoder()
        if pretrained:
            e.load_state_dict(best_sd)
        self.trunk, self.head = e.mlp, head()
    def forward(self, x):
        return self.head(self.trunk(x.transpose(1, 2)).amax(-1)).squeeze(-1)

ymu, ysd = ALL_cd[ALL_tr].mean(), ALL_cd[ALL_tr].std()
tr_ids = np.where(ALL_tr)[0]
va_ids = np.where(ALL_split == "val")[0]
te_ids = np.where(ALL_te)[0]
Xva = torch.from_numpy(ALL_pts[va_ids]); Xte = torch.from_numpy(ALL_pts[te_ids])
res["finetune"] = {}
for frac in [0.05, 0.10, 0.25]:
    rng = np.random.default_rng(0)
    sub = rng.choice(tr_ids, int(len(tr_ids) * frac), replace=False)
    for tag, pre in [("scratch", False), ("pretrained", True)]:
        torch.manual_seed(0)
        net = Reg(pre).cuda()
        o = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(o, 60, eta_min=1e-5)
        lf = nn.SmoothL1Loss(beta=1.0)
        dlt = DataLoader(TensorDataset(torch.from_numpy(ALL_pts[sub]),
                                       torch.from_numpy(((ALL_cd[sub]-ymu)/ysd).astype(np.float32))),
                         batch_size=32, shuffle=True, drop_last=len(sub) > 32, num_workers=2)
        @torch.no_grad()
        def infer(X):
            net.eval()
            o_ = [net(X[i:i+128].cuda()).cpu() for i in range(0, len(X), 128)]
            return torch.cat(o_).numpy() * ysd + ymu
        bv, bsd2, bad = -9e9, None, 0
        for ep in range(60):
            net.train()
            for x, y in dlt:
                o.zero_grad(set_to_none=True)
                lf(net(x.cuda(non_blocking=True)), y.cuda(non_blocking=True)).backward()
                o.step()
            sc.step()
            vr = r2(infer(Xva), ALL_cd[va_ids])
            if vr > bv: bv, bsd2, bad = vr, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else: bad += 1
            if bad >= 15: break
        net.load_state_dict(bsd2)
        tr2 = r2(infer(Xte), ALL_cd[te_ids])
        res["finetune"][f"{int(frac*100)}pct_{tag}"] = tr2
        print(f"[4] 라벨 {int(frac*100):>2}% ({len(sub)}대) {tag:<10} test R² {tr2:+.3f}", flush=True)

with open(f"{OUT}/crossmodal_phase2.json", "w") as f:
    json.dump(res, f, indent=1)
try:
    import wandb
    run = wandb.init(project="cfa", name="crossmodal_phase2", tags=["crossmodal", "future-work"],
                     config={"n_pairs": len(pidx)})
    run.summary.update({k: v for k, v in res.items() if not isinstance(v, dict)}
                       | {f"ft/{k}": v for k, v in res["finetune"].items()}
                       | {"inverse_mlp_mean_r2": res["inverse_mlp_mean_r2"]})
    run.finish()
except Exception as e:
    print(f"(wandb 비활성: {e})")
print("\nPhase2 완료 ->", f"{OUT}/crossmodal_phase2.json")
