"""배포용 PointNet 학습 + ONNX 추출.

현재 평가 프로토콜(교집합 3,704대, K=5 rotating)과 **동일한 데이터·전처리**로 학습한 뒤
서빙용 ONNX를 만든다. Optuna로 찾은 최적 하이퍼파라미터를 기본값으로 쓴다.

핵심 설계 — 전처리를 ONNX 그래프에 내장:
  입력  (B, N, 3) float32, **미터 스케일 원본 좌표** (센터링 불필요)
  내부  학습 시 사용한 center 상수를 빼고 → PointNet → 표준화 해제
  출력  (B,) float32 = Cd
  → 서빙 코드가 전처리를 틀릴 여지를 없앤다 (가장 흔한 서빙 버그).

  python scripts/export_pointnet_onnx.py                    # 기본: 2048점, 튜닝된 파라미터
  python scripts/export_pointnet_onnx.py --npoints 1024
결과: outputs/pointnet_serving.onnx, outputs/pointnet_serving.pt, outputs/pointnet_serving.json
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

sys.path.insert(0, "/home/kwy00/qi")
sys.path.insert(0, "/home/kwy00/qi/scripts")
from models_pc import PointNet
from protocol import load_dataset, make_folds, split_indices, evaluate

OUT = "/home/kwy00/qi/outputs"
# Optuna 탐색 결과 (outputs/optuna_results.json, best val R² 0.892)
BEST = {"lr": 0.0012453, "weight_decay": 2.1069e-05, "bs": 32, "dropout": 0.4, "emb": 512}


class ServingPointNet(nn.Module):
    """전처리(센터링)와 후처리(역표준화)를 내장한 서빙 래퍼."""

    def __init__(self, net: nn.Module, center: np.ndarray, ymu: float, ysd: float):
        super().__init__()
        self.net = net
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32).view(1, 1, 3))
        self.register_buffer("ymu", torch.tensor(float(ymu), dtype=torch.float32))
        self.register_buffer("ysd", torch.tensor(float(ysd), dtype=torch.float32))

    def forward(self, pts):                 # (B, N, 3) 미터 스케일 원본
        z = self.net(pts - self.center)     # 학습과 동일한 상수 평행이동
        return z * self.ysd + self.ymu      # 표준화 해제 → Cd


def main(a):
    torch.manual_seed(0); np.random.seed(0)
    ds = load_dataset(npoints=a.npoints,
                      cache=("/home/kwy00/qi/data/fps4096.npz" if a.npoints > 2048
                             else "/home/kwy00/qi/data/fps2048.npz"))
    sets = split_indices(make_folds(ds["cls"]))
    pts, y, cls, keys = ds["pts"], ds["cd"], ds["cls"], ds["keys"]

    # 배포 모델: fold 1~4 학습 / fold 5 조기종료용 검증 (데이터 80% 활용)
    s = sets[0]
    tr = np.concatenate([s["train"], s["test"]])     # 세트1의 train(3fold) + test(1fold) = 4 fold
    va = s["val"]
    print(f"배포용 학습: train {len(tr):,} / val(조기종료) {len(va):,} · {a.npoints}점")
    print(f"하이퍼파라미터: {BEST}")

    center = pts[tr].reshape(-1, 3).mean(0)
    ymu, ysd = float(y[tr].mean()), float(y[tr].std())
    P = pts.astype(np.float32)                        # 센터링은 모델 내부에서
    yz = ((y - ymu) / ysd).astype(np.float32)

    net = PointNet(n_dims=0, emb=BEST["emb"]).cuda()
    for m in net.head.modules():
        if isinstance(m, nn.Dropout):
            m.p = BEST["dropout"]
    opt = torch.optim.AdamW(net.parameters(), lr=BEST["lr"], weight_decay=BEST["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-5)
    lossf = nn.SmoothL1Loss(beta=1.0)
    ctr = torch.tensor(center, dtype=torch.float32).view(1, 1, 3).cuda()
    dl = DataLoader(TensorDataset(torch.from_numpy(P[tr]), torch.from_numpy(yz[tr])),
                    batch_size=BEST["bs"], shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    @torch.no_grad()
    def infer(idx):
        net.eval()
        o = [net(torch.from_numpy(P[idx[i:i+128]]).cuda() - ctr).cpu() for i in range(0, len(idx), 128)]
        return torch.cat(o).numpy() * ysd + ymu

    best, best_sd, bad, t0 = -9e9, None, 0, time.time()
    for ep in range(a.epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            lossf(net(xb.cuda(non_blocking=True) - ctr), yb.cuda(non_blocking=True)).backward()
            opt.step()
        sch.step()
        yh = infer(va)
        vr = 1 - ((yh - y[va]) ** 2).sum() / ((y[va] - y[va].mean()) ** 2).sum()
        if vr > best:
            best, bad = vr, 0
            best_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 20 == 0 or ep == a.epochs - 1:
            print(f"  ep{ep:3d} valR2={vr:+.4f} best={best:+.4f} ({time.time()-t0:.0f}s)", flush=True)
        if bad >= a.patience:
            print(f"  early stop @ep{ep}"); break
    net.load_state_dict(best_sd)
    print(f"학습 완료: val R² {best:+.4f} ({time.time()-t0:.0f}s)")

    # ---------- 서빙 래퍼 + ONNX ----------
    serve = ServingPointNet(net.cpu().eval(), center, ymu, ysd).eval()
    dummy = torch.from_numpy(P[tr[:2]])                       # (2, N, 3) 미터 원본
    with torch.no_grad():
        ref = serve(dummy).numpy()

    onnx_path = f"{OUT}/pointnet_serving.onnx"
    torch.onnx.export(
        serve, (dummy,), onnx_path,
        input_names=["points"], output_names=["cd"],
        dynamic_axes={"points": {0: "batch", 1: "num_points"}, "cd": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    # 가중치를 단일 파일에 내장 (기본 exporter는 외부 .onnx.data로 분리 → 배포 시 깨짐)
    import onnx as _onnx, os as _os
    _m = _onnx.load(onnx_path)                       # 외부 데이터까지 로드
    _onnx.save_model(_m, onnx_path, save_as_external_data=False)
    _ext = onnx_path + ".data"
    if _os.path.exists(_ext):
        _os.remove(_ext)
    print(f"\nONNX 저장: {onnx_path} ({_os.path.getsize(onnx_path)/1048576:.2f} MB, 가중치 내장)")

    # ---------- 검증: ONNX vs PyTorch ----------
    import onnx, onnxruntime as ort
    onnx.checker.check_model(onnx.load(onnx_path))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(["cd"], {"points": dummy.numpy()})[0]
    diff = float(np.abs(got - ref).max())
    print(f"검증: PyTorch vs ONNX 최대 차이 {diff:.2e}  "
          f"{'✅ 일치' if diff < 1e-5 else '⚠️ 불일치'}")

    # 가변 배치/점수 확인
    for b, n in [(1, a.npoints), (5, a.npoints)]:
        out = sess.run(["cd"], {"points": P[tr[:b], :n]})[0]
        assert out.shape == (b,), out.shape
    print(f"가변 배치 확인: (1,{a.npoints},3) / (5,{a.npoints},3) 모두 정상")

    # ---------- 데모 5대 예측 ----------
    demo = json.load(open("/home/kwy00/qi/data/demo_holdout.json"))
    d2 = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
    dkeys = np.array([str(k) for k in d2["keys"]])
    print(f"\n=== 데모 홀드아웃 5대 (학습에 한 번도 안 쓰인 차) ===")
    print(f"{'ID':<22}{'차종':<12}{'실제 Cd':>10}{'예측 Cd':>10}{'오차(counts)':>14}")
    errs = []
    for it in demo["items"]:
        i = int(np.where(dkeys == it["id"])[0][0])
        x = d2["pts"][i:i+1, :a.npoints].astype(np.float32)
        p = float(sess.run(["cd"], {"points": x})[0][0])
        e = abs(p - it["true_cd"]) * 1000
        errs.append(e)
        print(f"{it['id']:<22}{it['body_type']:<12}{it['true_cd']:>10.4f}{p:>10.4f}{e:>14.1f}")
    print(f"{'평균 오차':<44}{np.mean(errs):>14.1f} counts")

    # ---------- 저장 ----------
    torch.save({"state_dict": net.state_dict(), "center": center,
                "ymu": ymu, "ysd": ysd, "npoints": a.npoints, "config": BEST},
               f"{OUT}/pointnet_serving.pt")
    meta = {
        "onnx": "pointnet_serving.onnx",
        "input": {"name": "points", "shape": ["batch", a.npoints, 3], "dtype": "float32",
                  "note": "미터 스케일 원본 좌표. 센터링/정규화 하지 말 것 (모델 내부 처리)"},
        "output": {"name": "cd", "shape": ["batch"], "dtype": "float32", "note": "Cd 예측값"},
        "preprocessing_required": "STL/메시 → 표면 샘플링 → FPS로 %d점 → 그대로 입력" % a.npoints,
        "trained_on": {"n_train": int(len(tr)), "n_val": int(len(va)), "npoints": a.npoints},
        "hyperparameters": BEST,
        "val_r2": float(best),
        "cv_reference": "동일 설정 5-fold 교차검증 R² 0.865 ± 0.038 (outputs/optuna_results.json)",
        "demo_mae_counts": float(np.mean(errs)),
        "onnx_pytorch_max_diff": diff,
    }
    with open(f"{OUT}/pointnet_serving.json", "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"\n저장: {OUT}/pointnet_serving.{{onnx,pt,json}}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npoints", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=30)
    main(p.parse_args())
