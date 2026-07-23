"""100k 전체 점군(샘플링 없음) 학습 — FPS-2048과의 성능 비교 실험.

train_r2.py와 동일 레시피 (AdamW 1e-3, SmoothL1, cosine, 120ep, patience 30, meter 유지).
배치는 32에서 시작해 OOM 시 절반으로 낮춰 재시도 (표준 하이퍼파라미터 조정).
bs=1에서도 OOM이면 그 백본은 불가로 기록한다 — gradient checkpointing / AMP /
accumulation 같은 메모리 우회 기법은 쓰지 않는다 (실험 조건을 바꾸므로).

  python scripts/train_100k.py --build-cache     # 1회: 7,713 × .paddle_tensor -> memmap 9.3GB
  python scripts/train_100k.py --backbone pointnet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/home/kwy00/qi")
from models_pc import BACKBONES

N_PTS = 100_000
_DATA = os.environ.get("QI_DATA", "/home/kwy00/qi/data")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FPS2048 = os.path.join(_DATA, "fps2048.npz")
CACHE = os.path.join(_DATA, "pc100k_f32.dat")
META = os.path.join(_DATA, "pc100k_meta.json")
CLASSES = ["Fastback", "Estate", "Notchback"]


def r2(yh, y):
    return 1 - float(((yh - y) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def mae_pct(yh, y):
    return float(np.mean(np.abs(yh - y) / y) * 100)


# ============================ 캐시 ============================

def build_cache():
    """fps2048.npz와 같은 sorted(files) 순서로 100k 원본 점군을 memmap에 적재."""
    from concurrent.futures import ProcessPoolExecutor
    import cd_common as C

    C.check_integrity()
    files = C.file_index()
    keys = sorted(files)
    d = np.load(_FPS2048, allow_pickle=True)
    assert len(keys) == len(d["keys"]), "fps2048.npz와 파일 수 불일치"
    assert all(f"{p}_{i}" == str(k) for (p, i), k in zip(keys, d["keys"])), "키 순서 불일치"

    mm = np.memmap(CACHE, dtype=np.float32, mode="w+", shape=(len(keys), N_PTS, 3))
    t0 = time.time()
    B = 64
    with ProcessPoolExecutor(max_workers=12) as ex:
        for b0 in range(0, len(keys), B):
            chunk = keys[b0:b0 + B]
            for j, p in enumerate(ex.map(C.safe_load_ndarray, [files[k] for k in chunk])):
                assert p.shape == (N_PTS, 3)
                mm[b0 + j] = p
            if b0 % (B * 20) == 0:
                done = b0 + len(chunk)
                el = time.time() - t0
                print(f"  {done:5d}/{len(keys)}  {el:6.0f}s  eta {el/done*(len(keys)-done):5.0f}s", flush=True)
    mm.flush()

    tr = d["split"] == "train"
    s = np.zeros(3, dtype=np.float64)
    for i in np.where(tr)[0]:
        s += np.asarray(mm[i], dtype=np.float64).mean(0)
    center = (s / tr.sum()).tolist()
    with open(META, "w") as f:
        json.dump({"n": len(keys), "center_train": center}, f)
    print(f"저장 {CACHE} ({len(keys)}×{N_PTS}×3, {len(keys)*N_PTS*12/2**30:.1f} GiB) "
          f"center={np.round(center,4).tolist()} ({time.time()-t0:.0f}s)")


class PC100k(Dataset):
    def __init__(self, idx, yz):
        self.idx, self.yz = idx, yz.astype(np.float32)
        self.mm = np.memmap(CACHE, dtype=np.float32, mode="r", shape=(len(np.load(
            _FPS2048, allow_pickle=True)["keys"]), N_PTS, 3))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return torch.from_numpy(np.array(self.mm[j])), self.yz[i]


# ============================ 학습 ============================

def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    d = np.load(_FPS2048, allow_pickle=True)
    cd, split, cls, keys = d["cd"], d["split"], d["cls"], d["keys"]
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    center = torch.tensor(json.load(open(META))["center_train"], dtype=torch.float32).cuda()

    ymu, ysd = cd[tr].mean(), cd[tr].std()
    yz = (cd - ymu) / ysd
    mm = np.memmap(CACHE, dtype=np.float32, mode="r", shape=(len(keys), N_PTS, 3))

    tag = f"{a.backbone}_n100000_dims0_meter"
    net = BACKBONES[a.backbone](n_dims=0).cuda()
    nparam = sum(p.numel() for p in net.parameters())
    print(f"=== {tag}  params={nparam/1e6:.2f}M  train={tr.sum()} val={va.sum()} test={te.sum()}", flush=True)

    # ---- 배치 프로브: 32 -> 1 절반씩. 실패 시 OOM 보고 후 종료 ----
    tr_idx = np.where(tr)[0]
    bs, oom_msgs = a.bs, []
    while True:
        try:
            xb = torch.from_numpy(np.array(mm[tr_idx[:bs]])).cuda() - center
            yb = torch.from_numpy(yz[tr_idx[:bs]].astype(np.float32)).cuda()
            loss = nn.functional.smooth_l1_loss(net(xb), yb)
            loss.backward()
            net.zero_grad(set_to_none=True)
            del xb, yb, loss
            torch.cuda.empty_cache()
            print(f"  배치 프로브 통과: bs={bs}", flush=True)
            break
        except torch.cuda.OutOfMemoryError as e:
            msg = str(e).split("\n")[0]
            oom_msgs.append(f"bs={bs}: {msg}")
            print(f"  OOM @ bs={bs}", flush=True)
            net.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if bs == 1:
                res = {"tag": tag, "params": nparam, "oom": True, "oom_log": oom_msgs,
                       "note": "bs=1에서도 OOM — 이 백본은 100k 입력으로 학습 불가 (우회 기법 미사용)"}
                with open(os.path.join(_REPO, "outputs", f"{tag}.json"), "w") as f:
                    json.dump(res, f, indent=1, ensure_ascii=False)
                print(f"  -> 불가 보고: outputs/{tag}.json", flush=True)
                return
            bs //= 2

    dl = DataLoader(PC100k(tr_idx, yz[tr]), batch_size=bs, shuffle=True, drop_last=True,
                    num_workers=6, pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-5)
    lossf = nn.SmoothL1Loss(beta=1.0)

    @torch.no_grad()
    def infer(mask):
        net.eval()
        ids = np.where(mask)[0]
        cs = max(bs, 4)
        out = []
        for i in range(0, len(ids), cs):
            xb = torch.from_numpy(np.array(mm[ids[i:i+cs]])).cuda() - center
            out.append(net(xb).cpu())
        return torch.cat(out).numpy() * ysd + ymu

    run = None
    if a.wandb:
        try:
            import wandb
            run = wandb.init(project="cfa", name=tag, tags=["n100k"],
                             config={**vars(a), "bs_effective": bs, "params": nparam})
        except Exception as e:
            print(f"  (wandb 비활성 — 학습은 계속: {e})")

    best, best_state, bad = -9e9, None, 0
    t0 = time.time()
    try:
        for ep in range(a.epochs):
            net.train()
            tot = 0.0
            for x, y in dl:
                x = x.cuda(non_blocking=True) - center
                y = y.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = lossf(net(x), y)
                loss.backward(); opt.step()
                tot += loss.item() * len(x)
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
            print(f"  ep{ep:3d} loss={tot/tr.sum():.4f} valR2={vr:+.4f} best={best:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if bad >= a.patience:
                print(f"  early stop @ep{ep}"); break
    except torch.cuda.OutOfMemoryError as e:
        res = {"tag": tag, "params": nparam, "oom": True, "bs": bs,
               "oom_log": oom_msgs + [f"학습 중 OOM: {str(e).split(chr(10))[0]}"],
               "note": "프로브는 통과했으나 학습 중 OOM — 불가로 기록"}
        with open(os.path.join(_REPO, "outputs", f"{tag}.json"), "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        print(f"  -> 학습 중 OOM, 불가 보고: outputs/{tag}.json", flush=True)
        if run:
            run.finish(exit_code=1)
        return

    net.load_state_dict(best_state)
    yh, y = infer(te), cd[te]
    res = {"tag": tag, "params": nparam, "val_r2": best, "bs_effective": bs,
           "epoch_time_s": round((time.time() - t0) / max(1, ep + 1), 1),
           "test": {"All": [r2(yh, y), mae_pct(yh, y)]}}
    for c in CLASSES:
        s = cls[te] == c
        res["test"][c] = [r2(yh[s], y[s]), mae_pct(yh[s], y[s])]
    print(f"\n  {'':<12}{'R2':>10}{'MAE%':>9}")
    for c in ["All"] + CLASSES:
        rr, m = res["test"][c]
        print(f"  {c:<12}{rr:>+10.3f}{m:>9.2f}")
    with open(os.path.join(_REPO, "outputs", f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    np.savez(os.path.join(_REPO, "outputs", f"{tag}_pred.npz"), yh=yh, y=y, keys=keys[te], cls=cls[te])
    print(f"\n  -> outputs/{tag}.json   ({time.time()-t0:.0f}s)")
    if run:
        run.summary.update({"val_r2_best": best, "bs_effective": bs}
                           | {f"test/{c}_r2": v[0] for c, v in res["test"].items()}
                           | {f"test/{c}_mae_pct": v[1] for c, v in res["test"].items()})
        run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-cache", action="store_true")
    p.add_argument("--backbone", default="pointnet", choices=list(BACKBONES))
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", type=int, default=1)
    if "--build-cache" in sys.argv:
        build_cache()
    else:
        main(p.parse_args())
