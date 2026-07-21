"""precompute_fps.py의 K 파라미터 버전 — 임의 점 개수 FPS 캐시.

FPS는 index 0에서 시작하는 결정적 그리디라, K=4096 캐시의 앞 2048점은
fps2048.npz와 동일(coarse-to-fine nested). 캐시 하나로 1024/2048/4096 비교 가능.

  python scripts/precompute_fps_k.py --k 4096 --out /home/kwy00/qi/data/fps4096.npz
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np, torch
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/kwy00/qi")
import cd_common as C

BATCH = 32


@torch.no_grad()
def fps_gpu(x: torch.Tensor, k: int) -> torch.Tensor:
    B, N, _ = x.shape
    idx = torch.zeros(B, k, dtype=torch.long, device=x.device)
    dist = torch.full((B, N), 1e10, device=x.device)
    far = torch.zeros(B, dtype=torch.long, device=x.device)
    ar = torch.arange(B, device=x.device)
    for i in range(k):
        idx[:, i] = far
        c = x[ar, far].unsqueeze(1)
        dist = torch.minimum(dist, ((x - c) ** 2).sum(-1))
        far = dist.argmax(-1)
    return x[ar.unsqueeze(1), idx]


def _load(path):
    p = C.safe_load_ndarray(path)
    return p, C.robust_feats(p.astype(np.float64)).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    C.check_integrity()
    files, cd, sp = C.file_index(), C.drag_table(), C.splits()
    keys = sorted(files)
    split_of = {k: s for s, ids in sp.items() for k in ids}

    pts = np.zeros((len(keys), a.k, 3), dtype=np.float32)
    dims = np.zeros((len(keys), 6), dtype=np.float32)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        for b0 in range(0, len(keys), BATCH):
            chunk = keys[b0:b0 + BATCH]
            res = list(ex.map(_load, [files[k] for k in chunk]))
            raw = torch.from_numpy(np.stack([r[0] for r in res])).cuda()
            pts[b0:b0 + len(chunk)] = fps_gpu(raw, a.k).cpu().numpy()
            dims[b0:b0 + len(chunk)] = np.stack([r[1] for r in res])
            if b0 % (BATCH * 40) == 0:
                done = b0 + len(chunk); el = time.time() - t0
                print(f"  {done:5d}/{len(keys)}  {el:6.1f}s  eta {el/done*(len(keys)-done):5.1f}s", flush=True)

    np.savez(a.out, pts=pts, dims=dims,
             cd=np.array([cd[k] for k in keys], dtype=np.float32),
             cls=np.array([C.body_type(k) for k in keys]),
             split=np.array([split_of[k] for k in keys]),
             keys=np.array([f"{p}_{i}" for p, i in keys]))
    print(f"\n저장 {a.out}  pts={pts.shape}  {pts.nbytes/2**20:.0f} MiB  ({time.time()-t0:.0f}s)")
