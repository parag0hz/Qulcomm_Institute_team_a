"""공식 split에 '쌍둥이 설계'가 걸쳐 있는가?

DrivAerNet++는 기본 차체를 파라미터로 변형해 만든다. split이 설계 ID 기준 랜덤이면
파라미터 공간에서 거의 붙은 쌍이 train/test로 갈릴 수 있다. 그렇다면 딥모델의 높은 R2는
'형상 이해'가 아니라 'train에서 닮은 차 찾아 Cd 베끼기'일 수 있다.

검증: test 각 설계를 복셀 점유 격자로 만들고 train에서 최근접 이웃을 찾는다.
      그 이웃의 Cd를 그대로 복사하는 1-NN 예측기의 R2를 잰다.

  python check_leakage.py
"""
import sys
import numpy as np, torch

sys.path.insert(0, "/home/kwy00/qi")

GRID = (48, 24, 20)          # x, y, z — 차 비율에 대충 맞춤
CLASSES = ["Fastback", "Estate", "Notchback"]
DEV = "cpu"                  # GPU는 학습이 쓰는 중


def r2(yh, y):
    return 1 - float(((yh - y) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def mae_pct(yh, y):
    return float(np.mean(np.abs(yh - y) / y) * 100)


def voxelize(pts):
    """(M,N,3) -> (M, prod(GRID)) float32 점유 격자. 전역 bbox 기준 = 스케일 유지."""
    lo = pts.reshape(-1, 3).min(0)
    hi = pts.reshape(-1, 3).max(0)
    idx = ((pts - lo) / (hi - lo + 1e-9) * (np.array(GRID) - 1)).astype(np.int32)
    M = len(pts)
    flat = idx[..., 0] * GRID[1] * GRID[2] + idx[..., 1] * GRID[2] + idx[..., 2]
    out = np.zeros((M, int(np.prod(GRID))), dtype=np.float32)
    for i in range(M):
        out[i, flat[i]] = 1.0
    return out


if __name__ == "__main__":
    d = np.load("/home/kwy00/qi/data/fps2048.npz", allow_pickle=True)
    pts, cd, split, cls, keys = d["pts"], d["cd"], d["split"], d["cls"], d["keys"]

    V = voxelize(pts)
    tr, te = split == "train", split == "test"
    A = torch.from_numpy(V[te]).to(DEV)
    B = torch.from_numpy(V[tr]).to(DEV)
    # 코사인 유사도 (점유 격자 -> 겹침 비율)
    An = A / A.norm(dim=1, keepdim=True)
    Bn = B / B.norm(dim=1, keepdim=True)
    sim = An @ Bn.T                              # (n_te, n_tr)
    top = sim.topk(5, dim=1)
    nn_i = top.indices[:, 0].cpu().numpy()
    nn_s = top.values[:, 0].cpu().numpy()

    y_te, y_tr = cd[te], cd[tr]
    yh_1nn = y_tr[nn_i]
    yh_5nn = y_tr[top.indices.cpu().numpy()].mean(1)

    print("=" * 78)
    print("1-NN 검색 예측기 — 형상 이해 없이 'train에서 제일 닮은 차의 Cd 복사'")
    print("=" * 78)
    print(f"{'':<14}{'R2':>10}{'MAE%':>9}")
    for nm, yh in (("1-NN", yh_1nn), ("5-NN 평균", yh_5nn)):
        print(f"{nm:<14}{r2(yh, y_te):>+10.3f}{mae_pct(yh, y_te):>9.2f}")
    print()
    print(f"{'차종':<14}{'1-NN R2':>10}{'MAE%':>9}")
    for c in CLASSES:
        s = cls[te] == c
        print(f"{c:<14}{r2(yh_1nn[s], y_te[s]):>+10.3f}{mae_pct(yh_1nn[s], y_te[s]):>9.2f}")

    print("\n" + "=" * 78)
    print("최근접 이웃이 얼마나 닮았나 (복셀 점유 코사인 유사도)")
    print("=" * 78)
    q = np.percentile(nn_s, [1, 25, 50, 75, 99])
    print(f"  p1={q[0]:.4f}  p25={q[1]:.4f}  median={q[2]:.4f}  p75={q[3]:.4f}  p99={q[4]:.4f}")
    same = np.array([cls[tr][i] for i in nn_i]) == cls[te]
    print(f"  최근접 이웃이 같은 차종인 비율: {same.mean()*100:.1f}%")
    # 같은 variant prefix인가 (파라메트릭 쌍둥이)
    pre_te = np.array(["_".join(k.split("_")[:-1]) for k in keys[te]])
    pre_tr = np.array(["_".join(k.split("_")[:-1]) for k in keys[tr]])
    print(f"  최근접 이웃이 같은 variant인 비율: {(pre_tr[nn_i] == pre_te).mean()*100:.1f}%")

    print("\n  가장 닮은 쌍 5개:")
    for j in np.argsort(-nn_s)[:5]:
        print(f"    {keys[te][j]:<22} <- {keys[tr][nn_i[j]]:<22} "
              f"sim={nn_s[j]:.4f}  ΔCd={abs(y_te[j]-y_tr[nn_i[j]])*1000:.1f}e-3")
