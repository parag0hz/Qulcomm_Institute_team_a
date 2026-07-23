"""DrivAerNet++ 공용 모듈 — 안전 로더 + ID 정규화.

이 데이터셋은 서로 다른 두 출처가 합쳐진 것이라 design ID 표기가 어긋난다.
raw 문자열로 조인하면 7,713개 중 609개가 **에러 없이** 사라진다. 자세한 건 아래 norm_id 참고.

원칙: ID로 파일명을 재구성하지 않는다. 디렉토리를 스캔해 만든 인덱스에서 조회한다.
      (패딩 자릿수를 코드에 박아두면 상류가 규약을 바꿀 때 또 조용히 깨진다)
"""
from __future__ import annotations

import csv
import glob
import os
import pickletools
import re
from functools import lru_cache

import numpy as np

DATA_DIR = os.environ.get("QI_DATA", "/home/kwy00/qi/data")
PC_DIR = os.path.join(DATA_DIR, "DrivAerNetPlusPlus_Processed_Point_Clouds_100k_paddle")
CSV_PATH = os.path.join(DATA_DIR, "DrivAerNetPlusPlus_Drag_8k.csv")
SPLIT_DIR = os.path.join(DATA_DIR, "subset_dir")

N_DESIGNS = 7713
N_POINTS = 100_000
SPLIT_SIZES = {"train": 5398, "val": 1157, "test": 1158}

DesignID = tuple[str, int]  # (prefix, index) — 정규화된 키

_ID_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)$")
_BODY = {"F": "Fastback", "E": "Estate", "N": "Notchback"}


# ----------------------------------------------------------------------------
# ID 정규화
# ----------------------------------------------------------------------------
def norm_id(name: str) -> DesignID:
    """어떤 표기로 오든 정규 키 (prefix, index)로 바꾼다.

    이 데이터셋에 세 가지 표기가 공존한다:
        CSV       E_S_WWC_WM_1        (패딩 없음)
        split     E_S_WWC_WM_001      (3자리)
        파일명     E_S_WWC_WM_001.paddle_tensor
    그리고 상속된 fastback 계열만 어디서나 4자리다:
        DrivAer_F_D_WM_WW_0001

    'DrivAer_' 접두사도 떼어낸다. 상류 저장소가 이미 이 접두사를 제거했고
    (Mohamedelrefaie/DrivAerNet issue #21), 우리가 받은 PaddleScience 미러는
    그 이전의 스냅샷이다. 떼어내면 body type이 항상 prefix의 첫 토큰이 된다.

    표기를 못 알아보면 조용히 넘기지 않고 예외를 던진다. 핸드오프 §4의 원본은
    실패 시 (name, -1)을 돌려주는데, 그러면 가짜 키가 만들어져 오염이 퍼진다.
    """
    s = os.path.basename(str(name)).strip()
    if s.endswith(".paddle_tensor"):
        s = s[: -len(".paddle_tensor")]
    if s.startswith("DrivAer_"):
        s = s[len("DrivAer_") :]
    m = _ID_RE.match(s)
    if not m:
        raise ValueError(f"design id로 해석할 수 없음: {name!r}")
    return m.group("prefix"), int(m.group("idx"))


def body_type(key: DesignID) -> str:
    """(prefix, idx) -> 'Fastback' | 'Estate' | 'Notchback'"""
    return _BODY[key[0].split("_")[0]]


# ----------------------------------------------------------------------------
# 안전 로더 — pickle.load 금지 (임의 코드 실행)
# ----------------------------------------------------------------------------
def safe_load_ndarray(path: str, expect_points: int = N_POINTS) -> np.ndarray:
    """.paddle_tensor(pickle)에서 좌표 배열만 코드 실행 없이 복원해 (N,3) float32로 반환."""
    with open(path, "rb") as f:
        blob = f.read()
    ops = [(op.name, arg) for op, arg, _ in pickletools.genops(blob)]

    bufs = [a for _, a in ops if isinstance(a, (bytes, bytearray))]
    if not bufs:
        raise ValueError(f"바이트 버퍼가 없음: {path}")
    raw = max(bufs, key=len)  # 가장 큰 bytes = 배열 버퍼

    typestr = None
    for i, (_, a) in enumerate(ops):
        if a == "dtype":
            for _, a2 in ops[i + 1 : i + 6]:
                if isinstance(a2, str):
                    try:
                        np.dtype(a2)
                        typestr = a2
                        break
                    except TypeError:
                        pass
            if typestr:
                break
    if typestr is None:  # opcode에서 못 찾으면 버퍼 길이로 추론
        for itemsize, ts in ((4, "<f4"), (8, "<f8")):
            if len(raw) % itemsize == 0 and (len(raw) // itemsize) % 3 == 0:
                typestr = ts
                break
    if typestr is None:
        raise ValueError(f"dtype 판별 실패: {path}")

    dt = np.dtype(typestr)
    if len(raw) % (dt.itemsize * 3):
        raise ValueError(f"버퍼 크기가 (N,3)과 안 맞음: {len(raw)}B, dtype={dt}")
    pts = len(raw) // dt.itemsize // 3
    if expect_points and pts != expect_points:
        raise ValueError(f"점 개수 {pts} != 기대 {expect_points}: {path}")

    # frombuffer는 read-only view -> astype으로 쓰기 가능한 복사본
    return np.frombuffer(raw, dtype=dt).reshape(pts, 3).astype(np.float32)


# ----------------------------------------------------------------------------
# 인덱스 — 파일시스템이 진실의 원천
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def file_index() -> dict[DesignID, str]:
    """정규 키 -> .paddle_tensor 절대경로."""
    idx: dict[DesignID, str] = {}
    for p in glob.glob(os.path.join(PC_DIR, "*.paddle_tensor")):
        k = norm_id(p)
        if k in idx:
            raise ValueError(f"정규화 후 키 충돌: {k} <- {idx[k]}, {p}")
        idx[k] = p
    if not idx:
        raise FileNotFoundError(f"포인트클라우드가 없음: {PC_DIR}")
    return idx


@lru_cache(maxsize=1)
def drag_table() -> dict[DesignID, float]:
    """정규 키 -> Average Cd."""
    out: dict[DesignID, float] = {}
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            k = norm_id(row["Design"])
            if k in out:
                raise ValueError(f"CSV에 중복 design: {k}")
            out[k] = float(row["Average Cd"])
    return out


@lru_cache(maxsize=1)
def splits() -> dict[str, list[DesignID]]:
    """'train'|'val'|'test' -> 정규 키 리스트 (공식 split)."""
    out = {}
    for s in ("train", "val", "test"):
        with open(os.path.join(SPLIT_DIR, f"{s}_design_ids.txt"), encoding="utf-8-sig") as f:
            out[s] = [norm_id(l) for l in f if l.strip()]
    return out


def load_points(key: DesignID | str) -> np.ndarray:
    """정규 키(또는 아무 표기의 이름)로 (100000, 3) float32 포인트클라우드를 읽는다."""
    if isinstance(key, str):
        key = norm_id(key)
    return safe_load_ndarray(file_index()[key])


# ----------------------------------------------------------------------------
# 무결성 — 조용한 누락을 시끄러운 실패로 바꾼다
# ----------------------------------------------------------------------------
def check_integrity() -> None:
    files, cd, sp = file_index(), drag_table(), splits()

    assert len(files) == N_DESIGNS, f"파일 {len(files)} != {N_DESIGNS}"
    assert len(cd) == N_DESIGNS, f"라벨 {len(cd)} != {N_DESIGNS}"

    missing_label = set(files) - set(cd)
    missing_file = set(cd) - set(files)
    assert not missing_label, f"라벨 없는 파일 {len(missing_label)}개: {sorted(missing_label)[:3]}"
    assert not missing_file, f"파일 없는 라벨 {len(missing_file)}개: {sorted(missing_file)[:3]}"

    seen: set[DesignID] = set()
    for s, ids in sp.items():
        assert len(ids) == SPLIT_SIZES[s], f"{s} {len(ids)} != {SPLIT_SIZES[s]}"
        assert len(set(ids)) == len(ids), f"{s}에 중복 id"
        assert not (set(ids) - set(files)), f"{s}에 파일 없는 id"
        assert not (seen & set(ids)), f"{s}가 다른 split과 겹침"
        seen |= set(ids)
    assert seen == set(files), f"split 합집합이 전체와 불일치 ({len(seen)} vs {len(files)})"


# ----------------------------------------------------------------------------
# 피처 — 핸드오프 §6
# ----------------------------------------------------------------------------
FEAT_NAMES = ["H", "L", "W", "FA", "H/L", "drop"]


def raw_feats(p: np.ndarray) -> np.ndarray:
    """v1: 생 극값(min/max) + 마스크 없는 hull. 노이즈에 약하다."""
    from scipy.spatial import ConvexHull

    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    H, L, W = z.max(), np.ptp(x), np.ptp(y)
    try:
        FA = ConvexHull(p[:, 1:3]).volume  # 2D hull에서 .volume == 면적
    except Exception:
        FA = np.nan
    xlo, xhi = x.min(), x.max()
    span = xhi - xlo
    mid = p[(x > xlo + 0.4 * span) & (x < xlo + 0.6 * span)]
    rear = p[x > xhi - 0.10 * span]
    drop = mid[:, 2].max() - rear[:, 2].max() if len(mid) > 5 and len(rear) > 5 else np.nan
    return np.array([H, L, W, FA, H / L if L > 1e-6 else np.nan, drop])


def robust_feats(p: np.ndarray) -> np.ndarray:
    """v2: 퍼센타일 기반. 노이즈에 강하다.

    주의: 여기서 나오는 drop은 §6 상관표의 +0.58을 재현하지 않는다 (전수 r=+0.191).
    §6의 표는 raw_feats 쪽 정의로 계산된 것이고, drop은 정의에 따라 r이 0.16~0.80까지 흔들린다.
    """
    from scipy.spatial import ConvexHull

    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    H = np.percentile(z, 99.5)
    L = np.percentile(x, 99.5) - np.percentile(x, 0.5)
    W = np.percentile(y, 99.5) - np.percentile(y, 0.5)
    ylo, yhi = np.percentile(y, [0.5, 99.5])
    zlo, zhi = np.percentile(z, [0.5, 99.5])
    m = (y >= ylo) & (y <= yhi) & (z >= zlo) & (z <= zhi)
    try:
        FA = ConvexHull(p[:, 1:3][m]).volume
    except Exception:
        FA = np.nan
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    span = xhi - xlo
    mid = p[(x > xlo + 0.4 * span) & (x < xlo + 0.6 * span)]
    rear = p[x > xhi - 0.10 * span]
    drop = (np.percentile(mid[:, 2], 99) - np.percentile(rear[:, 2], 99)
            if len(mid) > 5 and len(rear) > 5 else np.nan)
    return np.array([H, L, W, FA, H / L if L > 1e-6 else np.nan, drop])


# ----------------------------------------------------------------------------
# 열화 — 폰스캔 모사 (핸드오프 §6)
# ----------------------------------------------------------------------------
def degrade_noise(p, rng, sigma=0.015):
    return p + rng.normal(0, sigma, p.shape)


def degrade_sparse(p, rng, n=4000):
    return p[rng.choice(len(p), min(n, len(p)), replace=False)]


def degrade_glass(p, rng, z_thr=1.0, keep=0.4):
    high = p[:, 2] > z_thr
    return p[~high | (rng.random(len(p)) < keep)]


def degrade_oneside(p, y_cut=-0.15):
    return p[p[:, 1] > y_cut]


def degrade_phone(p, rng):
    """§6 원본: 노이즈 + 한쪽만 + 유리소실 + 희소(5k)."""
    q = degrade_noise(p, rng)
    q = degrade_oneside(q)
    q = degrade_glass(q, rng)
    if len(q) > 5000:
        q = degrade_sparse(q, rng, 5000)
    return q


def mirror_complete(q, plane=0.0):
    """좌우대칭 복원. plane=0.0 은 데이터셋의 **정답** 대칭평면(실스캔엔 없다)."""
    m = q.copy()
    m[:, 1] = 2.0 * plane - m[:, 1]
    return np.concatenate([q, m], 0)


def estimate_symmetry_plane(q):
    """스캔에서 대칭평면을 추정. 센트로이드 = PCA가 원점으로 잡는 지점."""
    return float(q[:, 1].mean())


if __name__ == "__main__":
    check_integrity()
    print(f"OK  designs={len(file_index())}  "
          f"splits={ {k: len(v) for k, v in splits().items()} }")
