"""Safe .paddle_tensor (pickle) parsing + CPU farthest-point sampling for uploads.

`.paddle_tensor` files are raw Python pickles from a third-party mirror. We MUST NOT
unpickle them — `pickle.load` executes arbitrary opcodes (remote code execution on an
uploaded file). This module walks the pickle with `pickletools.genops` and rebuilds the
coordinate buffer with `np.frombuffer`, executing nothing. Ported from
`ml/cd_common.safe_load_ndarray`, adapted to read bytes and accept any point count.
"""

from __future__ import annotations

import pickletools

import numpy as np

# Sanity bound: DrivAerNet clouds are 100k points. Reject absurd sizes early.
MAX_INPUT_POINTS = 2_000_000
MODEL_POINTS = 2048


def parse_paddle_tensor(data: bytes) -> np.ndarray:
    """Reconstruct an ``(N, 3)`` float32 cloud from ``.paddle_tensor`` bytes.

    Executes nothing — walks pickle opcodes and copies the largest byte buffer as the
    array. Raises ``ValueError`` on anything that is not a recognisable point cloud.
    """
    try:
        ops = [(op.name, arg) for op, arg, _ in pickletools.genops(data)]
    except Exception as exc:  # malformed pickle stream
        raise ValueError("File is not a readable .paddle_tensor stream.") from exc

    buffers = [a for _, a in ops if isinstance(a, (bytes, bytearray))]
    if not buffers:
        raise ValueError("No coordinate buffer found; not a point-cloud tensor.")
    raw = bytes(max(buffers, key=len))  # largest bytes object = the array buffer

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
    if typestr is None:  # infer from buffer length when opcodes don't say
        for itemsize, ts in ((4, "<f4"), (8, "<f8")):
            if len(raw) % itemsize == 0 and (len(raw) // itemsize) % 3 == 0:
                typestr = ts
                break
    if typestr is None:
        raise ValueError("Could not determine tensor dtype.")

    dt = np.dtype(typestr)
    if len(raw) % (dt.itemsize * 3):
        raise ValueError("Buffer size is not a multiple of (N, 3).")
    n = len(raw) // dt.itemsize // 3
    if n < 3 or n > MAX_INPUT_POINTS:
        raise ValueError(f"Unexpected point count ({n}).")

    # frombuffer yields a read-only view; astype makes a writable float32 copy.
    return np.frombuffer(raw, dtype=dt).reshape(n, 3).astype(np.float32)


def farthest_point_sample(points: np.ndarray, k: int = MODEL_POINTS) -> np.ndarray:
    """Farthest-point sampling to ``k`` points (CPU, numpy).

    Deterministic — starts from index 0, matching the training-time FPS cache, so an
    uploaded DrivAerNet cloud reproduces the same 2048 points (and the same Cd) as the
    bundled demo. If the cloud already has ``<= k`` points it is returned unchanged.
    """
    pts = np.ascontiguousarray(points, dtype=np.float32)
    n = len(pts)
    if n <= k:
        return pts
    selected = np.empty(k, dtype=np.int64)
    distance = np.full(n, np.inf, dtype=np.float64)
    far = 0
    for i in range(k):
        selected[i] = far
        d = ((pts - pts[far]) ** 2).sum(axis=1)
        np.minimum(distance, d, out=distance)
        far = int(distance.argmax())
    return pts[selected]


def cloud_from_paddle_bytes(data: bytes, k: int = MODEL_POINTS) -> tuple[np.ndarray, int]:
    """Full upload pipeline: safe-parse → FPS to ``k``. Returns ``(sampled, n_input)``."""
    cloud = parse_paddle_tensor(data)
    n_input = int(len(cloud))
    return farthest_point_sample(cloud, k), n_input
