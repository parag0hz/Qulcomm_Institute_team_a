"""STL parsing utilities for Paragon.

The web MVP only needs a bounded point cloud sample and lightweight mesh
metadata. This parser accepts common binary and ASCII STL files without
requiring the original research dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import struct
from typing import Iterable, List, Sequence, Tuple


Point = Tuple[float, float, float]


@dataclass(frozen=True)
class MeshPointCloud:
    points: List[Point]
    triangle_count: int
    source_format: str

    @property
    def point_count(self) -> int:
        return len(self.points)


# Bounding every numeric segment keeps malformed ASCII input from triggering
# ambiguous, quadratic backtracking in Python's regex engine. STL vertices are
# line-oriented, so anchoring the match also prevents partial numeric tokens.
_COORDINATE_TOKEN = rb"[-+]?(?:\d{1,32}(?:\.\d{0,32})?|\.\d{1,32})(?:[eE][-+]?\d{1,3})?"
_VERTEX_RE = re.compile(
    rb"(?m)^[ \t]*vertex[ \t]+("
    + _COORDINATE_TOKEN
    + rb")[ \t]+("
    + _COORDINATE_TOKEN
    + rb")[ \t]+("
    + _COORDINATE_TOKEN
    + rb")[ \t]*\r?$"
)


def parse_stl_bytes(data: bytes, max_points: int = 6000) -> MeshPointCloud:
    """Parse an STL file into a sampled point cloud.

    Args:
        data: Raw STL bytes.
        max_points: Maximum vertices to keep in memory for the response.

    Returns:
        MeshPointCloud with a deterministic sample of vertices.

    Raises:
        ValueError: If the file is empty or cannot be parsed as STL.
    """

    if not data:
        raise ValueError("Uploaded file is empty.")

    binary = _try_parse_binary_stl(data, max_points)
    if binary is not None:
        return binary

    ascii_cloud = _parse_ascii_stl(data, max_points)
    if ascii_cloud.point_count == 0:
        raise ValueError("Could not find STL vertices in the uploaded file.")
    return ascii_cloud


def normalize_preview_points(points: Sequence[Point], max_points: int = 700) -> List[Point]:
    """Normalize a point sample into roughly [-1, 1] for canvas preview."""

    sample = deterministic_sample(points, max_points)
    if not sample:
        return []

    xs, ys, zs = zip(*sample)
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-9)

    return [((x - cx) / span * 2.0, (y - cy) / span * 2.0, (z - cz) / span * 2.0) for x, y, z in sample]


def deterministic_sample(points: Sequence[Point], max_points: int) -> List[Point]:
    if len(points) <= max_points:
        return list(points)

    stride = max(1, len(points) // max_points)
    sampled = list(points[::stride])
    return sampled[:max_points]


def _try_parse_binary_stl(data: bytes, max_points: int) -> MeshPointCloud | None:
    if len(data) < 84:
        return None

    triangle_count = struct.unpack("<I", data[80:84])[0]
    expected_size = 84 + triangle_count * 50
    if triangle_count <= 0 or expected_size != len(data):
        return None

    points: List[Point] = []
    offset = 84
    for triangle_index in range(triangle_count):
        if offset + 50 > len(data):
            break
        vertex_offset = offset + 12
        for vertex_index in range(3):
            start = vertex_offset + vertex_index * 12
            point = struct.unpack("<fff", data[start : start + 12])
            if _is_finite_point(point):
                _append_bounded(points, point, max_points)
        offset += 50

    if not points:
        return None
    return MeshPointCloud(points=points, triangle_count=triangle_count, source_format="binary")


def _parse_ascii_stl(data: bytes, max_points: int) -> MeshPointCloud:
    points: List[Point] = []
    for match in _VERTEX_RE.finditer(data):
        point = (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        if _is_finite_point(point):
            _append_bounded(points, point, max_points)

    return MeshPointCloud(points=points, triangle_count=len(points) // 3, source_format="ascii")


def _append_bounded(points: List[Point], point: Point, max_points: int) -> None:
    if len(points) < max_points:
        points.append(point)


def _is_finite_point(point: Iterable[float]) -> bool:
    return all(math.isfinite(value) for value in point)
