#!/usr/bin/env python3
"""Prepare a topology-checked, multi-part DrivAer GLB using QEM."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Iterator

import numpy as np

try:
    from fast_simplification import simplify
except ImportError as error:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "fast-simplification is required. Install it with: "
        "python3 -m pip install fast-simplification==0.1.13"
    ) from error


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BACKEND_ROOT / "stl/3DMeshesSTL/F_D_WM_WW_8/F_D_WM_WW_3532.stl"
DEFAULT_OUTPUT = BACKEND_ROOT / "cfa_service/static/models/drivaer_reference.glb"


@dataclass
class MeshPart:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    source_faces: int
    boundary_edges: int
    nonmanifold_edges: int

    @property
    def bounds_min(self) -> list[float]:
        return self.vertices.min(axis=0).astype(float).tolist()

    @property
    def bounds_max(self) -> list[float]:
        return self.vertices.max(axis=0).astype(float).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a QEM DrivAer GLB reference mesh.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--body-faces", type=int, default=80_000)
    parser.add_argument("--wheel-faces", type=int, default=8_000)
    parser.add_argument("--aggressiveness", type=int, default=3)
    parser.add_argument("--source-label", default="DrivAer F_D_WM_WW reference")
    return parser.parse_args()


def read_binary_stl(path: Path) -> tuple[bytes, int]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError("STL is too small to be a binary STL file.")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    if triangle_count <= 0 or 84 + triangle_count * 50 != len(data):
        raise ValueError("Expected a valid binary STL file.")
    return data, triangle_count


def iter_triangles(data: bytes, triangle_count: int) -> Iterator[tuple[tuple[float, float, float], ...]]:
    for index in range(triangle_count):
        offset = 84 + index * 50 + 12
        yield tuple(struct.unpack_from("<fff", data, offset + vertex * 12) for vertex in range(3))


def indexed_mesh(data: bytes, triangle_count: int) -> tuple[np.ndarray, np.ndarray]:
    vertex_ids: dict[tuple[float, float, float], int] = {}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for triangle in iter_triangles(data, triangle_count):
        face = []
        for point in triangle:
            if point not in vertex_ids:
                vertex_ids[point] = len(vertices)
                vertices.append(point)
            face.append(vertex_ids[point])
        faces.append(tuple(face))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def split_components(vertices: np.ndarray, faces: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    parent = np.arange(len(vertices), dtype=np.int64)
    rank = np.zeros(len(vertices), dtype=np.uint8)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            following = int(parent[value])
            parent[value] = root
            value = following
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for a, b, c in faces:
        union(int(a), int(b))
        union(int(a), int(c))

    groups: dict[int, list[np.ndarray]] = {}
    for face in faces:
        groups.setdefault(find(int(face[0])), []).append(face)

    components = []
    for grouped_faces in sorted(groups.values(), key=len, reverse=True):
        component_faces = np.asarray(grouped_faces, dtype=np.int32)
        used = np.unique(component_faces)
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        components.append((vertices[used].copy(), remap[component_faces].astype(np.int32)))
    return components


def topology_stats(faces: np.ndarray) -> tuple[int, int]:
    edges: Counter[tuple[int, int]] = Counter()
    for a, b, c in faces:
        for left, right in ((a, b), (b, c), (c, a)):
            edge = (int(left), int(right))
            edges[edge if edge[0] < edge[1] else (edge[1], edge[0])] += 1
    return (
        sum(count == 1 for count in edges.values()),
        sum(count > 2 for count in edges.values()),
    )


def wheel_name(vertices: np.ndarray) -> str:
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    axle = "F" if center[0] < 1.0 else "R"
    side = "L" if center[1] > 0 else "R"
    return f"Wheel_{axle}{side}"


def prepare_parts(
    vertices: np.ndarray,
    faces: np.ndarray,
    body_faces: int,
    wheel_faces: int,
    aggressiveness: int,
) -> list[MeshPart]:
    components = split_components(vertices, faces)
    if len(components) != 5:
        raise ValueError(f"Expected body plus four wheels; found {len(components)} components.")

    parts = []
    source_boundary_total = 0
    source_nonmanifold_total = 0
    output_boundary_total = 0
    output_nonmanifold_total = 0
    for index, (part_vertices, part_faces) in enumerate(components):
        name = "Body" if index == 0 else wheel_name(part_vertices)
        target = body_faces if index == 0 else wheel_faces
        if target < 1_000 or target >= len(part_faces):
            raise ValueError(f"Invalid target {target} for {name} with {len(part_faces)} faces.")
        source_boundary, source_nonmanifold = topology_stats(part_faces)
        simplified_vertices, simplified_faces = simplify(
            part_vertices,
            part_faces,
            target_count=target,
            agg=aggressiveness,
        )
        simplified_vertices = np.asarray(simplified_vertices, dtype=np.float32)
        simplified_faces = np.asarray(simplified_faces, dtype=np.uint32)
        boundary, nonmanifold = topology_stats(simplified_faces)
        source_boundary_total += source_boundary
        source_nonmanifold_total += source_nonmanifold
        output_boundary_total += boundary
        output_nonmanifold_total += nonmanifold
        parts.append(
            MeshPart(
                name=name,
                vertices=simplified_vertices,
                faces=simplified_faces,
                source_faces=len(part_faces),
                boundary_edges=boundary,
                nonmanifold_edges=nonmanifold,
            )
        )

    if output_boundary_total > source_boundary_total:
        raise ValueError(
            f"QEM introduced open edges: {source_boundary_total} -> {output_boundary_total}."
        )
    if output_nonmanifold_total > source_nonmanifold_total:
        raise ValueError(
            "QEM increased non-manifold edges: "
            f"{source_nonmanifold_total} -> {output_nonmanifold_total}. "
            "Try a lower --aggressiveness or larger face targets."
        )
    return parts


def calculate_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(b - a, c - a)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1
    return (normals / lengths[:, None]).astype(np.float32)


def align4(blob: bytearray) -> None:
    blob.extend(b"\x00" * ((-len(blob)) % 4))


def write_glb(output: Path, parts: list[MeshPart], source_label: str) -> None:
    binary = bytearray()
    buffer_views = []
    accessors = []
    meshes = []
    nodes = []

    def add_view(payload: bytes, target: int) -> int:
        align4(binary)
        offset = len(binary)
        binary.extend(payload)
        buffer_views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(payload), "target": target}
        )
        return len(buffer_views) - 1

    all_minimum = np.min(np.asarray([part.bounds_min for part in parts]), axis=0)
    all_maximum = np.max(np.asarray([part.bounds_max for part in parts]), axis=0)
    wheel_specs = []
    for part in parts:
        vertices = np.ascontiguousarray(part.vertices, dtype="<f4")
        normals = np.ascontiguousarray(calculate_normals(vertices, part.faces), dtype="<f4")
        indices = np.ascontiguousarray(part.faces.reshape(-1), dtype="<u4")
        position_view = add_view(vertices.tobytes(), 34962)
        normal_view = add_view(normals.tobytes(), 34962)
        index_view = add_view(indices.tobytes(), 34963)
        position_accessor = len(accessors)
        accessors.extend(
            [
                {
                    "bufferView": position_view,
                    "componentType": 5126,
                    "count": len(vertices),
                    "type": "VEC3",
                    "min": part.bounds_min,
                    "max": part.bounds_max,
                },
                {
                    "bufferView": normal_view,
                    "componentType": 5126,
                    "count": len(normals),
                    "type": "VEC3",
                },
                {
                    "bufferView": index_view,
                    "componentType": 5125,
                    "count": len(indices),
                    "type": "SCALAR",
                },
            ]
        )
        material = 0 if part.name == "Body" else 1
        meshes.append(
            {
                "name": part.name,
                "primitives": [
                    {
                        "attributes": {"POSITION": position_accessor, "NORMAL": position_accessor + 1},
                        "indices": position_accessor + 2,
                        "material": material,
                    }
                ],
            }
        )
        nodes.append({"mesh": len(meshes) - 1, "name": part.name})
        if part.name.startswith("Wheel_"):
            minimum, maximum = np.asarray(part.bounds_min), np.asarray(part.bounds_max)
            wheel_specs.append(
                {
                    "name": part.name,
                    "center": ((minimum + maximum) * 0.5).round(6).tolist(),
                    "radius": round(float((maximum[2] - minimum[2]) * 0.5), 6),
                    "width": round(float(maximum[1] - minimum[1]), 6),
                }
            )

    face_count = sum(len(part.faces) for part in parts)
    document = {
        "asset": {"version": "2.0", "generator": "Paragon QEM DrivAer mesh preparer"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [
            {
                "name": "Paragon teal body finish",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.035, 0.42, 0.40, 1.0],
                    "metallicFactor": 0.35,
                    "roughnessFactor": 0.28,
                },
            },
            {
                "name": "Dataset detailed wheel",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.035, 0.045, 0.05, 1.0],
                    "metallicFactor": 0.15,
                    "roughnessFactor": 0.55,
                },
            },
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "extras": {
            "source": source_label,
            "simplifier": "fast-simplification QEM",
            "component_count": len(parts),
            "face_count": face_count,
            "vertex_count": sum(len(part.vertices) for part in parts),
            "boundary_edges": sum(part.boundary_edges for part in parts),
            "nonmanifold_edges": sum(part.nonmanifold_edges for part in parts),
            "axis_convention": "X length, Y width, Z height",
            "bounds_min": all_minimum.astype(float).tolist(),
            "bounds_max": all_maximum.astype(float).tolist(),
            "wheel_specs": sorted(wheel_specs, key=lambda item: item["name"]),
            "parts": [
                {
                    "name": part.name,
                    "source_faces": part.source_faces,
                    "output_faces": len(part.faces),
                    "boundary_edges": part.boundary_edges,
                    "nonmanifold_edges": part.nonmanifold_edges,
                }
                for part in parts
            ],
        },
    }
    json_blob = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_blob += b" " * ((-len(json_blob)) % 4)
    align4(binary)
    total_length = 12 + 8 + len(json_blob) + 8 + len(binary)
    glb = b"".join(
        [
            struct.pack("<III", 0x46546C67, 2, total_length),
            struct.pack("<I4s", len(json_blob), b"JSON"),
            json_blob,
            struct.pack("<I4s", len(binary), b"BIN\x00"),
            bytes(binary),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(glb)
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input STL not found: {args.input}")
    data, original_faces = read_binary_stl(args.input)
    vertices, faces = indexed_mesh(data, original_faces)
    parts = prepare_parts(
        vertices,
        faces,
        body_faces=args.body_faces,
        wheel_faces=args.wheel_faces,
        aggressiveness=args.aggressiveness,
    )
    write_glb(args.output, parts, args.source_label)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "simplifier": "fast-simplification QEM",
                "original_faces": original_faces,
                "output_faces": sum(len(part.faces) for part in parts),
                "components": [part.name for part in parts],
                "boundary_edges": sum(part.boundary_edges for part in parts),
                "nonmanifold_edges": sum(part.nonmanifold_edges for part in parts),
                "output_mb": round(args.output.stat().st_size / 1024 / 1024, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
