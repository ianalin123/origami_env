"""Origami fold simulator — analytical rotation with cumulative transforms.

BFS from face 0 through the face adjacency graph. Each face accumulates
a rotation transform (R, t) such that: folded_pos = R @ flat_pos + t.
When crossing a fold edge, the fold rotation is composed with the parent
face's transform. Non-fold edges inherit the parent's transform directly.

This correctly handles multiple intersecting folds (e.g. quarter fold)
because each face's transform captures ALL upstream folds.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .fold_parser import parse_fold


@dataclass
class SimResult:
    """Result of a fold simulation."""

    positions: np.ndarray  # (N, 3) final vertex positions
    converged: bool
    steps_taken: int
    max_strain: float
    total_energy: float


def simulate(
    fold_data: dict,
    crease_percent: float = 1.0,
    max_steps: int = 500,
    params: dict | None = None,
) -> SimResult:
    """Simulate a FOLD crease pattern and return final 3D positions.

    Uses cumulative rotation transforms per face. BFS from face 0,
    composing fold rotations at each crease edge.

    Args:
        fold_data: FOLD-format dict with vertices, edges, assignments, angles.
        crease_percent: 0.0 = flat, 1.0 = fully folded.
        max_steps: Unused (kept for API compat).
        params: Unused (kept for API compat).

    Returns:
        SimResult with final positions, strain info.
    """
    parsed = parse_fold(fold_data)
    flat_pos = parsed["vertices"].copy()
    edges = parsed["edges"]
    assignments = parsed["assignments"]
    fold_angles = parsed["fold_angles"]
    faces = parsed["faces"]
    positions = flat_pos.copy()

    if len(faces) == 0:
        return SimResult(
            positions=positions, converged=True,
            steps_taken=0, max_strain=0.0, total_energy=0.0,
        )

    # Build face adjacency: edge -> [face_idx, ...]
    face_adj = _build_face_adjacency(faces)

    # Build crease map: (v_min, v_max) -> fold_angle_rad * crease_percent
    crease_map: dict[tuple[int, int], float] = {}
    for i, (v1, v2) in enumerate(edges):
        key = (min(int(v1), int(v2)), max(int(v1), int(v2)))
        if assignments[i] in ("M", "V"):
            crease_map[key] = fold_angles[i] * crease_percent

    # Per-face cumulative transform: folded = R @ flat + t
    n_faces = len(faces)
    face_R = [None] * n_faces
    face_t = [None] * n_faces

    # Face 0 is fixed (identity transform)
    face_R[0] = np.eye(3)
    face_t[0] = np.zeros(3)

    visited = [False] * n_faces
    visited[0] = True

    placed: set[int] = set()
    for vi in faces[0]:
        placed.add(int(vi))

    queue = [0]
    while queue:
        fi = queue.pop(0)
        face = faces[fi]

        for j in range(len(face)):
            v1, v2 = int(face[j]), int(face[(j + 1) % len(face)])
            edge_key = (min(v1, v2), max(v1, v2))

            for fj in face_adj.get(edge_key, []):
                if visited[fj]:
                    continue
                visited[fj] = True
                queue.append(fj)

                angle = crease_map.get(edge_key, 0.0)

                if abs(angle) > 1e-10:
                    # Fold rotation around the edge in folded space
                    p1 = positions[v1].copy()
                    axis = positions[v2] - p1
                    axis_len = np.linalg.norm(axis)
                    if axis_len > 1e-12:
                        axis_unit = axis / axis_len
                        fold_rot = Rotation.from_rotvec(
                            angle * axis_unit,
                        ).as_matrix()
                    else:
                        fold_rot = np.eye(3)

                    # Compose: R_fj = fold_rot @ R_fi, t_fj adjusted for pivot
                    face_R[fj] = fold_rot @ face_R[fi]
                    face_t[fj] = fold_rot @ (face_t[fi] - p1) + p1
                else:
                    # No fold — inherit parent's transform
                    face_R[fj] = face_R[fi].copy()
                    face_t[fj] = face_t[fi].copy()

                # Place unplaced vertices using this face's transform
                for vi in faces[fj]:
                    vi_int = int(vi)
                    if vi_int not in placed:
                        positions[vi_int] = face_R[fj] @ flat_pos[vi_int] + face_t[fj]
                        placed.add(vi_int)

    # Compute strain (deviation from rest edge lengths)
    max_strain = _compute_strain(positions, parsed)

    return SimResult(
        positions=positions,
        converged=True,
        steps_taken=1,
        max_strain=max_strain,
        total_energy=0.0,
    )


def _build_face_adjacency(
    faces: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    """Map each edge (sorted vertex pair) to list of face indices."""
    adj: dict[tuple[int, int], list[int]] = {}
    for fi, face in enumerate(faces):
        n = len(face)
        for j in range(n):
            v1, v2 = int(face[j]), int(face[(j + 1) % n])
            key = (min(v1, v2), max(v1, v2))
            if key not in adj:
                adj[key] = []
            adj[key].append(fi)
    return adj


def _compute_strain(positions: np.ndarray, parsed: dict) -> float:
    """Compute max axial strain across all edges."""
    edges = parsed["edges"]
    vertices_flat = parsed["vertices"]
    max_strain = 0.0
    for v1, v2 in edges:
        rest = np.linalg.norm(vertices_flat[v2] - vertices_flat[v1])
        curr = np.linalg.norm(positions[v2] - positions[v1])
        if rest > 1e-12:
            strain = abs(curr - rest) / rest
            max_strain = max(max_strain, strain)
    return max_strain
