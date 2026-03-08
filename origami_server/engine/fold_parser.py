"""FOLD JSON parsing and validation.

Validates LLM-generated FOLD crease patterns before simulation.
FOLD spec: https://github.com/edemaine/fold
"""

from typing import Any

import numpy as np


def validate_fold(fold_data: dict[str, Any]) -> tuple[bool, str]:
    """Validate a FOLD JSON object. Returns (is_valid, error_message)."""

    # Required fields
    for key in ("vertices_coords", "edges_vertices", "edges_assignment"):
        if key not in fold_data:
            return False, f"Missing required field: {key}"

    verts = fold_data["vertices_coords"]
    edges = fold_data["edges_vertices"]
    assignments = fold_data["edges_assignment"]

    # Must have at least 3 vertices (a triangle)
    if len(verts) < 3:
        return False, f"Need at least 3 vertices, got {len(verts)}"

    # Must have at least 3 edges
    if len(edges) < 3:
        return False, f"Need at least 3 edges, got {len(edges)}"

    # Edges and assignments must match length
    if len(edges) != len(assignments):
        return False, (
            f"edges_vertices ({len(edges)}) and "
            f"edges_assignment ({len(assignments)}) must match length"
        )

    # Fold angles must match if present
    if "edges_foldAngle" in fold_data:
        angles = fold_data["edges_foldAngle"]
        if len(angles) != len(edges):
            return False, (
                f"edges_foldAngle ({len(angles)}) must match "
                f"edges_vertices ({len(edges)})"
            )

    # Validate vertex coordinates (2D or 3D)
    num_verts = len(verts)
    for i, v in enumerate(verts):
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            return False, f"Vertex {i} must be [x, y] or [x, y, z], got {v}"

    # Validate edge indices
    for i, e in enumerate(edges):
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            return False, f"Edge {i} must be [v1, v2], got {e}"
        v1, v2 = e
        if v1 < 0 or v1 >= num_verts or v2 < 0 or v2 >= num_verts:
            return False, f"Edge {i} references invalid vertex: {e}"
        if v1 == v2:
            return False, f"Edge {i} is degenerate (same vertex): {e}"

    # Validate assignments
    valid_assignments = {"M", "V", "B", "F", "U", "C"}
    for i, a in enumerate(assignments):
        if a not in valid_assignments:
            return False, f"Edge {i} has invalid assignment '{a}'"

    # Must have at least one fold crease (M or V)
    has_fold = any(a in ("M", "V") for a in assignments)
    if not has_fold:
        return False, "No fold creases (M or V) found"

    # Must have boundary edges
    has_boundary = any(a == "B" for a in assignments)
    if not has_boundary:
        return False, "No boundary edges (B) found"

    return True, ""


def parse_fold(fold_data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Parse validated FOLD JSON into numpy arrays for simulation.

    Returns dict with:
        vertices: (N, 3) float64 — vertex positions (z=0 for 2D input)
        edges: (E, 2) int — edge vertex indices
        assignments: list[str] — edge type per edge
        fold_angles: (E,) float64 — target fold angle per edge (degrees)
        faces: (F, 3) int — triangulated face vertex indices
    """
    verts = fold_data["vertices_coords"]

    # Ensure 3D (add z=0 if 2D)
    vertices = np.zeros((len(verts), 3), dtype=np.float64)
    for i, v in enumerate(verts):
        vertices[i, 0] = v[0]
        vertices[i, 1] = v[1]
        if len(v) > 2:
            vertices[i, 2] = v[2]

    edges = np.array(fold_data["edges_vertices"], dtype=np.int32)
    assignments = list(fold_data["edges_assignment"])

    # Fold angles: default based on assignment if not provided
    if "edges_foldAngle" in fold_data:
        fold_angles = np.array(fold_data["edges_foldAngle"], dtype=np.float64)
    else:
        fold_angles = np.zeros(len(edges), dtype=np.float64)
        for i, a in enumerate(assignments):
            if a == "V":
                fold_angles[i] = 180.0
            elif a == "M":
                fold_angles[i] = -180.0

    # Convert degrees to radians for simulation
    fold_angles_rad = np.radians(fold_angles)

    # Triangulate faces
    if "faces_vertices" in fold_data:
        raw_faces = fold_data["faces_vertices"]
        faces = _triangulate_faces(raw_faces)
    else:
        faces = _compute_faces(vertices, edges)

    return {
        "vertices": vertices,
        "edges": edges,
        "assignments": assignments,
        "fold_angles": fold_angles_rad,
        "faces": faces,
    }


def _triangulate_faces(raw_faces: list[list[int]]) -> np.ndarray:
    """Fan-triangulate polygon faces into triangles."""
    triangles = []
    for face in raw_faces:
        if len(face) < 3:
            continue
        for i in range(1, len(face) - 1):
            triangles.append([face[0], face[i], face[i + 1]])
    if not triangles:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(triangles, dtype=np.int32)


def _compute_faces(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Compute triangulated faces from vertices and edges using adjacency.

    Finds all triangles formed by the edge connectivity.
    """
    from collections import defaultdict

    n_verts = len(vertices)
    adj = defaultdict(set)
    for v1, v2 in edges:
        adj[v1].add(v2)
        adj[v2].add(v1)

    triangles = set()
    for v1, v2 in edges:
        common = adj[v1] & adj[v2]
        for v3 in common:
            tri = tuple(sorted([v1, v2, v3]))
            triangles.add(tri)

    if not triangles:
        # Fallback: create faces using Delaunay on 2D projection
        from scipy.spatial import Delaunay

        pts_2d = vertices[:, :2]
        try:
            tri = Delaunay(pts_2d)
            return tri.simplices.astype(np.int32)
        except Exception:
            return np.zeros((0, 3), dtype=np.int32)

    return np.array(list(triangles), dtype=np.int32)
