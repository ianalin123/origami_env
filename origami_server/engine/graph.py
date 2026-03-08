import numpy as np
from typing import Optional

BOUNDARY_TOL = 1e-9
VERTEX_TOL = 1e-9


class CreaseGraph:
    """
    Planar graph representing an origami crease pattern on a unit square.

    Vertices: points in [0,1]x[0,1], deduplicated by proximity.
    Edges: segments between vertices, labeled M (mountain), V (valley), or B (boundary).
    """

    def __init__(self):
        self.vertices: dict[int, tuple[float, float]] = {}
        self.edges: dict[int, tuple[int, int, str]] = {}
        self.vertex_edges: dict[int, list[int]] = {}
        self._next_vertex_id: int = 0
        self._next_edge_id: int = 0

        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        for x, y in corners:
            vid = self._next_vertex_id
            self.vertices[vid] = (x, y)
            self.vertex_edges[vid] = []
            self._next_vertex_id += 1

        boundary_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for v1, v2 in boundary_pairs:
            eid = self._next_edge_id
            self.edges[eid] = (v1, v2, 'B')
            self.vertex_edges[v1].append(eid)
            self.vertex_edges[v2].append(eid)
            self._next_edge_id += 1

    def add_vertex(self, x: float, y: float) -> int:
        for vid, (vx, vy) in self.vertices.items():
            if abs(vx - x) < VERTEX_TOL and abs(vy - y) < VERTEX_TOL:
                return vid
        vid = self._next_vertex_id
        self.vertices[vid] = (float(x), float(y))
        self.vertex_edges[vid] = []
        self._next_vertex_id += 1
        return vid

    def add_edge(self, v1_id: int, v2_id: int, assignment: str) -> int:
        pair = frozenset((v1_id, v2_id))
        for eid, (ev1, ev2, _) in self.edges.items():
            if frozenset((ev1, ev2)) == pair:
                return eid
        eid = self._next_edge_id
        self.edges[eid] = (v1_id, v2_id, assignment)
        self.vertex_edges[v1_id].append(eid)
        self.vertex_edges[v2_id].append(eid)
        self._next_edge_id += 1
        return eid

    def get_cyclic_edges(self, vertex_id: int) -> list[int]:
        vx, vy = self.vertices[vertex_id]
        edge_ids = self.vertex_edges[vertex_id]

        def angle_of_edge(eid: int) -> float:
            ev1, ev2, _ = self.edges[eid]
            other_id = ev2 if ev1 == vertex_id else ev1
            ox, oy = self.vertices[other_id]
            return float(np.arctan2(oy - vy, ox - vx))

        return sorted(edge_ids, key=angle_of_edge)

    def interior_vertices(self) -> list[int]:
        result = []
        for vid, (x, y) in self.vertices.items():
            if (
                x > BOUNDARY_TOL
                and x < 1.0 - BOUNDARY_TOL
                and y > BOUNDARY_TOL
                and y < 1.0 - BOUNDARY_TOL
            ):
                result.append(vid)
        return result

    def split_edge(self, edge_id: int, new_vertex_id: int) -> tuple[int, int]:
        ev1, ev2, assignment = self.edges[edge_id]

        del self.edges[edge_id]
        if edge_id in self.vertex_edges[ev1]:
            self.vertex_edges[ev1].remove(edge_id)
        if edge_id in self.vertex_edges[ev2]:
            self.vertex_edges[ev2].remove(edge_id)

        eid1 = self._next_edge_id
        self.edges[eid1] = (ev1, new_vertex_id, assignment)
        self.vertex_edges[ev1].append(eid1)
        self.vertex_edges[new_vertex_id].append(eid1)
        self._next_edge_id += 1

        eid2 = self._next_edge_id
        self.edges[eid2] = (new_vertex_id, ev2, assignment)
        self.vertex_edges[new_vertex_id].append(eid2)
        self.vertex_edges[ev2].append(eid2)
        self._next_edge_id += 1

        return (eid1, eid2)

    def crease_edges(self) -> list[int]:
        return [eid for eid, (_, _, a) in self.edges.items() if a in ('M', 'V')]

    def boundary_midpoints(self) -> list[tuple[float, float]]:
        midpoints = []
        for eid, (v1, v2, assignment) in self.edges.items():
            if assignment == 'B':
                x1, y1 = self.vertices[v1]
                x2, y2 = self.vertices[v2]
                midpoints.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        return midpoints
