import numpy as np
from shapely.geometry import LineString, Point, Polygon
from typing import Optional
from .graph import CreaseGraph, VERTEX_TOL

UNIT_SQUARE_CORNERS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

_UNIT_SQUARE = Polygon(UNIT_SQUARE_CORNERS)


class PaperState:
    """
    Represents the evolving crease pattern on a unit square [0,1]x[0,1].
    Uses CreaseGraph for the underlying data structure.
    """

    def __init__(self):
        self.graph = CreaseGraph()
        self.fold_history: list[dict] = []

    def anchor_points(self) -> list[tuple[float, float]]:
        points: dict[tuple[float, float], None] = {}
        for corner in UNIT_SQUARE_CORNERS:
            points[corner] = None
        for vid, (x, y) in self.graph.vertices.items():
            points[(float(x), float(y))] = None
        return list(points.keys())

    def _is_anchor(self, pt: tuple[float, float]) -> bool:
        px, py = pt
        for ax, ay in self.anchor_points():
            if abs(ax - px) < VERTEX_TOL and abs(ay - py) < VERTEX_TOL:
                return True
        return False

    def _edge_exists(self, v1_id: int, v2_id: int) -> bool:
        """Check if an edge already exists between the two vertex IDs."""
        pair = frozenset((v1_id, v2_id))
        for ev1, ev2, _ in self.graph.edges.values():
            if frozenset((ev1, ev2)) == pair:
                return True
        return False

    def add_crease(self, p1: list, p2: list, assignment: str) -> dict:
        errors: list[str] = []

        if assignment not in ('M', 'V'):
            return {
                'valid': False,
                'anchored': False,
                'new_vertices': [],
                'errors': ['invalid_assignment'],
                'duplicate': False,
            }

        p1 = (float(p1[0]), float(p1[1]))
        p2 = (float(p2[0]), float(p2[1]))

        anchored = self._is_anchor(p1) and self._is_anchor(p2)

        seg_len = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if seg_len < VERTEX_TOL:
            errors.append('zero_length')
            return {'valid': False, 'anchored': anchored, 'new_vertices': [], 'errors': errors, 'duplicate': False}

        new_line = LineString([p1, p2])

        if not _UNIT_SQUARE.contains(new_line) and not _UNIT_SQUARE.boundary.contains(new_line):
            clipped = new_line.intersection(_UNIT_SQUARE)
            if clipped.is_empty:
                errors.append('outside_bounds')
                return {'valid': False, 'anchored': anchored, 'new_vertices': [], 'errors': errors, 'duplicate': False}

        intersection_points: list[tuple[float, float]] = []

        for eid, (ev1, ev2, _) in list(self.graph.edges.items()):
            ex1, ey1 = self.graph.vertices[ev1]
            ex2, ey2 = self.graph.vertices[ev2]
            existing_line = LineString([(ex1, ey1), (ex2, ey2)])
            inter = new_line.intersection(existing_line)

            if inter.is_empty:
                continue

            if inter.geom_type == 'Point':
                ix, iy = inter.x, inter.y
                ep1 = (ex1, ey1)
                ep2 = (ex2, ey2)
                if (
                    abs(ix - ep1[0]) < VERTEX_TOL and abs(iy - ep1[1]) < VERTEX_TOL
                    or abs(ix - ep2[0]) < VERTEX_TOL and abs(iy - ep2[1]) < VERTEX_TOL
                ):
                    continue
                intersection_points.append((ix, iy))
            # MultiPoint or LineString intersections (collinear) are skipped

        new_vertex_coords: list[tuple[float, float]] = []
        for ix, iy in intersection_points:
            before = set(self.graph.vertices.keys())
            vid = self.graph.add_vertex(ix, iy)
            if vid not in before:
                new_vertex_coords.append((ix, iy))

            for eid in list(self.graph.edges.keys()):
                if eid not in self.graph.edges:
                    continue
                ev1, ev2, _ = self.graph.edges[eid]
                ex1, ey1 = self.graph.vertices[ev1]
                ex2, ey2 = self.graph.vertices[ev2]
                seg = LineString([(ex1, ey1), (ex2, ey2)])
                pt = Point(ix, iy)
                if seg.distance(pt) < VERTEX_TOL:
                    if ev1 != vid and ev2 != vid:
                        self.graph.split_edge(eid, vid)

        v1_id = self.graph.add_vertex(p1[0], p1[1])
        v2_id = self.graph.add_vertex(p2[0], p2[1])

        waypoints = [p1] + sorted(
            intersection_points,
            key=lambda pt: np.hypot(pt[0] - p1[0], pt[1] - p1[1]),
        ) + [p2]

        waypoint_ids = []
        for wp in waypoints:
            wid = self.graph.add_vertex(wp[0], wp[1])
            waypoint_ids.append(wid)

        duplicate = any(
            self._edge_exists(waypoint_ids[i], waypoint_ids[i + 1])
            for i in range(len(waypoint_ids) - 1)
        )

        for i in range(len(waypoint_ids) - 1):
            wa = waypoint_ids[i]
            wb = waypoint_ids[i + 1]
            if wa != wb:
                self.graph.add_edge(wa, wb, assignment)

        record = {
            'p1': p1,
            'p2': p2,
            'assignment': assignment,
            'anchored': anchored,
            'new_vertices': new_vertex_coords,
        }
        self.fold_history.append(record)

        return {
            'valid': True,
            'anchored': anchored,
            'new_vertices': new_vertex_coords,
            'errors': errors,
            'duplicate': duplicate,
        }

    def crease_edges(self) -> list[dict]:
        result = []
        for eid in self.graph.crease_edges():
            v1, v2, assignment = self.graph.edges[eid]
            x1, y1 = self.graph.vertices[v1]
            x2, y2 = self.graph.vertices[v2]
            result.append({'v1': (x1, y1), 'v2': (x2, y2), 'assignment': assignment})
        return result


def hash_paper_state(state: PaperState) -> int:
    edges = state.crease_edges()
    canonical = []
    for e in edges:
        p1 = tuple(round(c, 6) for c in e["v1"])
        p2 = tuple(round(c, 6) for c in e["v2"])
        canonical.append((min(p1, p2), max(p1, p2), e["assignment"]))
    canonical.sort()
    return hash(tuple(canonical))
