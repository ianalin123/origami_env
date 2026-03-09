"""Per-step reward computation for V2 multi-step origami episodes.

Combines verifier (Kawasaki/Maekawa/BLB) and coverage-based reward from optigami.
"""

import numpy as np
from .graph import CreaseGraph
from .paper_state import PaperState

COMPLETION_BONUS = {1: 2.0, 2: 5.0, 3: 10.0, 4: 15.0}


def _compute_sector_angles(vertex_id: int, graph: CreaseGraph) -> list[float]:
    """Compute consecutive sector angles (CCW) at a vertex from its cyclic edges."""
    cyclic_edges = graph.get_cyclic_edges(vertex_id)
    n = len(cyclic_edges)
    vx, vy = graph.vertices[vertex_id]

    angles = []
    for eid in cyclic_edges:
        ev1, ev2, _ = graph.edges[eid]
        other_id = ev2 if ev1 == vertex_id else ev1
        ox, oy = graph.vertices[other_id]
        angles.append(np.arctan2(oy - vy, ox - vx))

    sectors = []
    for i in range(n):
        diff = angles[(i + 1) % n] - angles[i]
        if diff < 0:
            diff += 2 * np.pi
        if diff > 2 * np.pi:
            diff -= 2 * np.pi
        sectors.append(diff)

    return sectors


def check_kawasaki_at_vertex(vertex_id: int, graph: CreaseGraph) -> tuple[bool, float]:
    """
    Checks Kawasaki-Justin theorem at a single vertex.

    Kawasaki: at an interior vertex with 2n creases, the alternating sum
    of consecutive sector angles = 0.
    Equivalently: sum(odd-indexed sectors) == sum(even-indexed sectors) == π.

    Returns (satisfied: bool, |alternating_sum|: float).
    Returns (True, 0.0) for vertices with degree < 4 (not an interior fold vertex yet).
    Returns (False, inf) for odd-degree vertices (impossible for flat folds).
    """
    cyclic_edges = graph.get_cyclic_edges(vertex_id)
    n = len(cyclic_edges)

    if n % 2 != 0:
        return (False, float('inf'))

    if n < 4:
        return (True, 0.0)

    sectors = _compute_sector_angles(vertex_id, graph)
    alt_sum = sum(s * ((-1) ** i) for i, s in enumerate(sectors))
    return (abs(alt_sum) < 1e-9, abs(alt_sum))


def check_maekawa_at_vertex(vertex_id: int, graph: CreaseGraph) -> bool:
    """
    Checks Maekawa-Justin theorem at a single vertex.

    Maekawa: |M - V| == 2 where M, V are counts of mountain/valley fold edges
    at the vertex. BOUNDARY edges ('B') are NOT counted.

    Returns True if satisfied or if vertex has fewer than 4 fold edges (not yet active).
    """
    edge_ids = graph.vertex_edges[vertex_id]
    fold_edges = [
        eid for eid in edge_ids
        if graph.edges[eid][2] in ('M', 'V')
    ]

    if len(fold_edges) < 4:
        return True

    m_count = sum(1 for eid in fold_edges if graph.edges[eid][2] == 'M')
    v_count = sum(1 for eid in fold_edges if graph.edges[eid][2] == 'V')
    return abs(m_count - v_count) == 2


def check_blb_at_vertex(vertex_id: int, graph: CreaseGraph) -> list[tuple[int, int]]:
    """
    Checks Big-Little-Big lemma at a single vertex.

    BLB: if sector angle i is a strict local minimum (smaller than both neighbors),
    the fold edges bounding that sector must have OPPOSITE MV assignments.

    Returns list of (edge_a_id, edge_b_id) pairs where BLB is violated.
    Empty list = no violations.
    """
    cyclic_edges = graph.get_cyclic_edges(vertex_id)
    n = len(cyclic_edges)

    if n < 4:
        return []

    sectors = _compute_sector_angles(vertex_id, graph)
    violations = []

    for i in range(n):
        prev_sector = sectors[(i - 1) % n]
        next_sector = sectors[(i + 1) % n]

        if sectors[i] < prev_sector and sectors[i] < next_sector:
            edge_a = cyclic_edges[i]
            edge_b = cyclic_edges[(i + 1) % n]

            assign_a = graph.edges[edge_a][2]
            assign_b = graph.edges[edge_b][2]

            if assign_a in ('M', 'V') and assign_b in ('M', 'V'):
                if assign_a == assign_b:
                    violations.append((edge_a, edge_b))

    return violations


def _angle_diff(a1: float, a2: float) -> float:
    """Minimum angle difference between two directed lines (considering 180° symmetry)."""
    diff = abs(a1 - a2) % np.pi
    return min(diff, np.pi - diff)


def geometric_crease_coverage(
    state: PaperState,
    target_edges: list[dict],
    tol_pos: float = 0.05,
    tol_angle_deg: float = 5.0,
) -> tuple[float, float, float]:
    """
    Computes how well the current crease pattern matches the target.

    Args:
        state: current paper state with crease graph
        target_edges: list of {'v1': (x1,y1), 'v2': (x2,y2), 'assignment': 'M'|'V'}
        tol_pos: position tolerance for midpoint matching
        tol_angle_deg: angle tolerance in degrees for direction matching

    Returns:
        (coverage, economy, assignment_accuracy)
        coverage: weighted fraction of target creases matched [0, 1];
                  1.0 if position+assignment match, 0.5 if position matches but assignment doesn't
        economy: penalty for excess creases [0, 1], 1.0 = no excess
        assignment_accuracy: fraction of positionally matched edges that also have correct M/V assignment [0, 1];
                            returns 1.0 if no positional matches (vacuous case)
    """
    current_edges = state.crease_edges()
    tol_angle_rad = np.deg2rad(tol_angle_deg)

    total_score = 0.0
    position_matches = 0
    assignment_correct = 0

    for target in target_edges:
        tx1, ty1 = target['v1']
        tx2, ty2 = target['v2']
        t_mid = ((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0)
        t_angle = np.arctan2(ty2 - ty1, tx2 - tx1)
        t_assign = target.get('assignment', 'M')

        for current in current_edges:
            cx1, cy1 = current['v1']
            cx2, cy2 = current['v2']
            c_mid = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
            c_angle = np.arctan2(cy2 - cy1, cx2 - cx1)
            c_assign = current.get('assignment', 'M')

            mid_dist = np.hypot(c_mid[0] - t_mid[0], c_mid[1] - t_mid[1])
            angle_distance = _angle_diff(c_angle, t_angle)

            if mid_dist <= tol_pos and angle_distance <= tol_angle_rad:
                position_matches += 1
                assign_match = (t_assign == c_assign)
                if assign_match:
                    total_score += 1.0
                    assignment_correct += 1
                else:
                    total_score += 0.5
                break

    coverage = total_score / max(len(target_edges), 1)
    n_excess = max(0, len(current_edges) - len(target_edges))
    economy = max(0.0, 1.0 - n_excess / max(len(target_edges), 1))
    assignment_accuracy = (
        assignment_correct / position_matches if position_matches > 0 else 1.0
    )
    return (coverage, economy, assignment_accuracy)


def check_degree_sanity(graph: CreaseGraph) -> float:
    """
    Checks that interior vertices have even degree (required for flat-foldability).

    Returns:
        Fraction of interior vertices with even degree [0, 1].
        1.0 = all interior vertices have even degree.
        0.0 = none do.
        Returns 1.0 if there are no interior vertices (vacuous case).
    """
    interior = graph.interior_vertices()
    if not interior:
        return 1.0
    even_count = sum(
        1 for vid in interior
        if len(graph.vertex_edges[vid]) % 2 == 0
    )
    return even_count / len(interior)


def check_all_vertices(graph: CreaseGraph) -> dict:
    """
    Run all vertex-level checks on every interior vertex.

    Returns dict with:
        'kawasaki': float  # fraction of interior vertices passing Kawasaki [0,1]
        'maekawa': float   # fraction passing Maekawa [0,1]
        'blb': float       # fraction with no BLB violations [0,1]
        'n_interior': int  # number of interior vertices checked
        'per_vertex': list[dict]  # per-vertex details
    """
    interior = graph.interior_vertices()

    if not interior:
        return {
            'kawasaki': 1.0,
            'maekawa': 1.0,
            'blb': 1.0,
            'n_interior': 0,
            'per_vertex': [],
        }

    per_vertex = []
    kaw_pass = 0
    mae_pass = 0
    blb_pass = 0

    for vid in interior:
        kaw_ok, kaw_val = check_kawasaki_at_vertex(vid, graph)
        mae_ok = check_maekawa_at_vertex(vid, graph)
        blb_violations = check_blb_at_vertex(vid, graph)
        blb_ok = len(blb_violations) == 0

        kaw_pass += int(kaw_ok)
        mae_pass += int(mae_ok)
        blb_pass += int(blb_ok)

        per_vertex.append({
            'vertex_id': vid,
            'kawasaki_ok': kaw_ok,
            'kawasaki_error': kaw_val,
            'maekawa_ok': mae_ok,
            'blb_violations': blb_violations,
        })

    n = len(interior)
    return {
        'kawasaki': kaw_pass / n,
        'maekawa': mae_pass / n,
        'blb': blb_pass / n,
        'n_interior': n,
        'per_vertex': per_vertex,
    }


def target_crease_edges(target: dict) -> list[dict]:
    """
    Extract crease edges from a FOLD target dict as list of
    {'v1': (x1,y1), 'v2': (x2,y2), 'assignment': 'M'|'V'} dicts.
    """
    verts = target['vertices_coords']
    result = []
    for i, (v1_idx, v2_idx) in enumerate(target['edges_vertices']):
        assignment = target['edges_assignment'][i]
        if assignment in ('M', 'V'):
            result.append({
                'v1': tuple(verts[v1_idx]),
                'v2': tuple(verts[v2_idx]),
                'assignment': assignment,
            })
    return result


def compute_reward(
    prev_state: PaperState,
    action_result: dict,
    new_state: PaperState,
    target: dict,
    step: int,
    max_steps: int,
) -> dict:
    """
    Compute the full reward dict for a fold action (lexicographically gated).

    Args:
        prev_state: PaperState BEFORE the action was applied
        action_result: {'valid': bool, 'anchored': bool, 'duplicate': bool, ...}
        new_state: PaperState AFTER the action was applied
        target: FOLD target dict
        step: current step index
        max_steps: maximum steps in episode

    Returns dict with keys:
        format, anchored, novelty, kawasaki, maekawa, blb, degree_sanity,
        progress, economy, assignment_accuracy, delta, regression,
        completion, efficiency, total
    """
    r = {}

    # GATE 1: Format — did the action parse and apply?
    r['format'] = 1.0 if action_result.get('valid', False) else 0.0
    if not r['format']:
        r['total'] = -0.1
        return r

    # GATE 2: Structural sanity
    r['anchored'] = 1.0 if action_result.get('anchored', False) else 0.3
    r['novelty'] = 0.0 if action_result.get('duplicate', False) is True else 0.2

    # LEVEL 3: Local flat-foldability
    vertex_scores = check_all_vertices(new_state.graph)
    r['kawasaki'] = vertex_scores['kawasaki']
    r['maekawa'] = vertex_scores['maekawa']
    r['blb'] = vertex_scores['blb']
    r['degree_sanity'] = check_degree_sanity(new_state.graph)

    # LEVEL 4: Progress (absolute + delta)
    fold_target = target.get("target_fold", target)
    t_edges = target_crease_edges(fold_target)
    old_coverage, _, _ = geometric_crease_coverage(prev_state, t_edges)
    new_coverage, economy, assignment_accuracy = geometric_crease_coverage(new_state, t_edges)

    r['progress'] = new_coverage
    r['economy'] = economy
    r['assignment_accuracy'] = assignment_accuracy
    r['delta'] = max(0.0, new_coverage - old_coverage)
    r['regression'] = min(0.0, new_coverage - old_coverage)

    # LEVEL 5: Completion bonus
    all_valid = (
        r['kawasaki'] == 1.0
        and r['maekawa'] == 1.0
        and r['blb'] == 1.0
    )
    difficulty = target.get("difficulty", 1)
    bonus = COMPLETION_BONUS.get(difficulty, 10.0)
    r['completion'] = bonus if (r['progress'] > 0.9 and all_valid) else 0.0

    # LEVEL 6: Efficiency — escalating step cost
    r['efficiency'] = -0.01 * (1 + step / max_steps)

    # Weighted total
    r['total'] = (
        0.05 * r['anchored']
        + 0.05 * r['novelty']
        + 0.06 * r['kawasaki']
        + 0.06 * r['maekawa']
        + 0.04 * r['blb']
        + 0.04 * r['degree_sanity']
        + 0.25 * r['progress']
        + 0.05 * r['economy']
        + 0.05 * r['assignment_accuracy']
        + 0.20 * r['delta']
        + 0.10 * r['regression']
        + r['completion']
        + r['efficiency']
    )
    return r


def compute_terminal_reward(
    state: PaperState,
    target: dict,
    max_steps: int,
) -> dict:
    """
    Compute reward for the final state after a complete fold sequence.
    Uses fresh PaperState as baseline and step = max_steps.
    """
    fake_result = {
        'valid': True,
        'anchored': True,
        'duplicate': False,
    }
    return compute_reward(
        prev_state=PaperState(),
        action_result=fake_result,
        new_state=state,
        target=target,
        step=max_steps,
        max_steps=max_steps,
    )
