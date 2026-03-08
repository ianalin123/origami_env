"""GRPO reward functions for origami RL training.

Follows the OpenEnv 2048 pattern exactly:
- launch_openenv() spawns/reuses the origami server
- Reward functions call the server via EnvClient
- Server computes simulation + shape matching, returns reward

These functions are also importable for use in notebooks.
"""

import json
import math
import re
from typing import Any


def extract_fold_json(response: str) -> dict | None:
    """Extract FOLD JSON from LLM response text.

    Looks for JSON between ```json ... ``` or raw JSON object.
    """
    # Try fenced code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON object
    match = re.search(r"\{[^{}]*\"vertices_coords\"[^{}]*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try parsing the whole response
    try:
        data = json.loads(response.strip())
        if isinstance(data, dict) and "vertices_coords" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def valid_fold(completions: list, **kwargs: Any) -> list[float]:
    """Reward 1: Does the LLM output parse as valid FOLD JSON?

    +1.0  valid FOLD JSON with correct structure
    -0.5  parseable JSON but invalid FOLD structure
    -2.0  not parseable as JSON at all

    Local check — no server needed.
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        fold_data = extract_fold_json(response)

        if fold_data is None:
            scores.append(-2.0)
            continue

        # Basic structural validation
        required = {"vertices_coords", "edges_vertices", "edges_assignment"}
        if not required.issubset(fold_data.keys()):
            scores.append(-0.5)
            continue

        verts = fold_data.get("vertices_coords", [])
        edges = fold_data.get("edges_vertices", [])
        assigns = fold_data.get("edges_assignment", [])

        if len(edges) != len(assigns):
            scores.append(-0.5)
            continue

        has_fold = any(a in ("M", "V") for a in assigns)
        has_boundary = any(a == "B" for a in assigns)
        if not has_fold or not has_boundary:
            scores.append(-0.5)
            continue

        n = len(verts)
        valid_indices = all(
            0 <= e[0] < n and 0 <= e[1] < n and e[0] != e[1]
            for e in edges
        )
        if not valid_indices:
            scores.append(-0.5)
            continue

        scores.append(1.0)

    return scores


# ── Flat-foldability reward ────────────────────────────────────────────────────
# Ported from optigami/env/verifier.py — operates directly on raw FOLD JSON
# so it runs locally with no server round-trip.
#
# Three theorems checked at every interior vertex (vertex not on paper boundary):
#   Kawasaki: alternating sector angles must each sum to π
#   Maekawa:  |mountain_count - valley_count| = 2
#   BLB:      smallest sector must be bounded by folds of opposite type


def _cyclic_incident(vertex_idx: int, fold_data: dict) -> list[tuple[float, str]]:
    """Return (angle, assignment) pairs for edges around vertex_idx, sorted CCW."""
    verts = fold_data["vertices_coords"]
    vx, vy = verts[vertex_idx]
    result = []
    for (v1, v2), assign in zip(fold_data["edges_vertices"], fold_data["edges_assignment"]):
        if v1 == vertex_idx or v2 == vertex_idx:
            other = v2 if v1 == vertex_idx else v1
            ox, oy = verts[other]
            result.append((math.atan2(oy - vy, ox - vx), assign))
    result.sort(key=lambda t: t[0])
    return result


def _sector_angles(incident: list[tuple[float, str]]) -> list[float]:
    n = len(incident)
    sectors = []
    for i in range(n):
        diff = incident[(i + 1) % n][0] - incident[i][0]
        if diff < 0:
            diff += 2 * math.pi
        sectors.append(diff)
    return sectors


def _kawasaki_ok(vertex_idx: int, fold_data: dict) -> bool:
    inc = _cyclic_incident(vertex_idx, fold_data)
    n = len(inc)
    if n % 2 != 0:
        return False
    if n < 4:
        return True
    sectors = _sector_angles(inc)
    alt_sum = sum(s * ((-1) ** i) for i, s in enumerate(sectors))
    return abs(alt_sum) < 1e-6


def _maekawa_ok(vertex_idx: int, fold_data: dict) -> bool:
    inc = _cyclic_incident(vertex_idx, fold_data)
    folds = [a for _, a in inc if a in ("M", "V")]
    if len(folds) < 4:
        return True
    m = sum(1 for a in folds if a == "M")
    v = len(folds) - m
    return abs(m - v) == 2


def _blb_ok(vertex_idx: int, fold_data: dict) -> bool:
    inc = _cyclic_incident(vertex_idx, fold_data)
    n = len(inc)
    if n < 4:
        return True
    sectors = _sector_angles(inc)
    for i in range(n):
        if sectors[i] < sectors[(i - 1) % n] and sectors[i] < sectors[(i + 1) % n]:
            a_left = inc[i][1]
            a_right = inc[(i + 1) % n][1]
            if a_left in ("M", "V") and a_right in ("M", "V") and a_left == a_right:
                return False
    return True


def _interior_vertices(fold_data: dict) -> list[int]:
    """Vertices strictly inside the paper boundary (not at x/y = 0 or 1)."""
    eps = 1e-6
    width = max(x for x, y in fold_data["vertices_coords"])
    height = max(y for x, y in fold_data["vertices_coords"])
    return [
        i for i, (x, y) in enumerate(fold_data["vertices_coords"])
        if eps < x < width - eps and eps < y < height - eps
    ]


def flat_foldable_reward(completions: list, **kwargs: Any) -> list[float]:
    """Reward 3: flat-foldability at interior vertices (Kawasaki + Maekawa + BLB).

    Score is the weighted fraction of interior vertices passing all three theorems.
    Returns 0.0 if no interior vertices exist (nothing to check yet).
    Returns -0.5 if the JSON is unparseable.

    Ported from optigami/env/verifier.py — runs locally, no server needed.
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        fold_data = extract_fold_json(response)

        if fold_data is None:
            scores.append(-0.5)
            continue

        required = {"vertices_coords", "edges_vertices", "edges_assignment"}
        if not required.issubset(fold_data.keys()):
            scores.append(-0.5)
            continue

        try:
            interior = _interior_vertices(fold_data)
            if not interior:
                scores.append(0.0)
                continue

            n = len(interior)
            kaw = sum(1 for v in interior if _kawasaki_ok(v, fold_data)) / n
            mae = sum(1 for v in interior if _maekawa_ok(v, fold_data)) / n
            blb = sum(1 for v in interior if _blb_ok(v, fold_data)) / n

            # Kawasaki and Maekawa are equally fundamental; BLB is a corollary
            scores.append(0.4 * kaw + 0.4 * mae + 0.2 * blb)
        except Exception:
            scores.append(0.0)

    return scores


# ── V2 single-crease helpers ───────────────────────────────────────────────────


def extract_crease_json(response: str) -> dict | None:
    """Extract single-crease JSON from LLM response.

    Looks for {"from": ..., "to": ..., "assignment": ...} object.
    """
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    raw = response[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if not all(k in data for k in ("from", "to", "assignment")):
        return None
    return data


def valid_crease(completions: list, **kwargs: Any) -> list[float]:
    """V2 Reward: does the LLM output parse as a valid single-crease JSON?

    +1.0  valid {"from": [x,y], "to": [x,y], "assignment": "M"|"V"}
    -0.5  parseable JSON but missing required fields or wrong assignment value
    -2.0  not parseable as JSON
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        data = extract_crease_json(response)

        if data is None:
            scores.append(-2.0)
            continue

        from_pt = data.get("from", [])
        to_pt = data.get("to", [])
        assignment = data.get("assignment", "")

        if (
            not isinstance(from_pt, list) or len(from_pt) != 2
            or not all(isinstance(v, (int, float)) for v in from_pt)
        ):
            scores.append(-0.5)
            continue

        if (
            not isinstance(to_pt, list) or len(to_pt) != 2
            or not all(isinstance(v, (int, float)) for v in to_pt)
        ):
            scores.append(-0.5)
            continue

        if assignment not in ("M", "V"):
            scores.append(-0.5)
            continue

        scores.append(1.0)

    return scores
