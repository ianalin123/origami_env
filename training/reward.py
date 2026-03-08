"""GRPO reward functions for origami RL training.

Follows the OpenEnv 2048 pattern exactly:
- launch_openenv() spawns/reuses the origami server
- Reward functions call the server via EnvClient
- Server computes simulation + shape matching, returns reward

These functions are also importable for use in notebooks.
"""

import json
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
