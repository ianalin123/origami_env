"""GRPO reward functions for origami RL training.

Two reward functions (matching the 2048 pattern):
1. valid_fold: Does the LLM output parse as valid FOLD JSON?
2. shape_match: Simulate and compare to target shape.
"""

import json
import re
from typing import Any

import numpy as np

from origami_server.engine.fold_parser import validate_fold
from origami_server.engine.shape_match import compute_shape_match
from origami_server.engine.simulate import simulate
from origami_server.tasks import get_task


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
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        fold_data = extract_fold_json(response)

        if fold_data is None:
            scores.append(-2.0)
            continue

        is_valid, error = validate_fold(fold_data)
        if is_valid:
            scores.append(1.0)
        else:
            scores.append(-0.5)

    return scores


def shape_match(
    completions: list,
    task_name: str = "triangle",
    **kwargs: Any,
) -> list[float]:
    """Reward 2: Simulate the fold and compare to target shape.

    Score = similarity × 20.0 (range: 0 to 20)
    -1.0  if simulation fails/diverges
    -2.0  if FOLD data is invalid

    This is the main reward signal — AlphaFold-style shape comparison.
    """
    task = get_task(task_name)
    target_fold = task["target_fold"]

    # Pre-compute target positions
    try:
        target_result = simulate(target_fold, crease_percent=1.0)
        target_positions = target_result.positions
    except Exception:
        # Target itself fails — all scores 0
        return [0.0] * len(completions)

    scores = []
    for completion in completions:
        response = completion[0]["content"]
        fold_data = extract_fold_json(response)

        if fold_data is None:
            scores.append(-2.0)
            continue

        is_valid, error = validate_fold(fold_data)
        if not is_valid:
            scores.append(-1.0)
            continue

        try:
            result = simulate(fold_data, crease_percent=1.0)
            similarity = compute_shape_match(result.positions, target_positions)
            scores.append(similarity * 20.0)
        except Exception:
            scores.append(-1.0)

    return scores
