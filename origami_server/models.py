"""OpenEnv types for the Origami RL environment.

OrigamiAction: LLM submits a FOLD crease pattern.
OrigamiObservation: Result of simulating that pattern against a target.
OrigamiState: Internal episode state.
"""

from typing import Any, Optional

from openenv.core import Action, Observation, State
from pydantic import Field


class OrigamiAction(Action):
    """LLM submits a FOLD crease pattern as its action.

    The fold_data dict must contain:
      - vertices_coords: [[x, y], ...] — 2D vertex positions on flat paper
      - edges_vertices: [[v1, v2], ...] — edge connectivity
      - edges_assignment: ["B"|"M"|"V", ...] — boundary/mountain/valley
      - edges_foldAngle: [angle, ...] — target fold angles in degrees
        (optional — defaults from assignment: M=-180, V=+180, B=0)
    """

    fold_data: dict[str, Any] = Field(
        ..., description="FOLD-format crease pattern JSON"
    )


class OrigamiObservation(Observation):
    """Result of simulating the LLM's crease pattern.

    Contains everything the viewer and reward function need:
    - The submitted fold data and simulation results
    - Target shape for overlay comparison
    - Shape similarity score (the reward signal)
    """

    task: dict[str, Any] = Field(default_factory=dict)
    fold_data: dict[str, Any] = Field(default_factory=dict)
    final_positions: list[list[float]] = Field(default_factory=list)
    target_positions: list[list[float]] = Field(default_factory=list)
    shape_similarity: float = 0.0
    max_strain: float = 0.0
    is_stable: bool = True
    error: Optional[str] = None


class OrigamiState(State):
    """Internal state for an origami episode."""

    task_name: str = ""
    shape_similarity: float = 0.0
    is_stable: bool = True
