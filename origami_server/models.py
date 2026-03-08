"""OpenEnv types for the Origami RL environment.

OrigamiAction: LLM submits a FOLD crease pattern.
OrigamiObservation: Result of simulating that pattern against a target.
OrigamiState: Internal episode state.
"""

from typing import Any, Optional

from openenv.core import Action, Observation, State
from pydantic import Field


class OrigamiAction(Action):
    """LLM action — either a complete FOLD pattern (V1) or a single crease (V2).

    V1 (single-shot): set fold_data with complete FOLD-format crease pattern.
    V2 (multi-step):  set crease with {"from": [x,y], "to": [x,y], "assignment": "M"|"V"}.
    Exactly one of fold_data or crease must be set.
    """

    fold_data: dict[str, Any] | None = Field(
        default=None, description="V1: complete FOLD-format crease pattern JSON"
    )
    crease: dict[str, Any] | None = Field(
        default=None,
        description='V2: single crease {"from": [x,y], "to": [x,y], "assignment": "M"|"V"}',
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

    # V2 multi-step fields
    step_count: int = 0
    max_steps: int = 1
    current_creases: list[dict[str, Any]] = Field(default_factory=list)
    anchor_points: list[list[float]] = Field(default_factory=list)
    reward_breakdown: dict[str, float] = Field(default_factory=dict)


class OrigamiState(State):
    """Internal state for an origami episode."""

    task_name: str = ""
    mode: str = "single"  # "single" (V1) | "step" (V2)
    step_count: int = 0
    shape_similarity: float = 0.0
    is_stable: bool = True
