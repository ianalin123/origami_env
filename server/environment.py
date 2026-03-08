"""Origami RL Environment — OpenEnv Environment subclass.

Single-shot episodes: LLM submits a FOLD crease pattern, physics simulates it,
reward = shape similarity to target. Like AlphaFold for origami.
"""

import uuid
from typing import Any, Optional

import numpy as np
from openenv.core import Environment

from .engine.fold_parser import validate_fold
from .engine.shape_match import compute_shape_match
from .engine.simulate import SimResult, simulate
from .models import OrigamiAction, OrigamiObservation, OrigamiState
from .tasks import get_task


class OrigamiEnvironment(
    Environment[OrigamiAction, OrigamiObservation, OrigamiState]
):
    """Origami folding environment.

    Episode flow:
        1. reset(task_name="triangle") → returns task description + target info
        2. step(OrigamiAction(fold_data={...})) → simulates, scores, returns done=True

    Single action per episode. The action IS the complete crease pattern.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._state = OrigamiState()
        self._task: dict = {}
        self._target_positions: np.ndarray = np.zeros((0, 3))

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OrigamiObservation:
        """Start a new episode with a target shape task."""
        self._state = OrigamiState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )

        # Get task
        task_name = kwargs.get("task_name", "triangle")
        self._task = get_task(task_name)
        self._state.task_name = self._task["name"]

        # Simulate the target FOLD to get target positions
        target_fold = self._task["target_fold"]
        try:
            target_result = simulate(target_fold, crease_percent=1.0)
            self._target_positions = target_result.positions
        except Exception as e:
            self._target_positions = np.zeros((0, 3))

        return OrigamiObservation(
            done=False,
            reward=None,
            task={
                "name": self._task["name"],
                "description": self._task["description"],
                "difficulty": self._task["difficulty"],
                "paper": self._task["paper"],
            },
            fold_data={},
            final_positions=[],
            target_positions=self._target_positions.tolist(),
            shape_similarity=0.0,
            max_strain=0.0,
            is_stable=True,
            error=None,
        )

    def step(
        self,
        action: OrigamiAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OrigamiObservation:
        """Evaluate the LLM's crease pattern.

        1. Validate FOLD data
        2. Run physics simulation (creasePercent=1.0)
        3. Compare final shape to target
        4. Return observation with reward = similarity × 20
        """
        self._state.step_count += 1
        fold_data = action.fold_data

        # Validate
        is_valid, error_msg = validate_fold(fold_data)
        if not is_valid:
            self._state.is_stable = False
            return OrigamiObservation(
                done=True,
                reward=-2.0,
                task=self._task_info(),
                fold_data=fold_data,
                final_positions=[],
                target_positions=self._target_positions.tolist(),
                shape_similarity=0.0,
                max_strain=0.0,
                is_stable=False,
                error=f"Invalid FOLD data: {error_msg}",
            )

        # Simulate
        try:
            result: SimResult = simulate(fold_data, crease_percent=1.0)
        except Exception as e:
            self._state.is_stable = False
            return OrigamiObservation(
                done=True,
                reward=-2.0,
                task=self._task_info(),
                fold_data=fold_data,
                final_positions=[],
                target_positions=self._target_positions.tolist(),
                shape_similarity=0.0,
                max_strain=0.0,
                is_stable=False,
                error=f"Simulation error: {str(e)}",
            )

        # Shape match
        similarity = compute_shape_match(
            result.positions, self._target_positions
        )
        reward = similarity * 20.0

        self._state.shape_similarity = similarity
        self._state.is_stable = result.converged

        return OrigamiObservation(
            done=True,
            reward=reward,
            task=self._task_info(),
            fold_data=fold_data,
            final_positions=result.positions.tolist(),
            target_positions=self._target_positions.tolist(),
            shape_similarity=similarity,
            max_strain=result.max_strain,
            is_stable=result.converged,
            error=None,
        )

    @property
    def state(self) -> OrigamiState:
        return self._state

    def _task_info(self) -> dict:
        """Task info dict for observations."""
        if not self._task:
            return {}
        return {
            "name": self._task.get("name", ""),
            "description": self._task.get("description", ""),
            "difficulty": self._task.get("difficulty", 0),
            "paper": self._task.get("paper", {}),
        }
