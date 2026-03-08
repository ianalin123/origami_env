"""Origami RL Environment — OpenEnv Environment subclass.

Origami folding environment — supports single-shot (V1) and multi-step (V2) modes.

V1 (mode='single'): LLM submits complete FOLD JSON, gets Chamfer-distance reward. Done=True after 1 step.
V2 (mode='step'): LLM submits one crease per step, gets per-step reward (progress + geometry). Done=True when max_folds reached or completion bonus triggered.
"""

import copy
import uuid
from typing import Any, Optional

import numpy as np
from openenv.core import Environment

from .engine.fold_parser import validate_fold
from .engine.paper_state import PaperState
from .engine.shape_match import compute_shape_match
from .engine.simulate import SimResult, simulate
from .engine.step_reward import compute_reward
from .models import OrigamiAction, OrigamiObservation, OrigamiState
from .tasks import get_task


class OrigamiEnvironment(
    Environment[OrigamiAction, OrigamiObservation, OrigamiState]
):
    """Origami folding environment — supports single-shot (V1) and multi-step (V2) modes.

    V1 (mode='single'): LLM submits complete FOLD JSON, gets Chamfer-distance reward. Done=True after 1 step.
    V2 (mode='step'): LLM submits one crease per step, gets per-step reward (progress + geometry). Done=True when max_folds reached or completion bonus triggered.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, mode: str = "step", **kwargs: Any):
        super().__init__(**kwargs)
        self._mode = mode  # "step" (V2 default) | "single" (V1 compat)
        self._state = OrigamiState()
        self._task: dict = {}
        self._target_positions: np.ndarray = np.zeros((0, 3))
        self._paper_state: Optional[PaperState] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OrigamiObservation:
        """Start a new episode with a target shape task."""
        task_name = kwargs.get("task_name", "triangle")
        self._task = get_task(task_name)

        self._state = OrigamiState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            mode=self._mode,
            task_name=self._task["name"],
        )

        # Simulate target FOLD to get target positions
        target_fold = self._task["target_fold"]
        try:
            target_result = simulate(target_fold, crease_percent=1.0)
            self._target_positions = target_result.positions
        except Exception:
            self._target_positions = np.zeros((0, 3))

        # V2: initialize empty paper state
        if self._mode == "step":
            self._paper_state = PaperState()
            anchor_pts = [[x, y] for x, y in self._paper_state.anchor_points()]
            return OrigamiObservation(
                done=False,
                reward=None,
                task=self._task_info(),
                fold_data={},
                final_positions=[],
                target_positions=self._target_positions.tolist(),
                shape_similarity=0.0,
                max_strain=0.0,
                is_stable=True,
                error=None,
                step_count=0,
                max_steps=self._task.get("max_folds", 1),
                current_creases=[],
                anchor_points=anchor_pts,
                reward_breakdown={},
            )

        # V1: return initial observation (unchanged behavior)
        return OrigamiObservation(
            done=False,
            reward=None,
            task=self._task_info(),
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
        """Dispatch to V2 crease step or V1 full-fold step."""
        # V2: single-crease step
        if action.crease is not None:
            return self._step_crease(action.crease)
        # V1: complete FOLD JSON (backward compat)
        if action.fold_data is not None:
            return self._step_fold(action.fold_data)
        # Neither set
        return OrigamiObservation(
            done=True,
            reward=-2.0,
            task=self._task_info(),
            fold_data={},
            final_positions=[],
            target_positions=self._target_positions.tolist(),
            shape_similarity=0.0,
            max_strain=0.0,
            is_stable=False,
            error="OrigamiAction must set either fold_data (V1) or crease (V2)",
        )

    def _step_crease(self, crease: dict) -> OrigamiObservation:
        """V2: apply one crease, compute per-step reward."""
        if self._paper_state is None:
            self._paper_state = PaperState()

        # Validate crease fields
        assignment = crease.get("assignment", "")
        from_pt = crease.get("from")
        to_pt = crease.get("to")
        if assignment not in ("M", "V") or from_pt is None or to_pt is None:
            done = self._state.step_count >= self._task.get("max_folds", 1)
            return OrigamiObservation(
                done=done,
                reward=-0.1,
                task=self._task_info(),
                fold_data={},
                final_positions=[],
                target_positions=self._target_positions.tolist(),
                shape_similarity=0.0,
                max_strain=0.0,
                is_stable=False,
                error=f"Invalid crease: {crease}",
                step_count=self._state.step_count,
                max_steps=self._task.get("max_folds", 1),
                current_creases=self._paper_state.crease_edges(),
                anchor_points=[[x, y] for x, y in self._paper_state.anchor_points()],
                reward_breakdown={},
            )

        prev_state = copy.deepcopy(self._paper_state)
        result = self._paper_state.add_crease(from_pt, to_pt, assignment)
        self._state.step_count += 1

        reward_dict = compute_reward(
            prev_state=prev_state,
            action_result=result,
            new_state=self._paper_state,
            target=self._task,
            step=self._state.step_count,
            max_steps=self._task.get("max_folds", 1),
        )

        max_folds = self._task.get("max_folds", 1)
        done = (
            self._state.step_count >= max_folds
            or reward_dict.get("completion", 0) > 0
        )

        self._state.shape_similarity = reward_dict.get("progress", 0.0)

        # On final step, run full simulation for viewer
        final_positions: list = []
        if done and self._paper_state.crease_edges():
            try:
                fold_data = self._paper_state_to_fold()
                sim = simulate(fold_data, crease_percent=1.0)
                final_positions = sim.positions.tolist()
            except Exception:
                pass

        return OrigamiObservation(
            done=done,
            reward=reward_dict["total"],
            task=self._task_info(),
            fold_data={},
            final_positions=final_positions,
            target_positions=self._target_positions.tolist(),
            shape_similarity=reward_dict.get("progress", 0.0),
            max_strain=0.0,
            is_stable=True,
            error=None,
            step_count=self._state.step_count,
            max_steps=max_folds,
            current_creases=self._paper_state.crease_edges(),
            anchor_points=[[x, y] for x, y in self._paper_state.anchor_points()],
            reward_breakdown={k: float(v) for k, v in reward_dict.items() if isinstance(v, (int, float))},
        )

    def _step_fold(self, fold_data: dict) -> OrigamiObservation:
        """V1: evaluate a complete FOLD crease pattern.

        1. Validate FOLD data
        2. Run physics simulation (creasePercent=1.0)
        3. Compare final shape to target
        4. Return observation with reward = similarity × 20
        """
        self._state.step_count += 1

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

    def _paper_state_to_fold(self) -> dict:
        """Convert current PaperState crease graph to a minimal FOLD dict for simulation."""
        if self._paper_state is None:
            return {}
        graph = self._paper_state.graph
        # Build vertex list
        vid_to_idx = {}
        vertices = []
        for vid, (x, y) in graph.vertices.items():
            vid_to_idx[vid] = len(vertices)
            vertices.append([x, y])
        # Build edge lists
        edges_vertices = []
        edges_assignment = []
        for eid, (v1, v2, assign) in graph.edges.items():
            edges_vertices.append([vid_to_idx[v1], vid_to_idx[v2]])
            edges_assignment.append(assign)
        return {
            "vertices_coords": vertices,
            "edges_vertices": edges_vertices,
            "edges_assignment": edges_assignment,
        }

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
