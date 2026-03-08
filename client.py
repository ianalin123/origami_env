"""Origami environment client — connects to a running origami_env server."""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from origami_server.models import OrigamiAction, OrigamiObservation, OrigamiState


class OrigamiEnv(EnvClient[OrigamiAction, OrigamiObservation, OrigamiState]):
    """
    Client for the origami RL environment.

    Example:
        >>> with OrigamiEnv(base_url="https://origami-env-production.up.railway.app") as env:
        ...     result = env.reset(task_name="triangle")
        ...     result = env.step(OrigamiAction(fold_data={...}))
        ...     print(result.observation.shape_similarity)
    """

    def _step_payload(self, action: OrigamiAction) -> Dict[str, Any]:
        return {k: v for k, v in action.model_dump().items() if v is not None}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[OrigamiObservation]:
        obs_data = payload.get("observation", payload)
        return StepResult(
            observation=OrigamiObservation(**obs_data),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> OrigamiState:
        return OrigamiState(**payload)
