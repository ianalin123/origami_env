from origami_server.environment import OrigamiEnvironment
from origami_server.models import OrigamiAction, OrigamiObservation


class OrigamiEnvPool:
    """Pool of in-process origami environments for parallel rollouts.

    Environments run in-process (no HTTP) since env.step() is pure CPU math (~1ms).
    Sequential stepping is fine since 64 envs x 1ms = 64ms, negligible vs ~2s generation.
    """

    def __init__(self, pool_size: int = 64):
        self.envs = [OrigamiEnvironment(mode="step") for _ in range(pool_size)]
        self.pool_size = pool_size

    def reset(self, idx: int, task_name: str) -> OrigamiObservation:
        return self.envs[idx].reset(task_name=task_name)

    def step(self, idx: int, crease: dict) -> OrigamiObservation:
        return self.envs[idx].step(OrigamiAction(crease=crease))

    def step_batch(
        self, indices: list[int], creases: list[dict]
    ) -> list[OrigamiObservation]:
        return [self.step(i, c) for i, c in zip(indices, creases)]

    def get_paper_state(self, idx: int):
        return self.envs[idx]._paper_state
