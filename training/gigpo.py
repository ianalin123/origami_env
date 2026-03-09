from collections import defaultdict

import numpy as np

from .trajectory import Trajectory


def compute_gigpo_advantages(
    trajectories: list[Trajectory],
    alpha: float = 0.7,
) -> list[list[float]]:
    """Two-level advantage estimation (GiGPO).

    Level 1 — Episode-level (standard GRPO):
      Group trajectories by task. Within each group:
      A_episode_i = (R_i - mean(R_group)) / std(R_group)

    Level 2 — Step-level (anchor state grouping):
      Group steps by (task, step_index, paper_state_hash).
      Steps that start from the same paper state are comparable.
      A_step_t = (r_t - mean(r_group_t)) / std(r_group_t)

    Combined:
      A_t = alpha * A_episode + (1 - alpha) * A_step_t

    Args:
        trajectories: list of completed episode trajectories
        alpha: weight for episode-level vs step-level advantage.
               1.0 = pure episode-level, 0.0 = pure step-level.
               Annealed from 1.0 -> 0.3 over training.
    """
    if not trajectories:
        return []

    # Level 1: Episode-level advantages — group by task
    task_groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for i, traj in enumerate(trajectories):
        task_groups[traj.task].append((i, traj.total_reward))

    episode_adv: dict[int, float] = {}
    for task, group in task_groups.items():
        rewards = np.array([r for _, r in group])
        mean_r = float(np.mean(rewards))
        std_r = max(float(np.std(rewards)), 1e-8)
        for idx, r in group:
            episode_adv[idx] = (r - mean_r) / std_r

    # Level 2: Step-level advantages — group by (task, step_index, state_hash)
    step_groups: dict[tuple, list[tuple[int, int, float]]] = defaultdict(list)
    for i, traj in enumerate(trajectories):
        for t, step in enumerate(traj.steps):
            key = (traj.task, t, step.state_hash)
            step_groups[key].append((i, t, step.reward))

    step_adv: dict[tuple[int, int], float] = {}
    for key, group in step_groups.items():
        if len(group) < 2:
            for idx, t, _ in group:
                step_adv[(idx, t)] = 0.0
            continue
        rewards = np.array([r for _, _, r in group])
        mean_r = float(np.mean(rewards))
        std_r = max(float(np.std(rewards)), 1e-8)
        for idx, t, r in group:
            step_adv[(idx, t)] = (r - mean_r) / std_r

    # Combine
    combined = []
    for i, traj in enumerate(trajectories):
        traj_adv = []
        for t in range(len(traj.steps)):
            a_ep = episode_adv[i]
            a_st = step_adv.get((i, t), 0.0)
            traj_adv.append(alpha * a_ep + (1 - alpha) * a_st)
        combined.append(traj_adv)

    return combined


class GiGPORewardManager:
    """Manages GiGPO advantage computation with alpha annealing."""

    def __init__(
        self,
        alpha_start: float = 1.0,
        alpha_end: float = 0.3,
        warmup_steps: int = 200,
        total_steps: int = 1500,
    ):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.global_step = 0

    @property
    def alpha(self) -> float:
        if self.global_step < self.warmup_steps:
            return self.alpha_start
        progress = min(
            (self.global_step - self.warmup_steps)
            / max(self.total_steps - self.warmup_steps, 1),
            1.0,
        )
        return self.alpha_start + (self.alpha_end - self.alpha_start) * progress

    def compute_advantages(
        self, trajectories: list[Trajectory]
    ) -> list[list[float]]:
        return compute_gigpo_advantages(trajectories, alpha=self.alpha)

    def step(self):
        self.global_step += 1
