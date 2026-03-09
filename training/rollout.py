import random
from typing import Callable

from origami_server.engine.paper_state import hash_paper_state
from origami_server.tasks import get_task
from training.env_pool import OrigamiEnvPool
from training.prompt_builder import build_prompt_from_obs
from training.reward import extract_crease_json
from training.trajectory import Trajectory


def run_rollout_batch(
    generate_fn: Callable[[list[str]], list[str]],
    task_pool: list[str],
    batch_size: int = 16,
) -> list[Trajectory]:
    """Run a batch of multi-step episodes, return trajectories.

    Args:
        generate_fn: takes list of prompt strings, returns list of completion strings.
                     During training this wraps vLLM/sglang; for testing use a mock.
        task_pool: list of task names to sample from
        batch_size: number of parallel episodes
    """
    pool = OrigamiEnvPool(pool_size=batch_size)
    tasks = [random.choice(task_pool) for _ in range(batch_size)]
    task_infos = {t: get_task(t) for t in set(tasks)}

    # Reset all envs
    observations = [pool.reset(i, tasks[i]) for i in range(batch_size)]
    trajectories = [Trajectory(task=tasks[i]) for i in range(batch_size)]
    active = [True] * batch_size

    # Find max episode length across all tasks in this batch
    max_folds = max(task_infos[t].get("max_folds", 1) for t in set(tasks))

    for step in range(max_folds):
        # Filter to active episodes that haven't hit their per-task max_folds
        active_idx = [
            i for i in range(batch_size)
            if active[i] and trajectories[i].length < task_infos[tasks[i]].get("max_folds", 1)
        ]
        if not active_idx:
            break

        # Build prompts from current observations
        prompts = [
            build_prompt_from_obs(tasks[i], task_infos[tasks[i]], observations[i])
            for i in active_idx
        ]

        # Generate completions (batched)
        completions = generate_fn(prompts)

        # Step envs
        for j, i in enumerate(active_idx):
            state_hash = hash_paper_state(pool.get_paper_state(i))
            crease = extract_crease_json(completions[j])

            if crease is None:
                trajectories[i].add_step(
                    prompt=prompts[j],
                    completion=completions[j],
                    reward=-2.0,
                    done=False,
                    state_hash=state_hash,
                )
                # Don't mark as done -- let the model try again next step
                continue

            obs = pool.step(i, crease)
            observations[i] = obs

            trajectories[i].add_step(
                prompt=prompts[j],
                completion=completions[j],
                reward=obs.reward if obs.reward is not None else -1.0,
                done=obs.done,
                state_hash=state_hash,
                reward_breakdown=(
                    {k: v for k, v in obs.reward_breakdown.items()}
                    if obs.reward_breakdown
                    else {}
                ),
            )

            if obs.done:
                active[i] = False

    return trajectories
