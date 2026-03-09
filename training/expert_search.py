"""Expert search: enumerate all valid creases to find the best action.

For each paper state, generates all (anchor_pair × assignment) combinations,
scores each with the reward function, and returns the best one as an SFT target.
This bypasses the GRPO exploration problem entirely.
"""

import copy
import json
from itertools import combinations

from origami_server.engine.paper_state import PaperState
from origami_server.engine.step_reward import compute_reward


def find_best_crease(
    paper_state: PaperState,
    target: dict,
    step: int,
    max_steps: int,
) -> tuple[dict | None, float]:
    """Enumerate all valid creases and return the best one.

    Considers both anchor points AND target vertices as candidate endpoints,
    since target crease endpoints (e.g., midpoints) may not be anchor points yet.

    Returns:
        (best_crease, best_reward) where best_crease is
        {"from": [x1,y1], "to": [x2,y2], "assignment": "M"|"V"}
        or None if no valid crease found.
    """
    anchors = set(paper_state.anchor_points())

    fold_target = target.get("target_fold", target)
    for v in fold_target.get("vertices_coords", []):
        anchors.add((float(v[0]), float(v[1])))

    anchors = list(anchors)

    best_crease = None
    best_reward = float('-inf')

    for (p1, p2) in combinations(anchors, 2):
        for assignment in ("M", "V"):
            state_copy = copy.deepcopy(paper_state)
            result = state_copy.add_crease(list(p1), list(p2), assignment)

            if not result.get('valid', False):
                continue

            reward_dict = compute_reward(
                prev_state=paper_state,
                action_result=result,
                new_state=state_copy,
                target=target,
                step=step,
                max_steps=max_steps,
            )

            total = reward_dict['total']
            if total > best_reward:
                best_reward = total
                best_crease = {
                    "from": list(p1),
                    "to": list(p2),
                    "assignment": assignment,
                }

    return best_crease, best_reward


def _get_candidate_points(paper_state: PaperState, target: dict) -> list[tuple[float, float]]:
    """Get all candidate points: anchors + target vertices."""
    points = set(paper_state.anchor_points())
    fold_target = target.get("target_fold", target)
    for v in fold_target.get("vertices_coords", []):
        points.add((float(v[0]), float(v[1])))
    return list(points)


def expert_trajectory(
    task_name: str,
    task_info: dict,
    beam_width: int = 5,
) -> list[dict]:
    """Generate an expert trajectory using beam search across all steps.

    For multi-step tasks, evaluates all sequences of creases (up to beam_width
    candidates per step) and returns the sequence with the highest total reward.

    Returns list of {"obs": ..., "completion": str, "reward": float} per step.
    """
    from origami_server.environment import OrigamiEnvironment
    from origami_server.models import OrigamiAction

    max_folds = task_info.get("max_folds", 1)

    # Each beam entry: (total_reward, [(obs, crease_dict, step_reward), ...], env)
    initial_env = OrigamiEnvironment(mode="step")
    initial_obs = initial_env.reset(task_name=task_name)
    beams = [(0.0, [], initial_env, initial_obs)]

    for step_idx in range(max_folds):
        next_beams = []

        for total_reward, history, env, obs in beams:
            paper_state = env._paper_state
            candidates = _get_candidate_points(paper_state, task_info)

            scored = []
            for (p1, p2) in combinations(candidates, 2):
                for assignment in ("M", "V"):
                    state_copy = copy.deepcopy(paper_state)
                    result = state_copy.add_crease(list(p1), list(p2), assignment)
                    if not result.get('valid', False):
                        continue

                    reward_dict = compute_reward(
                        prev_state=paper_state,
                        action_result=result,
                        new_state=state_copy,
                        target=task_info,
                        step=step_idx + 1,
                        max_steps=max_folds,
                    )
                    crease = {"from": list(p1), "to": list(p2), "assignment": assignment}
                    scored.append((reward_dict['total'], crease))

            scored.sort(key=lambda x: -x[0])
            for step_reward, crease in scored[:beam_width]:
                env_copy = copy.deepcopy(env)
                obs_copy = copy.deepcopy(obs)
                new_obs = env_copy.step(OrigamiAction(crease=crease))
                new_history = history + [(obs_copy, crease, step_reward)]
                next_beams.append((
                    total_reward + step_reward,
                    new_history,
                    env_copy,
                    new_obs,
                ))

        if not next_beams:
            break

        next_beams.sort(key=lambda x: -x[0])
        beams = next_beams[:beam_width]

    if not beams or not beams[0][1]:
        return []

    best_total, best_history, _, _ = beams[0]
    return [
        {
            "obs": obs,
            "completion": json.dumps(crease),
            "reward": reward,
        }
        for obs, crease, reward in best_history
    ]
