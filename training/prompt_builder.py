from origami_server.models import OrigamiObservation

STEP_PROMPT = """You are an origami designer. Add the next fold crease.

Target: {description}
Paper: {width} x {height} unit square

CURRENT STATE (step {step} of {max_folds}):
  Creases placed: {crease_history}

AVAILABLE ANCHOR POINTS:
  {anchor_points}

Flat-foldability rules at every interior vertex:
  - Kawasaki: alternating sector angles each sum to 180 degrees
  - Maekawa: |mountain_count - valley_count| = 2
  - BLB: smallest sector bounded by opposite M/V types

Output ONLY this JSON (no explanation):
{{"from": [x1, y1], "to": [x2, y2], "assignment": "M" or "V"}}"""


def build_prompt_from_obs(
    task_name: str,
    task_info: dict,
    obs: OrigamiObservation,
) -> str:
    w = task_info["paper"]["width"]
    h = task_info["paper"]["height"]

    if obs.current_creases:
        history_parts = []
        for c in obs.current_creases:
            v1, v2, a = c["v1"], c["v2"], c["assignment"]
            history_parts.append(f"({v1[0]},{v1[1]})->({v2[0]},{v2[1]}) {a}")
        crease_history = "; ".join(history_parts)
    else:
        crease_history = "none"

    anchors = [f"({p[0]},{p[1]})" for p in obs.anchor_points]
    anchor_str = "  ".join(anchors)

    return STEP_PROMPT.format(
        description=task_info["description"],
        width=w,
        height=h,
        step=obs.step_count,
        max_folds=obs.max_steps,
        crease_history=crease_history,
        anchor_points=anchor_str,
    )
