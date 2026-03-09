import random

from origami_server.models import OrigamiObservation

TEMPLATES = [
    """You are an origami designer. Add the next fold crease.

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
{{"from": [x1, y1], "to": [x2, y2], "assignment": "M" or "V"}}""",

    """Fold a piece of paper. Place one crease line for this step.

Goal: {description}
Sheet dimensions: {width} x {height}

Progress (step {step}/{max_folds}):
  Existing creases: {crease_history}

Points you can use:
  {anchor_points}

Rules for flat-foldability:
  - Kawasaki theorem: alternating angles around any interior vertex sum to 180°
  - Maekawa theorem: |M - V| = 2 at each interior vertex
  - Big-Little-Big: smallest angle sector bounded by opposite fold types

Respond with ONLY this JSON:
{{"from": [x1, y1], "to": [x2, y2], "assignment": "M" or "V"}}""",

    """You are folding origami. Choose the next crease to add.

Task: {description}
Paper size: {width} x {height}

Step {step} of {max_folds}:
  Current creases: {crease_history}

Available points:
  {anchor_points}

Constraints (flat-foldability):
  - Kawasaki: alternate sector angles sum to π each side
  - Maekawa: mountain and valley count differ by exactly 2
  - BLB: smallest sector angle flanked by opposite assignments

Return ONLY JSON:
{{"from": [x1, y1], "to": [x2, y2], "assignment": "M" or "V"}}""",

    """Design the next origami fold crease.

Objective: {description}
Paper: {width} × {height} square

State (step {step}/{max_folds}):
  Creases so far: {crease_history}

Anchor points available:
  {anchor_points}

Flat-foldability constraints at interior vertices:
  - Kawasaki: alternating sector angles each sum to 180°
  - Maekawa: |mountain - valley| = 2
  - BLB: smallest sector bounded by opposite types

Output JSON only:
{{"from": [x1, y1], "to": [x2, y2], "assignment": "M" or "V"}}""",
]


def build_prompt_from_obs(
    task_name: str,
    task_info: dict,
    obs: OrigamiObservation,
    randomize: bool = True,
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
    if randomize:
        random.shuffle(anchors)
    anchor_str = "  ".join(anchors)

    template = random.choice(TEMPLATES) if randomize else TEMPLATES[0]

    return template.format(
        description=task_info["description"],
        width=w,
        height=h,
        step=obs.step_count,
        max_folds=obs.max_steps,
        crease_history=crease_history,
        anchor_points=anchor_str,
    )
