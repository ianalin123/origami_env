# V2 Implementation Plan — Multi-Step Origami Episodes

## Goal
Upgrade from single-shot episodes (complete FOLD JSON in one action) to multi-step episodes
(one crease per step, per-step reward, evolving paper state in observation).

## Key Design Decision
Training stays compatible with `GRPOTrainer`. Each training sample is still a single
(prompt → completion → reward) tuple. The difference: the prompt now shows the current
paper state (initially empty) and the completion is a single crease JSON, not a full FOLD.
V2 MVP trains on step-0 only (empty paper). At inference, steps are chained sequentially.

---

## Step 1 — Add `shapely` to requirements
**File:** `requirements.txt`

Add `shapely>=2.0` to requirements. PaperState uses Shapely for intersection detection and
bounds clipping. This is the only new Python dependency.

Also add `shapely>=2.0` to the `run_commands` block in `modal_train.py` so the Modal image
includes it. Changing `run_commands` triggers a full image rebuild (~10 min), so do this step
first.

---

## Step 2 — Port `CreaseGraph` → `origami_server/engine/graph.py`
**New file:** `origami_server/engine/graph.py`
**Source:** `optigami/env/graph.py` (direct port, minimal changes)

Copy verbatim. No changes needed — the class is self-contained, pure Python + numpy.

```
CreaseGraph:
  - Pre-initializes unit-square corners (4 vertices) + boundary edges (4 B edges)
  - add_vertex(x, y): deduplicates by proximity (VERTEX_TOL = 1e-9)
  - add_edge(v1, v2, assignment): idempotent
  - split_edge(edge_id, new_vertex_id): for intersection handling
  - get_cyclic_edges(vertex_id): sorted by angle (used in verifier)
  - interior_vertices(): vertices not on boundary
  - crease_edges(): edges with assignment M or V
  - boundary_midpoints(): midpoints of B edges
```

---

## Step 3 — Port `PaperState` → `origami_server/engine/paper_state.py`
**New file:** `origami_server/engine/paper_state.py`
**Source:** `optigami/env/paper_state.py` (direct port, minimal changes)

Copy verbatim. PaperState:
- Wraps CreaseGraph
- `add_crease(p1, p2, assignment)` — validates, clips to unit square, finds intersections,
  splits existing edges, adds new waypoint edges
- `anchor_points()` — corners + all current vertices
- `crease_edges()` — returns list of dicts for serialization

One change from optigami: make paper dimensions configurable (default 1×1). The existing tasks
all use 1×1 paper so this is not urgent for V2 MVP.

---

## Step 4 — Port step reward → `origami_server/engine/step_reward.py`
**New file:** `origami_server/engine/step_reward.py`
**Sources:** `optigami/env/rewards.py` + `optigami/env/verifier.py`

Port `compute_reward()` and its dependencies:
- `target_crease_edges(target)` — extract M/V creases from FOLD target dict
- `check_all_vertices(graph)` — Kawasaki, Maekawa, BLB at all interior vertices
- `check_degree_sanity(graph)` — even crease count at interior vertices
- `geometric_crease_coverage(paper_state, target_edges)` — progress, economy, assignment_accuracy

The verifier functions (`_kawasaki_ok`, `_maekawa_ok`, `_blb_ok`) already exist in
`training/reward.py`, but they operate on raw FOLD JSON. The step_reward versions operate on
`CreaseGraph` directly. Keep both — they serve different purposes.

`compute_reward` signature:
```python
def compute_reward(
    prev_state: PaperState,
    action_result: dict,   # from PaperState.add_crease()
    new_state: PaperState,
    target: dict,          # FOLD task target dict
    step: int,
    max_steps: int,
) -> dict:
    # Returns dict with keys:
    # format, anchored, novelty, kawasaki, maekawa, blb, degree_sanity,
    # progress, economy, assignment_accuracy, delta, regression,
    # completion, efficiency, total
```

Weights (from optigami, validated):
```
total = (
    0.05 * anchored
    + 0.05 * novelty
    + 0.06 * kawasaki + 0.06 * maekawa + 0.04 * blb + 0.04 * degree_sanity
    + 0.25 * progress
    + 0.05 * economy + 0.05 * assignment_accuracy
    + 0.20 * delta
    + 0.10 * regression
    + completion      # 10.0 if progress > 0.9 and all geometry valid
    + efficiency      # -0.01 * (1 + step/max_steps)
)
```

The 10.0× completion bonus is the primary learning signal for hard tasks.

---

## Step 5 — Update `origami_server/models.py`
**File:** `origami_server/models.py`

Three changes:

### OrigamiAction
Make `fold_data` optional (backward compat) and add `crease` for V2:
```python
class OrigamiAction(Action):
    fold_data: dict[str, Any] | None = Field(
        default=None, description="V1: complete FOLD-format crease pattern"
    )
    crease: dict[str, Any] | None = Field(
        default=None,
        description='V2: single crease {"from": [x,y], "to": [x,y], "assignment": "M"|"V"}'
    )
```

Server validates that exactly one of `fold_data` or `crease` is set.

### OrigamiObservation
Add V2 fields (keep all V1 fields for backward compat):
```python
class OrigamiObservation(Observation):
    # V1 fields (unchanged)
    task: dict[str, Any] = Field(default_factory=dict)
    fold_data: dict[str, Any] = Field(default_factory=dict)
    final_positions: list[list[float]] = Field(default_factory=list)
    target_positions: list[list[float]] = Field(default_factory=list)
    shape_similarity: float = 0.0
    max_strain: float = 0.0
    is_stable: bool = True
    error: Optional[str] = None
    # V2 fields (new)
    step_count: int = 0
    max_steps: int = 1
    current_creases: list[dict] = Field(default_factory=list)  # placed so far
    anchor_points: list[list[float]] = Field(default_factory=list)
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
```

### OrigamiState
Add mode and step tracking:
```python
class OrigamiState(State):
    task_name: str = ""
    mode: str = "single"   # "single" | "step"
    step_count: int = 0
    shape_similarity: float = 0.0
    is_stable: bool = True
```

---

## Step 6 — Update `origami_server/environment.py`
**File:** `origami_server/environment.py`

Major upgrade. Add `mode` parameter and multi-step logic.

### Constructor
```python
def __init__(self, mode: str = "step", **kwargs):
    # mode: "step" (V2 default) | "single" (V1 backward compat)
    self._mode = mode
    self._paper_state: PaperState | None = None
    self._step_reward_prev: PaperState | None = None  # for delta computation
    # ... existing fields
```

### reset()
In step mode, initialize `_paper_state = PaperState()` and return initial observation
with empty `current_creases`, all anchor points, `done=False`.

In single mode, behavior unchanged from V1.

Grab `max_folds` from task definition (new task field, see Step 7).

### step() — V2 path (when `action.crease` is set)
```python
# 1. Parse crease
crease = action.crease  # {"from": [x,y], "to": [x,y], "assignment": "M"|"V"}

# 2. Validate
if crease["assignment"] not in ("M", "V"):
    return error observation, reward=-0.1

# 3. Apply crease to paper state
import copy
prev_state = copy.deepcopy(self._paper_state)
result = self._paper_state.add_crease(
    crease["from"], crease["to"], crease["assignment"]
)

# 4. Compute per-step reward
self._state.step_count += 1
reward_dict = compute_reward(
    prev_state=prev_state,
    action_result=result,
    new_state=self._paper_state,
    target=self._task,
    step=self._state.step_count,
    max_steps=self._task["max_folds"],
)

# 5. Check done
done = (
    self._state.step_count >= self._task["max_folds"]
    or reward_dict.get("completion", 0) > 0
)

# 6. Return observation
return OrigamiObservation(
    done=done,
    reward=reward_dict["total"],
    task=self._task_info(),
    fold_data={},           # empty in step mode
    final_positions=[],     # only populated on done=True
    target_positions=self._target_positions.tolist(),
    shape_similarity=reward_dict.get("progress", 0.0),
    max_strain=0.0,
    is_stable=True,
    error=None,
    step_count=self._state.step_count,
    max_steps=self._task["max_folds"],
    current_creases=self._paper_state.crease_edges(),
    anchor_points=[[x, y] for x, y in self._paper_state.anchor_points()],
    reward_breakdown=reward_dict,
)
```

When `done=True`, optionally run the full simulation (`simulate()`) to populate
`final_positions` and `shape_similarity` — useful for the viewer but not required for training.

### step() — V1 path (when `action.fold_data` is set)
Identical to current V1 implementation. No changes.

---

## Step 7 — Update `origami_server/tasks.py`
**File:** `origami_server/tasks.py`

Two changes:

### Add `max_folds` to all existing tasks
```python
"triangle":      max_folds=1
"half_fold":     max_folds=1
"quarter_fold":  max_folds=2
"letter_fold":   max_folds=2
```

`max_folds` is the maximum number of step() calls before done=True.

### Add two harder tasks

**`waterbomb_base`** (difficulty 3, max_folds=4):
Two diagonal valley folds (corner to corner) + two perpendicular valley folds (midpoint to
midpoint). Classic base that requires all four folds to be correct simultaneously.
```
target_fold: 9 vertices (4 corners + 4 midpoints + 1 center), 8 crease edges (all V)
Creases:
  (0,0)→(1,1)  V  (diagonal)
  (1,0)→(0,1)  V  (diagonal)
  (0.5,0)→(0.5,1)  V  (vertical)
  (0,0.5)→(1,0.5)  V  (horizontal)
```

**`map_fold`** (difficulty 4, max_folds=8):
Accordion fold into 4 strips horizontally + 4 strips vertically (8 total creases,
alternating M/V). The most demanding task for V2.
```
target_fold: Creases at y=0.25, 0.5, 0.75 (alternating V/M/V) + x=0.25, 0.5, 0.75 (V/M/V)
plus corner diagonals for proper map fold behavior
```

Add `get_task_for_step_mode(name)` helper that returns the task with `max_folds` validated.

---

## Step 8 — Update `training/reward.py`
**File:** `training/reward.py`

### Add `valid_crease()` reward function
New reward for V2 single-crease format:
```python
def valid_crease(completions: list, **kwargs) -> list[float]:
    """V2: Does the LLM output parse as valid single-crease JSON?

    +1.0  valid {"from": [x,y], "to": [x,y], "assignment": "M"|"V"}
    -0.5  parseable JSON but missing fields or wrong types
    -2.0  not parseable JSON
    """
```

### Add `extract_crease_json()` helper
```python
def extract_crease_json(response: str) -> dict | None:
    """Extract single-crease JSON from LLM response.
    Looks for {"from": ..., "to": ..., "assignment": ...} object.
    """
```

Keep all existing V1 functions (`valid_fold`, `flat_foldable_reward`, `extract_fold_json`)
unchanged for backward compat.

---

## Step 9 — Update `training/train_grpo.py`
**File:** `training/train_grpo.py`

### New prompt template
Replace `PROMPT_TEMPLATE` with a step-level format. Key difference: no FOLD fields listed,
just "output the next crease as JSON":

```python
STEP_PROMPT_TEMPLATE = """You are an origami designer. Add the next fold crease.

Target: {description}
Paper: {width} × {height} unit square

CURRENT STATE (step {step} of {max_folds}):
  Creases placed: {crease_history}

AVAILABLE ANCHOR POINTS:
  Corners:      {corners}
  Boundary pts: {boundary_pts}
  Intersections:{intersections}

Flat-foldability rules at every interior vertex:
  - Kawasaki: alternating sector angles each sum to 180°
  - Maekawa: |mountain_count - valley_count| = 2
  - BLB: smallest sector bounded by opposite M/V types

Output ONLY this JSON (no explanation):
{{"from": [x1, y1], "to": [x2, y2], "assignment": "M" or "V"}}"""
```

For V2 MVP (step-0 training), `step=0`, `crease_history="none"`, anchor points = corners + midpoints.

### New `per_step_reward()` function
Replace `shape_match_reward`:
```python
def per_step_reward(completions, task_name, **kwargs):
    scores = []
    for completion, tname in zip(completions, task_name):
        response = completion[0]["content"]
        crease = extract_crease_json(response)
        if crease is None:
            scores.append(-2.0)
            continue
        try:
            port, openenv_process = launch_openenv(port, openenv_process)
            openenv_process.reset(task_name=tname)
            result = openenv_process.step(OrigamiAction(crease=crease))
            scores.append(result.reward if result.reward is not None else 0.0)
        except TimeoutError:
            scores.append(-1.0)
        except Exception:
            scores.append(-2.0)
    return scores
```

### Updated reward function list
```python
trainer = GRPOTrainer(
    reward_funcs=[valid_crease, per_step_reward],  # removed flat_foldable_reward (server handles it now)
    ...
)
```

### Updated task list
Add new tasks to `ALL_TASKS`:
```python
ALL_TASKS = ["triangle", "half_fold", "quarter_fold", "letter_fold", "waterbomb_base", "map_fold"]
```

### Updated GRPO config
Increase `max_completion_length` since single crease JSON is shorter (~50 tokens):
```python
max_prompt_length=512,
max_completion_length=128,  # single crease JSON is ~50 tokens
max_steps=1200,             # more steps since harder tasks
```

---

## Step 10 — Update `client.py`
**File:** `client.py`

Minor update: `_step_payload` already calls `action.model_dump()`. With optional fields,
this will naturally include `crease` or `fold_data` depending on which is set. No change needed
unless OpenEnv has strict serialization requirements.

If OpenEnv rejects None fields, filter them:
```python
def _step_payload(self, action: OrigamiAction) -> Dict[str, Any]:
    return {k: v for k, v in action.model_dump().items() if v is not None}
```

---

## Implementation Order

Execute in this order to minimize broken states:

```
1. requirements.txt + modal_train.py (deps first, triggers image rebuild)
2. origami_server/engine/graph.py (new file, no dependencies)
3. origami_server/engine/paper_state.py (depends on graph.py)
4. origami_server/engine/step_reward.py (depends on paper_state.py)
5. origami_server/models.py (API types — do before environment)
6. origami_server/environment.py (depends on models + paper_state + step_reward)
7. origami_server/tasks.py (add max_folds + new tasks)
8. training/reward.py (new valid_crease, extract_crease_json)
9. training/train_grpo.py (new prompts + per_step_reward)
10. client.py (minor defensive fix)
```

After step 7: run `curl http://localhost:8000/tasks` and verify new tasks appear.
After step 9: run a single training step locally with `--model unsloth/Qwen2.5-3B-Instruct --max_steps 5`
to verify reward functions fire.

---

## Files Changed / Created

| File | Status | Notes |
|------|--------|-------|
| `requirements.txt` | modified | add shapely>=2.0 |
| `modal_train.py` | modified | add shapely to run_commands |
| `origami_server/engine/graph.py` | **new** | port from optigami |
| `origami_server/engine/paper_state.py` | **new** | port from optigami |
| `origami_server/engine/step_reward.py` | **new** | port from optigami |
| `origami_server/models.py` | modified | OrigamiAction crease field, observation V2 fields |
| `origami_server/environment.py` | modified | multi-step mode |
| `origami_server/tasks.py` | modified | max_folds + waterbomb_base + map_fold |
| `training/reward.py` | modified | valid_crease + extract_crease_json |
| `training/train_grpo.py` | modified | step prompt + per_step_reward |
| `client.py` | modified | optional fields in _step_payload |

**V1 backward compat preserved:** All existing API routes, observation fields, and reward
functions remain unchanged. `mode='single'` continues to work for existing training runs.

---

## Risk Notes

- **shapely requirement**: PaperState intersection detection uses shapely. If Railway/Modal
  build fails, can fall back to numpy-only intersection (more code, but avoids the dep).
  Suggest testing locally first with `pip install shapely`.

- **OrigamiAction change**: Making `fold_data` optional is a breaking change for clients
  sending the field as required. Any existing V1 clients that always set `fold_data` will
  continue to work since pydantic accepts it as optional-with-value.

- **step-0 only training**: V2 MVP trains exclusively from empty paper state (step 0).
  The model learns "first crease for task X" but doesn't train on step 1+. This means
  chained inference (running multiple steps at eval time) may degrade at step 2+ because
  the policy was never trained on non-empty paper states. Acceptable for V2 MVP — a future
  V3 adds episode rollout collection to the training loop.

- **completion bonus scale**: The 10.0× completion bonus means episodes where the model
  hits >90% coverage + valid geometry will dominate the reward signal. For easy tasks
  (triangle, half_fold) this will happen quickly. For map_fold it may never happen in early
  training. Consider starting with only triangle/waterbomb_base for first training run.
