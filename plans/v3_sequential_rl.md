# Implementation Plan: Multi-Step RL Training (V3)

## Context

Replace TRL's GRPOTrainer (single-turn bandit) with veRL + verl-agent (multi-turn
sequential rollouts) using GiGPO for per-step credit assignment. Architecture is in
`PLAN_V3_SEQUENTIAL_TRAINING.md`. This plan is the step-by-step build order.

**Framework**: veRL ≥0.7 with verl-agent extension (GiGPO)
**Hardware**: 4× B200 on Modal (single-node Ray)
**Model**: Qwen3-32B with LoRA rank 32

---

## Phase 0 — Framework Validation (no origami code)

### Step 0.1: Verify veRL runs on Modal with 4× B200

**Goal**: Confirm veRL + Ray + vLLM + sglang work inside a Modal container before
writing any origami integration.

**File**: `modal_verl_smoke.py` (new, temporary)

```
- Create Modal image with verl, vllm, sglang, ray, torch cu128
- Run veRL's built-in GSM8K example with Qwen2.5-3B (small model, fast)
- Config: 10 training steps, 4 generations, multi_turn: false
- Success = gradient step completes, checkpoint saved to volume
```

**Why first**: If veRL doesn't work on Modal (Ray init issues, GPU detection, sglang
compatibility), nothing else matters. This takes ~30 min to debug.

**Validation**:
- `ray.init()` detects 4 GPUs
- vLLM loads model with tensor_parallel=4
- 10 training steps complete without OOM
- Checkpoint directory exists

**Risk**: Modal's container networking may conflict with Ray. Mitigation: use
`ray.init(num_gpus=4)` with local-only mode, no multi-node.

---

### Step 0.2: Verify veRL multi-turn rollout with a toy environment

**Goal**: Run veRL's multi-turn rollout with a trivial tool (not origami) to confirm
the AgentLoop → tool → reward pipeline works.

**File**: `tests/test_verl_multiturn_smoke.py` (new, temporary)

```
- Implement ToyTool(BaseTool) that returns a random number
- Implement ToyAgentLoop(AgentLoopBase) that calls generate → tool → return
- Config: multi_turn: True, name: sglang, 5 training steps
- Success = multi-turn trajectories generated, loss computed
```

**Why**: Isolates veRL multi-turn mechanics from origami complexity.

**Validation**:
- AgentLoopOutput has correct prompt_ids, response_ids, response_mask
- response_mask correctly masks tool-response tokens (0) vs LLM tokens (1)
- Training step completes with non-zero loss

---

## Phase 1 — Core Building Blocks (pure Python, no veRL dependency)

### Step 1.1: Paper state hashing

**File**: `origami_server/engine/paper_state.py` (modify)

Add `hash_paper_state(state: PaperState) -> int` function.

**Implementation**:
```python
def hash_paper_state(state: PaperState) -> int:
    edges = state.crease_edges()
    canonical = []
    for e in edges:
        p1 = tuple(round(c, 6) for c in e["v1"])
        p2 = tuple(round(c, 6) for c in e["v2"])
        canonical.append((min(p1, p2), max(p1, p2), e["assignment"]))
    canonical.sort()
    return hash(tuple(canonical))
```

**Key details**:
- `crease_edges()` returns `[{'v1': (x,y), 'v2': (x,y), 'assignment': str}]`
  — already strips internal vertex/edge IDs
- `round(c, 6)` prevents floating-point divergence from Shapely intersection
  order (e.g., 0.49999999 vs 0.50000001)
- Canonicalize edge direction: `min(p1, p2)` ensures (0,0)→(1,1) and
  (1,1)→(0,0) hash identically
- Sort edges so insertion order doesn't matter

**Validation** — add to `tests/test_origami.py`:
```python
class TestPaperStateHash:
    def test_empty_papers_same_hash(self):
        s1, s2 = PaperState(), PaperState()
        assert hash_paper_state(s1) == hash_paper_state(s2)

    def test_same_creases_different_order(self):
        s1 = PaperState()
        s1.add_crease([0.5, 0], [0.5, 1], "V")
        s1.add_crease([0, 0.5], [1, 0.5], "V")
        s2 = PaperState()
        s2.add_crease([0, 0.5], [1, 0.5], "V")
        s2.add_crease([0.5, 0], [0.5, 1], "V")
        assert hash_paper_state(s1) == hash_paper_state(s2)

    def test_different_creases_different_hash(self):
        s1 = PaperState()
        s1.add_crease([0, 0], [1, 1], "V")
        s2 = PaperState()
        s2.add_crease([1, 0], [0, 1], "V")
        assert hash_paper_state(s1) != hash_paper_state(s2)

    def test_different_assignment_different_hash(self):
        s1 = PaperState()
        s1.add_crease([0.5, 0], [0.5, 1], "V")
        s2 = PaperState()
        s2.add_crease([0.5, 0], [0.5, 1], "M")
        assert hash_paper_state(s1) != hash_paper_state(s2)
```

---

### Step 1.2: Trajectory data structures

**File**: `training/trajectory.py` (new)

Pure dataclasses, no framework dependency.

```python
from dataclasses import dataclass, field

@dataclass
class Step:
    prompt: str
    completion: str
    reward: float
    done: bool
    state_hash: int           # hash_paper_state BEFORE this step
    reward_breakdown: dict = field(default_factory=dict)
    log_prob: float = 0.0     # filled by vLLM/sglang

@dataclass
class Trajectory:
    task: str
    steps: list[Step] = field(default_factory=list)

    def add_step(self, **kwargs) -> None:
        self.steps.append(Step(**kwargs))

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    @property
    def length(self) -> int:
        return len(self.steps)
```

**Validation**: unit test constructing trajectories, verifying total_reward.

---

### Step 1.3: Environment pool

**File**: `training/env_pool.py` (new)

```python
from origami_server.environment import OrigamiEnvironment
from origami_server.models import OrigamiAction, OrigamiObservation

class OrigamiEnvPool:
    def __init__(self, pool_size: int = 64):
        self.envs = [OrigamiEnvironment(mode="step") for _ in range(pool_size)]

    def reset(self, idx: int, task_name: str) -> OrigamiObservation:
        return self.envs[idx].reset(task_name=task_name)

    def step(self, idx: int, crease: dict) -> OrigamiObservation:
        return self.envs[idx].step(OrigamiAction(crease=crease))

    def step_batch(
        self, indices: list[int], creases: list[dict]
    ) -> list[OrigamiObservation]:
        return [self.step(i, c) for i, c in zip(indices, creases)]

    @property
    def paper_state(self, idx: int):
        return self.envs[idx]._paper_state
```

**Why sequential, not threaded?** Env step is CPU-bound (~1ms, Shapely + numpy).
ThreadPoolExecutor doesn't help with CPU-bound work due to GIL. For 64 envs,
sequential = 64ms total — negligible vs ~2s vLLM generation. Keep it simple.

**Validation**:
```python
class TestEnvPool:
    def test_reset_and_step(self):
        pool = OrigamiEnvPool(pool_size=4)
        obs = pool.reset(0, "triangle")
        assert not obs.done
        assert len(obs.anchor_points) == 4  # corners

        obs = pool.step(0, {"from": [0, 0], "to": [1, 1], "assignment": "V"})
        assert obs.reward is not None

    def test_batch_step(self):
        pool = OrigamiEnvPool(pool_size=4)
        for i in range(4):
            pool.reset(i, "triangle")
        creases = [{"from": [0,0], "to": [1,1], "assignment": "V"}] * 4
        results = pool.step_batch([0,1,2,3], creases)
        assert len(results) == 4
        assert all(r.reward is not None for r in results)
```

---

### Step 1.4: GiGPO advantage computation

**File**: `training/gigpo.py` (new)

```python
from collections import defaultdict
import numpy as np
from .trajectory import Trajectory

def compute_gigpo_advantages(
    trajectories: list[Trajectory],
    alpha: float = 0.7,
) -> list[list[float]]:
    """Two-level advantage: episode-level GRPO + step-level anchor grouping."""

    # Level 1: Episode-level — group by task
    task_groups = defaultdict(list)
    for i, traj in enumerate(trajectories):
        task_groups[traj.task].append((i, traj.total_reward))

    episode_adv = {}
    for task, group in task_groups.items():
        rewards = [r for _, r in group]
        mean_r = np.mean(rewards)
        std_r = max(np.std(rewards), 1e-8)
        for idx, r in group:
            episode_adv[idx] = (r - mean_r) / std_r

    # Level 2: Step-level — group by (task, step_index, state_hash)
    step_groups = defaultdict(list)
    for i, traj in enumerate(trajectories):
        for t, step in enumerate(traj.steps):
            key = (traj.task, t, step.state_hash)
            step_groups[key].append((i, t, step.reward))

    step_adv = {}
    for key, group in step_groups.items():
        if len(group) < 2:
            # Singleton group — no relative signal, use 0
            for idx, t, _ in group:
                step_adv[(idx, t)] = 0.0
            continue
        rewards = [r for _, _, r in group]
        mean_r = np.mean(rewards)
        std_r = max(np.std(rewards), 1e-8)
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
```

**Key edge case**: singleton step groups (size 1) at late steps. When only 1 trajectory
has a given state, step-level advantage = 0 (no comparison possible). The episode-level
advantage still provides signal via `alpha * a_ep`. This degrades gracefully.

**Validation**:
```python
class TestGiGPO:
    def test_same_task_different_rewards(self):
        # 4 trajectories for "triangle", rewards [1, 2, 3, 4]
        # Episode advantages should be [-1.34, -0.45, 0.45, 1.34] (z-scores)
        trajs = [Trajectory(task="triangle") for _ in range(4)]
        for i, t in enumerate(trajs):
            t.add_step(prompt="", completion="", reward=float(i+1),
                       done=True, state_hash=0)
        advantages = compute_gigpo_advantages(trajs, alpha=1.0)
        # With alpha=1 (episode only), step 0 advantages = episode z-scores
        assert advantages[0][0] < advantages[3][0]

    def test_step_level_grouping(self):
        # 4 trajectories, same task, same state_hash at step 0
        # Different rewards at step 0 → step-level advantages differ
        trajs = [Trajectory(task="triangle") for _ in range(4)]
        for i, t in enumerate(trajs):
            t.add_step(prompt="", completion="", reward=float(i),
                       done=True, state_hash=42)
        advantages = compute_gigpo_advantages(trajs, alpha=0.0)
        # With alpha=0 (step only), advantages = step-level z-scores
        assert advantages[0][0] < advantages[3][0]

    def test_singleton_group_returns_zero(self):
        # 2 trajectories with different state hashes → each group has size 1
        trajs = [Trajectory(task="t") for _ in range(2)]
        trajs[0].add_step(prompt="", completion="", reward=10.0,
                          done=True, state_hash=1)
        trajs[1].add_step(prompt="", completion="", reward=0.0,
                          done=True, state_hash=2)
        advantages = compute_gigpo_advantages(trajs, alpha=0.0)
        # Step-level advantages = 0 for singletons
        assert advantages[0][0] == 0.0
        assert advantages[1][0] == 0.0
```

---

### Step 1.5: Curriculum scheduler

**File**: `training/curriculum.py` (new)

```python
CURRICULUM = [
    {"steps": (0, 200),    "tasks": ["triangle", "half_fold"]},
    {"steps": (200, 500),  "tasks": ["triangle", "half_fold", "quarter_fold", "letter_fold"]},
    {"steps": (500, 900),  "tasks": ["quarter_fold", "letter_fold", "waterbomb_base"]},
    {"steps": (900, 1500), "tasks": ["waterbomb_base", "map_fold"]},
]

def get_task_pool(global_step: int) -> list[str]:
    for phase in CURRICULUM:
        if phase["steps"][0] <= global_step < phase["steps"][1]:
            return phase["tasks"]
    return CURRICULUM[-1]["tasks"]
```

**Validation**: `assert get_task_pool(0) == ["triangle", "half_fold"]`

---

### Step 1.6: Reward shaping — difficulty-scaled completion bonus

**File**: `origami_server/engine/step_reward.py` (modify)

**Change**: Replace hardcoded `10.0` completion bonus with difficulty-scaled values.

Current code (line 347):
```python
r['completion'] = 10.0 if (r['progress'] > 0.9 and all_valid) else 0.0
```

New code:
```python
COMPLETION_BONUS = {1: 2.0, 2: 5.0, 3: 10.0, 4: 15.0}

# In compute_reward(), add difficulty parameter
difficulty = target.get("difficulty", 1)
bonus = COMPLETION_BONUS.get(difficulty, 10.0)
r['completion'] = bonus if (r['progress'] > 0.9 and all_valid) else 0.0
```

**Why**: Without scaling, triangle (difficulty 1) gives 10.0 completion bonus in 1 step
while map_fold (difficulty 4) rarely achieves it. GiGPO's episode-level advantage
sees triangle episodes as always-positive → model only learns triangle.

**Validation**: Existing step_reward tests still pass. Add test confirming
`compute_reward(..., target=triangle_task)` gives bonus=2.0 not 10.0.

---

## Phase 2 — Prompt Building & Rollout Logic

### Step 2.1: Observation-aware prompt builder

**File**: `training/prompt_builder.py` (new)

The existing `build_step_prompt()` in `train_grpo.py` takes raw task dict fields.
V3 needs a version that takes an `OrigamiObservation` directly, since each step
returns updated anchor points and crease history.

```python
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

    # Format crease history
    if obs.current_creases:
        history_lines = []
        for c in obs.current_creases:
            v1, v2, a = c["v1"], c["v2"], c["assignment"]
            history_lines.append(
                f"({v1[0]},{v1[1]})→({v2[0]},{v2[1]}) {a}"
            )
        crease_history = "; ".join(history_lines)
    else:
        crease_history = "none"

    # Format anchor points
    anchors = [f"({p[0]},{p[1]})" for p in obs.anchor_points]
    anchor_str = "  ".join(anchors)

    return STEP_PROMPT.format(
        description=task_info["description"],
        width=w, height=h,
        step=obs.step_count,
        max_folds=obs.max_steps,
        crease_history=crease_history,
        anchor_points=anchor_str,
    )
```

**Why separate file**: `train_grpo.py` is V1/V2. V3 uses a different training loop
entirely. Keep prompt logic importable without pulling in TRL dependencies.

**Validation**: Build prompt from a real observation, verify it contains anchor points
that include intersection points (not just corners).

---

### Step 2.2: Rollout function (standalone, no veRL yet)

**File**: `training/rollout.py` (new)

This is the core multi-turn rollout logic, written as a standalone function first
(before wiring into veRL's AgentLoop). This lets us test it independently.

```python
import random
from .env_pool import OrigamiEnvPool
from .trajectory import Trajectory
from .prompt_builder import build_prompt_from_obs
from origami_server.engine.paper_state import hash_paper_state
from origami_server.tasks import get_task
from training.reward import extract_crease_json

def run_rollout_batch(
    generate_fn,          # callable: list[str] -> list[str]
    task_pool: list[str],
    batch_size: int = 16,
    max_steps: int = 6,   # global cap across all tasks
) -> list[Trajectory]:
    """Run a batch of multi-step episodes, return trajectories."""

    pool = OrigamiEnvPool(pool_size=batch_size)
    tasks = [random.choice(task_pool) for _ in range(batch_size)]
    task_infos = {t: get_task(t) for t in set(tasks)}

    # Reset all envs
    observations = [pool.reset(i, tasks[i]) for i in range(batch_size)]
    trajectories = [Trajectory(task=tasks[i]) for i in range(batch_size)]
    active = [True] * batch_size

    for step in range(max_steps):
        active_idx = [i for i in range(batch_size) if active[i]]
        if not active_idx:
            break

        # Check per-task max_folds
        active_idx = [
            i for i in active_idx
            if trajectories[i].length < task_infos[tasks[i]].get("max_folds", 1)
        ]
        if not active_idx:
            break

        # Build prompts from current observations
        prompts = [
            build_prompt_from_obs(tasks[i], task_infos[tasks[i]], observations[i])
            for i in active_idx
        ]

        # Generate (batched)
        completions = generate_fn(prompts)

        # Step envs
        for j, i in enumerate(active_idx):
            state_hash = hash_paper_state(pool.envs[i]._paper_state)
            crease = extract_crease_json(completions[j])

            if crease is None:
                trajectories[i].add_step(
                    prompt=prompts[j],
                    completion=completions[j],
                    reward=-2.0,
                    done=False,
                    state_hash=state_hash,
                )
                continue

            obs = pool.step(i, crease)
            observations[i] = obs

            trajectories[i].add_step(
                prompt=prompts[j],
                completion=obs.completion if hasattr(obs, 'completion') else completions[j],
                reward=obs.reward if obs.reward is not None else -1.0,
                done=obs.done,
                state_hash=state_hash,
                reward_breakdown={
                    k: v for k, v in obs.reward_breakdown.items()
                } if obs.reward_breakdown else {},
            )

            if obs.done:
                active[i] = False

    return trajectories
```

**Validation** — test with a mock generate_fn:
```python
def test_rollout_with_mock_generate():
    def mock_gen(prompts):
        return ['{"from": [0,0], "to": [1,1], "assignment": "V"}'] * len(prompts)

    trajs = run_rollout_batch(
        generate_fn=mock_gen,
        task_pool=["triangle"],
        batch_size=4,
    )
    assert len(trajs) == 4
    assert all(t.length == 1 for t in trajs)  # triangle has max_folds=1
    assert all(t.steps[0].done for t in trajs)
    assert all(t.total_reward > 0 for t in trajs)

def test_rollout_multistep_task():
    def mock_gen(prompts):
        return ['{"from": [0.5,0], "to": [0.5,1], "assignment": "V"}'] * len(prompts)

    trajs = run_rollout_batch(
        generate_fn=mock_gen,
        task_pool=["quarter_fold"],
        batch_size=4,
    )
    assert len(trajs) == 4
    assert all(t.length == 2 for t in trajs)  # quarter_fold max_folds=2
```

---

## Phase 3 — veRL Integration

### Step 3.1: Origami tool (veRL BaseTool wrapper)

**File**: `training/verl_tool.py` (new)

veRL's multi-turn rollout calls tools via `BaseTool.execute()`. Wrap the origami
environment step as a tool.

```python
from verl.tools.base_tool import BaseTool, ToolResponse
from origami_server.environment import OrigamiEnvironment
from origami_server.models import OrigamiAction
from origami_server.engine.paper_state import hash_paper_state
from training.reward import extract_crease_json
import json

class OrigamiTool(BaseTool):
    """veRL tool wrapper for origami environment step."""

    def __init__(self, config: dict = None):
        super().__init__()
        self.envs: dict[str, OrigamiEnvironment] = {}

    def execute(self, tool_input: str, **kwargs) -> ToolResponse:
        session_id = kwargs.get("session_id", "default")

        # Parse LLM output as crease JSON
        crease = extract_crease_json(tool_input)
        if crease is None:
            return ToolResponse(text=json.dumps({
                "error": "Invalid crease JSON",
                "reward": -2.0,
                "done": False,
            }))

        env = self.envs.get(session_id)
        if env is None:
            return ToolResponse(text=json.dumps({
                "error": "No active session. Call reset first.",
                "reward": -2.0,
                "done": True,
            }))

        obs = env.step(OrigamiAction(crease=crease))

        return ToolResponse(text=json.dumps({
            "reward": obs.reward,
            "done": obs.done,
            "step": obs.step_count,
            "max_steps": obs.max_steps,
            "anchor_points": obs.anchor_points,
            "current_creases": obs.current_creases,
            "reward_breakdown": obs.reward_breakdown,
            "state_hash": hash_paper_state(env._paper_state),
        }))
```

**Tool config YAML**: `config/origami_tool.yaml`
```yaml
tools:
  - class_name: "training.verl_tool.OrigamiTool"
    config:
      type: native
    tool_schema:
      type: function
      function:
        name: add_crease
        description: "Add a fold crease to the paper"
        parameters:
          type: object
          properties:
            from:
              type: array
              items: {type: number}
              description: "[x, y] start point"
            to:
              type: array
              items: {type: number}
              description: "[x, y] end point"
            assignment:
              type: string
              enum: ["M", "V"]
          required: [from, to, assignment]
```

---

### Step 3.2: Origami AgentLoop

**File**: `training/agent_loop.py` (new)

This is where we decide between two integration approaches:

**Option A — Tool-call style**: Model outputs a function call `add_crease(...)`,
veRL parses it, calls OrigamiTool.execute(), appends tool response to conversation.
Pros: native veRL pattern, automatic response masking. Cons: requires the model to
learn tool-call format (extra tokens, not just raw JSON).

**Option B — Custom AgentLoop**: Implement AgentLoopBase.run() directly. Model
outputs raw crease JSON, we parse it, step the env, build next prompt manually.
Pros: simpler output format (just JSON), matches V2 prompt template. Cons: must
manually construct prompt_ids, response_ids, response_mask.

**Decision: Option B (custom AgentLoop)** — origami's output is a single JSON object,
not a tool call. Adding tool-call formatting overhead wastes tokens and requires
the model to learn a syntax it doesn't need.

```python
from verl.tools.agent_loop import AgentLoopBase, AgentLoopOutput
from training.rollout import run_rollout_batch
from training.curriculum import get_task_pool

class OrigamiAgentLoop(AgentLoopBase):
    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        global_step = kwargs.get("global_step", 0)
        task_pool = get_task_pool(global_step)
        batch_size = kwargs.get("batch_size", 16)
        tokenizer = kwargs.get("tokenizer")

        # generate_fn wraps the vLLM/sglang server
        async def generate_fn(prompts):
            # veRL provides this via the rollout worker
            return await self.generate(prompts, sampling_params)

        trajectories = run_rollout_batch(
            generate_fn=generate_fn,
            task_pool=task_pool,
            batch_size=batch_size,
        )

        # Convert trajectories to token-level format
        # Each trajectory → concatenated [prompt][response][prompt][response]...
        all_prompt_ids = []
        all_response_ids = []
        all_response_mask = []

        for traj in trajectories:
            prompt_ids = []
            response_ids = []
            response_mask = []

            for step in traj.steps:
                p_ids = tokenizer.encode(step.prompt, add_special_tokens=False)
                r_ids = tokenizer.encode(step.completion, add_special_tokens=False)
                prompt_ids.extend(p_ids)
                response_ids.extend(r_ids)
                response_mask.extend([1] * len(r_ids))  # all LLM-generated

            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_mask.append(response_mask)

        return AgentLoopOutput(
            prompt_ids=all_prompt_ids,
            response_ids=all_response_ids,
            response_mask=all_response_mask,
        )
```

**IMPORTANT**: This is a sketch. The actual veRL AgentLoopBase API may differ in how
`generate()` is called and how batches are structured. Step 0.2 validates this before
we commit to the implementation.

**Validation**: Run 1 training step with Qwen2.5-3B on a single GPU.

---

### Step 3.3: GiGPO reward function integration

**File**: `training/reward_fn.py` (new)

veRL expects a reward function that takes trajectories and returns per-step rewards.
Wire GiGPO advantage computation here.

```python
from training.gigpo import compute_gigpo_advantages
from training.trajectory import Trajectory

class OrigamiRewardManager:
    def __init__(self, alpha_start=1.0, alpha_end=0.3, warmup_steps=200):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.warmup_steps = warmup_steps
        self.global_step = 0

    def compute_rewards(self, trajectories: list[Trajectory]) -> list[list[float]]:
        """Returns per-step advantages for each trajectory."""
        alpha = self._get_alpha()
        return compute_gigpo_advantages(trajectories, alpha=alpha)

    def _get_alpha(self) -> float:
        if self.global_step >= self.warmup_steps:
            progress = min(
                (self.global_step - self.warmup_steps)
                / (1500 - self.warmup_steps),
                1.0,
            )
            return self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        return self.alpha_start

    def step(self):
        self.global_step += 1
```

**Alpha schedule**: Start at 1.0 (episode-level only — simpler signal for early
training). Anneal to 0.3 (step-level dominates — fine-grained credit assignment
once policy is stable enough to produce consistent early steps).

---

### Step 3.4: veRL training config

**File**: `config/origami_gigpo.yaml` (new)

```yaml
data:
  return_raw_chat: true
  max_prompt_length: 512
  max_response_length: 128

actor_rollout_ref:
  model:
    path: unsloth/Qwen3-32B
    lora:
      rank: 32
      alpha: 64
      target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
        - gate_proj
        - up_proj
        - down_proj

  actor:
    optim:
      lr: 2e-4
      weight_decay: 0.01
    ppo_mini_batch_size: 16
    ppo_micro_batch_size: 4

  rollout:
    name: sglang
    multi_turn: true
    mode: async
    gpu_memory_utilization: 0.85
    tensor_model_parallel_size: 4
    temperature: 0.7
    top_p: 0.95
    max_tokens: 128
    multi_turn:
      tokenization_sanity_check_mode: ignore_strippable

  ref:
    offload: true

algorithm:
  adv_estimator: grpo          # GiGPO uses GRPO as base
  grpo:
    num_generations: 8
    clip_range: 0.2
  kl_ctrl:
    kl_coef: 0.02

trainer:
  total_training_steps: 1500
  save_steps: 100
  log_steps: 10
  project_name: origami-v3
  experiment_name: gigpo-sequential
```

**Note**: verl-agent may use a different config schema for GiGPO vs standard GRPO.
Consult verl-agent's example configs during Step 0.2. The custom GiGPO advantage
computation (Step 1.4) may need to be injected via a callback or by subclassing
veRL's advantage estimator.

---

## Phase 4 — Modal Deployment

### Step 4.1: Modal training entrypoint

**File**: `modal_train_v3.py` (new)

```python
import modal

app = modal.App("origami-v3-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128",
        "pip install -q verl[all]>=0.7 sglang vllm>=0.8",
        "pip install -q transformers peft ray[default]",
        "pip install -q numpy scipy shapely pydantic 'openenv-core[core]>=0.2.1'",
    )
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(".", remote_path="/app", copy=True,
                   ignore=[".git", "__pycache__", "outputs", "*.egg-info", ".venv"])
)

volume = modal.Volume.from_name("origami-checkpoints-v3", create_if_missing=True)

@app.function(
    image=image,
    gpu=modal.gpu.B200(count=4),
    timeout=6 * 3600,
    volumes={"/outputs": volume},
)
def train(max_steps: int = 1500, resume: bool = False):
    import subprocess
    import ray

    ray.init(num_gpus=4)

    cmd = [
        "python", "-m", "verl.trainer.main",
        "--config", "/app/config/origami_gigpo.yaml",
        f"--trainer.total_training_steps={max_steps}",
        "--trainer.save_path=/outputs/checkpoints",
    ]
    if resume:
        cmd.append("--trainer.resume_from=/outputs/checkpoints/latest")

    subprocess.run(cmd, cwd="/app", check=True)
    volume.commit()

@app.local_entrypoint()
def main(max_steps: int = 1500, resume: bool = False):
    train.remote(max_steps=max_steps, resume=resume)
```

**Validation**: `modal run modal_train_v3.py --max-steps 5` completes without error.

---

### Step 4.2: Multi-step evaluation

**File**: `modal_eval_v3.py` (new)

Key difference from V1 eval: run full multi-step episodes, not single-shot.

```python
def evaluate_multistep(model, tokenizer, task_name, n_episodes=20):
    """Run complete multi-step episodes and report per-step + final metrics."""
    env = OrigamiEnvironment(mode="step")
    task_info = get_task(task_name)
    results = []

    for ep in range(n_episodes):
        obs = env.reset(task_name=task_name)
        episode_reward = 0.0
        steps = []

        for step in range(task_info["max_folds"]):
            prompt = build_prompt_from_obs(task_name, task_info, obs)
            completion = generate_single(model, tokenizer, prompt)
            crease = extract_crease_json(completion)

            if crease is None:
                steps.append({"step": step, "reward": -2.0, "valid": False})
                episode_reward -= 2.0
                break

            obs = env.step(OrigamiAction(crease=crease))
            steps.append({
                "step": step,
                "reward": obs.reward,
                "progress": obs.reward_breakdown.get("progress", 0),
                "valid": True,
            })
            episode_reward += obs.reward if obs.reward else 0

            if obs.done:
                break

        results.append({
            "total_reward": episode_reward,
            "steps": steps,
            "completed": obs.done,
            "final_progress": obs.reward_breakdown.get("progress", 0),
        })

    return results
```

**Reported metrics per task**:
- Mean total reward (across episodes)
- Mean steps to completion
- Completion rate (% episodes reaching progress > 0.9)
- Per-step reward curve (mean reward at step 0, 1, 2, ...)

---

## Phase 5 — Validation & First Run

### Step 5.1: Local smoke test (single GPU, small model)

```bash
# Run 5 training steps with Qwen2.5-3B
python -m training.train_v3 \
  --model unsloth/Qwen2.5-3B-Instruct \
  --max-steps 5 \
  --num-generations 4 \
  --tasks triangle,half_fold
```

**Check**:
- Trajectories generated (print lengths)
- GiGPO advantages computed (print mean, std)
- Gradient step completes (loss is finite)
- Checkpoint saved

### Step 5.2: Modal smoke test (4× B200, Qwen3-32B)

```bash
modal run modal_train_v3.py --max-steps 50
```

**Check**:
- All 4 GPUs utilized (nvidia-smi)
- No OOM
- Curriculum phase 1 active (triangle, half_fold only)
- Checkpoint committed to volume

### Step 5.3: Full training run

```bash
modal run modal_train_v3.py --max-steps 1500
```

**Monitor**:
- Reward curves per task (should rise monotonically within each curriculum phase)
- Step-level advantage variance (should be non-zero at step 0, may be 0 at late steps)
- Completion bonus frequency (should increase over training)
- Economy scores (should stay > 0.7 — model not spamming excess creases)
- KL divergence (should stay bounded, not explode)

**Expected timeline**: ~5 hours, ~$135

### Step 5.4: Evaluation

```bash
modal run modal_eval_v3.py --checkpoint latest --n-episodes 20 --tasks all
```

**Success criteria**:
- triangle: >90% completion rate, mean reward > 1.5
- half_fold: >90% completion rate
- quarter_fold: >70% completion rate, mean 2-step reward > 2.0
- waterbomb_base: >30% completion rate (hard task, 4 steps)
- map_fold: any completion = major win (6 steps, hardest task)

---

## Files Summary

| File | Phase | Status | Notes |
|------|-------|--------|-------|
| `modal_verl_smoke.py` | 0 | new (temp) | Verify veRL on Modal |
| `tests/test_verl_multiturn_smoke.py` | 0 | new (temp) | Verify multi-turn pipeline |
| `origami_server/engine/paper_state.py` | 1 | modify | Add hash_paper_state() |
| `origami_server/engine/step_reward.py` | 1 | modify | Difficulty-scaled completion bonus |
| `training/trajectory.py` | 1 | new | Step + Trajectory dataclasses |
| `training/env_pool.py` | 1 | new | In-process env pool |
| `training/gigpo.py` | 1 | new | GiGPO advantage computation |
| `training/curriculum.py` | 1 | new | ScalingInter-RL scheduler |
| `training/prompt_builder.py` | 2 | new | Observation-aware prompts |
| `training/rollout.py` | 2 | new | Standalone rollout function |
| `training/verl_tool.py` | 3 | new | BaseTool wrapper (if Option A) |
| `training/agent_loop.py` | 3 | new | AgentLoopBase implementation |
| `training/reward_fn.py` | 3 | new | GiGPO reward manager |
| `config/origami_gigpo.yaml` | 3 | new | veRL training config |
| `modal_train_v3.py` | 4 | new | Modal V3 entrypoint |
| `modal_eval_v3.py` | 4 | new | Multi-step evaluation |
| `tests/test_origami.py` | 1-2 | modify | Add V3 tests |

**No existing files broken**: V1/V2 training pipeline untouched. `train_grpo.py`,
`modal_train.py`, `modal_eval.py` continue to work for backward compat.

---

## Dependency on veRL API Discovery

Steps 0.1 and 0.2 will likely reveal that veRL's actual API differs from the
documentation sketches above. Expect to adjust:

1. **AgentLoopBase.run() signature** — may be sync, not async. May receive
   `generate_fn` differently than shown.
2. **AgentLoopOutput fields** — may need additional fields (rewards, advantages,
   metadata) beyond prompt_ids/response_ids/response_mask.
3. **GiGPO integration** — verl-agent may have its own advantage estimator class
   that we should subclass rather than computing advantages manually.
4. **Config schema** — verl-agent's YAML may use different keys than vanilla veRL.

The phased approach (Phase 0 first) is specifically designed to surface these
discrepancies before we write origami-specific code on top of wrong assumptions.

---

## Critical Path

```
Phase 0 (framework validation)  ←── BLOCKING, do first
    ↓
Phase 1 (building blocks)  ←── can parallelize all 6 steps
    ↓
Phase 2 (rollout logic)  ←── depends on Phase 1
    ↓
Phase 3 (veRL integration)  ←── depends on Phase 0 + Phase 2
    ↓
Phase 4 (Modal deployment)  ←── depends on Phase 3
    ↓
Phase 5 (validation)  ←── depends on Phase 4
```

**Parallelizable**: Phase 1 steps (1.1-1.6) are all independent. Phase 0 can run
in parallel with Phase 1 since Phase 0 doesn't touch origami code.

**Estimated total effort**: ~3-4 days of focused implementation.
- Phase 0: 0.5 day (mostly waiting for Modal builds)
- Phase 1: 0.5 day (pure Python, well-defined)
- Phase 2: 0.5 day (prompt building + rollout)
- Phase 3: 1-1.5 days (veRL API integration, most uncertain)
- Phase 4: 0.5 day (Modal config)
- Phase 5: 0.5 day (runs + monitoring)
