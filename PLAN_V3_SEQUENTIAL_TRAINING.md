# V3 Training Architecture — Sequential Rollouts with GiGPO

## Prior Art

This plan is informed by five recent papers. Read these before implementing.

### 1. GiGPO — Group-in-Group Policy Optimization (NeurIPS 2025)
**Paper:** https://arxiv.org/abs/2505.10978
**Code:** https://github.com/langfengQ/verl-agent

The single most relevant paper. GiGPO extends GRPO to multi-turn agent environments with
two-level advantage estimation:
- **Episode-level**: standard GRPO grouping (same prompt → N trajectories → relative advantage)
- **Step-level**: "anchor state grouping" — retroactively finds steps across trajectories that
  share the same environment state, then computes step-local advantages within those groups

This maps perfectly to origami: at step 0, all trajectories share the same state (empty paper).
At step 1, trajectories that placed the same first crease share a state. GiGPO automatically
discovers these groups and assigns per-step credit without needing a critic network.

Key properties: critic-free, same GPU memory as GRPO, no additional rollout cost.
Results: +12% over GRPO on ALFWorld, +9% on WebShop.

### 2. AgentGym-RL — ScalingInter-RL (2025)
**Paper:** https://arxiv.org/abs/2509.08755
**Code:** https://github.com/WooooDyy/AgentGym-RL

Framework for training LLM agents across diverse environments with progressive horizon
expansion (ScalingInter-RL). Key insight: start with short episodes (1-2 steps), then
gradually increase max_steps as the policy improves.

Directly applicable: start training on triangle/half_fold (1 step), then add quarter_fold
(2 steps), then waterbomb_base (4 steps), then map_fold (6 steps). This prevents the model
from being overwhelmed by long horizons before learning basic crease placement.

### 3. veRL / verl-agent — Engineering Framework
**Docs:** https://verl.readthedocs.io/en/latest/start/agentic_rl.html
**Code:** https://github.com/volcengine/verl

ByteDance's RL framework with native multi-turn rollout support. verl-agent extends it with
GiGPO + multi-turn AgentLoop. Handles the hard distributed training problems:
- vLLM ↔ training weight synchronization
- Ray-based multi-GPU allocation
- Custom environment integration via BaseTool / AgentLoop

### 4. mmGRPO — Multi-Module GRPO (2025)
**Paper:** https://arxiv.org/abs/2508.04660

Handles variable-length and interrupted trajectories in modular LM programs. Relevant for
origami because episodes can end early (completion bonus triggers done=True before max_folds).
mmGRPO's trajectory padding/masking strategy applies directly.

### 5. ArCHer — Hierarchical Multi-Turn RL (ICML 2024)
**Paper:** https://arxiv.org/abs/2402.19446

Two-level RL: high-level value function aggregates reward across turns, low-level policy
gradient within each turn. More complex than GiGPO (requires a critic), but relevant if
GiGPO underperforms — ArCHer's off-policy value function can handle longer horizons
(map_fold's 6 steps).

### 6. Agent-R1 — End-to-End RL for Agents (2025)
**Paper:** https://arxiv.org/abs/2511.14460
**Code:** https://github.com/0russwest0/Agent-R1

Clean Tool/ToolEnv abstraction for multi-turn RL. Process rewards + outcome rewards.
Good reference implementation for the rollout ↔ environment interface.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     Modal Container (4× B200)                     │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                   veRL Training Loop                       │   │
│  │                                                            │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │   │
│  │  │ vLLM Engine  │   │   Trainer    │   │  Ref Model   │  │   │
│  │  │ (generation) │   │  (gradients) │   │  (KL anchor) │  │   │
│  │  │  GPU 0-3 TP  │   │  GPU 0-3 DP  │   │  GPU 0 only  │  │   │
│  │  └──────┬───────┘   └──────────────┘   └──────────────┘  │   │
│  │         │                                                  │   │
│  │  ┌──────▼─────────────────────────────────────────────┐   │   │
│  │  │           OrigamiAgentLoop                          │   │   │
│  │  │                                                     │   │   │
│  │  │  for step in range(max_folds):                      │   │   │
│  │  │    prompts = build_step_prompt(observations)        │   │   │
│  │  │    completions = vllm.generate(prompts, batch=N)    │   │   │
│  │  │    creases = extract_crease_json(completions)       │   │   │
│  │  │    observations = env_pool.step_all(creases)        │   │   │
│  │  │    trajectory.append(prompts, completions, rewards)  │   │   │
│  │  │    mask done episodes                               │   │   │
│  │  └──────────────────────┬──────────────────────────────┘   │   │
│  │                         │                                   │   │
│  │  ┌──────────────────────▼──────────────────────────────┐   │   │
│  │  │           GiGPO Advantage Estimation                 │   │   │
│  │  │                                                      │   │   │
│  │  │  1. Episode-level: R_i = Σ r_t for trajectory i     │   │   │
│  │  │     A_episode_i = (R_i - mean(R_group)) / std(R)    │   │   │
│  │  │                                                      │   │   │
│  │  │  2. Step-level: group steps by (task, step_idx,      │   │   │
│  │  │     paper_state_hash)                                │   │   │
│  │  │     A_step_t = (r_t - mean(r_group_t)) / std(r_t)  │   │   │
│  │  │                                                      │   │   │
│  │  │  3. Combined: A_t = α·A_episode + (1-α)·A_step_t   │   │   │
│  │  │     α annealed from 1.0 → 0.3 over training         │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              Environment Pool (CPU threads)                │   │
│  │                                                            │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐        ┌────────┐      │   │
│  │  │ Env 0  │ │ Env 1  │ │ Env 2  │  ...   │ Env 63 │      │   │
│  │  │ Paper  │ │ Paper  │ │ Paper  │        │ Paper  │      │   │
│  │  │ State  │ │ State  │ │ State  │        │ State  │      │   │
│  │  └────────┘ └────────┘ └────────┘        └────────┘      │   │
│  │                                                            │   │
│  │  Each env: OrigamiEnvironment(mode="step")                │   │
│  │  In-process, no HTTP — direct Python calls                 │   │
│  │  ~1ms per step (PaperState + step_reward)                  │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### GPU Allocation Strategy

veRL uses time-sharing (colocate) by default: all GPUs switch between generation and training.
This is simpler than partitioning and works well when generation is fast (our completions are
~50 tokens).

```
Phase 1: ROLLOUT (all 4 GPUs via vLLM tensor-parallel)
  → Generate N trajectories × max_folds steps
  → ~8 sec for 64 trajectories × 4 steps (waterbomb_base)

Phase 2: REWARD (CPU, overlapped with weight transfer)
  → GiGPO advantage computation
  → ~100ms

Phase 3: TRAIN (all 4 GPUs via FSDP)
  → Policy gradient step on trajectory batch
  → ~3 sec

Phase 4: WEIGHT SYNC (automatic in veRL)
  → Push updated LoRA weights to vLLM engine
  → ~500ms

Total per training step: ~12 sec (vs ~5 sec single-shot V1)
```

---

## Component Design

### 1. OrigamiAgentLoop

The core multi-turn rollout logic. Implements veRL's `AgentLoopBase`.

```python
# training/agent_loop.py

from verl.tools.agent_loop import AgentLoopBase
from origami_server.environment import OrigamiEnvironment
from origami_server.models import OrigamiAction

class OrigamiAgentLoop(AgentLoopBase):
    """Multi-turn rollout: generate one crease per step, accumulate trajectory."""

    def __init__(self, task_pool: list[str], max_folds_map: dict[str, int]):
        self.task_pool = task_pool
        self.max_folds_map = max_folds_map

    def run(self, generate_fn, prompts: list[str], **kwargs) -> list[Trajectory]:
        """
        Called by veRL's rollout worker. Runs full episodes for a batch of prompts.

        generate_fn: calls vLLM under the hood (batched, all GPUs)
        prompts: initial prompts (one per episode)
        """
        batch_size = len(prompts)

        # 1. Initialize environments
        envs = []
        tasks = []
        for i in range(batch_size):
            task = random.choice(self.task_pool)
            env = OrigamiEnvironment(mode="step")
            obs = env.reset(task_name=task)
            envs.append(env)
            tasks.append(task)

        # 2. Build initial prompts from observations
        current_prompts = [
            build_step_prompt(
                task=tasks[i],
                obs=obs,
                step=0,
                max_folds=self.max_folds_map[tasks[i]]
            )
            for i, obs in enumerate([e._last_obs for e in envs])
        ]

        # 3. Sequential rollout
        trajectories = [Trajectory() for _ in range(batch_size)]
        active_mask = [True] * batch_size

        for step in range(max(self.max_folds_map.values())):
            # Skip finished episodes
            active_indices = [i for i, a in enumerate(active_mask) if a]
            if not active_indices:
                break

            active_prompts = [current_prompts[i] for i in active_indices]

            # Generate completions (batched across all active episodes)
            completions = generate_fn(active_prompts)

            # Step environments
            for j, i in enumerate(active_indices):
                crease = extract_crease_json(completions[j])
                if crease is None:
                    # Failed parse → penalty, episode continues
                    trajectories[i].add_step(
                        prompt=current_prompts[i],
                        completion=completions[j],
                        reward=-2.0,
                        done=False,
                        state_hash=hash_paper_state(envs[i]._paper_state),
                    )
                    continue

                action = OrigamiAction(crease=crease)
                obs = envs[i].step(action)

                trajectories[i].add_step(
                    prompt=current_prompts[i],
                    completion=completions[j],
                    reward=obs.reward,
                    done=obs.done,
                    state_hash=hash_paper_state(envs[i]._paper_state),
                    reward_breakdown=obs.reward_breakdown,
                )

                if obs.done:
                    active_mask[i] = False
                else:
                    # Update prompt with new observation
                    current_prompts[i] = build_step_prompt(
                        task=tasks[i],
                        obs=obs,
                        step=step + 1,
                        max_folds=self.max_folds_map[tasks[i]],
                    )

        return trajectories
```

**Key design decision:** environments run in-process (no HTTP). The origami env's step() is
pure CPU math (~1ms). No need for a server when training — import OrigamiEnvironment directly.
This eliminates 64 × 4 = 256 HTTP round-trips per training step.

### 2. GiGPO Advantage Estimation

GiGPO's two-level grouping adapted for origami:

```python
# training/gigpo.py

def compute_gigpo_advantages(
    trajectories: list[Trajectory],
    alpha: float = 0.7,  # episode vs step weight, annealed over training
) -> list[list[float]]:
    """
    Two-level advantage estimation (GiGPO, §3.2).

    Level 1 — Episode-level (standard GRPO):
      Group trajectories by task. Within each group:
      A_episode_i = (R_i - mean(R_group)) / std(R_group)

    Level 2 — Step-level (anchor state grouping):
      Group steps by (task, step_index, paper_state_hash).
      Steps that start from the same paper state are comparable.
      A_step_t = (r_t - mean(r_group_t)) / std(r_group_t)

    Combined:
      A_t = α · A_episode + (1 - α) · A_step_t
    """
    # --- Level 1: Episode advantages ---
    task_groups = defaultdict(list)
    for i, traj in enumerate(trajectories):
        task_groups[traj.task].append((i, traj.total_reward))

    episode_advantages = {}
    for task, group in task_groups.items():
        rewards = [r for _, r in group]
        mean_r, std_r = np.mean(rewards), max(np.std(rewards), 1e-8)
        for idx, r in group:
            episode_advantages[idx] = (r - mean_r) / std_r

    # --- Level 2: Step advantages ---
    # Group by (task, step_index, state_hash) — the "anchor state" grouping
    step_groups = defaultdict(list)
    for i, traj in enumerate(trajectories):
        for t, step in enumerate(traj.steps):
            key = (traj.task, t, step.state_hash)
            step_groups[key].append((i, t, step.reward))

    step_advantages = {}
    for key, group in step_groups.items():
        rewards = [r for _, _, r in group]
        mean_r, std_r = np.mean(rewards), max(np.std(rewards), 1e-8)
        for idx, t, r in group:
            step_advantages[(idx, t)] = (r - mean_r) / std_r

    # --- Combine ---
    combined = []
    for i, traj in enumerate(trajectories):
        traj_advantages = []
        for t in range(len(traj.steps)):
            a_ep = episode_advantages[i]
            a_step = step_advantages.get((i, t), 0.0)
            traj_advantages.append(alpha * a_ep + (1 - alpha) * a_step)
        combined.append(traj_advantages)

    return combined
```

**Why GiGPO works well for origami:**

The "anchor state grouping" is naturally strong here because:

1. **Step 0**: ALL trajectories for the same task share the same state (empty paper).
   This gives a large group → robust step-level advantage. The model learns which first
   crease is best for each task.

2. **Step 1+**: Trajectories that placed the same first crease share a state. If 8 out of
   64 trajectories all placed a diagonal V crease at step 0, those 8 form a step-1 group.
   The model learns which second crease is best given that first crease.

3. **State hashing**: PaperState can be hashed by sorting its crease edges. Two paper states
   with the same creases (regardless of order added) produce the same hash. This increases
   group sizes at later steps.

### 3. Paper State Hashing

For GiGPO's anchor state grouping to work, we need a fast, deterministic hash of PaperState:

```python
# origami_server/engine/paper_state.py  (addition)

def hash_paper_state(state: PaperState) -> int:
    """
    Deterministic hash of paper state for GiGPO anchor grouping.

    Two paper states with the same crease edges (same geometry, same assignments)
    produce the same hash, regardless of the order creases were added.
    """
    edges = state.crease_edges()  # list of {"from": [x,y], "to": [x,y], "assignment": str}

    # Canonicalize: sort endpoints within each edge, then sort edges
    canonical = []
    for e in edges:
        p1 = tuple(round(c, 6) for c in e["from"])
        p2 = tuple(round(c, 6) for c in e["to"])
        canonical.append((min(p1, p2), max(p1, p2), e["assignment"]))
    canonical.sort()

    return hash(tuple(canonical))
```

### 4. ScalingInter-RL: Progressive Horizon Expansion

From AgentGym-RL. Don't throw the model into 6-step map_fold episodes on day one.

```python
# training/curriculum.py

CURRICULUM = [
    # Phase 1: Single-step tasks (learn basic crease placement)
    {
        "steps": (0, 200),
        "tasks": ["triangle", "half_fold"],
        "max_folds_override": None,  # use task default (1)
    },
    # Phase 2: Two-step tasks (learn sequencing)
    {
        "steps": (200, 500),
        "tasks": ["triangle", "half_fold", "quarter_fold", "letter_fold"],
        "max_folds_override": None,  # 1-2 steps
    },
    # Phase 3: Medium episodes (learn geometry awareness)
    {
        "steps": (500, 900),
        "tasks": ["quarter_fold", "letter_fold", "waterbomb_base"],
        "max_folds_override": None,  # 2-4 steps
    },
    # Phase 4: Full difficulty (learn long-horizon planning)
    {
        "steps": (900, 1500),
        "tasks": ["waterbomb_base", "map_fold"],
        "max_folds_override": None,  # 4-6 steps
    },
]

def get_curriculum_config(global_step: int) -> dict:
    """Returns task pool and max_folds for the current training phase."""
    for phase in CURRICULUM:
        if phase["steps"][0] <= global_step < phase["steps"][1]:
            return phase
    return CURRICULUM[-1]  # default to final phase
```

**Why this matters:** Without curriculum, the model sees map_fold (6 steps) early when it
can't even place a single valid crease. The 6 failed steps generate garbage gradients.
With curriculum, the model masters 1-step tasks first (high reward signal → fast learning),
then transfers that knowledge to 2-step tasks, etc.

### 5. Reward Shaping Fixes

The V2 reward has a known problem: the 10.0× completion bonus dominates everything else.
Once the model discovers the completion bonus for easy tasks, it ignores per-step signals.

**Fix 1: Scale completion bonus by task difficulty**
```python
# origami_server/engine/step_reward.py  (modification)

COMPLETION_BONUS = {
    1: 2.0,   # triangle, half_fold — easy, small bonus
    2: 5.0,   # quarter_fold, letter_fold
    3: 10.0,  # waterbomb_base
    4: 15.0,  # map_fold — hardest, biggest bonus
}
```

**Fix 2: Normalize total reward to bounded range**

GiGPO's advantage estimation works best when rewards are in a similar range across tasks.
If triangle gives reward ~2.5 and map_fold gives ~0.3, the advantage for map_fold episodes
is always negative → the model avoids map_fold entirely.

```python
def normalize_episode_reward(total: float, task_difficulty: int) -> float:
    """Normalize reward to [0, 1] range per task difficulty."""
    # Expected reward ranges (empirical, calibrate after first run)
    expected_max = {1: 3.0, 2: 6.0, 3: 12.0, 4: 17.0}
    return min(total / expected_max[task_difficulty], 1.0)
```

**Fix 3: Entropy bonus for exploration**

Early in training, the model may collapse to always producing the same crease (e.g., diagonal
valley fold for every task). Add a small entropy bonus to encourage diversity:

```python
# In GiGPO config
entropy_coeff = 0.01  # standard, decayed over training
```

veRL supports this natively via `algorithm.kl_ctrl.kl_coef`.

---

## Trajectory Data Structure

Each training step produces a batch of trajectories. Each trajectory is a multi-step episode:

```python
@dataclass
class Step:
    prompt: str              # full prompt including paper state
    completion: str          # model output (crease JSON)
    reward: float            # per-step reward from compute_reward()
    done: bool               # episode ended?
    state_hash: int          # PaperState hash for GiGPO grouping
    reward_breakdown: dict   # {progress, delta, kawasaki, ...}
    log_prob: float          # log probability under current policy (from vLLM)

@dataclass
class Trajectory:
    task: str
    steps: list[Step]

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    @property
    def length(self) -> int:
        return len(self.steps)
```

### Trajectory → Token-Level Training Data

veRL's policy gradient operates on token-level log probs. Each trajectory must be flattened
into a single token sequence with per-token advantages:

```
[prompt_0][completion_0][prompt_1][completion_1]...[prompt_T][completion_T]
                ↑                      ↑                          ↑
           advantage_0           advantage_1                 advantage_T
           (applied to           (applied to                 (applied to
            completion            completion                  completion
            tokens only)          tokens only)                tokens only)
```

Prompt tokens get advantage = 0 (no gradient through prompts). Completion tokens get the
GiGPO advantage for that step.

For variable-length episodes (early termination via completion bonus):
- Pad to max_folds with masked tokens (loss = 0)
- Or use mmGRPO's approach: pack variable-length trajectories and mask at the step boundary

veRL handles this via its `attention_mask` and `loss_mask` tensors in the training batch.

---

## veRL Configuration

```yaml
# config/origami_gigpo.yaml

data:
  train_files: null  # prompts generated dynamically by AgentLoop
  max_prompt_length: 512
  max_response_length: 128  # single crease JSON ≈ 50 tokens

actor_rollout_ref:
  model:
    path: unsloth/Qwen3-32B
    lora:
      rank: 32
      alpha: 64
      target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

  actor:
    optim:
      lr: 2e-4
      weight_decay: 0.01
    ppo_mini_batch_size: 16
    ppo_micro_batch_size: 4

  rollout:
    name: vllm
    multi_turn: true
    agent_loop: training.agent_loop.OrigamiAgentLoop
    gpu_memory_utilization: 0.85
    tensor_model_parallel_size: 4  # all 4 B200s for generation
    temperature: 0.7
    top_p: 0.95
    max_tokens: 128

  ref:
    # Reference model for KL penalty (loaded on GPU 0, offloaded during generation)
    offload: true

algorithm:
  name: gigpo
  gigpo:
    num_generations: 8          # 8 trajectories per prompt group
    alpha_schedule:
      start: 1.0               # episode-level dominates early
      end: 0.3                 # step-level dominates late
      warmup_steps: 200
    clip_range: 0.2
    entropy_coeff: 0.01
  kl_ctrl:
    kl_coef: 0.02              # KL penalty to reference model

trainer:
  total_training_steps: 1500
  save_steps: 100
  log_steps: 10
  val_steps: 50
```

---

## Environment Pool

The origami env is lightweight (~1ms per step). No need for remote servers or Ray actors.
Just a pool of OrigamiEnvironment instances managed by the AgentLoop.

```python
# training/env_pool.py

class OrigamiEnvPool:
    """Pool of in-process origami environments for parallel rollouts."""

    def __init__(self, pool_size: int = 64):
        self.envs = [OrigamiEnvironment(mode="step") for _ in range(pool_size)]
        self.active = [False] * pool_size

    def reset(self, idx: int, task_name: str) -> OrigamiObservation:
        self.active[idx] = True
        return self.envs[idx].reset(task_name=task_name)

    def step(self, idx: int, crease: dict) -> OrigamiObservation:
        action = OrigamiAction(crease=crease)
        obs = self.envs[idx].step(action)
        if obs.done:
            self.active[idx] = False
        return obs

    def step_batch(self, indices: list[int], creases: list[dict]) -> list[OrigamiObservation]:
        """Step multiple envs in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=min(len(indices), 16)) as pool:
            futures = [
                pool.submit(self.step, idx, crease)
                for idx, crease in zip(indices, creases)
            ]
            return [f.result() for f in futures]
```

### Why in-process, not HTTP?

| Approach | Latency per step | 256 steps (64 episodes × 4 folds) |
|----------|------------------|------------------------------------|
| HTTP to localhost | ~5ms | ~1.3 sec |
| In-process Python call | ~1ms | ~0.26 sec (threaded) |
| In-process (sequential) | ~1ms | ~0.26 sec |

The env is CPU-bound (Shapely intersection + numpy angle math). ThreadPoolExecutor helps
with I/O-bound work but not CPU-bound. For 64 envs, even sequential is only 64ms per step
round — negligible vs the ~2 sec vLLM generation time.

**Exception:** If we later add vision rendering (render paper state as image for VLM input),
the env step becomes ~50ms (matplotlib/PIL). At that point, move to multiprocessing or
a separate Ray actor pool.

---

## Modal Integration

```python
# modal_train_v3.py

import modal

app = modal.App("origami-v3-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "verl>=0.7",
        "vllm>=0.8",
        "torch>=2.5",
        "transformers",
        "peft",
        "ray[default]",
        "shapely>=2.0",
        "numpy",
        "scipy",
    )
    .copy_local_dir("origami_server", "/app/origami_server")
    .copy_local_dir("training", "/app/training")
    .copy_local_file("config/origami_gigpo.yaml", "/app/config/origami_gigpo.yaml")
)

volume = modal.Volume.from_name("origami-checkpoints-v3", create_if_missing=True)

@app.function(
    image=image,
    gpu=modal.gpu.B200(count=4),
    timeout=6 * 3600,  # 6 hours (longer episodes = longer training)
    volumes={"/outputs": volume},
)
def train(
    max_steps: int = 1500,
    resume: bool = False,
    model: str = "unsloth/Qwen3-32B",
):
    import subprocess

    cmd = [
        "python", "-m", "verl.trainer.main",
        "--config", "/app/config/origami_gigpo.yaml",
        f"--trainer.total_training_steps={max_steps}",
        f"--actor_rollout_ref.model.path={model}",
        "--trainer.save_path=/outputs/checkpoints",
    ]

    if resume:
        latest = find_latest_checkpoint("/outputs/checkpoints")
        cmd.append(f"--trainer.resume_from={latest}")

    subprocess.run(cmd, check=True)
    volume.commit()
```

---

## Cost Estimate

| Phase | Tasks | Steps | Time per step | Wall clock | Cost (4×B200 @ ~$28/hr) |
|-------|-------|-------|---------------|------------|--------------------------|
| 1: Single-step | triangle, half_fold | 200 | ~6 sec | 20 min | $9 |
| 2: Two-step | + quarter, letter | 300 | ~8 sec | 40 min | $19 |
| 3: Medium | + waterbomb_base | 400 | ~12 sec | 80 min | $37 |
| 4: Full | + map_fold | 600 | ~15 sec | 150 min | $70 |
| **Total** | | **1500** | | **~5 hours** | **~$135** |

Compare to V1: 600 steps × 5 sec = 50 min, ~$23. V3 is ~6× more expensive per run but
should actually learn multi-step planning rather than plateauing at step 30.

---

## Key Engineering Risks

### Risk 1: veRL + Modal compatibility
veRL uses Ray internally. Modal containers don't have a pre-configured Ray cluster.
**Mitigation:** veRL supports single-node Ray (`ray.init()` with local resources).
4× B200 on one Modal container = one Ray node with 4 GPUs. No multi-node issues.

### Risk 2: vLLM + LoRA hot-swapping
veRL's weight sync pushes LoRA adapters to vLLM after each gradient step. If LoRA rank is
high (32) and model is large (32B), this transfer can be slow.
**Mitigation:** With rank=32, LoRA weights are ~100MB. Transfer at PCIe 5.0 speed: <100ms.
Not a bottleneck.

### Risk 3: GiGPO step groups too small at later steps
At step 3 of waterbomb_base (4 folds total), if all 8 trajectories placed different creases
at steps 0-2, each step-3 group has size 1 → no step-level advantage signal.
**Mitigation:** Increase num_generations from 8 to 16. With 16 trajectories, ~4-6 will share
a state at step 2 (because the model converges on good early creases). Also: the
paper_state_hash canonicalizes crease order, so different orderings of the same creases
produce the same hash → larger groups.

### Risk 4: Completion bonus reward hacking
The model might learn to output trivially "valid" creases that happen to trigger the
completion bonus through geometric coincidence (e.g., placing creases that overlap with
the target but are physically meaningless).
**Mitigation:** The step_reward already checks kawasaki + maekawa + blb validity.
Completion bonus requires progress > 0.9 AND all geometry checks pass. Hard to hack
both simultaneously. Monitor `economy` reward component — if it drops while `progress`
rises, the model is placing excess creases.

### Risk 5: Context length growth across steps
Each step appends ~100 tokens (observation + new anchors + crease history). For map_fold
(6 steps): 512 (initial prompt) + 6 × 100 = 1112 tokens input. Well within Qwen3-32B's
context window. Not a risk for origami, but would be for longer episodes.

---

## Implementation Order

```
Step 1: Install veRL + verl-agent, verify it runs on Modal with 4× B200
        (no origami code yet, just the hello-world GRPO example)

Step 2: Implement paper_state hash (hash_paper_state in paper_state.py)
        Test: same creases in different order → same hash

Step 3: Implement OrigamiEnvPool (training/env_pool.py)
        Test: reset 64 envs, step all with valid creases, verify observations

Step 4: Implement OrigamiAgentLoop (training/agent_loop.py)
        Test: mock generate_fn that returns hardcoded creases, verify trajectories

Step 5: Implement GiGPO advantage computation (training/gigpo.py)
        Test: hand-crafted trajectories with known rewards, verify advantages

Step 6: Implement curriculum (training/curriculum.py)
        Test: verify task pool changes at correct global_step boundaries

Step 7: Write veRL config (config/origami_gigpo.yaml)
        Wire OrigamiAgentLoop into veRL's rollout config

Step 8: Reward shaping fixes (step_reward.py)
        Scale completion bonus by difficulty, add reward normalization

Step 9: Local smoke test
        Run 5 training steps with Qwen2.5-3B on a single GPU
        Verify: trajectories generated, advantages computed, gradient step completes

Step 10: Modal integration (modal_train_v3.py)
         Run 50 steps on 4× B200 with Qwen3-32B
         Verify: no OOM, checkpoints saved, curriculum transitions work

Step 11: Full training run (1500 steps, ~5 hours)
         Monitor: reward curves per task, step-level advantage variance,
         completion bonus frequency, economy scores

Step 12: Evaluation
         Update modal_eval.py for multi-step episodes
         Run N=20 episodes per task, report per-step and final metrics
```

---

## Files Created / Modified

| File | Status | Notes |
|------|--------|-------|
| `training/agent_loop.py` | **new** | OrigamiAgentLoop (veRL AgentLoopBase) |
| `training/gigpo.py` | **new** | GiGPO advantage estimation |
| `training/env_pool.py` | **new** | In-process environment pool |
| `training/curriculum.py` | **new** | ScalingInter-RL progressive horizon |
| `config/origami_gigpo.yaml` | **new** | veRL training configuration |
| `modal_train_v3.py` | **new** | Modal entrypoint for V3 training |
| `origami_server/engine/paper_state.py` | modified | add hash_paper_state() |
| `origami_server/engine/step_reward.py` | modified | scale completion bonus by difficulty |
| `training/reward.py` | modified | add valid_crease(), extract_crease_json() |
| `modal_eval.py` | modified | multi-step episode evaluation |

V1 and V2 code untouched. V3 training is a separate pipeline.

---

## Open Questions

1. **Qwen3-32B vs Qwen2.5-7B?** GiGPO paper uses 3B and 7B models. 32B on 4× B200 is
   feasible but slower. Consider starting with 7B for faster iteration, then scaling to 32B
   once the pipeline is validated.

2. **Vision input?** optigami uses Qwen2.5-VL-7B — renders the current paper state as an
   image and includes it in the prompt. This gives the model spatial understanding that
   text-only prompts lack. If we add vision: (a) env.step() must render a paper image,
   (b) prompt becomes multimodal, (c) vLLM config changes for VLM. Defer to V4.

3. **Off-policy data?** GiGPO is on-policy (fresh rollouts every step). Could we mix in
   offline trajectories (e.g., from the preset solutions in viewer/index.html) as a warm
   start? verl-agent supports SFT → RL transitions. Consider SFT on 100 expert trajectories
   before RL.

4. **Multi-node?** 4× B200 on one Modal container should suffice for 32B. If we go to 70B+,
   need multi-node Ray cluster on Modal — significantly more complex. Not needed for V3.
