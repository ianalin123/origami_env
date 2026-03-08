# Origami Env — V2 Handoff

## What This Project Is

RL environment where an LLM learns to generate origami crease patterns (FOLD format JSON).
The model is rewarded based on how closely its folded shape matches a target shape.
Deployed on Modal (NVIDIA B200, 192GB HBM3e) using Unsloth + TRL GRPO.

Env server runs on Railway: `https://origami-env-production.up.railway.app`

---

## Current State (V1 — working)

### Architecture

**Single-shot episodes**: LLM submits a complete FOLD JSON crease pattern in one action. Physics simulates it. Reward = shape similarity × 20. `done=True` after step 1.

```
reset(task_name="quarter_fold")  →  task description + target positions
step(OrigamiAction(fold_data={complete FOLD JSON}))  →  reward, done=True
```

### Stack
- **Model**: `unsloth/Qwen3-32B` bfloat16 on B200 (no quantization)
- **Training**: TRL `GRPOTrainer` via `training/train_grpo.py`
- **Cloud**: Modal (`modal run modal_train.py`)
- **Checkpoints**: Modal volume `origami-checkpoints` at `/outputs`
- **Env server**: FastAPI via OpenEnv `create_app()`, hosted on Railway

### Reward Functions (3 signals)

| Function | Source | Range | What it measures |
|---|---|---|---|
| `valid_fold` | `training/reward.py` | -2 to +1 | Parseable FOLD JSON with correct structure |
| `flat_foldable_reward` | `training/reward.py` | -0.5 to +1 | Kawasaki + Maekawa + BLB at interior vertices |
| `shape_match_reward` | `training/train_grpo.py` | -2 to +20 | Chamfer distance to target shape (via env server) |

`flat_foldable_reward` is new as of the last session — ported from optigami. Runs locally, no server round-trip.

### Tasks (4 tasks, all single-step)

| Task | Difficulty | Description |
|---|---|---|
| `triangle` | 1 | Diagonal valley fold — trivially easy for Qwen3-32B, converges in ~5 steps |
| `half_fold` | 1 | Horizontal fold at y=0.5 |
| `quarter_fold` | 2 | Two perpendicular valley folds |
| `letter_fold` | 2 | Two parallel folds at y=1/3 and y=2/3 (valley + mountain) |

Default training uses `--task all`: all 4 tasks × 200 samples = 800 dataset rows.

### Key Files

```
origami_env/
├── modal_train.py              # Modal cloud training entrypoint
├── modal_eval.py               # Modal cloud eval entrypoint
├── client.py                   # OrigamiEnv OpenEnv client
├── Dockerfile                  # Railway env server (PORT env var for Railway)
├── origami_server/
│   ├── app.py                  # FastAPI server via create_app()
│   ├── environment.py          # OrigamiEnvironment: reset() + step()
│   ├── models.py               # OrigamiAction, OrigamiObservation, OrigamiState
│   ├── tasks.py                # TASKS dict — 4 target patterns
│   └── engine/
│       ├── simulate.py         # BFS + cumulative rotation transforms
│       ├── shape_match.py      # Chamfer distance + 24-rotation search
│       └── fold_parser.py      # FOLD validation + face triangulation
└── training/
    ├── train_grpo.py           # GRPOTrainer setup, multi-task dataset, prompts
    └── reward.py               # valid_fold, flat_foldable_reward, extract_fold_json
```

### Training Commands

```bash
# Full run (all tasks, 600 steps)
modal run modal_train.py

# Resume from checkpoint
modal run modal_train.py --resume --max-steps 1200

# Eval latest checkpoint vs base model
modal run modal_eval.py --checkpoint checkpoint-20 --n-samples 20
modal run modal_eval.py --checkpoint base --n-samples 20

# Check volume contents
modal volume ls origami-checkpoints
modal volume get origami-checkpoints checkpoint-20 ./outputs/checkpoint-20
```

### Known V1 Issues

1. **Converges too fast**: Qwen3-32B already knows these tasks. All 4 tasks hit max reward within ~30 steps. After that, `reward_std=0` → no GRPO gradient → training is a no-op.

2. **Reward ceiling**: `shape_match_reward` maxes at 20.0. The model hits it early and stays there. No harder signal to keep learning.

3. **Single-shot limits learning**: The model submits the complete pattern at once. GRPO only sees the final result, not individual fold decisions. Compare: training on a chess game vs just win/loss.

4. **KL drift without gradient**: When `reward_std=0`, the policy drifts from base (KL grows to ~0.1) without any learning. Pure degradation after convergence.

5. **`flat_foldable_reward` untested**: Added last session. Needs a training run to verify it actually fires and produces useful signal.

---

## V2 Goal: Multi-Step Episodes

The core upgrade: instead of submitting complete FOLD JSON in one shot, the model outputs **one fold crease at a time**, gets reward after each crease, and sees the updated paper state before deciding the next crease.

### Reference Implementation

`/Users/ianalin/Desktop/optigami/` has a working multi-step implementation. Key files:
- `env/environment.py` — `OrigamiEnvironment` with `mode='step'` for one-crease-per-step
- `env/paper_state.py` — `PaperState` tracks crease graph incrementally
- `env/graph.py` — `CreaseGraph` with vertex deduplication + edge splitting at intersections
- `env/rewards.py` — per-step reward: Kawasaki/Maekawa/BLB + progress + delta + efficiency
- `env/prompts.py` — step-level prompt showing current state + anchor points + last reward breakdown
- `env/verifier.py` — Kawasaki, Maekawa, BLB theorem checks (already ported to `training/reward.py`)

### V2 Action Format

Single crease per step (instead of complete FOLD JSON):

```json
{"from": [0.0, 0.5], "to": [1.0, 0.5], "assignment": "V"}
```

### V2 Episode Flow

```
reset(task_name="quarter_fold")
→ observation: task description + available anchor points + current state (empty)

step({"from": [0.5, 0], "to": [0.5, 1], "assignment": "V"})
→ observation: updated paper state + intermediate reward + available anchor points

step({"from": [0, 0.5], "to": [1, 0.5], "assignment": "V"})
→ observation: final shape + terminal reward, done=True
```

### V2 Reward (per-step, from optigami/env/rewards.py)

```python
total = (
    0.40 * progress          # fraction of target creases covered
    + 0.20 * delta           # improvement this step
    + 0.10 * kawasaki        # Kawasaki theorem compliance
    + 0.10 * maekawa         # Maekawa theorem compliance
    + 0.05 * blb             # BLB lemma compliance
    + 0.05 * economy         # penalty for excess creases
    + 0.05 * assignment_acc  # correct M/V types
    - 0.01 * step_penalty    # efficiency: finish in fewer steps
    + 10.0 * completion_bonus  # if progress > 0.9 and all geometry valid
)
```

This gives GRPO a gradient at every step, not just at the end.

### V2 Prompt (per-step, from optigami/env/prompts.py)

```
Target: quarter_fold — fold the paper into quarters

CURRENT STATE (step 1 of 5):
  Creases placed: none

AVAILABLE ANCHOR POINTS:
  Corners:      (0,0)  (1,0)  (1,1)  (0,1)
  Midpoints:    (0,0.5)  (0.5,0)  (1,0.5)  (0.5,1)

Output the NEXT crease as JSON:
{"from": [x1, y1], "to": [x2, y2], "assignment": "M" or "V"}
```

### V2 Implementation Plan

**Phase 1: New environment server (modify `origami_server/`)**

1. Add `PaperState` class to track crease graph across steps (port from `optigami/env/paper_state.py` + `graph.py`)
2. Modify `OrigamiAction` in `models.py` to accept single-crease format: `{"from": [...], "to": [...], "assignment": "M"|"V"}`
3. Modify `OrigamiEnvironment` in `environment.py` to:
   - Track `_paper_state: PaperState` between steps
   - Return `done=False` until max_folds reached or "stop" action
   - Compute per-step reward using the optigami reward formula
   - Include current crease state + available anchor points in observation
4. Keep backward compat: make single-step (complete FOLD JSON) mode still work as `mode='single'`

**Phase 2: Update training (modify `training/`)**

1. Update `train_grpo.py` prompt to step-level format (already in optigami)
2. Update `shape_match_reward` to accept the incremental observation — final shape only computed when `done=True`
3. Consider `max_folds` as a task parameter (e.g. triangle=1, quarter_fold=2, letter_fold=2)

**Phase 3: Add harder tasks**

From optigami's `server/tasks.py`, good candidates:
- `map_fold` — 8 folds, must be deployable (can unfold back flat)
- `waterbomb_base` — classic base requiring diagonal + perpendicular folds
- Custom tasks with `target_ratio` (compactness goals)

**Phase 4: Model upgrade**

optigami uses `Qwen2.5-VL-7B` (vision-language) — could let the model SEE a rendered view of the current paper state as part of the observation. This is the highest-ceiling path but requires significant extra work.

---

## Important Constraints

- **OpenEnv API**: `reset()` and `step()` must return types matching `OrigamiObservation`. The FastAPI server is generated by `create_app(OrigamiEnvironment, OrigamiAction, OrigamiObservation)`. Changing `OrigamiAction` shape requires updating models + server + client.
- **Modal image**: Adding new Python dependencies requires changing the `run_commands` block in `modal_train.py`. The image caches by content hash — changing deps triggers a full rebuild (~10 min).
- **Railway**: Env server auto-deploys from `main` branch. `Dockerfile` + `requirements.txt` must stay in root.
- **Unsloth quirk**: With `num_generations > per_device_train_batch_size`, Unsloth auto-bumps batch size. Keep `num_generations=4` (current default) to avoid 8×batch blowup.
- **Qwen3 thinking**: Always include `{"role": "system", "content": "/no_think"}` in prompts. Without it, `<think>` tokens fill the entire completion budget.

---

---

## HuggingFace Deployment

Two separate HF deployments are needed: the **env server** on HF Spaces, and the **trained model** on HF Hub.

### 1. Env Server → HF Spaces (Docker Space)

HF Spaces runs the `Dockerfile` automatically. The current `Dockerfile` is already compatible:
- Uses `${PORT:-8000}` — HF Spaces injects `PORT=7860` at runtime, so it auto-binds correctly
- No code changes needed to the server itself

**What needs to be added:**

`README.md` must have HF Spaces frontmatter (was stripped during Railway migration — needs to come back):

```yaml
---
title: Origami Env
emoji: 🦢
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---
```

**HF Spaces constraints to design around in V2:**

| Constraint | Impact |
|---|---|
| **Stateless** — container restarts wipe memory | No in-memory episode state. `OrigamiEnvironment` must be fully reconstructable from the session ID alone. This is already true for V1 (no cross-request state) but V2 multi-step will need to store `PaperState` per session somewhere (dict keyed by `episode_id`, or Redis). |
| **Free tier is CPU-only** | Simulation (`simulate.py`) is pure NumPy — fine on CPU. No GPU needed for the env server. |
| **No persistent disk** | Checkpoints live on Modal volume, not HF. The env server doesn't need checkpoints. |
| **Cold starts** | First request after inactivity spins up fresh. Health check endpoints (`/health`) are already present. |
| **`MAX_CONCURRENT_ENVS`** | Currently set to 16 in `Dockerfile`. On free-tier HF Spaces with limited RAM, lower this to 4-8 for V2 multi-step since each session will hold a `PaperState` object in memory. |

**V2-specific concern — session state for multi-step:**

V1 is stateless between steps (single-shot, `done=True` after step 1). V2 multi-step is NOT stateless — `PaperState` (the evolving crease graph) must persist across `reset()` → `step()` → `step()` calls within an episode.

OpenEnv's `create_app()` already handles concurrent sessions via `session_id`. The `OrigamiEnvironment` instance is kept alive per session. This works fine on a single container. On HF Spaces with auto-scaling or restarts, a session mid-episode would be dropped. For the hackathon / demo use case this is acceptable — just document that episodes are tied to a single container lifetime.

**Deployment steps:**

```bash
# 1. Add README.md frontmatter (see above)

# 2. Push to HF Space repo
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/origami-env
git push hf main

# 3. Verify health
curl https://YOUR_USERNAME-origami-env.hf.space/health

# 4. Update client.py base_url to HF Space URL
# Update server_url default in modal_train.py if using external server
```

---

### 2. Trained Model → HF Hub

After training on Modal, push the LoRA adapter (or merged model) to HF Hub so it's publicly usable.

**Option A: Push LoRA adapter only** (small, ~300MB, requires base model separately)

Add to the end of `training/train_grpo.py` after `model.save_pretrained(save_path)`:

```python
# Push to HF Hub
import os
hf_repo = os.environ.get("HF_REPO")  # e.g. "username/origami-qwen3-32b-lora"
if hf_repo:
    model.push_to_hub(hf_repo, token=os.environ["HF_TOKEN"])
    tokenizer.push_to_hub(hf_repo, token=os.environ["HF_TOKEN"])
    print(f"Model pushed to https://huggingface.co/{hf_repo}")
```

Add `HF_REPO` and `HF_TOKEN` to Modal secrets:
```bash
modal secret create huggingface HF_TOKEN=hf_xxx HF_REPO=username/origami-qwen3-32b-lora
```

Then reference the secret in `modal_train.py`:
```python
@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface")],  # add this
)
```

**Option B: Merge LoRA into base + push** (large, ~65GB, self-contained)

```python
# After training, merge and push
if USE_UNSLOTH:
    merged = model.merge_and_unload()
    merged.push_to_hub(hf_repo, token=os.environ["HF_TOKEN"])
```

For the demo use case, Option A is fine. Option B is only needed if users will run inference without the base model available.

**HF Hub model card:**

The pushed repo needs a `README.md` model card. Minimum viable:

```markdown
---
base_model: unsloth/Qwen3-32B
tags:
  - lora
  - origami
  - rl
  - grpo
license: apache-2.0
---

# Origami Qwen3-32B LoRA

LoRA adapter trained with GRPO on origami crease pattern generation.
Tasks: triangle, half_fold, quarter_fold, letter_fold.
```

---

### 3. Full Deployment Topology

```
┌─────────────────────┐     ┌─────────────────────┐
│   HF Spaces         │     │   HF Hub             │
│   (env server)      │     │   (trained model)    │
│   Docker + CPU      │     │   LoRA adapter       │
│   /health /reset    │     │   ~300MB             │
│   /step /tasks      │     └─────────────────────┘
└────────┬────────────┘              ▲
         │  WebSocket                │ push_to_hub()
         │  /ws                      │
         ▼                           │
┌─────────────────────┐     ┌────────┴────────────┐
│   Modal             │────▶│   Modal Volume       │
│   B200 training     │     │   origami-checkpoints│
│   GRPO + Unsloth    │     │   checkpoint-N/      │
└─────────────────────┘     └─────────────────────┘
```

---

## Environment Setup

```bash
# Install deps
pip install -r requirements.txt

# Start env server locally
uvicorn origami_server.app:app --host 0.0.0.0 --port 8000

# Run training locally (small model for testing)
python -m training.train_grpo --model unsloth/Qwen2.5-3B-Instruct --max_steps 50

# Deploy to Modal (B200)
modal run modal_train.py
```

## Quick Verification

```bash
# Check env server is healthy
curl http://localhost:8000/health

# Check tasks
curl http://localhost:8000/tasks

# Submit a fold manually
curl -X POST http://localhost:8000/reset -d '{"task_name": "triangle"}'
```
