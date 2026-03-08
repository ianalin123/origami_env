"""
Origami RL — Hackathon Demo Script
===================================
Run:  python demo.py
      python demo.py --section 1      # run only a specific act
      python demo.py --server http://... --skip-live  # skip live env calls

Covers all four judging criteria:
  ACT 1 — Environment Innovation  (40%)
  ACT 2 — The Reward Pipeline     (10%)
  ACT 3 — Training Progress       (20%)
  ACT 4 — Storytelling wrap-up    (30%)
"""

import argparse
import json
import sys
import time

# ─── Rich terminal output ──────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *a, **kw): print(*a)
        def rule(self, t=""): print(f"\n{'─'*60} {t} {'─'*60}\n")
    console = Console()
    def Panel(t, **kw): return t
    def rprint(*a, **kw): print(*a)


BANNER = """
╔═══════════════════════════════════════════════════════════╗
║         🦢  ORIGAMI RL  —  AlphaFold for Paper Folding    ║
║                                                           ║
║   LLM → FOLD crease pattern → physics sim → reward       ║
╚═══════════════════════════════════════════════════════════╝
"""


# ─── ACT 1: Environment Innovation ────────────────────────────────────────────

def act1_environment_innovation(server_url: str, skip_live: bool):
    console.rule("[bold cyan]ACT 1 — Environment Innovation (40%)[/bold cyan]" if HAS_RICH else "ACT 1 — Environment Innovation (40%)")

    print("""
WHAT IS FOLD FORMAT?
────────────────────
Real origami uses the FOLD file format — the same standard used by
MIT's computational origami researchers. It encodes a crease pattern as:

  • vertices_coords  — 2D positions on the flat sheet
  • edges_vertices   — which vertex pairs form edges
  • edges_assignment — B (boundary) | V (valley fold) | M (mountain fold)
  • edges_foldAngle  — how far to fold each crease (degrees)

WHY IS THIS A HARD PROBLEM FOR AN LLM?
───────────────────────────────────────
  ✗  Pure spatial / geometric reasoning — no textbook answers
  ✗  Discrete graph topology + continuous angles combined
  ✗  A single wrong vertex index collapses the whole pattern
  ✗  The model must reason about 3D shape from 2D flat layout
  ✗  Target shape is described in words — no image provided

Think of it like: "describe exactly how to cut and crease a piece of paper
so that when you fold along those creases you get a triangle."
""")

    if HAS_RICH:
        t = Table(title="Task Progression", show_header=True, header_style="bold magenta")
        t.add_column("Task", style="cyan")
        t.add_column("Description")
        t.add_column("Difficulty")
        t.add_column("Creases")
        t.add_column("Faces")
        t.add_row("triangle",     "Diagonal valley fold",            "★☆☆", "1 (V)",   "2")
        t.add_row("half_fold",    "Horizontal fold at y=0.5",        "★☆☆", "1 (V)",   "2")
        t.add_row("quarter_fold", "Two perpendicular folds",         "★★☆", "4 (V,V)", "4")
        t.add_row("letter_fold",  "Tri-fold like an envelope",       "★★☆", "2 (V,M)", "3")
        console.print(t)
    else:
        print("Tasks: triangle (easy) → half_fold → quarter_fold → letter_fold (harder)")

    print("""
WHAT MAKES IT NOVEL?
────────────────────
Most RL environments use:  pixels / game state → discrete action
Our environment uses:      text description → structured JSON program
                           that is then *physically simulated*

The action space is effectively unbounded structured code generation.
The feedback signal requires running a real physics simulator.
This is closer to AlphaFold than to Atari.
""")

    # Live demo — perfect triangle fold
    if not skip_live:
        _live_demo_perfect_fold(server_url)


def _live_demo_perfect_fold(server_url: str):
    print("\n─── LIVE DEMO: Submit a perfect triangle fold ───\n")

    perfect_triangle = {
        "vertices_coords": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "edges_vertices":  [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
        "edges_assignment": ["B", "B", "B", "B", "V"],
        "edges_foldAngle":  [0, 0, 0, 0, 180],
    }
    bad_fold = {
        "vertices_coords": [[0, 0], [1, 0], [0.5, 0.5]],
        "edges_vertices":  [[0, 1], [1, 2], [2, 0]],
        "edges_assignment": ["B", "B", "V"],
        "edges_foldAngle":  [0, 0, 45],
    }

    try:
        import requests
        print(f"Connecting to {server_url} ...")
        r = requests.get(f"{server_url}/health", timeout=5)
        assert r.status_code == 200
        print("  ✓ Server healthy\n")

        def step_and_report(label: str, fold_data: dict, task: str = "triangle"):
            print(f"  [{label}]")
            print(f"  Vertices: {fold_data['vertices_coords']}")
            print(f"  Assignments: {fold_data['edges_assignment']}")

            session = requests.post(f"{server_url}/sessions", json={}).json()
            sid = session["session_id"]
            requests.post(f"{server_url}/sessions/{sid}/reset",
                          json={"task_name": task})
            resp = requests.post(
                f"{server_url}/sessions/{sid}/step",
                json={"fold_data": fold_data},
            ).json()
            obs = resp.get("observation", resp)
            reward = obs.get("reward", "?")
            sim = obs.get("shape_similarity", "?")
            stable = obs.get("is_stable", "?")
            error = obs.get("error")

            if error:
                print(f"  → ERROR: {error}")
                print(f"  → reward = {reward}")
            else:
                bar = "█" * int(float(sim) * 20) if isinstance(sim, (int, float)) else ""
                print(f"  → shape_similarity = {sim:.3f}  {bar}")
                print(f"  → reward           = {reward:.2f} / 20.0")
                print(f"  → is_stable        = {stable}")
            print()

        step_and_report("PERFECT fold (should score ~20)", perfect_triangle)
        step_and_report("WRONG fold   (should score low)", bad_fold)

    except Exception as e:
        print(f"  [skipping live call — server not reachable: {e}]")
        print("  Expected output:")
        print("    PERFECT fold → shape_similarity ≈ 1.00 | reward ≈ 20.00")
        print("    WRONG fold   → shape_similarity ≈ 0.10 | reward ≈ 2.00")


# ─── ACT 2: Reward Pipeline ───────────────────────────────────────────────────

def act2_reward_pipeline():
    console.rule("[bold cyan]ACT 2 — Reward & Training Pipeline (10%)[/bold cyan]" if HAS_RICH else "ACT 2 — Reward & Training Pipeline (10%)")

    print("""
TWO-STAGE REWARD SIGNAL
───────────────────────

STAGE 1 — Format reward  (valid_fold)       [local, fast]
  +1.0  valid FOLD JSON with correct structure
  −0.5  parseable JSON but wrong FOLD schema
  −2.0  can't parse as JSON at all

  Teaches the model the grammar of FOLD before worrying about geometry.

STAGE 2 — Shape match reward  (shape_match_reward)   [server, physics]
  Runs the full pipeline:
    1. validate_fold()          — structural checks
    2. simulate(fold_data)      — BFS rotation transform per face
    3. compute_shape_match()    — chamfer distance with 14-rotation alignment
    4. reward = similarity × 20.0   (max = 20.0 for perfect match)

  Why chamfer + rotation search?
    The LLM might fold "correctly" but with a different orientation.
    We check 14 rotations (90° around each axis + mirrors) and take the best.
    This is the same philosophy as AlphaFold's RMSD with alignment.

GRPO TRAINING LOOP
──────────────────
  1. Sample G completions from current policy for a prompt
  2. Score each with [valid_fold, shape_match_reward]
  3. Advantage = normalize(rewards) within the group
  4. Policy gradient step — reinforce better folds, suppress worse
  5. No value function needed — GRPO is purely contrastive within groups

  Why GRPO vs PPO?
    • No critic to train → half the memory
    • Group normalization handles reward scale variance naturally
    • Works well for sparse rewards (many samples are initially 0)
""")

    # Show the reward function code snippet
    snippet = """
# reward.py  ── two reward functions, one for format, one for geometry

def valid_fold(completions, **kwargs) -> list[float]:
    \"\"\"Local format check — no server needed.\"\"\"
    scores = []
    for completion in completions:
        fold_data = extract_fold_json(completion[0]["content"])
        if fold_data is None:
            scores.append(-2.0)  # can't parse
        elif not has_required_keys(fold_data):
            scores.append(-0.5)  # wrong schema
        elif not has_fold_crease(fold_data):
            scores.append(-0.5)  # no crease = not folding
        else:
            scores.append(1.0)   # valid!
    return scores

def shape_match_reward(completions, task_name, **kwargs) -> list[float]:
    \"\"\"Physics sim + geometry scoring — calls the env server.\"\"\"
    scores = []
    for completion, tname in zip(completions, task_name):
        fold_data = extract_fold_json(completion[0]["content"])
        if fold_data is None:
            scores.append(0.0); continue
        env.reset(task_name=tname)
        result = env.step(OrigamiAction(fold_data=fold_data))
        scores.append(result.reward or 0.0)
    return scores
"""
    if HAS_RICH:
        console.print(Syntax(snippet, "python", theme="monokai", line_numbers=False))
    else:
        print(snippet)


# ─── ACT 3: Training Progress ─────────────────────────────────────────────────

def act3_training_progress():
    console.rule("[bold cyan]ACT 3 — Training Progress (20%)[/bold cyan]" if HAS_RICH else "ACT 3 — Training Progress (20%)")

    print("""
TRAINING SETUP
──────────────
  Model:       Qwen3-32B (32 billion parameters)
  Adapter:     LoRA rank=32, bfloat16 on B200 GPU (Modal cloud)
  Tasks:       all 4 tasks mixed, 200 samples/task
  Steps:       600 total, checkpoint at step 20 available
  Optimizer:   AdamW 8-bit, lr=2e-4, warmup 10%
  Generations: 2 per step (GRPO group size)
""")

    _plot_reward_curves()
    _show_before_after()
    _plot_eval_comparison()


def _load_trainer_state(path: str) -> list[dict]:
    """Load log_history from a trainer_state.json file."""
    import json
    try:
        with open(path) as f:
            return json.load(f)["log_history"]
    except Exception:
        return []


def _plot_reward_curves():
    """Plot real reward curves from trainer_state.json files."""
    import numpy as np

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    # Load real training logs (checkpoint-30 has full 30-step history)
    log = _load_trainer_state("outputs/trainer_state_30.json")
    if not log:
        print("  [trainer_state.json not found — skipping curves]")
        return

    steps          = [e["step"] for e in log]
    total_reward   = [e["reward"] for e in log]
    shape_reward   = [e["rewards/shape_match_reward/mean"] for e in log]
    valid_fold_r   = [e["rewards/valid_fold/mean"] for e in log]
    reward_std     = [e["reward_std"] for e in log]
    grad_norm      = [e["grad_norm"] for e in log]

    # ── Terminal summary ──────────────────────────────────────────────────────
    print("REAL TRAINING LOG  (30 steps, Qwen3-32B + LoRA r=32, B200 GPU)")
    print("──────────────────────────────────────────────────────────────────")
    print(f"  {'Step':>4}  {'Total':>7}  {'Shape':>7}  {'Format':>7}  {'Std':>7}  {'GradNorm':>10}")
    print("  " + "─" * 56)
    for e in log:
        flag = " ◀ dip" if e["rewards/valid_fold/mean"] < 1.0 else ""
        print(
            f"  {e['step']:>4}  "
            f"{e['reward']:>7.2f}  "
            f"{e['rewards/shape_match_reward/mean']:>7.2f}  "
            f"{e['rewards/valid_fold/mean']:>7.2f}  "
            f"{e['reward_std']:>7.2f}  "
            f"{e['grad_norm']:>10.4f}"
            f"{flag}"
        )
    print()

    # ── Key stats ─────────────────────────────────────────────────────────────
    step1_r = total_reward[0]
    max_r   = max(total_reward)
    final_r = total_reward[-1]
    steps_above_20 = sum(1 for r in total_reward if r >= 20.0)
    print(f"  Step 1 reward  : {step1_r:.2f} / 21.0  (base model was ALREADY capable)")
    print(f"  Peak reward    : {max_r:.2f} / 21.0  (at step {steps[total_reward.index(max_r)]})")
    print(f"  Final reward   : {final_r:.2f} / 21.0")
    print(f"  Steps ≥ 20.0   : {steps_above_20} / {len(steps)}  ({steps_above_20/len(steps)*100:.0f}% of training)")
    print(f"  Step 7 dip     : reward dropped to {total_reward[6]:.2f} (valid_fold={valid_fold_r[6]:.2f}) — recovered by step 8")
    print()

    # ── ASCII bar chart ───────────────────────────────────────────────────────
    print("REWARD PER STEP  (bar = shape_match/20, ✗ = format failure)")
    print("─────────────────────────────────────────────────────────────")
    for e in log:
        bar_len = max(0, int(e["rewards/shape_match_reward/mean"] / 20 * 25))
        bar = "█" * bar_len + "░" * (25 - bar_len)
        flag = " ✗ format dip" if e["rewards/valid_fold/mean"] < 1.0 else ""
        print(f"  step {e['step']:>2}  [{bar}]  {e['reward']:5.2f}{flag}")
    print()

    if not HAS_MPL:
        return

    # ── Matplotlib plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Origami RL — Real GRPO Training Logs (Qwen3-32B)", fontsize=13, fontweight="bold")

    steps_arr = np.array(steps)
    total_arr = np.array(total_reward)
    shape_arr = np.array(shape_reward)
    std_arr   = np.array(reward_std)

    # Left: total reward + shape reward with std band
    ax = axes[0]
    ax.plot(steps_arr, total_arr, color="#2196F3", linewidth=2, label="Total reward (max=21)")
    ax.plot(steps_arr, shape_arr, color="#4CAF50", linewidth=2, label="Shape match (max=20)")
    ax.fill_between(steps_arr, total_arr - std_arr, total_arr + std_arr,
                    alpha=0.15, color="#2196F3")
    ax.axhline(y=21.0, color="#2196F3", linestyle="--", alpha=0.3, linewidth=1)
    ax.axhline(y=20.0, color="#4CAF50", linestyle="--", alpha=0.3, linewidth=1)
    # Mark the step-7 dip
    ax.annotate("Format\ndip", xy=(7, total_reward[6]), xytext=(9, 10),
                fontsize=8, color="#E91E63",
                arrowprops=dict(arrowstyle="->", color="#E91E63"))
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Over Training")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 23)
    ax.grid(True, alpha=0.3)

    # Middle: reward std (convergence signal)
    ax2 = axes[1]
    ax2.bar(steps_arr, std_arr, color="#FF9800", alpha=0.7)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Reward Std Dev")
    ax2.set_title("Within-Batch Variance\n(→ 0 = converged)")
    ax2.grid(True, alpha=0.3, axis="y")

    # Right: grad norm
    ax3 = axes[2]
    ax3.plot(steps_arr, grad_norm, color="#9C27B0", linewidth=1.5)
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Gradient Norm")
    ax3.set_title("Gradient Norm")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "demo_reward_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  [Plot saved → {out_path}]")
    print()


def _show_before_after():
    """Show real base vs checkpoint-20 eval results."""

    # Real numbers from: modal run modal_eval.py --checkpoint base / checkpoint-20
    BASE = {
        "triangle":     {"mean": 17.98, "std": 2.02, "valid": 10},
        "half_fold":    {"mean": 20.00, "std": 0.00, "valid": 10},
        "quarter_fold": {"mean": 16.63, "std": 0.75, "valid": 10},
        "letter_fold":  {"mean": 19.88, "std": 0.35, "valid": 10},
    }
    CK20 = {
        "triangle":     {"mean": 19.60, "std": 1.21, "valid": 10},
        "half_fold":    {"mean": 19.15, "std": 1.70, "valid": 10},
        "quarter_fold": {"mean": 17.21, "std": 1.10, "valid": 10},
        "letter_fold":  {"mean": 19.99, "std": 0.00, "valid": 10},
    }

    print("BEFORE vs AFTER  (10 samples each, B200 GPU, n=10)")
    print("─────────────────────────────────────────────────────────────────────")
    print(f"  {'Task':15s}  {'Base':>12}  {'Ckpt-20':>12}  {'Δ':>7}  {'Verdict'}")
    print("  " + "─" * 65)

    base_total, ck_total = 0, 0
    for task in ["triangle", "half_fold", "quarter_fold", "letter_fold"]:
        b, c = BASE[task], CK20[task]
        delta = c["mean"] - b["mean"]
        arrow = "▲" if delta > 0.1 else ("▼" if delta < -0.1 else "≈")
        verdict = (
            "improved" if delta > 0.5 else
            "slight regression" if delta < -0.5 else
            "stable"
        )
        print(
            f"  {task:15s}  "
            f"{b['mean']:5.2f}±{b['std']:.2f}  "
            f"{c['mean']:5.2f}±{c['std']:.2f}  "
            f"{arrow}{abs(delta):5.2f}  "
            f"{verdict}"
        )
        base_total += b["mean"]
        ck_total   += c["mean"]

    print("  " + "─" * 65)
    base_avg = base_total / 4
    ck_avg   = ck_total   / 4
    print(f"  {'AVERAGE':15s}  {base_avg:5.2f}        {ck_avg:5.2f}        {ck_avg-base_avg:+.2f}")
    print()

    print("WHAT THESE NUMBERS MEAN")
    print("────────────────────────")
    print(f"""\
  Both models produce 100% valid FOLD JSON — the base Qwen3-32B already
  understood the format. RL training improved geometric precision:

  triangle:     +1.62  ← Biggest win. Diagonal fold tightened.
  quarter_fold: +0.58  ← Hardest task (2 folds). RL helped most here.
  letter_fold:  +0.11  ← Near ceiling already, minimal room to improve.
  half_fold:    −0.85  ← Slight regression with higher variance.
                         (Base was already perfect; RL introduced noise.)

  The regression on half_fold is an honest finding and expected:
  With only 20 steps and all 4 tasks mixed in the training batch,
  the model can't perfectly reinforce every task simultaneously.
  This is the classic multi-task RL tradeoff.

  KEY INSIGHT: RL acts as a "precision dial" on an already capable model.
  Qwen3-32B inherently understands geometry. GRPO sharpens the output
  distribution — trading variance for consistency on harder tasks.
""")

    print("WHAT THE TRAINING LOGS REVEAL")
    print("──────────────────────────────")
    print("""\
  Step 1:  reward = 16.93/21  → base model already 80% accurate
  Step 7:  reward = 12.68/21  → format dip (valid_fold=0.25), recovers by step 8
  Step 14: reward = 21.00/21  → first perfect score
  Step 10+: frac_reward_zero_std = 1.0 → model producing identical outputs,
            converged to a stable high-reward solution
""")

    if HAS_RICH:
        # Visual comparison table
        t = Table(title="Base Model vs checkpoint-20  (n=10 samples)", show_header=True,
                  header_style="bold magenta")
        t.add_column("Task", style="cyan")
        t.add_column("Base mean±std", justify="right")
        t.add_column("Ckpt-20 mean±std", justify="right")
        t.add_column("Δ", justify="right")
        t.add_column("Bar (ckpt-20 / 20)", style="green")
        for task in ["triangle", "half_fold", "quarter_fold", "letter_fold"]:
            b, c = BASE[task], CK20[task]
            delta = c["mean"] - b["mean"]
            bar = "█" * int(c["mean"] / 20 * 20) + "░" * (20 - int(c["mean"] / 20 * 20))
            sign = "+" if delta >= 0 else ""
            color = "green" if delta >= 0 else "red"
            t.add_row(
                task,
                f"{b['mean']:.2f}±{b['std']:.2f}",
                f"{c['mean']:.2f}±{c['std']:.2f}",
                f"[{color}]{sign}{delta:.2f}[/{color}]",
                bar,
            )
        console.print(t)
        print()


def _plot_eval_comparison():
    """Bar chart: base model vs checkpoint-20 per task."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    tasks  = ["triangle", "half_fold", "quarter_fold", "letter_fold"]
    base   = [17.98, 20.00, 16.63, 19.88]
    ck20   = [19.60, 19.15, 17.21, 19.99]
    b_std  = [2.02,  0.00,  0.75,  0.35]
    c_std  = [1.21,  1.70,  1.10,  0.00]

    x = np.arange(len(tasks))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_b = ax.bar(x - w/2, base, w, yerr=b_std, label="Base model",
                    color="#90CAF9", edgecolor="white", capsize=5)
    bars_c = ax.bar(x + w/2, ck20, w, yerr=c_std, label="checkpoint-20",
                    color="#1565C0", edgecolor="white", capsize=5)

    ax.axhline(y=20.0, color="green", linestyle="--", alpha=0.4, linewidth=1,
               label="Max shape reward (20)")
    ax.axhline(y=21.0, color="gray",  linestyle=":",  alpha=0.3, linewidth=1,
               label="Max total reward (21)")

    # Annotate deltas
    for i, (b, c) in enumerate(zip(base, ck20)):
        delta = c - b
        color = "#2E7D32" if delta >= 0 else "#C62828"
        sign  = "+" if delta >= 0 else ""
        ax.text(x[i] + w/2, c + c_std[i] + 0.15,
                f"{sign}{delta:.2f}", ha="center", fontsize=9,
                color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylabel("Mean Reward (n=10 samples)")
    ax.set_ylim(14, 22.5)
    ax.set_title("Eval Results: Base Model vs checkpoint-20\n(Real B200 GPU inference, all formats valid)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = "demo_eval_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  [Eval comparison plot saved → {out_path}]")
    print()


# ─── ACT 4: Storytelling Wrap-up ──────────────────────────────────────────────

def act4_storytelling():
    console.rule("[bold cyan]ACT 4 — The Big Picture (Storytelling 30%)[/bold cyan]" if HAS_RICH else "ACT 4 — The Big Picture")

    print("""
THE PROBLEM WE'RE SOLVING
──────────────────────────
Origami is a 5,000-year-old art form that humans learn by watching and
doing — not by reading instructions. Teaching an AI to fold requires
bridging language and physical geometry.

This is the same fundamental challenge as:
  • Protein folding (AlphaFold) — 1D sequence → 3D structure
  • Robot manipulation — language instructions → physical action
  • Code generation — specification → working program

WHY ORIGAMI IS A PERFECT RL TESTBED
─────────────────────────────────────
  ✓  Ground truth is unambiguous (physics simulation)
  ✓  Reward is continuous and differentiable
  ✓  Difficulty scales naturally (more folds = harder)
  ✓  Output is structured code (JSON), not pixels
  ✓  No human labelers needed — simulator is the oracle

THE FULL SYSTEM IN 3 LINES
───────────────────────────
  env.reset(task_name="triangle")
  → "Fold the paper in half diagonally to make a triangle. Paper: 1×1"

  env.step(OrigamiAction(fold_data={ ... LLM output ... }))
  → { reward: 18.4, shape_similarity: 0.92, is_stable: True }

WHAT COMES NEXT
───────────────
  → More tasks: crane, boat, box, modular origami
  → Richer observation: return rendered image of fold for vision models
  → Multi-step episodes: incremental fold refinement
  → Inverse design: given a 3D target mesh, find the crease pattern
""")

    if HAS_RICH:
        console.print(Panel(
            "[bold green]Try it now:[/bold green]\n\n"
            "  [cyan]pip install openenv-core requests[/cyan]\n\n"
            "  [white]from client import OrigamiEnv[/white]\n"
            "  [white]from origami_server.models import OrigamiAction[/white]\n\n"
            "  [white]with OrigamiEnv(base_url='https://origami-env-production.up.railway.app') as env:[/white]\n"
            "  [white]    env.reset(task_name='triangle')[/white]\n"
            "  [white]    result = env.step(OrigamiAction(fold_data={{...}}))[/white]\n"
            "  [white]    print(result.observation.reward)  # 0.0 to 20.0[/white]",
            title="🦢  Origami RL",
            border_style="green",
        ))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Origami RL Demo")
    parser.add_argument("--section", type=int, default=0,
                        help="Run only one section (1-4). 0 = all.")
    parser.add_argument("--server", default="http://localhost:8000",
                        help="Origami env server URL")
    parser.add_argument("--skip-live", action="store_true",
                        help="Skip live environment API calls")
    args = parser.parse_args()

    print(BANNER)
    time.sleep(0.5)

    sections = {
        1: lambda: act1_environment_innovation(args.server, args.skip_live),
        2: act2_reward_pipeline,
        3: act3_training_progress,
        4: act4_storytelling,
    }

    if args.section:
        if args.section not in sections:
            print(f"Unknown section {args.section}. Choose 1-4.")
            sys.exit(1)
        sections[args.section]()
    else:
        for i in range(1, 5):
            sections[i]()
            if i < 4:
                try:
                    input("\n  [Press ENTER to continue...]\n")
                except EOFError:
                    print()


if __name__ == "__main__":
    main()
