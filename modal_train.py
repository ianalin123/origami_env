"""Modal training script for origami GRPO.

Run:
    modal run modal_train.py                              # 600 steps, triangle
    modal run modal_train.py --task half_fold             # different task
    modal run modal_train.py --max-steps 1200 --resume    # extend a previous run

Checkpoints persist in a Modal volume named 'origami-checkpoints'.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

# ── Constants ──────────────────────────────────────────────────────────────────

APP_NAME = "origami-grpo"
GPU = "B200"
TIMEOUT = 4 * 3600       # 4h hard ceiling
OUTPUTS_DIR = "/outputs"  # mount point for the persistent volume

# B200 defaults — 192GB HBM3e means no quantization needed
# Qwen2.5-72B bfloat16 uses ~144GB, fits comfortably
B200_MODEL = "unsloth/Qwen3-32B"
B200_LORA_RANK = 32

# ── App + volume ───────────────────────────────────────────────────────────────

app = modal.App(APP_NAME)

volume = modal.Volume.from_name("origami-checkpoints", create_if_missing=True)

# ── Image ──────────────────────────────────────────────────────────────────────
# Build order matters for layer caching:
#   1. System deps
#   2. PyTorch (slow, rarely changes)
#   3. ML stack pinned to notebook versions
#   4. Unsloth
#   5. Env server deps
#   6. PYTHONPATH env var     ← must come before add_local_dir
#   7. Repo code (changes most often — must be last)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        # PyTorch with CUDA 12.8 — required for B200/Blackwell (sm_100)
        "pip install -q torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu128",
        # Pin to notebook versions so notebook + script are identical
        "pip install -q 'transformers==4.56.2' tokenizers 'trl==0.22.2' "
        "'datasets>=3.0' 'accelerate>=0.30,<2' 'peft>=0.10,<2' "
        "'bitsandbytes>=0.43' 'triton>=3.4.0'",
        # Unsloth: auto-detects Blackwell when cu128 torch is installed.
        # If this fails, use the explicit wheel, e.g.:
        #   pip install 'unsloth[cu128-blackwell-torch270]'
        "pip install -q unsloth unsloth_zoo",
        # Env server
        "pip install -q fastapi uvicorn requests numpy scipy pydantic 'openenv-core[core]>=0.2.1'",
    )
    .env({"PYTHONPATH": "/app"})            # must precede add_local_dir
    .add_local_dir(                         # copy=True bakes into image layer
        ".",
        remote_path="/app",
        copy=True,
        ignore=[".git", "__pycache__", "**/__pycache__", "*.pyc",
                ".pytest_cache", "outputs", "*.egg-info", ".venv", "venv"],
    )
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _wait_for_server(url: str, timeout_s: int = 45) -> None:
    import requests as req
    for i in range(timeout_s):
        try:
            if req.get(url, timeout=2).status_code == 200:
                print(f"Env server ready after {i + 1}s")
                return
        except Exception:
            pass
        time.sleep(1)  # sleep every iteration, not just on exception
    raise RuntimeError(f"Env server did not become healthy within {timeout_s}s")


def _latest_checkpoint(output_dir: str) -> str | None:
    ckpts = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


def _checkpoint_step(ckpt_path: str) -> int:
    return int(Path(ckpt_path).name.split("-")[-1])


# ── Training function ──────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: volume},
)
def train(
    task: str = "all",
    max_steps: int = 600,
    num_generations: int = 4,
    model: str = B200_MODEL,
    lora_rank: int = B200_LORA_RANK,
    load_in_4bit: bool = False,
    server_url: str = "",
    resume: bool = False,
):
    os.environ["OUTPUT_DIR"] = OUTPUTS_DIR

    # ── Env server — use external URL if provided, otherwise start locally ─────
    server_proc = None
    if not server_url:
        server_url = "http://localhost:8000"
        server_proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "origami_server.app:app",
                "--host", "0.0.0.0", "--port", "8000",
            ],
            cwd="/app",
        )
    else:
        print(f"Using external env server: {server_url}")

    try:
        # health check inside try so server_proc is always cleaned up
        _wait_for_server(f"{server_url}/health")
        # ── Checkpoint status ──────────────────────────────────────────────────
        ckpt = _latest_checkpoint(OUTPUTS_DIR)

        if resume:
            if ckpt is None:
                print("No checkpoint found in volume — starting from scratch.")
            else:
                step = _checkpoint_step(ckpt)
                remaining = max_steps - step
                if remaining <= 0:
                    raise ValueError(
                        f"max_steps={max_steps} but checkpoint is already at step {step}. "
                        f"Pass --max-steps {step + 600} (or higher) to continue."
                    )
                print(f"Resuming from {Path(ckpt).name} "
                      f"(step {step} → {max_steps}, {remaining} steps remaining)")
        else:
            if ckpt is not None:
                print(f"NOTE: checkpoint {Path(ckpt).name} exists but --resume not set. "
                      f"Starting fresh. Pass --resume to continue from that checkpoint.")

        # ── Run training ───────────────────────────────────────────────────────
        cmd = [
            sys.executable, "-m", "training.train_grpo",
            "--task", task,
            "--max_steps", str(max_steps),
            "--num_generations", str(num_generations),
            "--model", model,
            "--lora_rank", str(lora_rank),
            "--server", server_url,
        ]
        if load_in_4bit:
            cmd.append("--load_in_4bit")
        else:
            cmd.append("--no-load_in_4bit")
        if resume and ckpt:
            cmd.append("--resume")

        subprocess.run(cmd, cwd="/app", check=True)

    finally:
        if server_proc is not None:
            server_proc.terminate()
        volume.commit()
        print("Checkpoints committed to volume 'origami-checkpoints'.")


# ── Local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    task: str = "all",
    max_steps: int = 600,
    num_generations: int = 4,
    model: str = B200_MODEL,
    lora_rank: int = B200_LORA_RANK,
    load_in_4bit: bool = False,
    server_url: str = "",
    resume: bool = False,
):
    train.remote(
        task=task,
        max_steps=max_steps,
        num_generations=num_generations,
        model=model,
        lora_rank=lora_rank,
        load_in_4bit=load_in_4bit,
        server_url=server_url,
        resume=resume,
    )
