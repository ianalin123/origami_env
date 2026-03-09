"""Modal V3 training: multi-step GRPO with GiGPO advantage estimation.

Run:
    modal run modal_train_v3.py                          # full 1500 steps, curriculum
    modal run modal_train_v3.py --max-steps 50           # smoke test
    modal run modal_train_v3.py --tasks triangle         # single task
    modal run modal_train_v3.py --resume                 # resume from latest checkpoint
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "origami-v3-train"
GPU = "B200"
TIMEOUT = 6 * 3600  # 6h ceiling
OUTPUTS_DIR = "/outputs"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("origami-checkpoints-v3", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "pip install -q torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu128",
        "pip install -q 'transformers==4.56.2' tokenizers 'trl==0.22.2' "
        "'datasets>=3.0' 'accelerate>=0.30,<2' 'peft>=0.10,<2' "
        "'bitsandbytes>=0.43' 'triton>=3.4.0'",
        "pip install -q unsloth unsloth_zoo",
        "pip install -q numpy scipy shapely pydantic 'openenv-core[core]>=0.2.1'",
    )
    .env({"PYTHONPATH": "/app", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        ".",
        remote_path="/app",
        copy=True,
        ignore=[".git", "__pycache__", "**/__pycache__", "*.pyc",
                ".pytest_cache", "outputs", "*.egg-info", ".venv", "venv"],
    )
)


def _latest_checkpoint(output_dir: str) -> str | None:
    ckpts = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: volume},
)
def train(
    max_steps: int = 1500,
    batch_size: int = 8,
    tasks: str = "auto",
    model_name: str = "unsloth/Qwen3-32B",
    lora_rank: int = 32,
    load_in_4bit: bool = False,
    resume: bool = False,
    lr: float = 5e-5,
    noise_scale: float = 1.5,
    temperature: float = 1.5,
):
    os.environ["OUTPUT_DIR"] = OUTPUTS_DIR

    try:
        ckpt = _latest_checkpoint(OUTPUTS_DIR) if resume else None

        cmd = [
            sys.executable, "-u", "-m", "training.train_v3",
            "--model", model_name,
            "--max-steps", str(max_steps),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--lora-rank", str(lora_rank),
            "--noise-scale", str(noise_scale),
            "--temperature", str(temperature),
            "--tasks", tasks,
            "--output-dir", OUTPUTS_DIR,
            "--save-steps", "50",
            "--log-steps", "5",
        ]

        if load_in_4bit:
            cmd.append("--load-in-4bit")
        else:
            cmd.append("--no-load-in-4bit")

        if ckpt:
            cmd.extend(["--resume", ckpt])
            print(f"Resuming from {Path(ckpt).name}")

        subprocess.run(cmd, cwd="/app", check=True)

    finally:
        volume.commit()
        print("Checkpoints committed to volume 'origami-checkpoints-v3'.")


@app.local_entrypoint()
def main(
    max_steps: int = 1500,
    batch_size: int = 8,
    tasks: str = "auto",
    model: str = "unsloth/Qwen3-32B",
    lora_rank: int = 32,
    load_in_4bit: bool = False,
    resume: bool = False,
    lr: float = 5e-5,
    noise_scale: float = 1.5,
    temperature: float = 1.5,
):
    train.remote(
        max_steps=max_steps,
        batch_size=batch_size,
        tasks=tasks,
        model_name=model,
        lora_rank=lora_rank,
        load_in_4bit=load_in_4bit,
        resume=resume,
        lr=lr,
        noise_scale=noise_scale,
        temperature=temperature,
    )
