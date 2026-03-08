"""GRPO training script for origami RL.

Follows the OpenEnv 2048 pattern exactly:
- Environment runs as a FastAPI server (origami_server.app)
- Training connects via WebSocket client (OrigamiEnv)
- Reward functions call the server, never import engine code
- GRPOTrainer from TRL handles the RL loop

Usage:
    # 1. Start the environment server first:
    uvicorn origami_server.app:app --host 0.0.0.0 --port 8000

    # 2. Run training (connects to server):
    python -m training.train_grpo --task triangle --max_steps 600

    # Or specify server URL:
    python -m training.train_grpo --server http://gpu-host:8000
"""

import argparse
import functools
import os

import requests

PROMPT_TEMPLATE = """You are an origami designer. Generate a FOLD-format crease pattern
that, when folded, produces the target shape described below.

Target: {description}
Paper size: {width} x {height}

Output a JSON object with these exact fields:
- vertices_coords: [[x, y], ...] — 2D positions on the flat paper (0 to {width} for x, 0 to {height} for y)
- edges_vertices: [[v1, v2], ...] — pairs of vertex indices forming edges
- edges_assignment: ["B"|"M"|"V", ...] — B=boundary, M=mountain fold, V=valley fold
- edges_foldAngle: [angle, ...] — fold angles in degrees (M: negative like -180, V: positive like 180, B: 0)

Rules:
- Boundary edges (B) must outline the paper rectangle
- At least one fold crease (M or V) must exist
- Mountain fold angles are negative (-180 to 0)
- Valley fold angles are positive (0 to 180)
- All vertex indices in edges must be valid (0 to N-1)

Output ONLY the JSON object wrapped in ```json ... ``` markers."""


def build_prompt(task: dict) -> str:
    return PROMPT_TEMPLATE.format(
        description=task["description"],
        width=task["paper"]["width"],
        height=task["paper"]["height"],
    )


def main():
    parser = argparse.ArgumentParser(description="GRPO training for origami RL")
    parser.add_argument(
        "--task", default="all",
        help="Comma-separated task names, or 'all' for all tasks",
    )
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument(
        "--load_in_4bit", action=argparse.BooleanOptionalAction, default=True,
        help="4-bit quantization (--load_in_4bit / --no-load_in_4bit). "
             "Disable on B200/H100 where full bfloat16 fits in VRAM.",
    )
    parser.add_argument(
        "--server", default="http://localhost:8000",
        help="URL of the origami environment server",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume from latest checkpoint in OUTPUT_DIR",
    )
    args = parser.parse_args()

    # --- Verify server is running ---
    print(f"Connecting to environment server at {args.server}...")
    try:
        r = requests.get(f"{args.server}/health", timeout=5)
        assert r.status_code == 200
        print("Server is healthy.")
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {args.server}")
        print(f"Start it first: uvicorn origami_server.app:app --port 8000")
        raise SystemExit(1)

    # --- Get task info from server ---
    ALL_TASKS = ["triangle", "half_fold", "quarter_fold", "letter_fold"]
    task_names = ALL_TASKS if args.task == "all" else [t.strip() for t in args.task.split(",")]
    tasks = {}
    for name in task_names:
        tasks[name] = requests.get(f"{args.server}/tasks/{name}").json()
        print(f"Task: {tasks[name]['name']} — {tasks[name]['description']}")

    # --- Configure reward functions (OpenEnv pattern) ---
    from client import OrigamiEnv
    from origami_server.models import OrigamiAction
    from training.reward import extract_fold_json, valid_fold
    from unsloth import is_port_open, launch_openenv

    global port, openenv_process
    from urllib.parse import urlparse as _urlparse
    _parsed = _urlparse(args.server)
    port = _parsed.port or (443 if _parsed.scheme == "https" else 8000)
    openenv_process = None

    launch_openenv = functools.partial(
        launch_openenv,
        working_directory=os.getcwd(),
        server="origami_server.app:app",
        environment={**os.environ, "PYTHONPATH": os.getcwd()},
        openenv_class=OrigamiEnv,
    )

    def shape_match_reward(completions, task_name, **kwargs):
        global port, openenv_process
        scores = []
        for completion, tname in zip(completions, task_name):
            response = completion[0]["content"]
            fold_data = extract_fold_json(response)
            if fold_data is None:
                scores.append(0.0)
                continue
            try:
                port, openenv_process = launch_openenv(port, openenv_process)
                openenv_process.reset(task_name=tname)
                result = openenv_process.step(OrigamiAction(fold_data=fold_data))
                scores.append(result.reward if result.reward is not None else 0.0)
            except TimeoutError:
                scores.append(-1.0)
            except Exception:
                scores.append(-2.0)
        return scores

    # --- Build dataset (same prompt repeated, like 2048) ---
    from datasets import Dataset

    # Mix all tasks evenly. task_name column is passed to reward functions by TRL.
    # /no_think disables Qwen3 chain-of-thought so completions are pure JSON.
    samples_per_task = 200
    rows = []
    for tname, tinfo in tasks.items():
        prompt_text = build_prompt(tinfo)
        rows.extend([{
            "prompt": [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt_text},
            ],
            "task_name": tname,
        }] * samples_per_task)
    dataset = Dataset.from_list(rows)

    # --- Load model with QLoRA ---
    try:
        from unsloth import FastLanguageModel
        USE_UNSLOTH = True
    except ImportError:
        USE_UNSLOTH = False

    max_seq_length = 1024  # no thinking mode; JSON output is ~150-300 tokens

    quant_label = "4-bit QLoRA" if args.load_in_4bit else "bfloat16 LoRA"
    if USE_UNSLOTH:
        print(f"Loading {args.model} with Unsloth {quant_label} (rank={args.lora_rank})...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            load_in_4bit=args.load_in_4bit,
            max_seq_length=max_seq_length,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.lora_rank * 2,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    else:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if (args.load_in_4bit and torch.cuda.is_available()) else None

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.print_trainable_parameters()

    # --- GRPO config (matches 2048 pattern) ---
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=args.lr,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_prompt_length=512,
        max_completion_length=max_seq_length - 512,
        max_steps=args.max_steps,
        save_steps=10,
        output_dir=os.environ.get("OUTPUT_DIR", "outputs"),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[valid_fold, shape_match_reward],
        args=training_args,
        train_dataset=dataset,
    )

    resume_from = args.resume or None  # True → latest ckpt, None → scratch
    print(f"Training: {args.max_steps} steps, {args.num_generations} generations/step"
          + (" (resuming)" if resume_from else ""))
    trainer.train(resume_from_checkpoint=resume_from)

    # Save LoRA adapter
    run_label = args.task.replace(",", "-").replace(" ", "")
    save_path = os.path.join(
        os.environ.get("OUTPUT_DIR", "outputs"),
        f"origami-{run_label}-lora-final",
    )
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
