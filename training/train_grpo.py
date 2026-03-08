"""GRPO training script for origami RL.

Follows the 2048 OpenEnv + Unsloth pattern:
- LLM generates FOLD JSON crease patterns
- Two reward functions: valid_fold + shape_match
- GRPOTrainer from TRL handles the RL loop

Usage (Colab):
    python -m origami_env.training.train_grpo --task triangle --max_steps 600
"""

import argparse

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
    parser.add_argument("--task", default="triangle", help="Task name")
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--model", default="unsloth/gpt-oss-20b")
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    # --- These imports are heavy, only load when actually training ---
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    from server.tasks import get_task
    from training.reward import shape_match, valid_fold

    task = get_task(args.task)
    prompt_text = build_prompt(task)

    # Build dataset (1000 copies of same prompt, like 2048)
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "answer": 0,
            }
        ]
        * 1000
    )

    # Load model with LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        load_in_4bit=True,
        max_seq_length=2048,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
    )

    # Wrap shape_match to inject task_name
    def shape_match_reward(completions, **kwargs):
        return shape_match(completions, task_name=args.task, **kwargs)

    # GRPO config
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
        max_prompt_length=1024,
        max_completion_length=1024,
        max_steps=args.max_steps,
        save_steps=100,
        output_dir="outputs",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[valid_fold, shape_match_reward],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
