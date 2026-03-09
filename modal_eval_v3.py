"""Modal V3 evaluation: multi-step episode rollouts.

Run:
    modal run modal_eval_v3.py                                    # latest checkpoint, all tasks
    modal run modal_eval_v3.py --checkpoint checkpoint-100        # specific checkpoint
    modal run modal_eval_v3.py --checkpoint base                  # base model (no LoRA)
    modal run modal_eval_v3.py --n-episodes 20 --tasks triangle,quarter_fold
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

from modal_train_v3 import OUTPUTS_DIR, app, image, volume

ALL_TASKS = ["triangle", "half_fold", "quarter_fold", "letter_fold", "waterbomb_base", "map_fold"]


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={OUTPUTS_DIR: volume},
)
def evaluate(
    checkpoint: str = "",
    n_episodes: int = 20,
    tasks: str = "all",
    model_name: str = "unsloth/Qwen3-32B",
):
    import numpy as np
    import torch

    from origami_server.environment import OrigamiEnvironment
    from origami_server.models import OrigamiAction
    from origami_server.tasks import get_task
    from training.prompt_builder import build_prompt_from_obs
    from training.reward import extract_crease_json
    from unsloth import FastLanguageModel

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    if checkpoint == "base":
        ckpt_path = None
        print("Evaluating base model (no LoRA)")
    elif checkpoint:
        ckpt_path = str(Path(OUTPUTS_DIR) / checkpoint)
        print(f"Evaluating checkpoint: {checkpoint}")
    else:
        ckpts = sorted(
            Path(OUTPUTS_DIR).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        finals = list(Path(OUTPUTS_DIR).glob("*-lora-final"))
        if ckpts:
            ckpt_path = str(ckpts[-1])
            print(f"Using latest checkpoint: {Path(ckpt_path).name}")
        elif finals:
            ckpt_path = str(finals[-1])
            print(f"Using: {Path(ckpt_path).name}")
        else:
            raise ValueError("No checkpoint found. Pass --checkpoint base to eval base model.")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,
        max_seq_length=1024,
    )
    if ckpt_path:
        model.load_adapter(ckpt_path)
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate_single(prompt: str) -> str:
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

    # ── Evaluate each task ────────────────────────────────────────────────────
    task_list = ALL_TASKS if tasks == "all" else [t.strip() for t in tasks.split(",")]
    all_results = {}

    for task_name in task_list:
        task_info = get_task(task_name)
        max_folds = task_info.get("max_folds", 1)
        episodes = []

        for ep in range(n_episodes):
            env = OrigamiEnvironment(mode="step")
            obs = env.reset(task_name=task_name)
            ep_reward = 0.0
            steps = []

            for step_idx in range(max_folds):
                prompt = build_prompt_from_obs(task_name, task_info, obs)
                completion = generate_single(prompt)
                crease = extract_crease_json(completion)

                if crease is None:
                    steps.append({"step": step_idx, "reward": -2.0, "valid": False})
                    ep_reward -= 2.0
                    break

                obs = env.step(OrigamiAction(crease=crease))
                r = obs.reward if obs.reward is not None else 0.0
                ep_reward += r
                steps.append({
                    "step": step_idx,
                    "reward": r,
                    "progress": obs.reward_breakdown.get("progress", 0) if obs.reward_breakdown else 0,
                    "valid": True,
                })

                if obs.done:
                    break

            final_progress = 0.0
            got_completion = False
            if obs.reward_breakdown:
                final_progress = obs.reward_breakdown.get("progress", 0)
                got_completion = obs.reward_breakdown.get("completion", 0) > 0

            episodes.append({
                "total_reward": ep_reward,
                "steps": steps,
                "n_steps": len(steps),
                "completed": got_completion,
                "final_progress": final_progress,
            })

            valid_steps = sum(1 for s in steps if s["valid"])
            print(f"  [{task_name}] ep {ep+1}: reward={ep_reward:.2f}  "
                  f"steps={len(steps)}/{max_folds}  "
                  f"progress={final_progress:.2f}  "
                  f"valid={valid_steps}/{len(steps)}")

        # Aggregate
        rewards = [e["total_reward"] for e in episodes]
        completions = sum(1 for e in episodes if e["completed"])
        mean_steps = np.mean([e["n_steps"] for e in episodes])

        all_results[task_name] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "completion_rate": completions / n_episodes * 100,
            "mean_steps": float(mean_steps),
            "max_folds": max_folds,
        }

        print(f"  {task_name:15s}  "
              f"reward={np.mean(rewards):.2f}±{np.std(rewards):.2f}  "
              f"complete={completions}/{n_episodes}  "
              f"steps={mean_steps:.1f}/{max_folds}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== V3 EVALUATION SUMMARY ===")
    for name, r in all_results.items():
        bar = "█" * int(r["completion_rate"] / 5)
        print(f"  {name:15s}  "
              f"reward={r['mean_reward']:5.2f}  "
              f"complete={r['completion_rate']:5.1f}%  "
              f"steps={r['mean_steps']:.1f}/{r['max_folds']}  "
              f"{bar}")

    return all_results


@app.local_entrypoint()
def eval_main(
    checkpoint: str = "",
    n_episodes: int = 20,
    tasks: str = "all",
    model: str = "unsloth/Qwen3-32B",
):
    evaluate.remote(
        checkpoint=checkpoint,
        n_episodes=n_episodes,
        tasks=tasks,
        model_name=model,
    )
