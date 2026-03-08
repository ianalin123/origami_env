"""Modal eval script for origami GRPO checkpoints.

Run:
    modal run modal_eval.py                                   # latest checkpoint, all tasks
    modal run modal_eval.py --checkpoint checkpoint-20        # specific checkpoint
    modal run modal_eval.py --checkpoint base                 # base model (no LoRA)
    modal run modal_eval.py --n-samples 20 --tasks quarter_fold,letter_fold
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import modal
from modal_train import OUTPUTS_DIR, app, image, volume

ALL_TASKS = ["triangle", "half_fold", "quarter_fold", "letter_fold"]


@app.function(
    image=image,
    gpu="B200",
    timeout=3600,
    volumes={OUTPUTS_DIR: volume},
)
def evaluate(
    checkpoint: str = "",
    n_samples: int = 10,
    server_url: str = "",
    tasks: str = "all",
    model_name: str = "unsloth/Qwen3-32B",
):
    import torch
    import requests as req
    from training.train_grpo import build_prompt
    from training.reward import extract_fold_json
    from origami_server.models import OrigamiAction
    from client import OrigamiEnv

    # ── Env server ────────────────────────────────────────────────────────────
    server_proc = None
    if not server_url:
        server_url = "http://localhost:8000"
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "origami_server.app:app",
             "--host", "0.0.0.0", "--port", "8000"],
            cwd="/app",
        )
        for _ in range(45):
            try:
                if req.get(f"{server_url}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)

    try:
        # ── Resolve checkpoint path ───────────────────────────────────────────
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
                raise ValueError("No checkpoint found in volume. Pass --checkpoint base to eval base model.")

        # ── Load model ────────────────────────────────────────────────────────
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=False,
            max_seq_length=1024,
        )
        if ckpt_path:
            model.load_adapter(ckpt_path)
        FastLanguageModel.for_inference(model)

        # ── Evaluate each task ────────────────────────────────────────────────
        task_list = ALL_TASKS if tasks == "all" else [t.strip() for t in tasks.split(",")]
        results = {}

        for task_name in task_list:
            task_info = req.get(f"{server_url}/tasks/{task_name}").json()
            prompt_text = build_prompt(task_info)
            messages = [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": prompt_text},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")
            attention_mask = torch.ones_like(input_ids)

            rewards, valid = [], 0
            for i in range(n_samples):
                with torch.no_grad():
                    out = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(
                    out[0][input_ids.shape[1]:], skip_special_tokens=True
                )
                fold_data = extract_fold_json(response)
                if fold_data is None:
                    print(f"    [{task_name}] sample {i+1}: invalid JSON")
                    rewards.append(0.0)
                    continue
                valid += 1
                try:
                    with OrigamiEnv(base_url=server_url) as env:
                        env.reset(task_name=task_name)
                        result = env.step(OrigamiAction(fold_data=fold_data))
                        r = result.reward if result.reward is not None else 0.0
                        rewards.append(r)
                        print(f"    [{task_name}] sample {i+1}: reward={r:.2f}")
                except Exception as e:
                    print(f"    [{task_name}] sample {i+1}: env error — {e}")
                    rewards.append(-1.0)

            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            results[task_name] = {"mean": mean_r, "std": std_r, "valid_pct": valid / n_samples * 100}
            print(f"  {task_name:15s}  reward={mean_r:.2f}±{std_r:.2f}  valid={valid}/{n_samples}")

        print("\n=== SUMMARY ===")
        for name, r in results.items():
            bar = "█" * int(r["mean"] / 21 * 20)
            print(f"  {name:15s}  {r['mean']:5.2f}/21  {bar}")

        return results

    finally:
        if server_proc:
            server_proc.terminate()


@app.local_entrypoint()
def eval_main(
    checkpoint: str = "",
    n_samples: int = 10,
    server_url: str = "",
    tasks: str = "all",
    model: str = "unsloth/Qwen3-32B",
):
    evaluate.remote(
        checkpoint=checkpoint,
        n_samples=n_samples,
        server_url=server_url,
        tasks=tasks,
        model_name=model,
    )
