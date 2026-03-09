"""V3 multi-step GRPO training with GiGPO advantage estimation.

Custom training loop that handles multi-turn rollouts:
1. Generate completions using model.generate() (batched)
2. Run rollout batches through OrigamiEnvPool
3. Compute GiGPO two-level advantages
4. Policy gradient update with clipped surrogate loss

Usage:
    python -m training.train_v3 --model unsloth/Qwen2.5-3B-Instruct --max-steps 50
    python -m training.train_v3 --model unsloth/Qwen3-32B --max-steps 1500
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from transformers import LogitsProcessor, LogitsProcessorList

from training.curriculum import get_task_pool
from training.gigpo import GiGPORewardManager
from training.rollout import run_rollout_batch
from training.trajectory import Trajectory


class ExplorationNoiseProcessor(LogitsProcessor):
    """Add Gaussian noise to logits to force diverse outputs for GRPO.

    Without this, the model produces identical completions for similar prompts
    (even with temperature scaling), resulting in zero reward variance and
    no GRPO learning signal.
    """
    def __init__(self, noise_scale: float = 3.0):
        self.noise_scale = noise_scale

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores + self.noise_scale * torch.randn_like(scores)


def build_generate_fn(model, tokenizer, temperature=0.7, max_new_tokens=128,
                      noise_scale=3.0, top_k=0):
    """Wrap model.generate() to match rollout's expected interface."""
    logits_processor = LogitsProcessorList([
        ExplorationNoiseProcessor(noise_scale=noise_scale),
    ]) if noise_scale > 0 else None

    def generate_fn(prompts: list[str]) -> list[str]:
        messages_batch = [
            [{"role": "system", "content": "/no_think"},
             {"role": "user", "content": p}]
            for p in prompts
        ]

        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]

        encodings = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        gen_kwargs = dict(
            **encodings,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
        if logits_processor:
            gen_kwargs["logits_processor"] = logits_processor

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

        completions = []
        for i, out in enumerate(outputs):
            prompt_len = encodings["input_ids"][i].shape[0]
            completion_ids = out[prompt_len:]
            completions.append(
                tokenizer.decode(completion_ids, skip_special_tokens=True)
            )

        return completions

    return generate_fn


def compute_log_probs(model, tokenizer, prompt: str, completion: str) -> tuple[torch.Tensor, int]:
    """Compute log probabilities for a completion given a prompt.

    Returns (sum_log_probs, n_tokens) so caller can compute both
    total log_prob (for REINFORCE) and per-token KL (for penalty).
    """
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + completion

    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)

    prompt_len = prompt_ids.shape[1]
    n_tokens = full_ids.shape[1] - prompt_len
    if n_tokens <= 0:
        return torch.tensor(0.0, device=model.device), 0

    with autocast("cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits

    shift_logits = logits[:, prompt_len - 1:-1, :]
    shift_labels = full_ids[:, prompt_len:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum(), n_tokens



def policy_loss(
    model,
    tokenizer,
    trajectories: list[Trajectory],
    advantages: list[list[float]],
    clip_range: float = 0.2,
    kl_coef: float = 0.1,
    max_kl_per_token: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Compute GRPO-style policy gradient loss with per-token KL penalty.

    For LoRA models, reference log probs are computed by disabling the adapter.
    KL is normalized per-token for fair comparison across completion lengths.
    Steps where per-token KL > max_kl_per_token are skipped.

    Optimized: toggles LoRA adapter only once (not per-step) and pre-filters
    zero-advantage steps to avoid unnecessary forward passes.
    """
    device = next(model.parameters()).device

    # 1. Collect all steps with non-zero advantages
    step_items = []
    for i, traj in enumerate(trajectories):
        for t, step in enumerate(traj.steps):
            adv = advantages[i][t]
            if abs(adv) < 1e-10:
                continue
            step_items.append((step.prompt, step.completion, adv))

    if not step_items:
        return (
            torch.tensor(0.0, device=device, requires_grad=False),
            {"n_steps": 0, "mean_kl_per_token": 0.0, "kl_skipped": 0},
        )

    # 2. Pre-compute ALL reference log probs (adapter disabled once)
    ref_data = []
    with torch.no_grad():
        model.disable_adapter_layers()
        for prompt, completion, _ in step_items:
            ref_lp, n_tokens = compute_log_probs(model, tokenizer, prompt, completion)
            ref_data.append((ref_lp.item(), n_tokens))
        model.enable_adapter_layers()

    # 3. Compute policy log probs and loss (adapter enabled, with gradients)
    losses = []
    total_kl = 0.0
    n_steps = 0
    n_kl_skipped = 0

    for idx, (prompt, completion, adv) in enumerate(step_items):
        ref_lp_val, n_tokens = ref_data[idx]
        if n_tokens == 0:
            continue

        log_prob, _ = compute_log_probs(model, tokenizer, prompt, completion)

        kl_per_token = (log_prob.detach().item() - ref_lp_val) / n_tokens
        if abs(kl_per_token) > max_kl_per_token:
            n_kl_skipped += 1
            continue

        total_kl += abs(kl_per_token)

        adv_tensor = torch.tensor(adv, device=device, dtype=log_prob.dtype)
        mean_log_prob = log_prob / n_tokens
        ref_lp_tensor = torch.tensor(ref_lp_val, device=device, dtype=log_prob.dtype)
        kl_term = (log_prob - ref_lp_tensor) / n_tokens
        step_loss = -(adv_tensor * mean_log_prob) + kl_coef * kl_term

        losses.append(step_loss)
        n_steps += 1

    if losses:
        total_loss = torch.stack(losses).mean()
    else:
        total_loss = torch.tensor(0.0, device=device, requires_grad=False)

    metrics = {
        "n_steps": n_steps,
        "mean_kl_per_token": total_kl / max(n_steps, 1),
        "kl_skipped": n_kl_skipped,
    }
    return total_loss, metrics


def save_checkpoint(model, tokenizer, output_dir: str, step: int):
    ckpt_path = os.path.join(output_dir, f"checkpoint-{step}")
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="V3 multi-step GRPO training")
    parser.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-per-task", type=int, default=8,
                        help="Number of episodes per task (GRPO needs >=4 for reward variance)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--max-kl-per-token", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--noise-scale", type=float, default=1.5,
                        help="Gaussian noise added to logits for exploration (0=disabled)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k sampling to force token diversity (0=disabled)")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--log-steps", type=int, default=5)
    parser.add_argument("--tasks", default="auto",
                        help="Comma-separated task names, or 'auto' for curriculum")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    output_dir = args.output_dir or os.environ.get("OUTPUT_DIR", "outputs/v3")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    from unsloth import FastLanguageModel

    print(f"Loading {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        load_in_4bit=args.load_in_4bit,
        max_seq_length=1024,
    )

    if args.resume:
        print(f"Loading adapter from {args.resume}")
        model.load_adapter(args.resume)
    else:
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.print_trainable_parameters()

    # For LoRA, reference = base model with adapter disabled (handled in policy_loss)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=int(args.max_steps * 0.1),
    )

    # ── GiGPO reward manager ─────────────────────────────────────────────────
    # Alpha=1.0 (pure episode-level GRPO) throughout training.
    # Step-level advantages + exploration noise caused collapse in run 8.
    reward_manager = GiGPORewardManager(
        alpha_start=1.0,
        alpha_end=1.0,
        warmup_steps=200,
        total_steps=args.max_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting V3 training: {args.max_steps} steps, batch_size={args.batch_size}")
    print(f"Output dir: {output_dir}\n")

    start_step = 0
    if args.resume:
        # Try to extract step number from checkpoint name
        try:
            start_step = int(Path(args.resume).name.split("-")[-1])
            reward_manager.global_step = start_step
            print(f"Resuming from step {start_step}")
        except (ValueError, IndexError):
            pass

    for global_step in range(start_step, args.max_steps):
        step_start = time.time()

        # 1. Get task pool
        if args.tasks == "auto":
            task_pool = get_task_pool(global_step)
        else:
            task_pool = [t.strip() for t in args.tasks.split(",")]

        # 2. Generate rollouts (model in eval mode for generation)
        FastLanguageModel.for_inference(model)
        generate_fn = build_generate_fn(
            model, tokenizer,
            temperature=args.temperature,
            max_new_tokens=128,
            noise_scale=args.noise_scale,
            top_k=args.top_k,
        )

        trajectories = run_rollout_batch(
            generate_fn=generate_fn,
            task_pool=task_pool,
            batch_size=args.batch_size,
            num_per_task=args.num_per_task,
        )

        # 3. Compute GiGPO advantages
        advantages = reward_manager.compute_advantages(trajectories)
        reward_manager.step()

        # 4. Policy gradient update (model in train mode)
        FastLanguageModel.for_training(model)
        optimizer.zero_grad()

        loss, metrics = policy_loss(
            model=model,
            tokenizer=tokenizer,
            trajectories=trajectories,
            advantages=advantages,
            clip_range=args.clip_range,
            kl_coef=args.kl_coef,
            max_kl_per_token=args.max_kl_per_token,
        )

        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        step_time = time.time() - step_start

        # 5. Logging
        if (global_step + 1) % args.log_steps == 0 or global_step == start_step:
            mean_reward = np.mean([t.total_reward for t in trajectories])
            mean_length = np.mean([t.length for t in trajectories])
            max_reward = max(t.total_reward for t in trajectories)
            completion_rate = sum(
                1 for t in trajectories
                if any(s.reward_breakdown.get("completion", 0) > 0 for s in t.steps)
            ) / len(trajectories) * 100

            task_rewards = {}
            for t in trajectories:
                task_rewards.setdefault(t.task, []).append(t.total_reward)
            task_summary = "  ".join(
                f"{k}:{np.mean(v):.2f}" for k, v in sorted(task_rewards.items())
            )

            # Advantage diagnostics
            flat_advs = [a for adv_list in advantages for a in adv_list]
            adv_nonzero = sum(1 for a in flat_advs if abs(a) > 1e-10)
            adv_std = float(np.std(flat_advs)) if flat_advs else 0.0

            print(
                f"step {global_step + 1}/{args.max_steps}  "
                f"loss={loss.item():.4f}  "
                f"reward={mean_reward:.2f}±{np.std([t.total_reward for t in trajectories]):.2f}  "
                f"max={max_reward:.2f}  "
                f"len={mean_length:.1f}  "
                f"complete={completion_rate:.0f}%  "
                f"adv_nz={adv_nonzero}/{len(flat_advs)}  "
                f"adv_std={adv_std:.3f}  "
                f"alpha={reward_manager.alpha:.2f}  "
                f"kl/t={metrics.get('mean_kl_per_token', 0):.4f}  "
                f"kl_skip={metrics.get('kl_skipped', 0)}  "
                f"tasks=[{task_summary}]  "
                f"time={step_time:.1f}s"
            )

        # 6. Checkpointing
        if (global_step + 1) % args.save_steps == 0:
            save_checkpoint(model, tokenizer, output_dir, global_step + 1)

    # Final save
    save_checkpoint(model, tokenizer, output_dir, args.max_steps)
    final_path = os.path.join(output_dir, "origami-v3-lora-final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
