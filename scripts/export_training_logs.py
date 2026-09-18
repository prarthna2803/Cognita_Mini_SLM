#!/usr/bin/env python3
"""Export a Hugging Face ``trainer_state.json`` into a compact, plot-ready log.

The Trainer keeps the full loss history inside every checkpoint. This script
pulls it out so the curves in the README can be regenerated without the
multi-gigabyte checkpoints (and without a Weights & Biases account).

Example:
    python scripts/export_training_logs.py \
        --checkpoint pretrain/checkpoints/checkpoint-6000 \
        --output results/pretrain_log.json
"""

import argparse
import json
import math
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory containing trainer_state.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens-per-step", type=int, default=None,
                        help="Tokens consumed per optimizer step (adds a tokens-seen axis)")
    args = parser.parse_args()

    with open(os.path.join(args.checkpoint, "trainer_state.json"), encoding="utf-8") as f:
        state = json.load(f)

    train, evals = [], []
    for entry in state["log_history"]:
        if "eval_loss" in entry:
            evals.append({
                "step": entry["step"],
                "epoch": round(entry["epoch"], 4),
                "eval_loss": entry["eval_loss"],
                "eval_perplexity": math.exp(entry["eval_loss"]),
                "eval_samples": round(entry["eval_samples_per_second"] * entry["eval_runtime"]),
            })
        elif "loss" in entry:
            train.append({
                "step": entry["step"],
                "epoch": round(entry["epoch"], 4),
                "loss": entry["loss"],
                "learning_rate": entry["learning_rate"],
                "grad_norm": entry.get("grad_norm"),
            })

    log = {
        "global_step": state["global_step"],
        "max_steps": state["max_steps"],
        "num_train_epochs": state["num_train_epochs"],
        "per_device_train_batch_size": state.get("train_batch_size"),
        "best_eval_loss": state.get("best_metric"),
        "best_checkpoint": os.path.basename(state.get("best_model_checkpoint") or ""),
        "tokens_per_step": args.tokens_per_step,
        "train": train,
        "eval": evals,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)

    print(f"{len(train)} training points, {len(evals)} evaluations -> {args.output}")
    if evals:
        last = evals[-1]
        print(f"final eval loss {last['eval_loss']:.4f} (perplexity {last['eval_perplexity']:.2f}) at step {last['step']}")


if __name__ == "__main__":
    main()
