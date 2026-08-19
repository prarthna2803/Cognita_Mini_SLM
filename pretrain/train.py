#!/usr/bin/env python3
"""Pre-train Cognita-Mini-167M from random initialisation.

Pipeline:
    1. load the pre-tokenized shards written by pretokenize.py
    2. pack token streams into fixed 4,096-token sequences, so no position in
       any batch is spent on padding
    3. hold out 1% of the packed sequences for validation (seed 42)
    4. train with the Hugging Face Trainer; evaluate and checkpoint every
       1,000 steps, and keep the checkpoint with the lowest validation loss

Settings of the released run: AdamW, peak LR 3e-4 with 1% linear warm-up and
cosine decay, weight decay 0.1, bf16 autocast, 2 epochs. Batch 12 sequences per
GPU x 8 accumulation steps x 2 GPUs = 192 sequences (786,432 tokens) per update.

The script resumes from the newest checkpoint in --output-dir if one exists.

Example (2 GPUs, DDP):
    torchrun --nproc_per_node 2 pretrain/train.py \
        --tokenizer tokenizer/spm/cognita_mini_32k.model \
        --shards-dir pretrain/tokenized_shards \
        --output-dir pretrain/checkpoints

Set WANDB_PROJECT (and WANDB_ENTITY) to log to Weights & Biases, or pass
--report-to none.
"""

import argparse
import glob
import os
import sys
import time

import torch
from datasets import concatenate_datasets, load_from_disk
from transformers import LlamaTokenizer, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MAX_SEQ_LENGTH, build_model  # noqa: E402


def pack_sequences(examples):
    """Concatenate tokenized lines and cut them into MAX_SEQ_LENGTH blocks."""
    ids = [token for line in examples["input_ids"] for token in line]
    usable = (len(ids) // MAX_SEQ_LENGTH) * MAX_SEQ_LENGTH
    blocks = [ids[i:i + MAX_SEQ_LENGTH] for i in range(0, usable, MAX_SEQ_LENGTH)]
    return {"input_ids": blocks, "labels": [block.copy() for block in blocks]}


def latest_checkpoint(output_dir: str):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    return max(checkpoints, key=lambda path: int(path.rsplit("-", 1)[1])) if checkpoints else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="SentencePiece .model file")
    parser.add_argument("--shards-dir", required=True, help="Output folder of pretokenize.py")
    parser.add_argument("--output-dir", default="pretrain/checkpoints")
    parser.add_argument("--per-device-batch-size", type=int, default=12)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--num-proc", type=int, default=24, help="Processes for sequence packing")
    parser.add_argument("--report-to", default="wandb", choices=["wandb", "none"])
    args = parser.parse_args()

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", "cognita-mini-pretrain")

    tokenizer = LlamaTokenizer(vocab_file=args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(tokenizer.vocab_size)
    print(f"Model initialised from scratch: {model.num_parameters() / 1e6:.2f}M parameters")

    shard_dirs = sorted(glob.glob(os.path.join(args.shards_dir, "shard*")))
    if not shard_dirs:
        raise SystemExit(f"No tokenized shards in {args.shards_dir}")
    lines = concatenate_datasets([load_from_disk(path) for path in shard_dirs])

    packed = lines.map(
        pack_sequences, batched=True, batch_size=2000,
        remove_columns=lines.column_names, num_proc=args.num_proc, load_from_cache_file=True,
    )
    split = packed.train_test_split(test_size=args.val_fraction, seed=42)
    print(f"Train sequences: {len(split['train']):,} | validation sequences: {len(split['test']):,}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=f"cognita-mini-167m-{int(time.time())}",
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        bf16=use_bf16,
        fp16=not use_bf16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=25,
        report_to=args.report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
    )

    resume_from = latest_checkpoint(args.output_dir)
    if resume_from:
        print(f"Resuming from {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = f"{args.output_dir.rstrip('/')}-final"
    trainer.save_model(final_dir)
    print(f"Best model and tokenizer saved to {final_dir}")


if __name__ == "__main__":
    main()
