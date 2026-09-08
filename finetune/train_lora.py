#!/usr/bin/env python3
"""Instruction-tune the pre-trained model with LoRA.

The base model stays frozen. Rank-16 adapters (alpha 32, dropout 0.05) are added
to every attention projection (q, k, v, o) and every feed-forward projection
(gate, up, down): 3.17M trainable parameters, 1.9% of the model.

Examples are formatted as

    ### Instruction:
    {instruction}

    ### Response:
    {response}</s>

Loss is computed on the response tokens only, and batches are padded to their
longest example.

    --legacy-padded-loss reproduces the objective of the released adapter, which
    padded every example to 4,096 tokens and included prompt and padding
    positions in the loss. It is far slower and its eval loss is dominated by
    padding; it is kept only for reproducibility.

Settings of the released run: batch 8 per GPU x 8 accumulation steps x 2 GPUs,
LR 2e-4 with cosine decay, weight decay 0.1, 3 epochs, bf16.

Example (2 GPUs):
    torchrun --nproc_per_node 2 finetune/train_lora.py \
        --base-model pretrain/checkpoints-final \
        --train-file ft_data/train.jsonl --val-file ft_data/val.jsonl \
        --output-dir finetune/lora
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, LlamaTokenizer, Trainer, TrainingArguments

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
IGNORE_INDEX = -100


def build_tokenize_fn(tokenizer, max_length: int, legacy_padded_loss: bool):
    def tokenize(batch):
        prompts = [PROMPT_TEMPLATE.format(instruction=instruction) for instruction in batch["instruction"]]
        texts = [prompt + response + tokenizer.eos_token for prompt, response in zip(prompts, batch["response"])]

        if legacy_padded_loss:
            encoded = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
            encoded["labels"] = [ids.copy() for ids in encoded["input_ids"]]
            return encoded

        encoded = tokenizer(texts, truncation=True, max_length=max_length)
        prompt_lengths = [len(ids) for ids in tokenizer(prompts)["input_ids"]]
        encoded["labels"] = [
            [IGNORE_INDEX] * min(prompt_length, len(ids)) + ids[prompt_length:]
            for ids, prompt_length in zip(encoded["input_ids"], prompt_lengths)
        ]
        return encoded
    return tokenize


class PadToLongest:
    """Pad a batch to its longest example; padded positions are excluded from the loss."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full((len(features), width), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(features), width), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(features), width), dtype=torch.long)
        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(feature["input_ids"])
            labels[row, :length] = torch.tensor(feature["labels"])
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True, help="Pre-trained model directory or Hub id")
    parser.add_argument("--train-file", required=True, help="JSONL with instruction / response fields")
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-dir", default="finetune/lora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop early (debugging)")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--legacy-padded-loss", action="store_true")
    parser.add_argument("--report-to", default="wandb", choices=["wandb", "none"])
    args = parser.parse_args()

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", "cognita-mini-finetune")

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    tokenize = build_tokenize_fn(tokenizer, args.max_length, args.legacy_padded_loss)
    datasets = {}
    for name, path in (("train", args.train_file), ("val", args.val_file)):
        raw = load_dataset("json", data_files=path, split="train")
        datasets[name] = raw.map(tokenize, batched=True, num_proc=args.num_proc, remove_columns=raw.column_names)
    print(f"Train examples: {len(datasets['train']):,} | validation examples: {len(datasets['val']):,}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        weight_decay=0.1,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=args.report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=PadToLongest(tokenizer.pad_token_id),
    )
    trainer.train()

    best_dir = f"{args.output_dir.rstrip('/')}-best"
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Best LoRA adapter and tokenizer saved to {best_dir}")


if __name__ == "__main__":
    main()
