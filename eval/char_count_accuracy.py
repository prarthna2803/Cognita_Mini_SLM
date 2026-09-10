#!/usr/bin/env python3
"""Score the instruction-tuned model on the character-counting task.

Each example asks "How many times does the character 'x' appear in "..."?" and
the reference answer is an integer. The model answers with greedy decoding; the
first integer in its reply is compared with the reference.

Reported per file:
    exact        prediction == reference
    within_2     |prediction - reference| <= 2
    mae          mean absolute error over parseable answers
    unparseable  replies that contain no integer (counted as wrong)

Example:
    python eval/char_count_accuracy.py \
        --base-model release/Cognita-Mini-167M \
        --adapter release/Cognita-Mini-167M-instruct-lora \
        --data english=data/char_count_en_val.jsonl odia=data/char_count_or_val.jsonl \
        --max-examples 500 --output results/char_count_eval.json
"""

import argparse
import json
import os
import random
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
FIRST_INT = re.compile(r"-?\d+")


def load_examples(path: str, max_examples: int, seed: int) -> list:
    with open(path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if max_examples and len(examples) > max_examples:
        random.Random(seed).shuffle(examples)
        examples = examples[:max_examples]
    return examples


@torch.no_grad()
def answer(model, tokenizer, instruction: str, device: str, max_new_tokens: int) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def score(model, tokenizer, examples: list, device: str, max_new_tokens: int, name: str) -> dict:
    exact = within_2 = unparseable = 0
    abs_errors = []
    for example in tqdm(examples, desc=name):
        reference = int(example["response"])
        match = FIRST_INT.search(answer(model, tokenizer, example["instruction"], device, max_new_tokens))
        if match is None:
            unparseable += 1
            continue
        error = abs(int(match.group()) - reference)
        abs_errors.append(error)
        exact += error == 0
        within_2 += error <= 2
    n = len(examples)
    return {
        "examples": n,
        "exact": round(exact / n, 4),
        "within_2": round(within_2 / n, 4),
        "mae": round(sum(abs_errors) / len(abs_errors), 3) if abs_errors else None,
        "unparseable": round(unparseable / n, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True, help="Pre-trained model directory or Hub id")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory or Hub id (omit to score the base model)")
    parser.add_argument("--data", nargs="+", required=True, metavar="NAME=PATH",
                        help="JSONL files with 'instruction' and integer 'response' fields")
    parser.add_argument("--max-examples", type=int, default=500, help="Examples per file (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/char_count_eval.json")
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.to(args.device).eval()

    results = {
        "base_model": os.path.basename(os.path.normpath(args.base_model)),
        "adapter": os.path.basename(os.path.normpath(args.adapter)) if args.adapter else None,
        "decoding": "greedy",
        "seed": args.seed,
        "by_dataset": {},
    }
    for spec in args.data:
        name, path = spec.split("=", 1)
        examples = load_examples(path, args.max_examples, args.seed)
        results["by_dataset"][name] = score(model, tokenizer, examples, args.device, args.max_new_tokens, name)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for name, entry in results["by_dataset"].items():
        print(f"{name:10s} n={entry['examples']:<5d} exact {entry['exact']:.1%}  within±2 {entry['within_2']:.1%}  "
              f"MAE {entry['mae']}  unparseable {entry['unparseable']:.1%}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
