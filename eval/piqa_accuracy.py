#!/usr/bin/env python3
"""Two-way multiple-choice accuracy on PIQA (physical commonsense).

PIQA gives a goal and two candidate solutions. The model never sees the choices
side by side: each solution is scored as the response to the goal under the
fine-tuning prompt template, and the solution with the higher length-normalised
log-likelihood wins. Chance is 50%.

The summed (un-normalised) variant is stored alongside for reference.

Example:
    python eval/piqa_accuracy.py \
        --base-model release/Cognita-Mini-167M \
        --adapter release/Cognita-Mini-167M-instruct-lora \
        --data piqa/valid.jsonl --labels piqa/valid-labels.lst \
        --output results/piqa_eval.json
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


@torch.no_grad()
def response_log_likelihood(model, tokenizer, prompt_ids: list, response: str, device: str):
    """Return (sum, mean) log-probability of the response tokens given the prompt."""
    response_ids = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    input_ids = torch.tensor([prompt_ids + response_ids], device=device)
    logits = model(input_ids=input_ids).logits.float()
    # Position i predicts token i + 1, so the response is predicted from len(prompt) - 1 onwards.
    predictions = logits[0, len(prompt_ids) - 1:-1]
    log_probs = -F.cross_entropy(predictions, input_ids[0, len(prompt_ids):], reduction="none")
    return log_probs.sum().item(), log_probs.mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None, help="LoRA adapter (omit to score the base model)")
    parser.add_argument("--data", required=True, help="PIQA .jsonl with goal / sol1 / sol2")
    parser.add_argument("--labels", required=True, help="PIQA .lst with one 0/1 label per line")
    parser.add_argument("--max-examples", type=int, default=0, help="0 = all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/piqa_eval.json")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    with open(args.labels, encoding="utf-8") as f:
        labels = [int(line) for line in f if line.strip()]
    if len(examples) != len(labels):
        raise SystemExit(f"{len(examples)} examples but {len(labels)} labels")
    if args.max_examples:
        examples, labels = examples[:args.max_examples], labels[:args.max_examples]

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.to(args.device).eval()

    correct_mean = correct_sum = 0
    for example, label in zip(tqdm(examples, desc="piqa"), labels):
        prompt_ids = tokenizer.encode(PROMPT_TEMPLATE.format(instruction=example["goal"]))
        scores = [response_log_likelihood(model, tokenizer, prompt_ids, example[key], args.device)
                  for key in ("sol1", "sol2")]
        correct_sum += int(scores[1][0] > scores[0][0]) == label
        correct_mean += int(scores[1][1] > scores[0][1]) == label

    n = len(examples)
    result = {
        "base_model": os.path.basename(os.path.normpath(args.base_model)),
        "adapter": os.path.basename(os.path.normpath(args.adapter)) if args.adapter else None,
        "examples": n,
        "chance": 0.5,
        "accuracy_length_normalised": round(correct_mean / n, 4),
        "accuracy_summed": round(correct_sum / n, 4),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"PIQA n={n}: {result['accuracy_length_normalised']:.1%} length-normalised, "
          f"{result['accuracy_summed']:.1%} summed (chance 50%)")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
