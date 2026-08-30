#!/usr/bin/env python3
"""Free-form text continuation with the pre-trained base model.

The base model is a plain language model: give it the start of a passage in
English, Odia or Dogri and it continues in the same language. For
question-style prompts use chat.py with the instruction adapter.

Example:
    python inference/generate.py --prompt "The history of Odisha begins"
    python inference/generate.py                      # interactive
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

DEFAULT_MODEL = "ProtonPratt/Cognita-Mini-167M"
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।\n])\s+")


def drop_consecutive_repeats(text: str) -> str:
    """Remove sentences that immediately repeat the previous one."""
    kept = []
    for sentence in (s.strip() for s in SENTENCE_BOUNDARY.split(text)):
        if sentence and (not kept or sentence != kept[-1]):
            kept.append(sentence)
    return " ".join(kept)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, args) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode prompt and continuation together: the first new token may complete the prompt's last word.
    return drop_consecutive_repeats(tokenizer.decode(output[0], skip_special_tokens=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub id or local directory")
    parser.add_argument("--prompt", default=None, help="Omit for an interactive session")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None, help="Fix the sampling seed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    if args.prompt is not None:
        print(generate(model, tokenizer, args.prompt, args))
        return

    print("Type the start of a passage and the model continues it; 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.strip().lower() in ("exit", "quit"):
            break
        print(f"Model: {generate(model, tokenizer, prompt, args)}\n")


if __name__ == "__main__":
    main()
