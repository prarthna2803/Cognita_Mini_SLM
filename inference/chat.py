#!/usr/bin/env python3
"""Ask the instruction-tuned model a question (base model + LoRA adapter).

The adapter was trained on single-turn instruction-response pairs for two
tasks: short commonsense answers and character counting. Each question is
answered on its own; there is no conversation memory.

Example:
    python inference/chat.py --instruction "How many times does the character 'a' appear in \"banana\"?"
    python inference/chat.py                          # interactive
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer

DEFAULT_BASE = "ProtonPratt/Cognita-Mini-167M"
DEFAULT_ADAPTER = "ProtonPratt/Cognita-Mini-167M-instruct-lora"
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


@torch.no_grad()
def respond(model, tokenizer, instruction: str, args) -> str:
    inputs = tokenizer(PROMPT_TEMPLATE.format(instruction=instruction), return_tensors="pt").to(model.device)
    sampling = {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p} \
        if args.temperature > 0 else {"do_sample": False}
    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id, **sampling)
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default=DEFAULT_BASE, help="Hub id or local directory")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Hub id or local directory")
    parser.add_argument("--instruction", default=None, help="Omit for an interactive session")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.5, help="0 for greedy decoding")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None, help="Fix the sampling seed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload().to(args.device).eval()

    if args.instruction is not None:
        print(respond(model, tokenizer, args.instruction, args))
        return

    print("Ask a question; 'exit' to quit.")
    while True:
        instruction = input("You: ")
        if instruction.strip().lower() in ("exit", "quit"):
            break
        print(f"Model: {respond(model, tokenizer, instruction, args)}\n")


if __name__ == "__main__":
    main()
