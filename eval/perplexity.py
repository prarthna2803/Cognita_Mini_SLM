#!/usr/bin/env python3
"""Token-level perplexity of the pre-trained model on plain text.

Accepts .txt (one document per line) or .jsonl (a "text" field per line). Each
document is scored on its own, padding is excluded from the loss, and the result
is weighted by token count, so the number is comparable across files of
different sentence lengths:

    perplexity = exp( sum of token negative log-likelihoods / number of scored tokens )

Documents longer than the context window are truncated to --max-length tokens.
No BOS token is prepended, matching how the pre-training shards were tokenized.

Example:
    python eval/perplexity.py \
        --model release/Cognita-Mini-167M \
        --data english=heldout_en.txt odia=heldout_or.jsonl \
        --output results/perplexity.json
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer


def read_documents(path: str, max_documents: int) -> list:
    documents = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "").strip() if path.endswith(".jsonl") else line
            if text:
                documents.append(text)
            if max_documents and len(documents) >= max_documents:
                break
    return documents


@torch.no_grad()
def corpus_nll(model, tokenizer, documents: list, device: str, batch_size: int, max_length: int, name: str):
    """Return (summed negative log-likelihood, number of scored tokens)."""
    encoded = [tokenizer.encode(doc, add_special_tokens=False)[:max_length] for doc in documents]
    encoded = sorted((ids for ids in encoded if len(ids) >= 2), key=len)  # length-sort to minimise padding

    total_nll, total_tokens = 0.0, 0
    for start in tqdm(range(0, len(encoded), batch_size), desc=name):
        batch = encoded[start:start + batch_size]
        width = max(len(ids) for ids in batch)
        input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, ids in enumerate(batch):
            input_ids[row, :len(ids)] = torch.tensor(ids)
            attention_mask[row, :len(ids)] = 1
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        targets = input_ids[:, 1:].masked_fill(attention_mask[:, 1:] == 0, -100)
        nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), targets.reshape(-1),
                              ignore_index=-100, reduction="sum")
        total_nll += nll.item()
        total_tokens += int((targets != -100).sum())
    return total_nll, total_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model directory or Hub id")
    parser.add_argument("--data", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--max-documents", type=int, default=0, help="Documents per file (0 = all)")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/perplexity.json")
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    results = {}
    for spec in args.data:
        name, path = spec.split("=", 1)
        documents = read_documents(path, args.max_documents)
        nll, tokens = corpus_nll(model, tokenizer, documents, args.device, args.batch_size, args.max_length, name)
        results[name] = {
            "documents": len(documents),
            "scored_tokens": tokens,
            "loss": round(nll / tokens, 4),
            "perplexity": round(math.exp(nll / tokens), 3),
        }
        print(f"{name:10s} {tokens:,} tokens  loss {results[name]['loss']:.4f}  perplexity {results[name]['perplexity']:.2f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
