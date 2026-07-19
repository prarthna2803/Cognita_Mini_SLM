#!/usr/bin/env python3
"""Stream an English corpus from the Hub into token-budgeted JSONL shards.

Two sources are supported:
    wikipedia   wikimedia/wikipedia, 20231101.en dump (encyclopedic text)
    fineweb     HuggingFaceFW/fineweb, CC-MAIN-2025-26 crawl (general web text)

Documents are written as {"text": ...} lines until the token budget is reached.
A new shard is started every --shard-tokens tokens. The budget is counted with a
reference tokenizer because the project tokenizer does not exist yet at this
stage; exact counts under the final tokenizer come from pretrain/corpus_stats.py.

Example:
    python data/download_english.py --source wikipedia --out-dir corpus/wikipedia
    python data/download_english.py --source fineweb --out-dir corpus/fineweb
"""

import argparse
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer

SOURCES = {
    "wikipedia": {"path": "wikimedia/wikipedia", "name": "20231101.en"},
    "fineweb": {"path": "HuggingFaceFW/fineweb", "name": "CC-MAIN-2025-26"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--budget-tokenizer", default="Qwen/Qwen2-7B",
                        help="Tokenizer used only to count the download budget")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.budget_tokenizer)
    dataset = load_dataset(**SOURCES[args.source], split="train", streaming=True)

    token_count = 0
    shard_id = 0
    reported_millions = 0
    out = open(os.path.join(args.out_dir, f"shard_{shard_id}.jsonl"), "w", encoding="utf-8")

    for doc in dataset:
        n_tokens = len(tokenizer(doc["text"]).input_ids)
        if token_count + n_tokens > args.max_tokens:
            break

        out.write(json.dumps(doc, ensure_ascii=False) + "\n")
        token_count += n_tokens

        if token_count // 1_000_000 > reported_millions:
            reported_millions = token_count // 1_000_000
            print(f"{reported_millions}M tokens")

        if token_count // args.shard_tokens > shard_id:
            out.close()
            shard_id += 1
            out = open(os.path.join(args.out_dir, f"shard_{shard_id}.jsonl"), "w", encoding="utf-8")

    out.close()
    print(f"Saved ~{token_count:,} tokens in {shard_id + 1} shards to {args.out_dir}/")


if __name__ == "__main__":
    main()
