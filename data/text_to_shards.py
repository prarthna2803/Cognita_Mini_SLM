#!/usr/bin/env python3
"""Convert a large one-sentence-per-line text file into token-budgeted JSONL shards.

Used for the Odia split of AI4Bharat IndicCorp v2, which ships as a single
``or.txt``. The file is streamed, so it never has to fit in memory.

Example:
    python data/text_to_shards.py --input IndicCorpV2/or.txt --out-dir corpus/odia_indiccorp \
        --max-tokens 1500000000
"""

import argparse
import json
import os

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Plain-text file, one sentence per line")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=1_500_000_000)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--budget-tokenizer", default="Qwen/Qwen2-7B",
                        help="Tokenizer used only to count the budget")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.budget_tokenizer)

    token_count = 0
    shard_id = 0
    reported_millions = 0
    skipped_empty = skipped_over_budget = 0
    out = open(os.path.join(args.out_dir, f"shard_{shard_id}.jsonl"), "w", encoding="utf-8")

    try:
        with open(args.input, encoding="utf-8") as source:
            for raw_line in source:
                if token_count >= args.max_tokens:
                    break
                text = raw_line.strip()
                if not text:
                    skipped_empty += 1
                    continue

                n_tokens = len(tokenizer(text).input_ids)
                if token_count + n_tokens > args.max_tokens:
                    skipped_over_budget += 1
                    continue

                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                token_count += n_tokens

                if token_count // 1_000_000 > reported_millions:
                    reported_millions = token_count // 1_000_000
                    print(f"{reported_millions}M tokens")

                if token_count // args.shard_tokens > shard_id:
                    out.close()
                    shard_id += 1
                    out = open(os.path.join(args.out_dir, f"shard_{shard_id}.jsonl"), "w", encoding="utf-8")
    finally:
        out.close()

    print(f"Saved {token_count:,} tokens in {shard_id + 1} shards to {args.out_dir}/")
    print(f"Skipped {skipped_empty:,} empty lines and {skipped_over_budget:,} lines over budget")


if __name__ == "__main__":
    main()
