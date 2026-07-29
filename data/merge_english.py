#!/usr/bin/env python3
"""Merge the cleaned Wikipedia and FineWeb shards into one shuffled English corpus.

Rows from every input folder are pooled, shuffled at row level and re-written as
fixed-size shards. Everything is held in memory, which is fine for the ~2.6M
English documents used here (about 10 GB of text).

Example:
    python data/merge_english.py --input-dirs corpus/wikipedia_cleaned corpus/fineweb_cleaned \
        --out-dir corpus/english
"""

import argparse
import glob
import json
import os
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rows-per-shard", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for folder in args.input_dirs:
        for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
            print(f"reading {path}")
            with open(path, encoding="utf-8") as infile:
                for line in infile:
                    try:
                        text = json.loads(line).get("text", "").strip()
                    except json.JSONDecodeError:
                        continue
                    if text:
                        rows.append(text)
    print(f"{len(rows):,} documents")

    random.Random(args.seed).shuffle(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    n_shards = 0
    for start in range(0, len(rows), args.rows_per_shard):
        path = os.path.join(args.out_dir, f"shard_{n_shards:03d}.jsonl")
        with open(path, "w", encoding="utf-8") as outfile:
            for text in rows[start:start + args.rows_per_shard]:
                outfile.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        n_shards += 1
    print(f"Wrote {n_shards} shards to {args.out_dir}/")


if __name__ == "__main__":
    main()
