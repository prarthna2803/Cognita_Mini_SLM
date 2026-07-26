#!/usr/bin/env python3
"""Export a Hugging Face dataset (Hub id or save_to_disk folder) to JSONL shards.

Used for the Dogri sources, which are small enough to export whole. All splits
are concatenated, and only the text column is kept when one exists.

Example:
    python data/dataset_to_shards.py --dataset StephennFernandes/ciil_mega_corpus_dogri \
        --out-dir corpus/dogri --prefix ciil
    python data/dataset_to_shards.py --dataset ./alpaca-dogri-cleaned --out-dir corpus/dogri --prefix alpaca
"""

import argparse
import json
import os

from datasets import concatenate_datasets, load_dataset, load_from_disk


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Hub id or local save_to_disk directory")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--rows-per-shard", type=int, default=100_000)
    parser.add_argument("--prefix", default="shard",
                        help="Shard file prefix; use distinct prefixes to write several sources into one folder")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = load_from_disk(args.dataset) if os.path.isdir(args.dataset) else load_dataset(args.dataset)
    splits = list(dataset.values()) if hasattr(dataset, "values") else [dataset]
    combined = concatenate_datasets(splits)
    print(f"{len(combined):,} rows from {len(splits)} split(s)")

    shard_id = 0
    written = 0
    out = open(os.path.join(args.out_dir, f"{args.prefix}_{shard_id}.jsonl"), "w", encoding="utf-8")
    for example in combined:
        record = {"text": example[args.text_column]} if args.text_column in example else example
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
        if written >= args.rows_per_shard:
            out.close()
            shard_id += 1
            written = 0
            out = open(os.path.join(args.out_dir, f"{args.prefix}_{shard_id}.jsonl"), "w", encoding="utf-8")
    out.close()
    print(f"Wrote {shard_id + 1} shard(s) to {args.out_dir}/")


if __name__ == "__main__":
    main()
