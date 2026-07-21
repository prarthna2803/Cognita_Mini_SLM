#!/usr/bin/env python3
"""Merge every Odia source into one shuffled corpus of JSONL shards.

Odia text is scattered across many small datasets with different schemas. This
script normalises each one to a single "text" column, drops empty rows,
concatenates everything, shuffles with a fixed seed and writes fixed-size shards.

Expected layout under --root (each sub-folder is a datasets.save_to_disk dump):

    <root>/<subset>/            OdiaGenAI subsets listed in SUBSET_TEXT_COLUMN
    <root>/Bpcc_odia/*.jsonl    AI4Bharat BPCC, Odia side in the "tgt" field
    <root>/indicV2/*.jsonl      AI4Bharat IndicCorp v2 shards ("text" field)
    <root>/*.txt                optional raw text dumps, one sentence per line

Example:
    python data/build_odia_corpus.py --root Odia_huggingface --out-dir corpus/odia_combined
"""

import argparse
import glob
import math
import os

from datasets import concatenate_datasets, load_dataset, load_from_disk

# OdiaGenAI subset -> column holding the Odia text.
SUBSET_TEXT_COLUMN = {
    "cultureax": "text",
    "indicQA": "merged",
    "odiaencorp": "or",
    "oscar": "text",
    "paraphrasing": "combined",
    "pmo": "combined",
    "samanantar": "tgt",
    "sentiment_analysis": "merged",
    "varta": "text",
    "wiki": "text",
    "xp3": "targets",
}


def as_text_dataset(dataset, column: str):
    """Rename `column` to "text", drop empty rows and every other column."""
    if column != "text":
        dataset = dataset.rename_column(column, "text")
    dataset = dataset.filter(lambda row: bool(row["text"] and row["text"].strip()))
    return dataset.select_columns(["text"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Folder holding the downloaded Odia sources")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rows-per-shard", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    parts = []
    for subset, column in SUBSET_TEXT_COLUMN.items():
        path = os.path.join(args.root, subset)
        if not os.path.isdir(path):
            print(f"skipping missing subset: {subset}")
            continue
        print(f"loading {subset}")
        parts.append(as_text_dataset(load_from_disk(path)["train"], column))

    for folder, column in (("Bpcc_odia", "tgt"), ("indicV2", "text")):
        for path in sorted(glob.glob(os.path.join(args.root, folder, "*.jsonl"))):
            print(f"loading {path}")
            parts.append(as_text_dataset(load_dataset("json", data_files=path)["train"], column))

    for path in sorted(glob.glob(os.path.join(args.root, "*.txt"))):
        print(f"loading {path}")
        parts.append(load_dataset("text", data_files=path)["train"])

    if not parts:
        raise SystemExit(f"No Odia sources found under {args.root}")

    combined = concatenate_datasets(parts).shuffle(seed=args.seed)
    n_shards = math.ceil(len(combined) / args.rows_per_shard)
    print(f"{len(combined):,} rows -> {n_shards} shards")

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(n_shards):
        start = i * args.rows_per_shard
        shard = combined.select(range(start, min(start + args.rows_per_shard, len(combined))))
        shard.to_json(os.path.join(args.out_dir, f"shard_{i:03d}.jsonl"), force_ascii=False)
    print(f"Saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
