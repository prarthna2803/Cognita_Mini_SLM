#!/usr/bin/env python3
"""Decode a few packed training sequences back to text.

A quick check that the data the model will see looks right: languages are
interleaved, sequences are full length, and nothing was mangled by tokenization.

Example:
    python pretrain/inspect_batch.py --tokenizer tokenizer/spm/cognita_mini_32k.model \
        --shards-dir pretrain/tokenized_shards --output batch_snapshot.txt
"""

import argparse
import glob
import os
import sys

import sentencepiece as spm
from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MAX_SEQ_LENGTH  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--num-sequences", type=int, default=2)
    parser.add_argument("--output", default="batch_snapshot.txt")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    shard_dirs = sorted(glob.glob(os.path.join(args.shards_dir, "shard*")))
    if not shard_dirs:
        raise SystemExit(f"No tokenized shards in {args.shards_dir}")

    # The first shard alone holds far more than a few sequences' worth of tokens.
    needed = args.num_sequences * MAX_SEQ_LENGTH
    ids = []
    for line in load_from_disk(shard_dirs[0])["input_ids"]:
        ids.extend(line)
        if len(ids) >= needed:
            break

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num_sequences):
            block = ids[i * MAX_SEQ_LENGTH:(i + 1) * MAX_SEQ_LENGTH]
            f.write(f"--- sequence {i + 1} ({len(block)} tokens) ---\n{sp.decode(block)}\n\n")
    print(f"Wrote {args.num_sequences} decoded sequences to {args.output}")


if __name__ == "__main__":
    main()
