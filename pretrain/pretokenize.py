#!/usr/bin/env python3
"""Tokenize the mixed pre-training corpus once, ahead of training.

Each text shard (one document per line) becomes a Hugging Face dataset with a
single "input_ids" column, saved under --out-dir/<shard name>. Tokenizing up
front keeps the GPUs from waiting on CPU-bound tokenization during training.

No BOS or EOS tokens are inserted: lines are concatenated directly when
sequences are packed in train.py.

Example:
    python pretrain/pretokenize.py --shards-dir corpus/pretrain_mix \
        --tokenizer tokenizer/spm/cognita_mini_32k.model --out-dir pretrain/tokenized_shards
"""

import argparse
import glob
import os
from multiprocessing import cpu_count

import sentencepiece as spm
from datasets import Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards-dir", required=True, help="Folder of shard*.txt files")
    parser.add_argument("--tokenizer", required=True, help="SentencePiece .model file")
    parser.add_argument("--out-dir", default="pretrain/tokenized_shards")
    parser.add_argument("--num-proc", type=int, default=max(cpu_count() - 1, 1))
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    def encode(batch):
        return {"input_ids": sp.encode(batch["text"], out_type=int)}

    shard_files = sorted(glob.glob(os.path.join(args.shards_dir, "shard*.txt")))
    if not shard_files:
        raise SystemExit(f"No shard*.txt files in {args.shards_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    for path in shard_files:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        dataset = Dataset.from_dict({"text": lines})
        tokenized = dataset.map(encode, batched=True, num_proc=args.num_proc, remove_columns=["text"])
        tokenized.save_to_disk(os.path.join(args.out_dir, name))
        print(f"{name}: {len(lines):,} lines")

    print(f"Tokenized {len(shard_files)} shards into {args.out_dir}/")


if __name__ == "__main__":
    main()
