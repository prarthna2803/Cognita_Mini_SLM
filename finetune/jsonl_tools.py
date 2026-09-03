#!/usr/bin/env python3
"""Small JSONL utilities used to assemble the fine-tuning sets.

Subcommands:
    sample   randomly keep a percentage of lines (to pick the subset sent for translation)
    split    first N lines to one file, the rest to another (train / validation)
    merge    concatenate instruction-response files and shuffle

Example:
    python finetune/jsonl_tools.py sample --input qa_train/combined.jsonl --output to_translate.jsonl --percent 60
    python finetune/jsonl_tools.py split  --input to_translate.ory_Orya.jsonl --head qa_or_train.jsonl \
        --tail qa_or_val.jsonl --lines 30000
    python finetune/jsonl_tools.py merge  --inputs char_en_train.jsonl qa_train/combined.jsonl --output en_train.jsonl
"""

import argparse
import json
import random


def sample(args):
    with open(args.input, encoding="utf-8") as f:
        lines = f.readlines()
    kept = random.Random(args.seed).sample(lines, int(len(lines) * args.percent / 100))
    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(kept)
    print(f"Sampled {len(kept):,} of {len(lines):,} lines ({args.percent}%) -> {args.output}")


def split(args):
    head = tail = 0
    with open(args.input, encoding="utf-8") as source, \
            open(args.head, "w", encoding="utf-8") as head_file, \
            open(args.tail, "w", encoding="utf-8") as tail_file:
        for index, line in enumerate(source):
            if index < args.lines:
                head_file.write(line)
                head += 1
            else:
                tail_file.write(line)
                tail += 1
    print(f"{head:,} lines -> {args.head}, {tail:,} lines -> {args.tail}")


def merge(args):
    rows = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "instruction" in row and "response" in row:
                    rows.append(row)
    random.Random(args.seed).shuffle(rows)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Merged {len(rows):,} examples from {len(args.inputs)} files -> {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("sample")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--percent", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(run=sample)

    p = commands.add_parser("split")
    p.add_argument("--input", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--tail", required=True)
    p.add_argument("--lines", type=int, required=True)
    p.set_defaults(run=split)

    p = commands.add_parser("merge")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(run=merge)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
