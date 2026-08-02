#!/usr/bin/env python3
"""Draw a size-capped random text sample from one language's cleaned shards.

The tokenizer is trained on roughly equal amounts of text per language so that
English, which dominates the full corpus, does not also dominate the vocabulary.
Run once per language; the outputs feed train_tokenizer.py.

Example (sizes used for the released tokenizer):
    python tokenizer/sample_corpus.py --input-dir corpus/english               --output english.txt --megabytes 140
    python tokenizer/sample_corpus.py --input-dir corpus/odia_combined_cleaned --output odia.txt    --megabytes 100
    python tokenizer/sample_corpus.py --input-dir corpus/dogri_cleaned         --output dogri.txt   --megabytes 100
"""

import argparse
import glob
import json
import os
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Folder of cleaned .jsonl shards")
    parser.add_argument("--output", required=True, help="Plain-text output, one document per line")
    parser.add_argument("--megabytes", type=float, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    lines = []
    for path in sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as infile:
            lines.extend(infile.readlines())
    print(f"{len(lines):,} lines in {args.input_dir}")
    random.Random(args.seed).shuffle(lines)

    target_bytes = int(args.megabytes * 1024 * 1024)
    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for line in lines:
            try:
                text = json.loads(line).get("text", "").strip()
            except json.JSONDecodeError:
                continue
            if not text:
                continue
            out.write(text + "\n")
            written += len(text.encode("utf-8")) + 1
            if written >= target_bytes:
                break
    print(f"Wrote {written / 1024 / 1024:.1f} MB to {args.output}")


if __name__ == "__main__":
    main()
