#!/usr/bin/env python3
"""Clean and normalise raw JSONL shards for every language in the corpus.

Each document goes through the same script-agnostic pipeline:

    1. Unicode NFC normalisation, so visually identical Odia / Devanagari
       sequences map to the same code points (and therefore the same tokens)
    2. HTML tag removal
    3. URL removal (http(s):// and www.)
    4. control-character removal (C0 and C1 ranges); letters of every script are kept
    5. whitespace collapsing

Documents with five words or fewer after cleaning are dropped. Shards are
processed in parallel, one thread per shard.

Example:
    python data/clean_shards.py --input-dir corpus/odia_combined --output-dir corpus/odia_combined_cleaned
"""

import argparse
import glob
import json
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

HTML_TAG = re.compile(r"<.*?>")
URL = re.compile(r"https?://\S+|www\.\S+")
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
MIN_WORDS = 5


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = HTML_TAG.sub("", text)
    text = URL.sub("", text)
    text = CONTROL_CHARS.sub("", text)
    return " ".join(text.split())


def clean_shard(shard_path: str, output_dir: str):
    """Clean one shard. Returns (shard name, lines read, lines written)."""
    name = os.path.basename(shard_path)
    lines_read = lines_written = 0
    with open(shard_path, encoding="utf-8") as infile, \
            open(os.path.join(output_dir, name), "w", encoding="utf-8") as outfile:
        for line in infile:
            lines_read += 1
            try:
                doc = json.loads(line)
                text = doc.get("text", "").strip()
            except (json.JSONDecodeError, AttributeError):
                continue
            if not text:
                continue
            cleaned = clean_text(text)
            if len(cleaned.split()) > MIN_WORDS:
                doc["text"] = cleaned
                outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")
                lines_written += 1
    return name, lines_read, lines_written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None, help="Defaults to <input-dir>_cleaned")
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--threads", type=int, default=None, help="Defaults to one per shard, capped at CPU count")
    args = parser.parse_args()

    output_dir = args.output_dir or f"{args.input_dir.rstrip('/')}_cleaned"
    os.makedirs(output_dir, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not shards:
        raise SystemExit(f"No shards matching {args.pattern} in {args.input_dir}")
    threads = args.threads or min(len(shards), os.cpu_count() or 4)
    print(f"Cleaning {len(shards)} shards with {threads} threads -> {output_dir}")

    total_read = total_written = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(clean_shard, shard, output_dir) for shard in shards]
        for future in tqdm(as_completed(futures), total=len(futures), desc="shards"):
            name, lines_read, lines_written = future.result()
            total_read += lines_read
            total_written += lines_written
            tqdm.write(f"{name}: read {lines_read:,}, kept {lines_written:,}")

    print(f"Total: read {total_read:,} lines, kept {total_written:,} "
          f"({total_written / max(total_read, 1):.1%})")


if __name__ == "__main__":
    main()
