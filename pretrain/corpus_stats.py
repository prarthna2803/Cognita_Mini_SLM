#!/usr/bin/env python3
"""Measure the pre-training corpus exactly as the model saw it.

Reads the pre-tokenized Arrow shards produced by ``pretokenize.py`` and reports
the number of lines and tokens per language. Each line is assigned to a language
by the majority Unicode script of its tokens (Latin -> English, Odia -> Odia,
Devanagari -> Dogri), so the numbers are in units of the final 32k tokenizer
rather than an external one.

Only Arrow metadata and the token-id buffers are touched (memory-mapped), so the
full 2.4B-token corpus is scanned in a couple of minutes on CPU.

Example:
    python pretrain/corpus_stats.py \
        --shards-dir pretrain/tokenized_shards \
        --tokenizer tokenizer/spm/cognita_mini_32k.model \
        --output results/corpus_stats.json
"""

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import sentencepiece as spm

SCRIPTS = ("latin", "odia", "devanagari", "other")
LANGUAGE_OF_SCRIPT = {"latin": "english", "odia": "odia", "devanagari": "dogri", "other": "other"}


def script_of_piece(piece: str) -> int:
    """Index into SCRIPTS for the first letter found in a vocabulary piece."""
    for ch in piece:
        code = ord(ch)
        if 0x0B00 <= code <= 0x0B7F:
            return 1
        if 0x0900 <= code <= 0x097F:
            return 2
        if ch.isascii() and ch.isalpha():
            return 0
        if 0x00C0 <= code <= 0x024F:  # Latin-1 supplement and Latin extended letters
            return 0
    return 3


def build_script_table(tokenizer_path: str) -> np.ndarray:
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    return np.array([script_of_piece(sp.id_to_piece(i)) for i in range(sp.vocab_size())], dtype=np.int8)


def scan_shard(args):
    """Return (rows, tokens) per script for one tokenized shard."""
    shard_dir, script_table = args
    rows = np.zeros(len(SCRIPTS), dtype=np.int64)
    tokens = np.zeros(len(SCRIPTS), dtype=np.int64)

    for path in sorted(glob.glob(os.path.join(shard_dir, "data-*.arrow"))):
        reader = pa.ipc.open_stream(pa.memory_map(path, "r"))
        for batch in reader:
            column = batch.column(0)
            offsets = column.offsets.to_numpy()
            lengths = np.diff(offsets)
            ids = column.values.to_numpy(zero_copy_only=False)[offsets[0]:offsets[-1]]
            scripts = script_table[ids]

            keep = lengths > 0  # reduceat needs strictly increasing, in-range starts
            starts = (offsets[:-1] - offsets[0])[keep]
            if len(starts) == 0:
                continue
            # Tokens of each script per line; the "other" column is ignored when voting.
            per_line = np.stack([np.add.reduceat(scripts == s, starts) for s in range(3)], axis=1)
            winner = per_line.argmax(axis=1)
            winner[per_line.sum(axis=1) == 0] = 3

            rows += np.bincount(winner, minlength=len(SCRIPTS))
            tokens += np.bincount(winner, weights=lengths[keep], minlength=len(SCRIPTS)).astype(np.int64)
    return rows, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards-dir", required=True, help="Directory of tokenized shard_* folders")
    parser.add_argument("--tokenizer", required=True, help="SentencePiece .model file")
    parser.add_argument("--output", default="results/corpus_stats.json")
    parser.add_argument("--seq-length", type=int, default=4096, help="Packed sequence length used for training")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    shard_dirs = sorted(glob.glob(os.path.join(args.shards_dir, "shard*")))
    if not shard_dirs:
        raise SystemExit(f"No shard* directories found in {args.shards_dir}")

    script_table = build_script_table(args.tokenizer)
    rows = np.zeros(len(SCRIPTS), dtype=np.int64)
    tokens = np.zeros(len(SCRIPTS), dtype=np.int64)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for shard_rows, shard_tokens in pool.map(scan_shard, [(d, script_table) for d in shard_dirs]):
            rows += shard_rows
            tokens += shard_tokens

    total_tokens = int(tokens.sum())
    stats = {
        "tokenized_shards": len(shard_dirs),
        "total_lines": int(rows.sum()),
        "total_tokens": total_tokens,
        "packed_sequences_upper_bound": total_tokens // args.seq_length,
        "seq_length": args.seq_length,
        "by_language": {
            LANGUAGE_OF_SCRIPT[script]: {
                "script": script,
                "lines": int(rows[i]),
                "tokens": int(tokens[i]),
                "token_share": round(float(tokens[i]) / total_tokens, 6),
            }
            for i, script in enumerate(SCRIPTS)
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"{stats['total_lines']:,} lines, {total_tokens:,} tokens in {len(shard_dirs)} shards")
    for language, entry in stats["by_language"].items():
        print(f"  {language:8s} {entry['tokens']:>15,} tokens ({entry['token_share']:.2%}), {entry['lines']:,} lines")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
