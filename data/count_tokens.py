#!/usr/bin/env python3
"""Count examples and tokens in a folder of JSONL shards, in parallel.

Works with either a SentencePiece ``.model`` file (the project tokenizer) or any
Hugging Face tokenizer. Per-shard and total counts are written to
``token_stats.json`` inside the shard folder.

Example:
    python data/count_tokens.py --shards-dir corpus/odia_combined_cleaned \
        --tokenizer tokenizer/spm/cognita_mini_32k.model
"""

import argparse
import glob
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

_tokenizer = None  # loaded lazily, once per worker process


def _encode_batch(texts: list, tokenizer_name: str) -> list:
    """Token count for each text."""
    global _tokenizer
    is_sentencepiece = tokenizer_name.endswith(".model")
    if _tokenizer is None:
        if is_sentencepiece:
            import sentencepiece as spm
            _tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_name)
        else:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=int(1e12))
    if is_sentencepiece:
        return [len(ids) for ids in _tokenizer.encode(texts)]
    return [len(ids) for ids in _tokenizer(texts, add_special_tokens=False)["input_ids"]]


def count_shard(path: str, tokenizer_name: str, batch_size: int):
    """Return (shard name, examples, tokens)."""
    opener = gzip.open if path.endswith(".gz") else open
    examples = tokens = 0
    batch = []

    def flush():
        nonlocal examples, tokens
        if batch:
            tokens += sum(_encode_batch(batch, tokenizer_name))
            examples += len(batch)
            batch.clear()

    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                text = (json.loads(line).get("text") or "").strip()
            except json.JSONDecodeError:
                continue
            if text:
                batch.append(text)
            if len(batch) >= batch_size:
                flush()
    flush()
    return os.path.basename(path), examples, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--tokenizer", required=True, help="SentencePiece .model path or Hugging Face tokenizer id")
    parser.add_argument("--pattern", default="*.jsonl*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-name", default="token_stats.json")
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(args.shards_dir, args.pattern)))
    if not shards:
        raise SystemExit(f"No shards matching {args.pattern} in {args.shards_dir}")
    print(f"Counting {len(shards)} shards with {args.tokenizer}")

    per_shard = {}
    worker = partial(count_shard, tokenizer_name=args.tokenizer, batch_size=args.batch_size)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, shard) for shard in shards]
        for future in as_completed(futures):
            name, examples, tokens = future.result()
            per_shard[name] = {"examples": examples, "tokens": tokens}
            print(f"{name}: {examples:,} examples, {tokens:,} tokens")

    total_examples = sum(entry["examples"] for entry in per_shard.values())
    total_tokens = sum(entry["tokens"] for entry in per_shard.values())
    summary = {
        "tokenizer": args.tokenizer,
        "total_examples": total_examples,
        "total_tokens": total_tokens,
        "avg_tokens_per_example": round(total_tokens / max(total_examples, 1), 2),
        "shards": dict(sorted(per_shard.items())),
    }
    out_path = os.path.join(args.shards_dir, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Total: {total_examples:,} examples, {total_tokens:,} tokens -> {out_path}")


if __name__ == "__main__":
    main()
