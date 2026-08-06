#!/usr/bin/env python3
"""Quantify what the custom tokenizer buys.

Two measurements:

1. Vocabulary composition: how many of the 32k pieces belong to each script.
2. Compression per language: characters per token and tokens per word on a text
   sample, optionally against a reference Hugging Face tokenizer.

Example:
    python tokenizer/analyze_tokenizer.py \
        --model tokenizer/spm/cognita_mini_32k.model \
        --sample english=eng_sample.txt odia=odia_sample.txt dogri=dogri_sample.txt \
        --baseline Qwen/Qwen3-0.6B \
        --output results/tokenizer_stats.json
"""

import argparse
import json
import os
from collections import Counter

import sentencepiece as spm


def script_of_piece(piece: str) -> str:
    """Script of the first letter in a piece; pieces without letters are symbols."""
    for ch in piece:
        code = ord(ch)
        if 0x0B00 <= code <= 0x0B7F:
            return "odia"
        if 0x0900 <= code <= 0x097F:
            return "devanagari"
        if (ch.isascii() and ch.isalpha()) or 0x00C0 <= code <= 0x024F:
            return "latin"
        if ch.isalpha():
            return "other_script"
    return "digits_punctuation_symbols"


def vocabulary_composition(sp: spm.SentencePieceProcessor) -> dict:
    counts = Counter(script_of_piece(sp.id_to_piece(i)) for i in range(sp.vocab_size()))
    return dict(counts.most_common())


def read_sample(path: str, max_chars: int) -> list:
    """Read whole lines from the start of a file until max_chars is reached."""
    lines, total = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            total += len(line)
            if total >= max_chars:
                break
    return lines


def compression(lines: list, encode) -> dict:
    chars = sum(len(line) for line in lines)
    words = sum(len(line.split()) for line in lines)
    utf8_bytes = sum(len(line.encode("utf-8")) for line in lines)
    tokens = sum(len(encode(line)) for line in lines)
    return {
        "tokens": tokens,
        "chars_per_token": round(chars / tokens, 3),
        "bytes_per_token": round(utf8_bytes / tokens, 3),
        "tokens_per_word": round(tokens / words, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="SentencePiece .model file")
    parser.add_argument("--sample", nargs="*", default=[], metavar="LANG=PATH",
                        help="Plain-text samples, one document or sentence per line")
    parser.add_argument("--max-chars", type=int, default=5_000_000, help="Characters to read per sample")
    parser.add_argument("--baseline", default=None, help="Hugging Face tokenizer to compare against")
    parser.add_argument("--output", default="results/tokenizer_stats.json")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.model)
    stats = {
        "vocab_size": sp.vocab_size(),
        "special_tokens": {"unk": sp.unk_id(), "bos": sp.bos_id(), "eos": sp.eos_id(), "pad": sp.pad_id()},
        "vocabulary_by_script": vocabulary_composition(sp),
        "compression": {},
    }

    baseline = None
    if args.baseline:
        from transformers import AutoTokenizer
        baseline = AutoTokenizer.from_pretrained(args.baseline)
        stats["baseline"] = {"name": args.baseline, "vocab_size": len(baseline)}

    for spec in args.sample:
        language, path = spec.split("=", 1)
        lines = read_sample(path, args.max_chars)
        entry = {
            "sample_lines": len(lines),
            "sample_chars": sum(len(line) for line in lines),
            "sample_words": sum(len(line.split()) for line in lines),
            "cognita_mini": compression(lines, lambda text: sp.encode(text)),
        }
        if baseline is not None:
            entry["baseline"] = compression(
                lines, lambda text: baseline(text, add_special_tokens=False)["input_ids"])
            entry["token_ratio_baseline_over_ours"] = round(
                entry["baseline"]["tokens"] / entry["cognita_mini"]["tokens"], 3)
        stats["compression"][language] = entry

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"vocab {stats['vocab_size']:,}: {stats['vocabulary_by_script']}")
    for language, entry in stats["compression"].items():
        ours = entry["cognita_mini"]
        line = f"  {language:8s} {ours['chars_per_token']:.2f} chars/token, {ours['tokens_per_word']:.2f} tokens/word"
        if "baseline" in entry:
            line += f" | baseline needs {entry['token_ratio_baseline_over_ours']:.2f}x the tokens"
        print(line)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
