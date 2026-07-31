#!/usr/bin/env python3
"""Sample the cleaned per-language corpora into the final mixed pre-training corpus.

Every language has a token budget measured with the project tokenizer. Lines are
drawn one at a time; the language of each draw is chosen with probability
proportional to the tokens that language still needs. Languages are therefore
interleaved evenly through the whole corpus instead of arriving in blocks, and a
language simply stops contributing when it runs out of text.

Output is plain text, one document per line, in shards of --shard-tokens tokens.

Example (budgets used for the released model):
    python data/build_pretrain_mix.py \
        --tokenizer tokenizer/spm/cognita_mini_32k.model \
        --corpus english=corpus/english odia=corpus/odia_combined_cleaned dogri=corpus/dogri_cleaned \
        --budget english=1600000000 odia=1200000000 dogri=1200000000 \
        --out-dir corpus/pretrain_mix

Odia and Dogri ran out of text before reaching their budgets, so the released
corpus contains every cleaned Odia and Dogri line that was available.
"""

import argparse
import glob
import json
import os
import random

import sentencepiece as spm
from tqdm import tqdm


def parse_pairs(pairs: list, cast=str) -> dict:
    return {key: cast(value) for key, value in (pair.split("=", 1) for pair in pairs)}


def load_lines(folder: str, language: str) -> list:
    lines = []
    for path in tqdm(sorted(glob.glob(os.path.join(folder, "*.jsonl"))), desc=f"reading {language}", unit="file"):
        with open(path, encoding="utf-8") as infile:
            for line in infile:
                try:
                    text = json.loads(line).get("text", "").strip()
                except (json.JSONDecodeError, AttributeError):
                    continue
                if text:
                    lines.append(text)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="SentencePiece .model file")
    parser.add_argument("--corpus", nargs="+", required=True, metavar="LANG=DIR")
    parser.add_argument("--budget", nargs="+", required=True, metavar="LANG=TOKENS")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-tokens", type=int, default=50_000_000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    folders = parse_pairs(args.corpus)
    budgets = parse_pairs(args.budget, int)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    buffers = {}
    for language in budgets:
        lines = load_lines(folders[language], language)
        rng.shuffle(lines)
        buffers[language] = lines
        print(f"{language}: {len(lines):,} lines")

    os.makedirs(args.out_dir, exist_ok=True)
    written = {language: 0 for language in budgets}
    shard_id = 0
    shard_tokens = 0
    out = open(os.path.join(args.out_dir, f"shard_{shard_id:05d}.txt"), "w", encoding="utf-8")
    progress = tqdm(total=sum(budgets.values()), desc="sampling", unit="tok")

    while True:
        active = [lang for lang in budgets if written[lang] < budgets[lang] and buffers[lang]]
        if not active:
            break
        remaining = [budgets[lang] - written[lang] for lang in active]
        language = rng.choices(active, weights=remaining, k=1)[0]

        line = buffers[language].pop()
        n_tokens = len(sp.encode(line))
        out.write(line + "\n")
        written[language] += n_tokens
        shard_tokens += n_tokens
        progress.update(n_tokens)

        if shard_tokens >= args.shard_tokens:
            out.close()
            shard_id += 1
            shard_tokens = 0
            out = open(os.path.join(args.out_dir, f"shard_{shard_id:05d}.txt"), "w", encoding="utf-8")

    out.close()
    progress.close()
    for language, count in written.items():
        print(f"{language}: {count:,} / {budgets[language]:,} tokens ({count / budgets[language]:.1%} of budget)")
    print(f"Wrote {shard_id + 1} shards to {args.out_dir}/")


if __name__ == "__main__":
    main()
