#!/usr/bin/env python3
"""Generate the synthetic character-counting instruction set.

Each example asks how many times one character occurs in a sentence:

    {"instruction": "How many times does the character 'z' appear in \\"...\\"?", "response": "7"}

Picking a random character from a random sentence gives an answer of 0 or 1
almost every time, which teaches nothing. Instead the target character is
injected at random positions (up to 20 times, 85% of the time), which flattens
the answer distribution across 0-20.

Modes:
    english   real sentences streamed from tiiuae/falcon-refinedweb
    odia      synthetic strings sampled from the Odia alphabet
    dogri     synthetic strings sampled from the Devanagari alphabet

The Odia and Dogri modes produce random character sequences, not natural
sentences: they test counting in each script without relying on vocabulary.

Example:
    python finetune/build_char_count_dataset.py --mode english --train-size 58500 --val-size 7000 \
        --output-prefix ft_data/char_count_en
"""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter

LATIN_LOWER = list("abcdefghijklmnopqrstuvwxyz")
LATIN_UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ASCII_DIGITS = list("0123456789")
ASCII_PUNCTUATION = list(".,!?;:'\"-()[]{}")
ASCII_SYMBOLS = list("@#$%^&*+=_~|/\\")
DANDA = list("।॥")

ODIA_VOWELS = list("ଅଆଇଈଉଊଋୠଌୡଏଐଓଔ")
ODIA_CONSONANTS = list("କଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମୟରଲଶଷସହଳ")
ODIA_DIGITS = list("୦୧୨୩୪୫୬୭୮୯")

DEVANAGARI_VOWELS = list("अआइईउऊएऐओऔऋॠॐ")
DEVANAGARI_CONSONANTS = list("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
DEVANAGARI_DIGITS = list("०१२३४५६७८९")

ASCII_POOLS = [
    ("latin_lowercase", LATIN_LOWER),
    ("latin_uppercase", LATIN_UPPER),
    ("digit", ASCII_DIGITS),
    ("punctuation", ASCII_PUNCTUATION),
    ("symbol", ASCII_SYMBOLS),
]
# Pools the target character (and synthetic text) is drawn from, per mode.
LANGUAGE_POOLS = {
    "english": ASCII_POOLS,
    "odia": [("vowel", ODIA_VOWELS), ("consonant", ODIA_CONSONANTS), ("digit", ODIA_DIGITS), ("danda", DANDA)],
    "dogri": [("vowel", DEVANAGARI_VOWELS), ("consonant", DEVANAGARI_CONSONANTS),
              ("digit", DEVANAGARI_DIGITS), ("danda", DANDA)],
}


def english_sentences():
    """Yield 3-30 word sentences from RefinedWeb."""
    from datasets import load_dataset
    for row in load_dataset("tiiuae/falcon-refinedweb", split="train", streaming=True):
        for sentence in re.split(r"[.!?]", row["content"]):
            sentence = sentence.strip()
            if 3 <= len(sentence.split()) <= 30:
                yield unicodedata.normalize("NFC", sentence)


def synthetic_sentences(mode: str, min_words: int, max_words: int, min_word_len: int, max_word_len: int):
    """Yield space-separated random 'words' over the language's alphabet (vowels weighted x2)."""
    pools = dict(LANGUAGE_POOLS[mode])
    alphabet = pools["vowel"] * 2 + pools["consonant"] + pools["digit"] + pools["danda"]
    while True:
        words = ["".join(random.choice(alphabet) for _ in range(random.randint(min_word_len, max_word_len)))
                 for _ in range(random.randint(min_words, max_words))]
        sentence = " ".join(words)
        if random.random() < 0.6:
            sentence += random.choice(DANDA + ASCII_PUNCTUATION)
        yield unicodedata.normalize("NFC", sentence)


def inject(sentence: str, pools: list, probability: float, max_inserts: int, char=None) -> str:
    """Insert `char` (or random characters from `pools`) at random positions."""
    if random.random() > probability:
        return sentence
    count = random.randint(1, max_inserts)
    inserts = [char] * count if char is not None else [random.choice(random.choice(pools)[1]) for _ in range(count)]
    chars = list(sentence)
    for ch in inserts:
        chars.insert(random.randint(0, len(chars)), ch)
    return "".join(chars)


def make_example(sentence: str, mode: str) -> dict:
    if mode == "english":
        target = random.choice(random.choice(ASCII_POOLS)[1])
        sentence = inject(sentence, ASCII_POOLS, probability=0.85, max_inserts=20, char=target)
    else:
        # Letters are weighted x2 over digits and danda when picking the target.
        candidates = [ch for name, pool in LANGUAGE_POOLS[mode]
                      for ch in pool * (2 if name in ("vowel", "consonant") else 1)]
        target = random.choice(candidates)
        noise_pools = ASCII_POOLS + LANGUAGE_POOLS[mode]
        sentence = inject(sentence, noise_pools, probability=0.85, max_inserts=20, char=target)
        sentence = inject(sentence, noise_pools, probability=0.50, max_inserts=8)
    return {
        "instruction": f"How many times does the character '{target}' appear in \"{sentence}\"?",
        "response": str(sentence.count(target)),
    }


def write_jsonl(path: str, examples: list) -> dict:
    """Write examples and return the answer distribution."""
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    return dict(sorted(Counter(int(example["response"]) for example in examples).items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=sorted(LANGUAGE_POOLS), default="english")
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--val-size", type=int, default=7100)
    parser.add_argument("--output-prefix", default="char_count")
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=30)
    parser.add_argument("--min-word-len", type=int, default=1)
    parser.add_argument("--max-word-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    total = args.train_size + args.val_size
    sentences = english_sentences() if args.mode == "english" else synthetic_sentences(
        args.mode, args.min_words, args.max_words, args.min_word_len, args.max_word_len)

    examples = []
    for sentence in sentences:
        if 3 <= len(sentence.split()) <= 30:
            examples.append(make_example(sentence, args.mode))
        if len(examples) >= total:
            break
    random.shuffle(examples)

    splits = {"train": examples[:args.train_size], "val": examples[args.train_size:]}
    for name, rows in splits.items():
        path = f"{args.output_prefix}_{name}.jsonl"
        distribution = write_jsonl(path, rows)
        with open(f"{args.output_prefix}_{name}_distribution.json", "w", encoding="utf-8") as f:
            json.dump(distribution, f, indent=2)
        mean = sum(count * n for count, n in distribution.items()) / max(len(rows), 1)
        print(f"{path}: {len(rows):,} examples, answers {min(distribution)}-{max(distribution)}, mean {mean:.2f}")


if __name__ == "__main__":
    main()
