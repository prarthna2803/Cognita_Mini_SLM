#!/usr/bin/env python3
"""Print how the tokenizer segments text in each language.

With no arguments it runs one built-in sentence per language; pass your own text
to see its pieces and ids.

Example:
    python tokenizer/inspect_tokenizer.py
    python tokenizer/inspect_tokenizer.py --text "ଓଡ଼ିଶା ଭାରତର ଏକ ରାଜ୍ୟ।"
"""

import argparse
import os

import sentencepiece as spm

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spm", "cognita_mini_32k.model")
SAMPLES = {
    "English": "This is a sample sentence in English.",
    "Odia": "ଏହା ଓଡ଼ିଆରେ ଏକ ନମୁନା ବାକ୍ୟ ଅଟେ।",
    "Dogri": "एह् डोगरी च इक नमूना वाक्य ऐ।",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text", nargs="*", help="Text to tokenize instead of the built-in samples")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.model)
    print(f"vocab size {sp.vocab_size():,} | unk={sp.unk_id()} bos={sp.bos_id()} eos={sp.eos_id()} pad={sp.pad_id()}\n")

    samples = {f"text {i + 1}": text for i, text in enumerate(args.text)} if args.text else SAMPLES
    for label, text in samples.items():
        pieces = sp.encode(text, out_type=str)
        ids = sp.encode(text)
        print(f"{label}: {text}")
        print(f"  {len(pieces)} tokens for {len(text.split())} words")
        print(f"  pieces: {pieces}")
        print(f"  ids:    {ids}")
        print(f"  round-trip ok: {sp.decode(ids) == text}\n")


if __name__ == "__main__":
    main()
