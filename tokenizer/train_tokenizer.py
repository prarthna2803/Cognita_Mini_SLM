#!/usr/bin/env python3
"""Train the shared English / Odia / Dogri SentencePiece BPE tokenizer.

One 32k vocabulary covers three scripts (Latin, Odia, Devanagari). The settings
that matter:

    vocab_size=32000           keeps the embedding and output matrices at
                               28.7M parameters each instead of dominating the model
    character_coverage=0.9999  retains the long tail of Odia and Devanagari
                               characters as base symbols rather than <unk>
    balanced input             100-140 MB per language (see sample_corpus.py)

Special tokens follow the SentencePiece defaults: <unk>=0, <s>=1, </s>=2.

Example:
    python tokenizer/train_tokenizer.py --input english.txt odia.txt dogri.txt \
        --model-prefix tokenizer/spm/cognita_mini_32k
"""

import argparse

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, help="Plain-text files, one document per line")
    parser.add_argument("--model-prefix", default="cognita_mini_32k")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--character-coverage", type=float, default=0.9999)
    parser.add_argument("--input-sentence-size", type=int, default=10_000_000)
    args = parser.parse_args()

    spm.SentencePieceTrainer.train(
        input=",".join(args.input),
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
    )
    print(f"Saved {args.model_prefix}.model and {args.model_prefix}.vocab")


if __name__ == "__main__":
    main()
