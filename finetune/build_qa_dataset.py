#!/usr/bin/env python3
"""Build the commonsense question-answering instruction set.

Four multiple-choice benchmarks are converted to open-ended
{"instruction": question, "response": text of the correct choice} pairs:

    mCSQA (en)       yusuke1997/mCSQA        loaded from the Hub
    CommonsenseQA    tau/commonsense_qa      loaded from the Hub
    Cosmos QA        <cosmos-dir>/{train,validation}.csv   (github.com/wilburOne/cosmosqa)
    PIQA             <piqa-dir>/{train,valid}.jsonl + {train,valid}-labels.lst

The answer choices are deliberately left out of the instruction, so the model
has to produce the answer rather than pick a letter. Cosmos QA rows keep only
the question (not the context paragraph), which makes those answers plausible
continuations rather than strictly recoverable facts.

Writes one JSONL per source plus combined.jsonl into --out-dir.

Example:
    python finetune/build_qa_dataset.py --split train --out-dir ft_data/qa_train \
        --cosmos-dir cosmosqa/data --piqa-dir piqa/data
    python finetune/build_qa_dataset.py --split validation --out-dir ft_data/qa_val \
        --cosmos-dir cosmosqa/data --piqa-dir piqa/data
"""

import argparse
import csv
import json
from pathlib import Path


def clean(value):
    return None if value is None else str(value).strip()


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def pick_choice(choices: list, key):
    """Resolve an answer key (letter, 0-based or 1-based index) to the choice text."""
    if isinstance(key, str) and len(key.strip()) == 1 and key.strip().isalpha():
        index = ord(key.strip().upper()) - ord("A")
        return choices[index] if 0 <= index < len(choices) else None
    index = to_int(key)
    if index is None:
        return None
    if 0 <= index < len(choices):
        return choices[index]
    if 1 <= index <= len(choices):
        return choices[index - 1]
    return None


def question_and_answer(example: dict):
    """Extract (question, correct answer text) from a Hub multiple-choice example."""
    question = clean(example.get("question"))
    choices = example.get("choices") or {}
    texts = [clean(text) for text in choices.get("text", [])] if isinstance(choices, dict) else []
    answer = pick_choice(texts, example.get("answerKey")) if texts else None
    return question, answer


def write_jsonl(path: Path, rows: list):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_hub_multiple_choice(repo: str, config, split: str, source: str) -> list:
    from datasets import load_dataset
    dataset = load_dataset(repo, config)[split] if config else load_dataset(repo)[split]
    rows = []
    for example in dataset:
        question, answer = question_and_answer(dict(example))
        if question and answer:
            rows.append({"instruction": question, "response": answer, "_dataset": source})
    return rows


def load_cosmos(cosmos_dir: Path, split: str) -> list:
    path = cosmos_dir / f"{split}.csv"
    if not path.exists():
        print(f"[cosmos] {path} not found, skipping")
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            question = clean(row.get("question"))
            answers = [clean(row[f"answer{i}"]) for i in range(4) if (row.get(f"answer{i}") or "").strip()]
            answer = pick_choice(answers, row.get("label")) if answers else None
            if question and answer:
                rows.append({"instruction": question, "response": answer, "_dataset": "cosmos"})
    return rows


def load_piqa(piqa_dir: Path, split: str) -> list:
    stem = "valid" if split == "validation" else split
    examples_path, labels_path = piqa_dir / f"{stem}.jsonl", piqa_dir / f"{stem}-labels.lst"
    if not examples_path.exists() or not labels_path.exists():
        print(f"[piqa] {examples_path} or {labels_path} not found, skipping")
        return []
    examples = [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for example, label in zip(examples, labels):
        goal = clean(example.get("goal"))
        answer = pick_choice([clean(example.get("sol1")), clean(example.get("sol2"))], label)  # labels are 0 / 1
        if goal and answer:
            rows.append({"instruction": goal, "response": answer, "_dataset": "piqa"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cosmos-dir", default="cosmosqa/data")
    parser.add_argument("--piqa-dir", default="piqa/data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "mcsqa": load_hub_multiple_choice("yusuke1997/mCSQA", "en", args.split, "mcsqa"),
        "commonsense": load_hub_multiple_choice("tau/commonsense_qa", None, args.split, "commonsense"),
        "cosmos": load_cosmos(Path(args.cosmos_dir), args.split),
        "piqa": load_piqa(Path(args.piqa_dir), args.split),
    }

    combined = []
    for name, rows in sources.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        combined.extend(rows)
        print(f"{name}: {len(rows):,} examples")
    write_jsonl(out_dir / "combined.jsonl", combined)
    print(f"combined: {len(combined):,} examples -> {out_dir / 'combined.jsonl'}")


if __name__ == "__main__":
    main()
