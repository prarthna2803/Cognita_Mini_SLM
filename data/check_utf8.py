#!/usr/bin/env python3
"""Sanity-check a folder of JSONL shards: every line parses and every text field
round-trips through UTF-8. Run after cleaning and before tokenizer training.

Example:
    python data/check_utf8.py corpus/odia_combined_cleaned
"""

import json
import os
import sys


def check_folder(folder: str) -> list:
    """Return (file, line number, problem) for every bad line."""
    problems = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(folder, name), encoding="utf-8", errors="replace") as f:
            for number, line in enumerate(f, 1):
                try:
                    text = json.loads(line).get("text", "")
                    text.encode("utf-8")
                    if "�" in text:
                        problems.append((name, number, "replacement character in text"))
                except json.JSONDecodeError as error:
                    problems.append((name, number, f"invalid JSON: {error}"))
                except UnicodeEncodeError as error:
                    problems.append((name, number, f"invalid UTF-8: {error}"))
    return problems


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python data/check_utf8.py <shards-dir>")
    found = check_folder(sys.argv[1])
    for name, number, problem in found[:50]:
        print(f"{name}:{number}: {problem}")
    print(f"{len(found)} problem line(s)" if found else "All text fields are valid UTF-8.")
    sys.exit(1 if found else 0)
