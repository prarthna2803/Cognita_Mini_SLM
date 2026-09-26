#!/usr/bin/env python3
"""Assemble Hub-ready folders for the base model and the LoRA adapter.

Copies only the files needed for inference (no optimizer state or training
arguments), points the adapter at the base model's Hub id, and drops in the
model cards. The result lives in release/, which is git-ignored.

Example:
    python scripts/prepare_release.py \
        --base-model path/to/pretrain/checkpoints-final \
        --adapter path/to/finetune/lora-best \
        --hub-user ProtonPratt
"""

import argparse
import json
import os
import shutil

BASE_NAME = "Cognita-Mini-167M"
ADAPTER_NAME = "Cognita-Mini-167M-instruct-lora"
BASE_FILES = ["config.json", "generation_config.json", "model.safetensors",
              "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors",
                 "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def copy_files(source: str, destination: str, names: list):
    os.makedirs(destination, exist_ok=True)
    for name in names:
        shutil.copy2(os.path.join(source, name), os.path.join(destination, name))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True, help="Final pre-trained model directory")
    parser.add_argument("--adapter", required=True, help="Best LoRA adapter directory")
    parser.add_argument("--hub-user", required=True, help="Hugging Face user or organisation")
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "release"))
    args = parser.parse_args()

    base_dir = os.path.join(args.out_dir, BASE_NAME)
    adapter_dir = os.path.join(args.out_dir, ADAPTER_NAME)
    copy_files(args.base_model, base_dir, BASE_FILES)
    copy_files(args.adapter, adapter_dir, ADAPTER_FILES)

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, encoding="utf-8") as f:
        adapter_config = json.load(f)
    adapter_config["base_model_name_or_path"] = f"{args.hub_user}/{BASE_NAME}"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)

    cards = {"MODEL_CARD.md": base_dir, "MODEL_CARD_LORA.md": adapter_dir}
    for card, destination in cards.items():
        source = os.path.join(REPO_ROOT, "docs", card)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(destination, "README.md"))

    for folder in (base_dir, adapter_dir):
        size = sum(os.path.getsize(os.path.join(folder, name)) for name in os.listdir(folder))
        print(f"{folder}: {len(os.listdir(folder))} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
