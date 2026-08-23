#!/usr/bin/env python3
"""Break a checkpoint's parameter count down by component.

Reads only the safetensors header, so it needs neither PyTorch nor the memory
to load the weights.

Example:
    python scripts/param_breakdown.py \
        --weights release/Cognita-Mini-167M/model.safetensors \
        --output results/model_params.json
"""

import argparse
import json
import math
import os
import struct
from collections import OrderedDict

# Substring of the tensor name -> component label, checked in order.
COMPONENTS = OrderedDict([
    ("embed_tokens", "token_embedding"),
    ("lm_head", "output_head"),
    ("self_attn.q_proj", "attention"),
    ("self_attn.k_proj", "attention"),
    ("self_attn.v_proj", "attention"),
    ("self_attn.o_proj", "attention"),
    ("mlp.", "feed_forward"),
    ("norm", "normalization"),
])


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


def component_of(name: str) -> str:
    for needle, label in COMPONENTS.items():
        if needle in name:
            return label
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="Path to model.safetensors")
    parser.add_argument("--output", default="results/model_params.json")
    args = parser.parse_args()

    header = read_header(args.weights)
    by_component = {}
    layers = set()
    for name, meta in header.items():
        count = math.prod(meta["shape"])
        by_component[component_of(name)] = by_component.get(component_of(name), 0) + count
        if ".layers." in name:
            layers.add(int(name.split(".layers.")[1].split(".")[0]))

    total = sum(by_component.values())
    result = {
        "total_parameters": total,
        "num_layers": len(layers),
        "num_tensors": len(header),
        "by_component": dict(sorted(by_component.items(), key=lambda kv: -kv[1])),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"{total:,} parameters in {len(header)} tensors, {len(layers)} layers")
    for label, count in result["by_component"].items():
        print(f"  {label:16s} {count:>12,}  {count / total:6.2%}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
