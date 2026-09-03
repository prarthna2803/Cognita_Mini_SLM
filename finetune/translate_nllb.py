#!/usr/bin/env python3
"""Translate an instruction-response JSONL file with NLLB-200 (3.3B).

Both the "instruction" and "response" fields are translated; every other field
is copied through and "_target_lang" is added. Output is written next to the
input as <name>.<target>.jsonl.

With several GPUs, --gpus spawns one worker per GPU, each with its own copy of
the model, and records are split round-robin between them.

Language codes are FLORES-200 codes. NLLB-200 has no Dogri: the released
"Dogri" QA set was produced with kas_Deva (Kashmiri in Devanagari script), the
closest available target. Treat that subset as Kashmiri, not Dogri.

Example:
    python finetune/translate_nllb.py --input to_translate.jsonl --target ory_Orya kas_Deva --gpus 0,1
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-3.3B"
FIELDS = ("instruction", "response")


def read_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def language_token_id(tokenizer, code: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(code)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise KeyError(f"'{code}' is not an NLLB-200 language code")
    return token_id


def load_model(device: str, model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
    return tokenizer, model.to(device).eval()


@torch.no_grad()
def translate(records: list, tokenizer, model, source: str, target: str, batch_size: int,
              max_new_tokens: int, device: str) -> list:
    """Return copies of `records` with FIELDS translated into `target`."""
    slots = [(i, field) for i, record in enumerate(records) for field in FIELDS
             if isinstance(record.get(field), str)]
    texts = [records[i][field] for i, field in slots]

    tokenizer.src_lang = source
    target_id = language_token_id(tokenizer, target)
    translations = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"-> {target} ({device})", unit="batch"):
        batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True, truncation=True)
        generated = model.generate(**batch.to(device), forced_bos_token_id=target_id,
                                   max_new_tokens=max_new_tokens, num_beams=4, early_stopping=True)
        translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    output = [dict(record) for record in records]
    for (i, field), text in zip(slots, translations):
        output[i][field] = text
    for record in output:
        record["_target_lang"] = target
    return output


def _worker(gpu: int, indexed_records: list, source: str, target: str, batch_size: int,
            max_new_tokens: int, queue):
    try:
        device = f"cuda:{gpu}"
        tokenizer, model = load_model(device)
        translated = translate([record for _, record in indexed_records], tokenizer, model,
                               source, target, batch_size, max_new_tokens, device)
        queue.put(("ok", [(index, record) for (index, _), record in zip(indexed_records, translated)]))
    except Exception as error:  # surfaced in the parent process
        queue.put(("error", f"GPU {gpu}: {error!r}"))


def translate_multi_gpu(records: list, source: str, target: str, batch_size: int,
                        max_new_tokens: int, gpus: list) -> list:
    """One process per GPU, records split round-robin, original order restored."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    shards = {gpu: [(i, r) for i, r in enumerate(records) if i % len(gpus) == slot]
              for slot, gpu in enumerate(gpus)}
    workers = [context.Process(target=_worker, args=(gpu, shard, source, target, batch_size, max_new_tokens, queue))
               for gpu, shard in shards.items() if shard]
    for worker in workers:
        worker.start()

    by_index = {}
    for _ in workers:
        status, payload = queue.get()
        if status != "ok":
            raise RuntimeError(payload)
        by_index.update(payload)
    for worker in workers:
        worker.join()
    return [by_index[i] for i in range(len(records))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="JSONL with instruction / response fields")
    parser.add_argument("--source", default="eng_Latn")
    parser.add_argument("--target", nargs="+", required=True, help="FLORES-200 codes, e.g. ory_Orya")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids; more than one enables multi-GPU mode")
    args = parser.parse_args()

    input_path = Path(args.input)
    records = read_jsonl(input_path)
    if not records:
        raise SystemExit(f"No records in {input_path}")
    gpus = [int(gpu) for gpu in args.gpus.split(",") if gpu.strip()]

    single = None  # (tokenizer, model, device), loaded once and reused across targets
    for target in args.target:
        if len(gpus) > 1 and torch.cuda.is_available():
            translated = translate_multi_gpu(records, args.source, target, args.batch_size,
                                             args.max_new_tokens, gpus)
        else:
            if single is None:
                device = f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu"
                single = (*load_model(device), device)
            tokenizer, model, device = single
            translated = translate(records, tokenizer, model, args.source, target,
                                   args.batch_size, args.max_new_tokens, device)

        out_path = input_path.with_name(f"{input_path.stem}.{target}.jsonl")
        write_jsonl(out_path, translated)
        print(f"Wrote {len(translated):,} records to {out_path}")


if __name__ == "__main__":
    main()
