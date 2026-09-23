---
license: cc-by-nc-4.0
language:
- en
- or
base_model: ProtonPratt/Cognita-Mini-167M
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- odia
- instruction-tuning
---

# Cognita-Mini-167M-instruct-lora

A LoRA adapter that turns [Cognita-Mini-167M](https://huggingface.co/ProtonPratt/Cognita-Mini-167M)
into a single-turn instruction follower for two tasks: short commonsense answers and character
counting, in English and Odia, with a small Devanagari-script subset.


| | |
|---|---|
| Method | LoRA, rank 16, α 32, dropout 0.05 |
| Target modules | q, k, v, o attention projections and gate, up, down feed-forward projections |
| Trainable parameters | 3,170,304 (1.9% of the base model) |
| Training data | 200,526 instruction-response pairs |
| Training | 3 epochs, 4,701 updates, effective batch 128, learning rate 2e-4 |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer

base = "ProtonPratt/Cognita-Mini-167M"
tokenizer = LlamaTokenizer.from_pretrained(base)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base),
                                  "ProtonPratt/Cognita-Mini-167M-instruct-lora")

prompt = "### Instruction:\nଭୋକ ଲାଗିଲେ ଲୋକମାନେ ସାଧାରଣତଃ କ'ଣ କରନ୍ତି?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The prompt format must match training exactly:

```text
### Instruction:
{instruction}

### Response:
```

## Training data

| Task | Script | Examples | Built from |
|---|---|---:|---|
| Commonsense QA | English | 62,026 | mCSQA, CommonsenseQA, Cosmos QA, PIQA |
| Commonsense QA | Odia | 30,000 | NLLB-200 translation of the English set |
| Commonsense QA | Kashmiri (Devanagari) | 10,000 | NLLB-200 translation, used as a proxy for Dogri |
| Character counting | English | 58,500 | Web sentences with injected characters |
| Character counting | Odia | 30,000 | Synthetic strings over the Odia alphabet |
| Character counting | Devanagari | 10,000 | Synthetic strings over the Devanagari alphabet |

## Evaluation

Character counting, 2,000 validation examples per script, greedy decoding:

| Script | Exact | Within ±2 | Mean abs. error |
|---|---:|---:|---:|
| English | 28.5% | 74.5% | 1.74 |
| Odia | 10.3% | 44.0% | 3.86 |
| Devanagari | 9.7% | 42.5% | 4.25 |

The base model without the adapter scores 0% exact match. PIQA validation (two-way likelihood scoring,
chance 50%): 59.0% with the adapter, 57.2% without.

## Limitations

- Free-form answers are often plausible but wrong; counting is approximate.
- The Devanagari QA subset is Kashmiri, not Dogri (NLLB-200 has no Dogri), and the Odia and Devanagari
  counting sets are random character strings rather than natural sentences.
- This adapter was trained with every example padded to 4,096 tokens and all positions included in the
  loss, so most of the training signal was padding. The training script in the repository now scores
  response tokens only; an adapter retrained that way should be better.

## License

CC BY-NC 4.0. The translated instruction data was produced with NLLB-200 (CC BY-NC 4.0).
