# Cognita-Mini: technical report

This document records how Cognita-Mini was built, why each decision was made, what was measured, and
what went wrong. The [README](../README.md) is the summary; this is the detail. Every number here comes
from a file in [`results/`](../results), produced by a script in this repository.

Contents: [1. Goal](#1-goal) · [2. Corpus](#2-corpus) · [3. Tokenizer](#3-tokenizer) ·
[4. Model](#4-model) · [5. Pre-training](#5-pre-training) · [6. Instruction tuning](#6-instruction-tuning) ·
[7. Evaluation](#7-evaluation) · [8. Lessons](#8-lessons) · [9. Reproducibility](#9-reproducibility)

## 1. Goal

Build a small autoregressive language model end to end, with nothing borrowed but the architecture:
collect a multilingual corpus, train a tokenizer on it, pre-train a model of roughly 150M parameters
from random initialisation, and adapt it to follow instructions.

The three languages were chosen to span the resource spectrum. English is the high-resource anchor.
Odia has substantial web and news text but is badly served by mainstream tokenizers. Dogri is
low-resource in the strict sense: all the clean text that could be found amounts to 10M tokens.

## 2. Corpus

### 2.1 Sources

| Language | Source | Notes | Licence on the Hub |
|---|---|---|---|
| English | `wikimedia/wikipedia`, `20231101.en` | encyclopedic text | CC BY-SA 3.0, GFDL |
| English | `HuggingFaceFW/fineweb`, `CC-MAIN-2025-26` | filtered web crawl | ODC-By |
| Odia | AI4Bharat IndicCorp v2, Odia split | news and web sentences | not tagged, see the dataset page |
| Odia | AI4Bharat BPCC, Odia side | parallel-corpus target sentences | not tagged, see the dataset page |
| Odia | OdiaGenAI collections (11 subsets) | culture, QA, OSCAR, PMO, Samanantar, Varta, Wikipedia and others | see each dataset page |
| Dogri | CIIL Dogri corpus | 14k passages | see the dataset page |
| Dogri | Dogri Alpaca-style instruction set | 52k instruction-response passages | see the dataset page |

English was downloaded by streaming and stopped at a budget of one billion tokens per source. Odia and
Dogri were taken whole.

### 2.2 Cleaning

[`data/clean_shards.py`](../data/clean_shards.py) applies one pipeline to every language:

1. Unicode NFC normalisation. Odia and Devanagari have characters with more than one valid encoding;
   without normalisation the same word produces different tokens.
2. HTML tag removal.
3. URL removal (`http(s)://` and `www.`).
4. Removal of C0 and C1 control characters. Nothing else is filtered by character class, so no script
   is damaged by a rule written with another script in mind.
5. Whitespace collapsing.

Lines with five words or fewer after cleaning are dropped. This removed 6% of raw Dogri lines and 10%
of raw IndicCorp Odia lines, and almost nothing from English, where a line is a whole document.

### 2.3 Mixing

[`data/build_pretrain_mix.py`](../data/build_pretrain_mix.py) draws one line at a time. The language of
each draw is sampled with probability proportional to the tokens that language still owes its budget.
The effect is that all languages are spread evenly through the corpus and finish together, and a
language that runs out of text simply stops being drawn.

Budgets were English 1.6B, Odia 1.2B, and "everything" for Dogri. Odia and Dogri both ran out of
text before reaching their budgets.

### 2.4 What the model actually saw

Measured on the pre-tokenized shards with the final tokenizer
([`pretrain/corpus_stats.py`](../pretrain/corpus_stats.py), [`results/corpus_stats.json`](../results/corpus_stats.json)):

| Language | Lines | Tokens | Share | Tokens per line |
|---|---:|---:|---:|---:|
| English | 2,015,071 | 1,602,942,297 | 66.21% | 795 |
| Odia | 15,071,896 | 807,950,933 | 33.37% | 54 |
| Dogri | 68,105 | 10,030,800 | 0.41% | 147 |
| Total | 17,155,073 | 2,420,924,049 | | |

Lines are assigned to a language by the majority script of their tokens, so the Dogri row is strictly
"Devanagari-script lines" and may include a small number of Hindi sentences from the Odia sources.

The original target was three billion tokens with 30 to 40% Odia and 10 to 20% Dogri. The corpus fell
short on both: there is not that much clean Odia text in the sources used, and nowhere near that much
Dogri anywhere. An earlier version of the pipeline upsampled Dogri by repetition; the final corpus does
not, so each Dogri line appears once per epoch.

## 3. Tokenizer

### 3.1 Design

SentencePiece BPE, 32,000 pieces, character coverage 0.9999, trained by
[`tokenizer/train_tokenizer.py`](../tokenizer/train_tokenizer.py).

- **Why train one.** The model uses the Qwen3 architecture, and reusing Qwen3's tokenizer would have
  been the easy path. Two measurements ruled it out. Its 151,669-piece vocabulary would put 272M
  parameters in the embedding and output matrices, against a total budget of about 150M. And it encodes
  Odia at 12.8 tokens per word.
- **Why 32k.** At hidden size 896, each 1k of vocabulary costs 1.8M parameters across the untied
  embedding and output matrices. 32k keeps them at 57M, 34% of the model.
- **Why 0.9999 coverage.** Three scripts have many rare characters (conjuncts, vowel signs, digits). A
  lower coverage would turn the rare ones into `<unk>`.
- **Why balance the training text.** English is two thirds of the corpus. Training the tokenizer on
  the corpus as-is would spend most of the vocabulary on English. Instead it was trained on a sample of
  about 140 MB English, 100 MB Odia and 96 MB Dogri (all the Dogri there is).

### 3.2 Result

[`tokenizer/analyze_tokenizer.py`](../tokenizer/analyze_tokenizer.py),
[`results/tokenizer_stats.json`](../results/tokenizer_stats.json):

| Script of the piece | Pieces | Share |
|---|---:|---:|
| Latin | 14,076 | 44.0% |
| Odia | 8,907 | 27.8% |
| Devanagari | 7,951 | 24.8% |
| Digits, punctuation, symbols, other | 1,066 | 3.3% |

| Language | Tokens per word, this tokenizer | Tokens per word, Qwen3 tokenizer | Ratio |
|---|---:|---:|---:|
| English | 1.455 | 1.370 | 0.94× |
| Odia | 1.488 | 12.812 | 8.61× |
| Dogri | 1.306 | 4.891 | 3.75× |

Measured on the first 5M characters of each language's tokenizer-training sample, so these are
in-sample figures for this tokenizer. The gap on Odia is far larger than any in-sample advantage.

## 4. Model

[`pretrain/config.py`](../pretrain/config.py) builds a `Qwen3ForCausalLM` from a fresh configuration.
The implementation is the one in Hugging Face Transformers; what this project chose is the shape.

| Setting | Value | Reason |
|---|---|---|
| Layers | 12 | standard depth for this parameter range |
| Hidden size | 896 | with 12 layers and a 32k vocabulary this lands near the parameter target |
| Query heads / KV heads | 12 / 4 | grouped-query attention: three query heads share one key/value head, cutting key/value projection parameters and the KV cache to a third |
| Head dimension | 128 | Qwen3 default; query and output projections are 896 ↔ 1,536 |
| Feed-forward width | 2,048 | SwiGLU with three matrices, about 2.3× the hidden size |
| Positions | RoPE, θ = 10,000 | no learned position table, so context length costs no parameters |
| Context | 4,096 | long enough to pack dozens of documents per sequence |
| Embeddings | untied | input and output matrices learn separately |

Total 167,469,952 parameters: feed-forward 66.1M, attention 44.0M, token embedding 28.7M, output head
28.7M ([`results/model_params.json`](../results/model_params.json)). This is above the original
100 to 150M target; the overshoot is the untied output head. Tying the embeddings would have brought
the model to 139M.

## 5. Pre-training

### 5.1 Data path

1. [`pretrain/pretokenize.py`](../pretrain/pretokenize.py) tokenizes each text shard once and stores
   the ids as Arrow files (49 shards of 50M tokens).
2. [`pretrain/train.py`](../pretrain/train.py) concatenates the token streams and cuts them into
   4,096-token sequences. No BOS or EOS tokens are inserted and no position is padding.
3. 1% of the packed sequences (5,868 sequences, 24M tokens) are held out with a fixed seed.

Packing was the largest throughput win. Odia lines average 54 tokens, so one-line-per-row batches at
any reasonable width would have been mostly padding.

### 5.2 Optimisation

| | |
|---|---|
| Optimiser | AdamW (Transformers defaults), weight decay 0.1 |
| Learning rate | peak 3e-4, 1% linear warm-up, cosine decay to zero |
| Batch | 12 sequences per GPU × 8 accumulation steps × 2 GPUs = 192 sequences, 786,432 tokens per update |
| Precision | bfloat16 autocast |
| Length | 2 epochs = 6,052 updates = 4.76B tokens processed |
| Hardware | 2× NVIDIA H100, DDP |
| Wall-clock | 8.63 h in two sessions (4.68 h, then 3.95 h resumed from the step-3,000 checkpoint) |
| Throughput | 169k tokens per second |

### 5.3 Curve

[`results/pretrain_log.json`](../results/pretrain_log.json):

| Update | Tokens processed | Validation loss | Perplexity |
|---:|---:|---:|---:|
| 1,000 | 0.79B | 3.820 | 45.6 |
| 2,000 | 1.57B | 3.480 | 32.5 |
| 3,000 | 2.36B | 3.337 | 28.1 |
| 4,000 | 3.15B | 3.259 | 26.0 |
| 5,000 | 3.93B | 3.215 | 24.9 |
| 6,000 | 4.72B | 3.203 | 24.6 |

Training loss at the same points is within about 0.02 of validation loss, in both directions. There is
no sign of overfitting after two epochs. The second epoch still bought 0.13 in loss, and the curve had
not flattened when the cosine schedule reached zero, so more data or more steps would have helped.

The one visible artefact is a spike in training loss just after update 3,000. That is where the second
session resumed from a checkpoint and where the second epoch begins.

## 6. Instruction tuning

### 6.1 Data

Two tasks, each in three scripts, 200,526 training examples in all.

**Commonsense QA.** [`finetune/build_qa_dataset.py`](../finetune/build_qa_dataset.py) converts four
multiple-choice benchmarks into open-ended pairs: the question becomes the instruction and the text of
the correct choice becomes the response.

| Source | Train | Validation |
|---|---:|---:|
| mCSQA (English) | 10,910 | 1,363 |
| CommonsenseQA | 9,741 | 1,221 |
| Cosmos QA | 25,262 | 2,985 |
| PIQA | 16,113 | 1,838 |
| Total | 62,026 | 7,407 |

A 60% sample of the English set was translated into Odia and a 20% sample into Kashmiri (Devanagari)
with NLLB-200 3.3B ([`finetune/translate_nllb.py`](../finetune/translate_nllb.py)), giving 30,000 and
10,000 training pairs.

**Character counting.** [`finetune/build_char_count_dataset.py`](../finetune/build_char_count_dataset.py)
asks how many times a character occurs in a sentence. Sampling a character from a natural sentence
gives 0 or 1 nearly every time, so the target character is injected at random positions, up to 20
times, in 85% of examples. The answer distribution becomes close to flat over 1 to 20, with extra mass
at 0. English uses real web sentences; the Odia and Devanagari sets use random strings over each
alphabet.

### 6.2 Method

LoRA with rank 16, α 32 and dropout 0.05 on all seven projection matrices of every block. That is
3,170,304 trainable parameters, 1.9% of the model. Effective batch 128, learning rate 2e-4 with cosine
decay, 3 epochs, 4,701 updates.

Prompt format:

```text
### Instruction:
{instruction}

### Response:
{response}</s>
```

### 6.3 A mistake in the released run

The released adapter was trained with every example padded to 4,096 tokens and with the loss computed
over every position, padding included. Examples average 59 tokens, of which 8 are the response
([`results/finetune_padding.json`](../results/finetune_padding.json)). So 98.5% of the scored positions
were padding and 0.2% were the response the model was meant to learn.

Consequences:

- The reported training and validation loss (0.054) is almost entirely the loss of predicting padding.
  It says nothing about answer quality, and it is not used anywhere in this repository's results.
- Each step processed about 70 times more positions than the examples contain. The run took 13.3 hours.
- The gradient signal from responses was diluted by the same factor.

[`finetune/train_lora.py`](../finetune/train_lora.py) now masks the prompt and pads each batch only to
its longest example. The old objective remains available as `--legacy-padded-loss` and was checked to
reproduce the original tokenization exactly. The adapter has not yet been retrained with the
corrected objective, so the results below are for the adapter as released.

## 7. Evaluation

### 7.1 Perplexity

Validation perplexity is 24.6 (loss 3.203) on the 24M held-out tokens, computed by the Trainer on
packed sequences. The held-out text is a random 1% of the same shuffled corpus, so this is an
in-distribution figure. Per-language perplexity on external text has not been measured;
[`eval/perplexity.py`](../eval/perplexity.py) is provided for it, and it excludes padding and weights by
token count.

### 7.2 Character counting

[`eval/char_count_accuracy.py`](../eval/char_count_accuracy.py), greedy decoding, 2,000 validation
examples per script ([`results/char_count_eval.json`](../results/char_count_eval.json)):

| Script | Exact | Within ±2 | MAE | Unparseable |
|---|---:|---:|---:|---:|
| English | 28.5% | 74.5% | 1.74 | 0.0% |
| Odia | 10.3% | 44.0% | 3.86 | 0.0% |
| Devanagari | 9.7% | 42.5% | 4.25 | 0.0% |

Reference points ([`results/char_count_baselines.json`](../results/char_count_baselines.json),
[`results/char_count_eval_base.json`](../results/char_count_eval_base.json)):

- Always answering the most common value gives 12.7% exact in English and 5.5% in the other two.
- The best constant answer gives 29.7%, 25.0% and 25.4% within ±2.
- The pre-trained model without the adapter gives 0.0% exact; over 95% of its replies contain no number.

So the adapter learned the answer format perfectly and learned real, approximate counting in English.
In the two Indic scripts it roughly doubles the constant-guess baseline.

Counting characters is structurally hard for this model. It reads subword pieces, not characters, and
injected characters break words into unusual piece sequences, so the count has to be inferred from the
identity of pieces rather than read off the input.

### 7.3 PIQA

[`eval/piqa_accuracy.py`](../eval/piqa_accuracy.py) scores each of the two candidate solutions as the
response to the goal and picks the one with the higher length-normalised log-likelihood. 1,838
validation questions, chance 50%.

| Model | Length-normalised | Summed |
|---|---:|---:|
| Pre-trained only | 57.2% | 56.7% |
| With LoRA adapter | 59.0% | 57.5% |

Both are more than six standard errors above chance. The difference between them is about one standard
error, so the adapter cannot be said to help here, even though PIQA's training split was part of the
instruction data.

## 8. Lessons

1. **Measure the corpus with the tokenizer you will train with.** Download budgets were counted with a
   stock tokenizer before the project tokenizer existed. For Odia the two differ by a factor of 8.6, so
   a budget of 1.5B stock tokens of IndicCorp bought roughly 175M tokens of training data, and Odia
   reached a third of the corpus only after every other Odia source was added.
2. **Check what the loss is averaged over.** A validation loss of 0.05 looked like success. It was the
   loss of predicting padding.
3. **Report from logs, not from memory.** Every figure in this repository is regenerated from JSON
   written by a script, and the scripts are in the repository.
4. **Pack sequences.** For short-sentence corpora it is the difference between a GPU that is busy and
   one that is mostly multiplying by masked positions.
5. **A translated proxy is not the language.** With no Dogri in NLLB-200, the Kashmiri subset was a
   stopgap. It should be labelled as Kashmiri, and it is.

What would come next, in order of expected value: retrain the adapter with the corrected objective;
evaluate perplexity per language on external text such as IN22 or FLORES; tie the embeddings and spend
the 29M parameters on depth; keep Cosmos QA context paragraphs in the instruction; find more Dogri.

## 9. Reproducibility

- Software for the released run: Python 3.12.11, PyTorch 2.8.0, Transformers 4.56.1, Datasets 4.1.0,
  PEFT 0.17.1, Accelerate 1.10.1, SentencePiece 0.2.1.
- The validation split uses seed 42. Corpus shuffling and language mixing in the released run were not
  seeded; the scripts now accept `--seed`.
- Evaluation subsets are drawn with seed 0 and decoded greedily, so the numbers in section 7 are
  deterministic given the weights.
- Evaluations were run on CPU in float32. GPU bfloat16 inference may differ in the last decimal.
- [`scripts/make_figures.py`](../scripts/make_figures.py) regenerates every figure from `results/`.
