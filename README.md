# DomainTune

**Fine-tuning a small open-source LLM to extract structured data from noisy real-world receipts, evaluated against zero-shot and few-shot prompting baselines.**

> Fine-tuned Qwen2.5-0.5B-Instruct (via QLoRA) to extract company, date, address, and total from receipt OCR text. Evaluated against the same base model prompted with zero and few examples, on 100 held-out receipts. Results and methodology below.

## What this project demonstrates

- **LLM fine-tuning end-to-end**: QLoRA (4-bit quantization + LoRA adapters), completion-only loss masking, hyperparameter iteration guided by loss curves, early stopping.
- **Rigorous evaluation**: base model tested both zero-shot and few-shot before concluding fine-tuning was worth it. Multiple metrics — exact match, fuzzy/token-level F1, schema validity, complete-record accuracy — chosen to avoid a single misleading number.
- **Data engineering discipline**: full-dataset distribution analysis before writing any normalization code, not assumptions from a handful of samples.
- **Iteration based on evidence**: found overfitting in the loss curves and a specific weak field in the eval results, then designed a targeted fix (data augmentation + regularization) — which measurably worked.
- **Production packaging**: FastAPI service, Dockerized, GPU/CPU auto-detection, dependency-locked with `uv`.

## The problem

Businesses need to pull structured data (who, when, where, how much) out of receipts. You could enforce a JSON schema with any LLM and call it done — but that doesn't tell you whether the _model_ actually understands the task, or is just following a format. This project asks a more useful engineering question: **does fine-tuning a small model on this specific task meaningfully outperform prompting the base model?**

The evaluation shows clear gains from fine-tuning across the extraction fields, while address remains the main limitation.

## Results

**Total amount, extracted correctly:**

|Approach|Total amount, exact match|
|---|---|
|Zero-shot base model|0%|
|Few-shot base model (3 examples)|63%|
|Fine-tuned, iteration 1|96%|
|**Fine-tuned, iteration 2**|**97%**|

Exact string match against ground truth, on 100 receipts held out from training.

A stricter bar — every field on the receipt correct at once (company, date, address, total together, not just one):

|Approach|Complete-record accuracy|
|---|---|
|Zero-shot base model|0%|
|Few-shot base model (3 examples)|0%|
|Fine-tuned, iteration 1|60%|
|**Fine-tuned, iteration 2**|**68%**|

_Methodology: greedy decoding, exact match after normalization (address allows fuzzy match ≥0.85 similarity), evaluated identically across all four approaches on the same 100 held-out receipts._

Field-by-field breakdown for the final model (iteration 2) against both baselines:

|Metric|Zero-shot base|Few-shot base|**Fine-tuned**|
|---|---|---|---|
|Valid JSON output|83%|96%|**100%**|
|Correct schema (flat, no nested junk)|0%|96%|**100%**|
|Company name exact match|27%|55%|**92%**|
|Date exact match|35%|68%|**97%**|
|Total amount exact match|0%|63%|**97%**|
|Address, fuzzy match|0%|0%|**74%**|
|Every field correct (strict)|0%|0%|**54%**|

The base model's failures aren't just wrong answers — it frequently invents its own JSON shape (nested objects instead of the requested flat fields), which is why both prompting baselines score 0% on schema and complete-record accuracy in this evaluation. Fine-tuning fixes this structurally, not just numerically.

**Known limitation**: address extraction is meaningfully weaker (74% fuzzy match) than the other three fields (92–97%). One iteration of targeted data augmentation improved it by 9 points; further gains may require more training-data diversity rather than additional prompting or hyperparameter tuning. Reported directly — see the notebooks for the full before/after comparison.

## How it was built

1. **Data**: [SROIE receipt dataset](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) (987 receipts) — inspected the _entire_ dataset's field distributions, date formats, and edge cases before writing normalization code, catching issues (inconsistent date formats, negative totals, duplicate leakage between splits) that a "looks fine from 5 examples" approach would have missed.
2. **Fine-tuning**: Qwen2.5-0.5B-Instruct, QLoRA (4-bit base + LoRA adapters), trained on Kaggle's free GPU tier. Loss only computed on the target JSON output, not the prompt — a fine-tuning detail that matters for training efficiency.
3. **Evaluation**: same 100 receipts run through all four approaches, scored on 7 metrics chosen to separate different failure modes (a low score could mean "wrong content" or "wrong shape" — these need different fixes, so they're measured separately).
4. **Iteration**: loss curves showed early overfitting; evaluation showed address as the weak field. Fixed both at once — light regularization (weight decay, dropout, early stopping) plus targeted data augmentation on the input text (address abbreviation variants, synthetic OCR noise) — and re-measured. Result: the table above, with no regressions on any other field.
5. **Serving**: FastAPI + Docker, auto-detecting GPU/CPU at startup so the same image works in either environment.

## Tech stack

Python, PyTorch, Hugging Face Transformers/PEFT/Accelerate, bitsandbytes (QLoRA), FastAPI, Docker, `uv`.

## Project structure

```
domaintune/
├── notebooks/          # data prep, fine-tuning, evaluation (fully documented, runnable)
├── src/domaintune/     # inference: model loading, JSON extraction/validation
├── api/                # FastAPI service
├── results/            # evaluation metrics (CSV)
└── Dockerfile
```

## Running it

```
uv sync

# unzip the trained adapter into models/domaintune_adapter_v2/
uv run uvicorn api.main:app --reload

curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"ocr_text": "..."}'
```

Or via Docker:

```
docker build -t domaintune .
docker run -p 8000:8000 domaintune
```