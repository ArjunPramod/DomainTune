# DomainTune

Fine-tuning Qwen2.5-0.5B-Instruct with QLoRA to extract structured fields
(company, date, address, total) from noisy OCR receipt text — and
measuring whether fine-tuning actually beats a prompted base model
(zero-shot and few-shot) on this task. See `HANDOVER.md` for the full
project history, findings, and decisions made along the way.

## Structure

```
domaintune/
├── notebooks/
│   ├── 01_data_preparation.ipynb   # SROIE → normalized, split, instruction-format data
│   └── 02_fine_tuning.ipynb        # QLoRA fine-tuning + evaluation (v1 and v2)
├── src/domaintune/
│   ├── model.py                    # load base model + LoRA adapter, generate()
│   ├── extraction.py               # JSON parsing/schema validation
│   └── schemas.py                  # pydantic request/response models
├── api/main.py                     # FastAPI app: POST /extract, GET /health
├── models/                         # adapter files go here (gitignored, see models/README.md)
├── examples/example_client.py      # minimal usage example
├── Dockerfile
└── pyproject.toml                  # uv-managed
```

## Notebooks

Run in Kaggle/Colab, independent of the local `uv` environment (plain
`pip install` inside the notebook). See `HANDOVER.md` for what each
notebook does and the results.

## API — local dev

```bash
uv sync
cd models && unzip /path/to/domaintune_adapter_v2.zip -d domaintune-adapter-v2 && cd ..
uv run uvicorn api.main:app --reload
```

GPU vs CPU is auto-detected at startup (4-bit + LoRA on GPU, float32 + LoRA
on CPU — same adapter either way).

Try it:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"ocr_text": "TAN WOON YANN\nBOOK TA .K(TAMAN DAYA) SDN BND\n...\nTOTAL: 9.00"}'
```

or:

```bash
uv run python examples/example_client.py
```

Interactive docs at `http://localhost:8000/docs` (FastAPI's built-in
Swagger UI).

## API — Docker

```bash
# adapter must already be unzipped under models/domaintune-adapter-v2/
docker build -t domaintune .
docker run -p 8000:8000 domaintune
```

The base model is downloaded once at build time and baked into the image,
so the container runs offline. For GPU inference, run on a CUDA-enabled
host with `docker run --gpus all ...` — the same image auto-detects and
uses it.

## Dataset

[`jsdnrs/ICDAR2019-SROIE`](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE)
on the Hugging Face Hub — 987 receipts with OCR text (`words`) and
ground-truth key fields (`entities`: company, date, address, total).

## Status

Data prep, fine-tuning (v1 + v2), and evaluation are done — see
`HANDOVER.md`. `domaintune-adapter-v2` is the adopted model. API/Docker
layer above is the current phase.
