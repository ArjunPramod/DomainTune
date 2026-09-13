# DomainTune inference API
#
# GPU vs CPU is auto-detected at runtime (src/domaintune/model.py), so this
# same image works whether or not `docker run --gpus all` is used. For GPU
# use, run against a CUDA-enabled base instead of python:slim and pass
# --gpus all; the CPU path (this Dockerfile) needs no such setup.
FROM python:3.12-slim

# uv itself, via the official static-binary image (no pip bootstrap needed).
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (better layer caching - this layer only
# rebuilds when pyproject.toml/uv.lock change, not on every code edit).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Warm the Hugging Face cache for the base model at build time, so the
# container runs fully offline afterward and startup isn't slowed down by
# a download on first launch. Requires network access during `docker
# build` only.
ARG DOMAINTUNE_BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
ENV HF_HOME=/app/.hf_cache
RUN uv run python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('${DOMAINTUNE_BASE_MODEL}'); \
AutoModelForCausalLM.from_pretrained('${DOMAINTUNE_BASE_MODEL}')"

# Now copy source and the adapter. Copied after the dependency/model-warm
# layers so editing code or swapping the adapter doesn't invalidate them.
COPY src ./src
COPY api ./api
COPY models ./models

RUN uv sync --frozen --no-dev

ENV DOMAINTUNE_BASE_MODEL="${DOMAINTUNE_BASE_MODEL}"
ENV DOMAINTUNE_ADAPTER_DIR="/app/models/domaintune-adapter-v2"
ENV PYTHONPATH="/app/src"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
