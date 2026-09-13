"""DomainTune inference API.

Run locally:
    uv run uvicorn api.main:app --reload

Run in Docker: see Dockerfile (same command, no --reload).
"""
from fastapi import FastAPI, HTTPException

from domaintune.model import get_model, BASE_MODEL, ADAPTER_DIR
from domaintune.schemas import ExtractRequest, ExtractResponse, ExtractedFieldsModel, HealthResponse

app = FastAPI(
    title="DomainTune",
    description="Receipt field extraction (company/date/address/total) "
    "via a QLoRA-fine-tuned Qwen2.5-0.5B-Instruct.",
    version="1.0.0",
)


@app.on_event("startup")
def _load_model_on_startup():
    # Load once at startup rather than on first request, so the first real
    # request isn't slowed down by model loading and any load failure is
    # visible immediately in the container logs.
    get_model()


@app.get("/health", response_model=HealthResponse)
def health():
    model = get_model()
    return HealthResponse(
        status="ok",
        device=model.device,
        base_model=BASE_MODEL,
        adapter_dir=ADAPTER_DIR,
    )


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest):
    model = get_model()
    try:
        result = model.extract(request.ocr_text)
    except Exception as exc:  # surface a clean 500 instead of a raw traceback
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    return ExtractResponse(
        fields=ExtractedFieldsModel(**result.fields),
        json_valid=result.json_valid,
        schema_valid=result.schema_valid,
        raw_output=result.raw_output,
        latency_sec=result.latency_sec,
    )
