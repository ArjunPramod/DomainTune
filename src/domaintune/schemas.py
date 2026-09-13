from typing import Optional

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    ocr_text: str = Field(
        ...,
        min_length=1,
        description="Raw OCR text from a receipt (newline-joined lines, "
        "same format used in training - see Notebook 1).",
        examples=["TAN WOON YANN\nBOOK TA .K(TAMAN DAYA) SDN BND\n..."],
    )


class ExtractedFieldsModel(BaseModel):
    company: Optional[str] = None
    date: Optional[str] = None
    address: Optional[str] = None
    total: Optional[str] = None


class ExtractResponse(BaseModel):
    fields: ExtractedFieldsModel
    json_valid: bool
    schema_valid: bool
    raw_output: str
    latency_sec: float


class HealthResponse(BaseModel):
    status: str
    device: str
    base_model: str
    adapter_dir: str
