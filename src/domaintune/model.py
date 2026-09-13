"""Load Qwen2.5-0.5B-Instruct + the DomainTune LoRA adapter, and run
extraction. Auto-detects GPU vs CPU at startup:

- GPU available: base model loaded in 4-bit (same as training) + LoRA
  adapter on top - matches the training/evaluation setup exactly.
- CPU only: base model loaded in float32 (bitsandbytes 4-bit isn't a CPU
  inference path) + the same LoRA adapter, unquantized.

The adapter is kept separate from the base model (not merged) - standard
LoRA serving pattern, smaller artifact to ship/update independently of the
base model.
"""
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from domaintune.extraction import (
    build_messages,
    extract_json_fields,
    check_schema_validity,
    ExtractedFields,
)

BASE_MODEL = os.environ.get("DOMAINTUNE_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = os.environ.get("DOMAINTUNE_ADAPTER_DIR", "models/domaintune_adapter_v2")
MAX_NEW_TOKENS = int(os.environ.get("DOMAINTUNE_MAX_NEW_TOKENS", "200"))
MAX_INPUT_TOKENS = int(os.environ.get("DOMAINTUNE_MAX_INPUT_TOKENS", "1024"))


@dataclass
class ExtractionResult:
    fields: ExtractedFields
    json_valid: bool
    schema_valid: bool
    raw_output: str
    latency_sec: float


class DomainTuneModel:
    """Loads once at startup; `extract()` is called per request."""

    def __init__(self, base_model: str = BASE_MODEL, adapter_dir: str = ADAPTER_DIR):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model, quantization_config=bnb_config, device_map="auto"
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_model, dtype=torch.float32, device_map="cpu"
            )

        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()

    def extract(self, ocr_text: str) -> ExtractionResult:
        messages = build_messages(ocr_text)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(self.model.device)

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - start

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        parsed = extract_json_fields(raw_output)
        json_valid = parsed is not None
        schema_valid = check_schema_validity(raw_output)
        fields = parsed or {"company": None, "date": None, "address": None, "total": None}

        return ExtractionResult(
            fields=fields,
            json_valid=json_valid,
            schema_valid=schema_valid,
            raw_output=raw_output,
            latency_sec=round(elapsed, 3),
        )


_model_instance: Optional[DomainTuneModel] = None


def get_model() -> DomainTuneModel:
    """Lazy singleton - loaded once on first request or at FastAPI startup."""
    global _model_instance
    if _model_instance is None:
        _model_instance = DomainTuneModel()
    return _model_instance
