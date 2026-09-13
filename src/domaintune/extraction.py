"""JSON extraction and schema validation for model outputs.

Ported directly from Notebook 2 (Sections 13.2/13.3) so behavior at
inference time matches what was measured during evaluation.
"""
import json
import re
from typing import Optional, TypedDict

EXPECTED_KEYS = ["company", "date", "address", "total"]

INSTRUCTION = (
    "Extract the company name, date, address, and total amount from this "
    "receipt OCR text. Respond with a JSON object with keys: "
    "company, date, address, total."
)


class ExtractedFields(TypedDict):
    company: Optional[str]
    date: Optional[str]
    address: Optional[str]
    total: Optional[str]


def _first_parseable_dict(raw_text: str) -> Optional[dict]:
    """Cascading parse: direct -> strip code fences -> regex-extract the
    first {...} block. Returns the parsed dict as-is (keys/types not
    filtered or coerced), or None if nothing parses."""
    candidates = [raw_text.strip()]

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def extract_json_fields(raw_text: str) -> Optional[ExtractedFields]:
    """Best-effort extraction of a {company, date, address, total} dict.
    Lenient about schema shape (a nested address dict still "succeeds"
    here, with the nested value passed through as-is) - use
    check_schema_validity() to catch schema-shape violations separately."""
    parsed = _first_parseable_dict(raw_text)
    if parsed is None:
        return None
    return {k: parsed.get(k) for k in EXPECTED_KEYS}  # type: ignore[return-value]


def check_schema_validity(raw_text: str) -> bool:
    """Exactly the 4 expected keys, each value a string or null - not a
    nested object/list/number. Fine-tuned model should satisfy this
    ~100% of the time per evaluation; useful as a runtime health signal."""
    parsed = _first_parseable_dict(raw_text)
    if parsed is None:
        return False
    if set(parsed.keys()) != set(EXPECTED_KEYS):
        return False
    for value in parsed.values():
        if value is not None and not isinstance(value, str):
            return False
    return True


def build_messages(ocr_text: str) -> list[dict]:
    """Build the chat messages list for generation (single user turn,
    matching training/evaluation exactly)."""
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n{ocr_text}"}]
