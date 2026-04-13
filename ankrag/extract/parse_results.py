"""Parse Gemini batch prediction JSONL into structured rows for BigQuery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ankrag.extract.schema import InvoiceExtractionResult


def _response_text(obj: dict[str, Any]) -> str | None:
    """Extract model text from various Vertex / GenAI batch response shapes."""
    if "response" in obj:
        r = obj["response"]
        if isinstance(r, dict):
            if "text" in r and isinstance(r["text"], str):
                return r["text"]
            cands = r.get("candidates") or []
            if cands and isinstance(cands[0], dict):
                content = cands[0].get("content") or {}
                parts = content.get("parts") or []
                for p in parts:
                    if isinstance(p, dict) and p.get("text"):
                        return str(p["text"])
    if "error" in obj:
        raise ValueError(f"Batch line error: {obj['error']}")
    return None


def parse_batch_line(line_dict: dict[str, Any]) -> tuple[str, InvoiceExtractionResult]:
    key = str(line_dict.get("key", ""))
    text = _response_text(line_dict)
    if not text:
        raise ValueError(f"No model text for batch key={key}: keys={list(line_dict)}")
    data = json.loads(text)
    return key, InvoiceExtractionResult.from_model_dict(data)


def iter_batch_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_batch_prediction_jsonl_to_extractions(path: Path) -> list[tuple[str, InvoiceExtractionResult]]:
    out: list[tuple[str, InvoiceExtractionResult]] = []
    errors: list[str] = []
    for row in iter_batch_jsonl(path):
        try:
            out.append(parse_batch_line(row))
        except Exception as e:  # noqa: BLE001 — collect row-level errors
            errors.append(str(e))
    if errors and not out:
        raise RuntimeError("Failed to parse any batch lines: " + "; ".join(errors[:5]))
    return out
