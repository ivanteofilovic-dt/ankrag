"""Build Vertex Gemini batch JSONL (GCS PDF + prompt)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ankrag.extract.prompts import HISTORICAL_EXTRACTION_SYSTEM, extraction_user_prompt


def build_batch_jsonl_for_pdfs(
    items: Iterable[tuple[str, str, str]],
) -> list[dict]:
    """
    items: (batch_key, gs_uri_pdf, document_id)

    Each output line matches Vertex batch prediction format:
    {"key": "...", "request": {GenerateContentRequest as dict}}
    """
    lines: list[dict] = []
    for batch_key, gs_uri, document_id in items:
        user_text = extraction_user_prompt(document_id)
        req = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": "application/pdf",
                                "fileUri": gs_uri,
                            }
                        },
                        {"text": user_text},
                    ],
                }
            ],
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": HISTORICAL_EXTRACTION_SYSTEM}],
            },
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2,
            },
        }
        lines.append({"key": batch_key, "request": req})
    return lines


def write_local_jsonl(records: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
