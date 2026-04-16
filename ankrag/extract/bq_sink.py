"""Write extraction rows to BigQuery."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from google.cloud import bigquery

from ankrag.config import require_settings
from ankrag.extract.schema import InvoiceExtractionResult


def _numeric_or_none(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def _parse_date(s: str | None) -> str | None:
    if not s:
        return None
    try:
        return str(date.fromisoformat(s[:10]))
    except ValueError:
        return None


def _invoice_join_key_from_extraction(invoice_number: str | None, document_id: str) -> str:
    """Align with GL load when ``join_key_mode=invoice`` (trimmed invoice #)."""
    s = (invoice_number or "").strip()
    return s if s else document_id


def extraction_to_rows(
    result: InvoiceExtractionResult,
    *,
    model_id: str,
    join_key_source: str = "invoice_number",
) -> list[dict[str, Any]]:
    if join_key_source not in ("invoice_number", "model"):
        raise ValueError("join_key_source must be 'invoice_number' or 'model'")
    raw = result.model_dump()
    base_json = json.dumps(raw, ensure_ascii=False)
    rows: list[dict[str, Any]] = []
    jk_invoice = (
        _invoice_join_key_from_extraction(result.invoice_number, result.document_id)
        if join_key_source == "invoice_number"
        else None
    )
    if not result.lines:
        rows.append(
            {
                "join_key": jk_invoice if join_key_source == "invoice_number" else result.document_id,
                "document_id": result.document_id,
                "line_index": 0,
                "supplier": result.supplier,
                "invoice_number": result.invoice_number,
                "invoice_date": _parse_date(result.invoice_date),
                "currency": result.currency,
                "line_description": None,
                "line_amount": None,
                "periodization_hint": result.periodization_hint,
                "extraction_json": base_json,
                "model_id": model_id,
            }
        )
        return rows

    for line in result.lines:
        jk = jk_invoice if join_key_source == "invoice_number" else line.join_key
        rows.append(
            {
                "join_key": jk,
                "document_id": result.document_id,
                "line_index": line.line_index,
                "supplier": result.supplier,
                "invoice_number": result.invoice_number,
                "invoice_date": _parse_date(result.invoice_date),
                "currency": result.currency,
                "line_description": line.description,
                "line_amount": _numeric_or_none(line.amount),
                "periodization_hint": result.periodization_hint,
                "extraction_json": base_json,
                "model_id": model_id,
            }
        )
    return rows


def insert_extractions(rows: list[dict[str, Any]], *, table: str = "invoice_extractions") -> None:
    settings = require_settings()
    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    full = f"{settings.gcp_project}.{settings.bq_dataset}.{table}"
    errors = client.insert_rows_json(full, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors[:3]}")


def log_extraction_error(
    *,
    gcs_uri: str | None,
    document_id: str | None,
    batch_key: str | None,
    message: str,
) -> None:
    settings = require_settings()
    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    full = f"{settings.gcp_project}.{settings.bq_dataset}.extraction_errors"
    row = {
        "document_id": document_id,
        "gcs_uri": gcs_uri,
        "error_message": message[:8192],
        "batch_key": batch_key,
    }
    errors = client.insert_rows_json(full, [row])
    if errors:
        raise RuntimeError(f"extraction_errors insert failed: {errors}")
