"""End-to-end helpers for batch extraction → BigQuery."""

from __future__ import annotations

import json
from pathlib import Path

from ankrag.extract.bq_sink import extraction_to_rows, insert_extractions, log_extraction_error
from ankrag.extract.parse_results import iter_batch_jsonl, parse_batch_line


def import_invoice_documents(rows: list[dict], *, table: str = "invoice_documents") -> None:
    from google.cloud import bigquery

    from ankrag.config import require_settings

    settings = require_settings()
    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    full = f"{settings.gcp_project}.{settings.bq_dataset}.{table}"
    errors = client.insert_rows_json(full, rows)
    if errors:
        raise RuntimeError(f"invoice_documents insert errors: {errors}")


def import_batch_prediction_jsonl(
    path: Path,
    *,
    model_id: str,
    gcs_by_key: dict[str, str] | None = None,
    register_documents: bool = True,
) -> tuple[int, int]:
    """
    Parse one Vertex batch output JSONL file; insert invoice_extractions (+ optional invoice_documents).

    gcs_by_key maps batch `key` -> gs:// URI for error logging and invoice_documents.
    Returns (success_count, error_count).
    """
    ok, err = 0, 0
    for row in iter_batch_jsonl(path):
        key = str(row.get("key", ""))
        gcs_uri = (gcs_by_key or {}).get(key)
        try:
            _, result = parse_batch_line(row)
            insert_extractions(extraction_to_rows(result, model_id=model_id))
            if register_documents and gcs_uri:
                import_invoice_documents(
                    [
                        {
                            "document_id": result.document_id,
                            "gcs_uri": gcs_uri,
                            "content_hash": None,
                            "source_filename": gcs_uri.rsplit("/", 1)[-1],
                        }
                    ],
                )
            ok += 1
        except Exception as e:  # noqa: BLE001
            log_extraction_error(
                gcs_uri=gcs_uri,
                document_id=None,
                batch_key=key or None,
                message=str(e),
            )
            err += 1
    return ok, err


def write_manifest(path: Path, items: list[tuple[str, str, str]]) -> Path:
    """items: (batch_key, gcs_uri_pdf, document_id)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for bk, uri, did in items:
            f.write(json.dumps({"key": bk, "gcs_uri": uri, "document_id": did}) + "\n")
    return path


def read_manifest(path: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            m[str(o["key"])] = str(o["gcs_uri"])
    return m
