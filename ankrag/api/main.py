"""FastAPI application: PDF analyze, similar search, BigQuery-backed lists."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import bigquery

from ankrag.config import get_settings, require_settings
from ankrag.rag.context import fetch_training_rows_for_join_keys
from ankrag.rag.suggest import (
    extract_invoice_online,
    neighbor_records,
    similar_invoices_for_local_pdf,
    suggest_coding_for_extraction,
    training_row_snippet,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="AnkReg API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ANKRAG_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(
        ","
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _confidence_status(confidence: float, high: float, low: float) -> str:
    if confidence >= high:
        return "Auto-Posted"
    if confidence >= low:
        return "Needs Review"
    return "Anomaly Flagged"


def _write_upload_to_temp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "invoice.pdf").suffix.lower()
    if suffix not in {".pdf"}:
        suffix = ".pdf"
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    path = Path(name)
    try:
        data = upload.file.read()
        path.write_bytes(data)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


@app.get("/api/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {
        "ok": True,
        "gcp_project_set": bool(s.gcp_project),
        "gcs_bucket_set": bool(s.gcs_bucket),
    }


@app.get("/api/config")
def public_config() -> dict[str, Any]:
    s = get_settings()
    return {
        "gcp_project": s.gcp_project or None,
        "gcp_region": s.gcp_region,
        "bq_dataset": s.bq_dataset,
        "gemini_model": s.gemini_model,
        "embedding_model": s.embedding_model,
        "rag_top_k": s.rag_top_k,
        "confidence_high_threshold": s.confidence_high_threshold,
        "confidence_low_threshold": s.confidence_low_threshold,
        "vector_search_backend": "matching_engine"
        if s.matching_engine_index_endpoint
        else "bigquery_ml",
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    s = get_settings()
    if not s.gcp_project:
        return {
            "configured": False,
            "counts": {},
            "error": "GCP_PROJECT is not set",
        }
    client = bigquery.Client(project=s.gcp_project, location=s.bq_location)
    ds = f"{s.gcp_project}.{s.bq_dataset}"
    tables = (
        "invoice_line_embeddings",
        "rag_suggestions",
        "invoice_extractions",
        "gl_lines",
    )
    counts: dict[str, int | None] = {}
    err: str | None = None
    for t in tables:
        try:
            sql = f"SELECT COUNT(*) AS c FROM `{ds}.{t}`"
            row = next(iter(client.query(sql).result()))
            counts[t] = int(row["c"])
        except Exception as e:  # noqa: BLE001
            logger.warning("stats query failed for %s: %s", t, e)
            counts[t] = None
            err = str(e)
    return {"configured": True, "counts": counts, "error": err}


@app.get("/api/suggestions")
def list_suggestions(limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
    try:
        s = require_settings()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    client = bigquery.Client(project=s.gcp_project, location=s.bq_location)
    table = f"`{s.gcp_project}.{s.bq_dataset}.rag_suggestions`"
    sql = f"""
    SELECT suggestion_id, document_id, gcs_uri, model_output_json, confidence,
           confidence_components, created_at
    FROM {table}
    ORDER BY created_at DESC
    LIMIT @lim
    """
    job = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("lim", "INT64", limit)]
    )
    items: list[dict[str, Any]] = []
    thr = get_settings()
    for r in client.query(sql, job_config=job).result():
        raw = r["model_output_json"]
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            parsed = {}
        conf = float(r["confidence"]) if r["confidence"] is not None else float(parsed.get("confidence") or 0)
        items.append(
            {
                "suggestion_id": r["suggestion_id"],
                "document_id": r["document_id"],
                "gcs_uri": r["gcs_uri"],
                "confidence": conf,
                "status": _confidence_status(conf, thr.confidence_high_threshold, thr.confidence_low_threshold),
                "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else r["created_at"],
                "rationale_preview": (parsed.get("rationale") or "")[:280],
                "journal_lines_preview": (parsed.get("journal_lines") or [])[:3],
            }
        )
    return {"items": items}


def _neighbors_from_stored_join_keys(join_keys: list[str]) -> list[dict[str, Any]]:
    """Rebuild neighbor panel from persisted suggestion join_keys (no distance scores)."""
    if not join_keys:
        return []
    rows = fetch_training_rows_for_join_keys(join_keys)
    out: list[dict[str, Any]] = []
    for i, jk in enumerate(join_keys, start=1):
        tr = rows.get(jk)
        out.append(
            {
                "rank": i,
                "join_key": jk,
                "invoice_line_id": tr.get("join_key", jk) if tr else jk,
                "document_id": str(tr.get("document_id", "")) if tr else "",
                "line_index": int(tr.get("line_index", 0)) if tr else 0,
                "cosine_distance": None,
                "similarity": None,
                "training": training_row_snippet(tr) if tr else None,
            }
        )
    return out


@app.get("/api/suggestions/{suggestion_id}")
def get_suggestion(suggestion_id: str) -> dict[str, Any]:
    try:
        s = require_settings()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    client = bigquery.Client(project=s.gcp_project, location=s.bq_location)
    table = f"`{s.gcp_project}.{s.bq_dataset}.rag_suggestions`"
    sql = f"""
    SELECT suggestion_id, document_id, gcs_uri, model_output_json, confidence,
           confidence_components, join_keys_suggested, created_at
    FROM {table}
    WHERE suggestion_id = @sid
    LIMIT 1
    """
    job = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("sid", "STRING", suggestion_id)]
    )
    rows = list(client.query(sql, job_config=job).result())
    if not rows:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    r = rows[0]
    raw = r["model_output_json"]
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        parsed = {}
    try:
        jks = json.loads(r["join_keys_suggested"] or "[]")
        if not isinstance(jks, list):
            jks = []
        jks = [str(x) for x in jks]
    except json.JSONDecodeError:
        jks = []
    conf = float(r["confidence"]) if r["confidence"] is not None else float(parsed.get("confidence") or 0)
    meta_raw = r["confidence_components"]
    try:
        conf_meta = json.loads(meta_raw) if isinstance(meta_raw, str) and meta_raw else {}
    except json.JSONDecodeError:
        conf_meta = {}
    thr = get_settings()
    return {
        "suggestion_id": r["suggestion_id"],
        "document_id": r["document_id"],
        "gcs_uri": r["gcs_uri"],
        "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else r["created_at"],
        "suggestion": parsed,
        "final_confidence": conf,
        "confidence_meta": conf_meta,
        "status": _confidence_status(conf, thr.confidence_high_threshold, thr.confidence_low_threshold),
        "neighbors": _neighbors_from_stored_join_keys(jks),
        "extraction": None,
    }


@app.post("/api/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    persist: bool = Query(True, description="Insert row into rag_suggestions when true"),
) -> dict[str, Any]:
    try:
        s = require_settings()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    path = _write_upload_to_temp(file)
    try:
        extraction = extract_invoice_online(local_pdf=path)
        suggestion, hits, meta = suggest_coding_for_extraction(
            extraction,
            gcs_uri=None,
            persist=persist,
        )
        rows = fetch_training_rows_for_join_keys([h.join_key for h in hits])
        neighbors = neighbor_records(hits, rows)
        final_conf = float(meta.get("final_confidence", suggestion.confidence))
        status = _confidence_status(final_conf, s.confidence_high_threshold, s.confidence_low_threshold)
        return {
            "extraction": extraction.model_dump(),
            "suggestion": suggestion.model_dump(),
            "neighbors": neighbors,
            "confidence_meta": meta.get("confidence_meta", {}),
            "final_confidence": final_conf,
            "status": status,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("analyze failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/similar")
async def similar_pdf(
    file: UploadFile = File(...),
    top_k: int | None = Query(None, ge=1, le=50),
) -> dict[str, Any]:
    try:
        require_settings()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    path = _write_upload_to_temp(file)
    try:
        return similar_invoices_for_local_pdf(
            path,
            top_k=top_k,
            log_neighbors=False,
            include_embed_text=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("similar failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        path.unlink(missing_ok=True)
