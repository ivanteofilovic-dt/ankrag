"""Online extraction + RAG coding suggestion."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from ankrag.config import require_settings
from ankrag.embeddings.embed import embed_texts
from ankrag.embeddings.text import canonical_embed_text
from ankrag.extract.prompts import HISTORICAL_EXTRACTION_SYSTEM, RAG_SYSTEM, extraction_user_prompt, rag_user_prompt
from ankrag.extract.schema import InvoiceExtractionResult
from ankrag.rag.confidence import apply_confidence_policy
from ankrag.rag.context import fetch_training_rows_for_join_keys, neighbors_block_text
from ankrag.rag.models import CodingSuggestion
from ankrag.rag.retrieve import NeighborHit, retrieve_similar


def _genai_client() -> genai.Client:
    s = require_settings()
    return genai.Client(vertexai=True, project=s.gcp_project, location=s.gcp_region)


def _gemini_model_id() -> str:
    s = require_settings()
    m = s.gemini_model
    return m if m.startswith("publishers/") else f"publishers/google/models/{m}"


def extract_invoice_online(*, gcs_uri: str | None = None, local_pdf: Path | None = None) -> InvoiceExtractionResult:
    if bool(gcs_uri) == bool(local_pdf):
        raise ValueError("Provide exactly one of gcs_uri or local_pdf")
    client = _genai_client()
    model = _gemini_model_id()
    if gcs_uri:
        doc_hint = gcs_uri.rsplit("/", 1)[-1].removesuffix(".pdf")
        parts = [
            types.Part.from_uri(file_uri=gcs_uri, mime_type="application/pdf"),
            types.Part.from_text(text=extraction_user_prompt(doc_hint)),
        ]
    else:
        assert local_pdf is not None
        data = local_pdf.read_bytes()
        doc_id = local_pdf.stem
        parts = [
            types.Part.from_bytes(data=data, mime_type="application/pdf"),
            types.Part.from_text(text=extraction_user_prompt(doc_id)),
        ]
    cfg = types.GenerateContentConfig(
        system_instruction=types.Content(parts=[types.Part.from_text(text=HISTORICAL_EXTRACTION_SYSTEM)]),
        response_mime_type="application/json",
        temperature=0.2,
    )
    resp = client.models.generate_content(model=model, contents=[types.Content(parts=parts)], config=cfg)
    text = resp.text
    if not text:
        raise RuntimeError("Empty extraction response")
    data = json.loads(text)
    return InvoiceExtractionResult.from_model_dict(data)


def _primary_line_for_embed(extraction: InvoiceExtractionResult) -> tuple[str, str]:
    """Returns (join_key, embed_text)."""
    if extraction.lines:
        line = extraction.lines[0]
        jk = line.join_key
        t = canonical_embed_text(
            supplier=extraction.supplier,
            invoice_number=extraction.invoice_number,
            line_description=line.description,
            line_amount=line.amount,
            currency=extraction.currency,
            periodization_hint=extraction.periodization_hint,
            join_key=jk,
        )
        return jk, t
    jk = extraction.document_id
    t = canonical_embed_text(
        supplier=extraction.supplier,
        invoice_number=extraction.invoice_number,
        line_description=None,
        line_amount=None,
        currency=extraction.currency,
        periodization_hint=extraction.periodization_hint,
        join_key=jk,
    )
    return jk, t


def suggest_coding_for_extraction(
    extraction: InvoiceExtractionResult,
    *,
    exclude_join_keys: list[str] | None = None,
    top_k: int | None = None,
    persist: bool = True,
    gcs_uri: str | None = None,
) -> tuple[CodingSuggestion, list[NeighborHit], dict[str, Any]]:
    settings = require_settings()
    k = top_k or settings.rag_top_k
    _, embed_txt = _primary_line_for_embed(extraction)
    qvec = embed_texts([embed_txt])[0]
    excl = list(exclude_join_keys or [])
    hits = retrieve_similar(qvec, top_k=k, exclude_join_keys=excl if excl else None)
    jks = list({h.join_key for h in hits})
    rows = fetch_training_rows_for_join_keys(jks)
    neighbor_rows = [rows[h.join_key] for h in hits if h.join_key in rows]
    block = neighbors_block_text(hits, rows)
    client = _genai_client()
    model = _gemini_model_id()
    user = rag_user_prompt(
        json.dumps(extraction.model_dump(), ensure_ascii=False, indent=2),
        block,
    )
    cfg = types.GenerateContentConfig(
        system_instruction=types.Content(parts=[types.Part.from_text(text=RAG_SYSTEM)]),
        response_mime_type="application/json",
        temperature=0.2,
    )
    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(parts=[types.Part.from_text(text=user)])],
        config=cfg,
    )
    if not resp.text:
        raise RuntimeError("Empty RAG response")
    raw = json.loads(resp.text)
    suggestion = CodingSuggestion.from_model_json(raw)
    suggestion, final_conf, meta = apply_confidence_policy(suggestion, hits, neighbor_rows)
    meta["neighbor_join_keys"] = jks

    if persist:
        _persist_suggestion(
            suggestion=suggestion,
            final_conf=final_conf,
            meta=meta,
            gcs_uri=gcs_uri,
            document_id=extraction.document_id,
        )

    return suggestion, hits, {"confidence_meta": meta, "final_confidence": final_conf}


def suggest_coding_for_gcs_pdf(gcs_uri: str, **kwargs: Any) -> tuple[CodingSuggestion, list[NeighborHit], dict[str, Any]]:
    ext = extract_invoice_online(gcs_uri=gcs_uri)
    return suggest_coding_for_extraction(ext, gcs_uri=gcs_uri, **kwargs)


def suggest_coding_for_local_pdf(path: Path, **kwargs: Any) -> tuple[CodingSuggestion, list[NeighborHit], dict[str, Any]]:
    ext = extract_invoice_online(local_pdf=path)
    return suggest_coding_for_extraction(ext, gcs_uri=None, **kwargs)


def _persist_suggestion(
    *,
    suggestion: CodingSuggestion,
    final_conf: float,
    meta: dict[str, Any],
    gcs_uri: str | None,
    document_id: str,
) -> None:
    from google.cloud import bigquery

    settings = require_settings()
    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    table = f"{settings.gcp_project}.{settings.bq_dataset}.rag_suggestions"
    row = {
        "suggestion_id": str(uuid.uuid4()),
        "document_id": document_id,
        "gcs_uri": gcs_uri,
        "join_keys_suggested": json.dumps(meta.get("neighbor_join_keys", [])),
        "model_output_json": suggestion.model_dump_json(),
        "confidence": final_conf,
        "confidence_components": json.dumps(meta, default=str),
    }
    errors = client.insert_rows_json(table, [row])
    if errors:
        raise RuntimeError(f"rag_suggestions insert failed: {errors}")
