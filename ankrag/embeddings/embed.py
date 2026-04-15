"""Vertex text embeddings and BigQuery upsert for invoice_line_embeddings."""

from __future__ import annotations

from google import genai
from google.cloud import bigquery

from ankrag.config import require_settings
from ankrag.embeddings.text import canonical_embed_text

# Streaming insert (insertAll) should stay small: large JSON bodies often hit ~10MB limits
# and are more likely to be cut off by proxies (SSLEOFError / EOF in violation of protocol).
_EMBED_INSERT_BATCH = 100


def _embedding_client() -> genai.Client:
    s = require_settings()
    return genai.Client(vertexai=True, project=s.gcp_project, location=s.gcp_region)


def embed_texts(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    s = require_settings()
    client = _embedding_client()
    mid = model or s.embedding_model
    if not mid.startswith("publishers/"):
        mid = f"publishers/google/models/{mid}"
    try:
        resp = client.models.embed_content(model=mid, contents=texts)
    except Exception:
        out: list[list[float]] = []
        for t in texts:
            r = client.models.embed_content(model=mid, contents=t)
            e = (r.embeddings or [None])[0]
            if e is None or e.values is None:
                raise
            out.append(list(e.values))
        return out
    out = []
    for emb in resp.embeddings or []:
        if emb.values is not None:
            out.append(list(emb.values))
    if len(out) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: got {len(out)} for {len(texts)} inputs")
    return out


def backfill_embeddings_from_extractions(
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """
    Read invoice_extractions, compute embeddings, MERGE into invoice_line_embeddings.
    Returns number of rows processed.
    """
    settings = require_settings()
    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    ds = f"`{settings.gcp_project}.{settings.bq_dataset}`"
    lim = f"LIMIT {int(limit)}" if limit else ""
    sql = f"""
    SELECT join_key, document_id, line_index, supplier, invoice_number, line_description,
           CAST(line_amount AS STRING) AS line_amount, currency, periodization_hint
    FROM {ds}.invoice_extractions
    {lim}
    """
    rows = list(client.query(sql).result())
    if not rows:
        return 0

    texts: list[str] = []
    meta: list[tuple[str, str, str, int]] = []
    for r in rows:
        jk, did, li = r["join_key"], r["document_id"], int(r["line_index"])
        t = canonical_embed_text(
            supplier=r["supplier"],
            invoice_number=r["invoice_number"],
            line_description=r["line_description"],
            line_amount=r["line_amount"],
            currency=r["currency"],
            periodization_hint=r["periodization_hint"],
            join_key=jk,
        )
        texts.append(t)
        invoice_line_id = f"{jk}#{li}"
        meta.append((jk, invoice_line_id, did, li))

    vectors = embed_texts(texts)
    if dry_run:
        return len(vectors)

    table = f"{settings.gcp_project}.{settings.bq_dataset}.invoice_line_embeddings"
    out_rows = []
    for i, vec in enumerate(vectors):
        jk, invoice_line_id, did, li = meta[i]
        out_rows.append(
            {
                "join_key": jk,
                "invoice_line_id": invoice_line_id,
                "document_id": did,
                "line_index": li,
                "embedding": vec,
                "embed_text": texts[i],
                "embedding_model": settings.embedding_model,
            }
        )
    for start in range(0, len(out_rows), _EMBED_INSERT_BATCH):
        chunk = out_rows[start : start + _EMBED_INSERT_BATCH]
        errors = client.insert_rows_json(table, chunk)
        if errors:
            raise RuntimeError(f"embedding insert errors (rows {start}-{start + len(chunk)}): {errors[:2]}")
    return len(out_rows)
