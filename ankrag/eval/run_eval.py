"""Held-out evaluation: RAG suggestion vs historical GL (account on first journal line)."""

from __future__ import annotations

import random
from typing import Any

from google.cloud import bigquery

from ankrag.config import require_settings
from ankrag.eval.metrics import account_match_rate, route_bucket
from ankrag.extract.schema import InvoiceExtractionResult, InvoiceLineItem
from ankrag.rag.suggest import suggest_coding_for_extraction


def _rows_to_extraction(rows: list[dict[str, Any]]) -> InvoiceExtractionResult:
    rows = sorted(rows, key=lambda r: int(r["line_index"]))
    first = rows[0]
    lines = [
        InvoiceLineItem(
            line_index=int(r["line_index"]),
            description=r.get("line_description"),
            amount=str(r["line_amount"]) if r.get("line_amount") is not None else None,
            join_key=str(r["join_key"]),
        )
        for r in rows
    ]
    inv_date = first.get("invoice_date")
    if hasattr(inv_date, "isoformat"):
        inv_date = inv_date.isoformat()
    return InvoiceExtractionResult(
        document_id=str(first["document_id"]),
        supplier=first.get("supplier"),
        invoice_number=first.get("invoice_number"),
        invoice_date=str(inv_date) if inv_date else None,
        currency=first.get("currency"),
        periodization_hint=first.get("periodization_hint"),
        lines=lines,
    )


def _sample_join_keys(n: int, seed: int) -> list[str]:
    settings = require_settings()
    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    view = f"`{settings.gcp_project}.{settings.bq_dataset}.invoice_gl_training_view`"
    sql = f"""
    SELECT join_key, COUNT(*) AS c
    FROM {view}
    WHERE account IS NOT NULL
    GROUP BY join_key
    HAVING c >= 1
    """
    keys = [str(r["join_key"]) for r in client.query(sql).result()]
    rng = random.Random(seed)
    rng.shuffle(keys)
    return keys[:n]


def _extractions_for_join_key(client: bigquery.Client, join_key: str) -> list[dict[str, Any]]:
    settings = require_settings()
    t = f"`{settings.gcp_project}.{settings.bq_dataset}.invoice_extractions`"
    sql = f"SELECT * FROM {t} WHERE join_key = @jk ORDER BY line_index"
    job = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("jk", "STRING", join_key)]
    )
    return [dict(r.items()) for r in client.query(sql, job_config=job).result()]


def _gl_account_for_join_key(client: bigquery.Client, join_key: str) -> str | None:
    settings = require_settings()
    t = f"`{settings.gcp_project}.{settings.bq_dataset}.gl_lines`"
    sql = f"SELECT account FROM {t} WHERE join_key = @jk LIMIT 1"
    job = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("jk", "STRING", join_key)]
    )
    rows = list(client.query(sql, job_config=job).result())
    if not rows:
        return None
    a = rows[0]["account"]
    return str(a) if a is not None else None


def run_heldout_eval(
    sample_size: int = 20,
    seed: int = 42,
    *,
    persist_suggestions: bool = False,
) -> dict[str, Any]:
    """
    For each sampled join_key, rebuild extraction from BigQuery, run RAG excluding that key
    from retrieval, and compare the first suggested journal line account to historical GL.
    """
    settings = require_settings()
    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    keys = _sample_join_keys(sample_size, seed)
    preds: list[str | None] = []
    actuals: list[str | None] = []
    routes: list[str] = []
    errors: list[str] = []

    for jk in keys:
        try:
            rows = _extractions_for_join_key(client, jk)
            if not rows:
                errors.append(f"no extraction for {jk}")
                continue
            ext = _rows_to_extraction(rows)
            acc_true = _gl_account_for_join_key(client, jk)
            sug, _hits, _meta = suggest_coding_for_extraction(
                ext,
                exclude_join_keys=[jk],
                persist=persist_suggestions,
            )
            pred = sug.journal_lines[0].account if sug.journal_lines else None
            preds.append(pred)
            actuals.append(acc_true)
            routes.append(route_bucket(sug.confidence))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{jk}: {e}")

    acc = account_match_rate(preds, actuals)
    return {
        "sampled_keys": keys,
        "account_match_rate": acc,
        "n_scored": len(preds),
        "route_distribution": {
            "auto_post": routes.count("auto_post"),
            "discretionary": routes.count("discretionary"),
            "human_review": routes.count("human_review"),
        },
        "errors": errors,
    }
