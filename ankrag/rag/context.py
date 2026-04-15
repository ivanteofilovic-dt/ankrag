"""Load BigQuery context rows for neighbor join_keys."""

from __future__ import annotations

import json
from typing import Any

from google.cloud import bigquery

from ankrag.config import require_settings


def fetch_training_rows_for_join_keys(join_keys: list[str]) -> dict[str, dict[str, Any]]:
    """Return map join_key -> row dict from invoice_gl_training_view."""
    if not join_keys:
        return {}
    settings = require_settings()
    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    view = f"`{settings.gcp_project}.{settings.bq_dataset}.invoice_gl_training_view`"
    sql = f"""
    SELECT *
    FROM {view}
    WHERE join_key IN UNNEST(@keys)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("keys", "STRING", join_keys)]
    )
    out: dict[str, dict[str, Any]] = {}
    for r in client.query(sql, job_config=job_config).result():
        row = dict(r.items())
        for k, v in list(row.items()):
            if hasattr(v, "isoformat"):
                row[k] = v.isoformat()
        out[str(row["join_key"])] = row
    return out


def neighbors_block_text(
    hits: list[Any],
    rows_by_jk: dict[str, dict[str, Any]],
) -> str:
    blocks: list[str] = []
    for h in hits:
        row = rows_by_jk.get(h.join_key)
        if not row:
            blocks.append(f"- join_key={h.join_key} distance={h.distance:.4f} (no GL row in view)")
            continue
        snippet = {
            "join_key": row.get("join_key"),
            "supplier": row.get("supplier"),
            "invoice_number": row.get("invoice_number"),
            "line_description": row.get("line_description"),
            "line_amount": str(row.get("line_amount")) if row.get("line_amount") is not None else None,
            "historical_gl": {
                "account": row.get("account"),
                "cost_center": row.get("cost_center"),
                "product_code": row.get("product_code"),
                "ic": row.get("ic"),
                "project": row.get("project"),
                "gl_system": row.get("gl_system"),
                "reserve": row.get("reserve"),
                "gl_amount": str(row.get("gl_amount")) if row.get("gl_amount") is not None else None,
                "periodization_start": row.get("periodization_start"),
                "periodization_end": row.get("periodization_end"),
            },
        }
        blocks.append(
            f"- similarity_distance={h.distance:.4f} case:\n{json.dumps(snippet, ensure_ascii=False, indent=2)}"
        )
    return "\n".join(blocks)
