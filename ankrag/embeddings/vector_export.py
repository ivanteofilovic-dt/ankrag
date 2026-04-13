"""Export embeddings in a Vertex Matching Engine–friendly JSONL format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_matching_engine_jsonl(
    rows: Iterable[dict[str, Any]],
    path: Path,
) -> Path:
    """
    Each line: {"id": "<invoice_line_id>", "embedding": [float, ...]}

    After upload to GCS, use the Vertex AI console or gcloud to build/update a
    streaming Matching Engine index (see Google Cloud docs). Datapoint `id` must
    match what you pass when upserting; use the same ids in retrieval mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def rows_from_bigquery_export(
    *,
    join_keys: list[str],
    invoice_line_ids: list[str],
    embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for jk, iid, emb in zip(join_keys, invoice_line_ids, embeddings, strict=True):
        out.append({"id": iid, "embedding": emb, "join_key": jk})
    return out
