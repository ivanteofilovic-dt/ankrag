"""Top-K similar invoice lines: BigQuery ML.DISTANCE or Vertex Matching Engine."""

from __future__ import annotations

from dataclasses import dataclass

from google.cloud import aiplatform, bigquery

from ankrag.config import require_settings


@dataclass
class NeighborHit:
    join_key: str
    invoice_line_id: str
    document_id: str
    line_index: int
    distance: float


def retrieve_similar_bigquery(
    query_embedding: list[float],
    *,
    top_k: int,
    exclude_join_keys: list[str] | None = None,
) -> list[NeighborHit]:
    settings = require_settings()
    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    table = f"`{settings.gcp_project}.{settings.bq_dataset}.invoice_line_embeddings`"
    exclude_join_keys = exclude_join_keys or []

    if exclude_join_keys:
        sql = f"""
        SELECT join_key, invoice_line_id, document_id, line_index,
               ML.DISTANCE(embedding, @q, 'COSINE') AS dist
        FROM {table}
        WHERE join_key NOT IN UNNEST(@exclude)
        ORDER BY dist ASC
        LIMIT @k
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("q", "FLOAT64", query_embedding),
                bigquery.ArrayQueryParameter("exclude", "STRING", exclude_join_keys),
                bigquery.ScalarQueryParameter("k", "INT64", top_k),
            ]
        )
    else:
        sql = f"""
        SELECT join_key, invoice_line_id, document_id, line_index,
               ML.DISTANCE(embedding, @q, 'COSINE') AS dist
        FROM {table}
        ORDER BY dist ASC
        LIMIT @k
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("q", "FLOAT64", query_embedding),
                bigquery.ScalarQueryParameter("k", "INT64", top_k),
            ]
        )

    hits: list[NeighborHit] = []
    for r in client.query(sql, job_config=job_config).result():
        hits.append(
            NeighborHit(
                join_key=r["join_key"],
                invoice_line_id=r["invoice_line_id"],
                document_id=r["document_id"],
                line_index=int(r["line_index"]),
                distance=float(r["dist"]),
            )
        )
    return hits


def retrieve_similar_matching_engine(
    query_embedding: list[float],
    *,
    top_k: int,
) -> list[NeighborHit]:
    settings = require_settings()
    if not settings.matching_engine_index_endpoint or not settings.matching_engine_deployed_index_id:
        raise ValueError("Matching Engine endpoint and deployed index id must be set")
    aiplatform.init(project=settings.gcp_project, location=settings.gcp_region)
    ep = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=settings.matching_engine_index_endpoint
    )
    resp = ep.find_neighbors(
        deployed_index_id=settings.matching_engine_deployed_index_id,
        queries=[query_embedding],
        num_neighbors=top_k,
        return_full_datapoint=False,
    )
    hits: list[NeighborHit] = []
    if not resp or not resp[0]:
        return hits
    for m in resp[0]:
        iid = str(m.id)
        parts = iid.rsplit("#", 1)
        jk = parts[0] if len(parts) == 2 else iid
        li = int(parts[1]) if len(parts) == 2 else 0
        hits.append(
            NeighborHit(
                join_key=jk,
                invoice_line_id=iid,
                document_id="",
                line_index=li,
                distance=float(m.distance) if m.distance is not None else 0.0,
            )
        )
    return hits


def retrieve_similar(
    query_embedding: list[float],
    *,
    top_k: int | None = None,
    exclude_join_keys: list[str] | None = None,
) -> list[NeighborHit]:
    settings = require_settings()
    k = top_k or settings.rag_top_k
    if settings.matching_engine_index_endpoint and settings.matching_engine_deployed_index_id:
        if exclude_join_keys:
            return retrieve_similar_bigquery(
                query_embedding, top_k=k, exclude_join_keys=exclude_join_keys
            )
        return retrieve_similar_matching_engine(query_embedding, top_k=k)
    return retrieve_similar_bigquery(query_embedding, top_k=k, exclude_join_keys=exclude_join_keys)
