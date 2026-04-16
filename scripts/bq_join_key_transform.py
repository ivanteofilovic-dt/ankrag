#!/usr/bin/env python3
"""
Create a new BigQuery dataset with gl_lines, invoice_extractions, and
invoice_line_embeddings copied from a source dataset, rewriting join_key (and
invoice_line_id for embeddings) so the key is derived from the invoice number in
raw_source_row JSON (Oracle GL export shape: $.INVOICE_NUM).

Does not load data from GCS or local files — only CREATE TABLE … AS SELECT
against existing tables.

Usage:
  export GCP_PROJECT=my-project
  python scripts/bq_join_key_transform.py --source-dataset ankreg --target-dataset ankreg_invjk

Then point AnkReg at the target dataset (BQ_DATASET=ankreg_invjk) or swap names
after validating.

Other tables (invoice_documents, rag_suggestions, extraction_errors) are not
copied. embed_text is unchanged (may still mention old join_key in text).
"""

from __future__ import annotations

import argparse
import os
import sys


def _new_join_key_expr(*, key_mode: str, json_path: str) -> str:
    """SQL expression: STRING new join_key from gl_lines row (uses join_key, company_code, raw_source_row)."""
    inv_json = f"JSON_VALUE(raw_source_row, '{json_path}')"
    trimmed = f"NULLIF(TRIM({inv_json}), '')"
    if key_mode == "invoice_num":
        return f"""CASE WHEN {trimmed} IS NOT NULL THEN TRIM({inv_json}) ELSE join_key END"""
    if key_mode == "company_invoice":
        return f"""CASE WHEN {trimmed} IS NOT NULL THEN CONCAT(
          COALESCE(NULLIF(TRIM(company_code), ''), 'UNK'),
          '|',
          TRIM({inv_json})
        ) ELSE join_key END"""
    raise ValueError(f"Unknown key_mode: {key_mode}")


def _tbl(project: str, dataset: str, table: str) -> str:
    return f"`{project}.{dataset}.{table}`"


def build_sql(
    *,
    project: str,
    source: str,
    target: str,
    key_mode: str,
    json_path: str,
) -> list[str]:
    src_gl = _tbl(project, source, "gl_lines")
    src_ex = _tbl(project, source, "invoice_extractions")
    src_em = _tbl(project, source, "invoice_line_embeddings")
    tgt_gl = _tbl(project, target, "gl_lines")
    tgt_ex = _tbl(project, target, "invoice_extractions")
    tgt_em = _tbl(project, target, "invoice_line_embeddings")
    tgt_vw = _tbl(project, target, "invoice_gl_training_view")

    jk_gl = _new_join_key_expr(key_mode=key_mode, json_path=json_path)

    # One row per gl_lines row (old join_key -> new key from invoice #).
    map_cte = f"""
    map_keys AS (
      SELECT
        join_key AS old_join_key,
        {jk_gl} AS new_join_key
      FROM {src_gl}
    )
    """

    stmts: list[str] = []

    stmts.append(
        f"""
CREATE OR REPLACE TABLE {tgt_gl}
PARTITION BY posting_date
CLUSTER BY join_key, company_code
AS
WITH {map_cte.strip()}
SELECT
  m.new_join_key AS join_key,
  g.gl_line_id,
  g.posting_date,
  g.company_code,
  g.account,
  g.cost_center,
  g.product_code,
  g.ic,
  g.project,
  g.gl_system,
  g.reserve,
  g.amount,
  g.currency,
  g.periodization_start,
  g.periodization_end,
  g.description,
  g.raw_source_row,
  g.ingested_at
FROM {src_gl} AS g
INNER JOIN map_keys AS m ON g.join_key = m.old_join_key
""".strip()
    )

    stmts.append(
        f"""
CREATE OR REPLACE TABLE {tgt_ex}
CLUSTER BY join_key, document_id
AS
WITH {map_cte.strip()}
SELECT
  COALESCE(m.new_join_key, e.join_key) AS join_key,
  e.document_id,
  e.line_index,
  e.supplier,
  e.invoice_number,
  e.invoice_date,
  e.currency,
  e.line_description,
  e.line_amount,
  e.periodization_hint,
  e.extraction_json,
  e.model_id,
  e.extracted_at
FROM {src_ex} AS e
LEFT JOIN map_keys AS m ON e.join_key = m.old_join_key
""".strip()
    )

    stmts.append(
        f"""
CREATE OR REPLACE TABLE {tgt_em}
CLUSTER BY join_key, document_id
AS
WITH {map_cte.strip()}
SELECT
  COALESCE(m.new_join_key, el.join_key) AS join_key,
  CONCAT(
    COALESCE(m.new_join_key, el.join_key),
    '#',
    CAST(el.line_index AS STRING)
  ) AS invoice_line_id,
  el.document_id,
  el.line_index,
  el.embedding,
  el.embed_text,
  el.embedding_model,
  el.created_at
FROM {src_em} AS el
LEFT JOIN map_keys AS m ON el.join_key = m.old_join_key
""".strip()
    )

    stmts.append(
        f"""
CREATE OR REPLACE VIEW {tgt_vw} AS
SELECT
  e.join_key,
  e.document_id,
  e.line_index,
  e.supplier,
  e.invoice_number,
  e.invoice_date,
  e.line_description,
  e.line_amount,
  e.currency,
  e.periodization_hint,
  g.account,
  g.cost_center,
  g.product_code,
  g.ic,
  g.project,
  g.gl_system,
  g.reserve,
  g.amount AS gl_amount,
  g.currency AS gl_currency,
  g.periodization_start,
  g.periodization_end,
  g.posting_date,
  g.company_code,
  g.description AS gl_description
FROM {tgt_ex} AS e
LEFT JOIN {tgt_gl} AS g
  ON e.join_key = g.join_key
""".strip()
    )

    return stmts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT"), help="GCP project id")
    p.add_argument("--source-dataset", required=True, help="Existing dataset (e.g. ankreg)")
    p.add_argument("--target-dataset", required=True, help="New dataset to create/populate (must differ from source)")
    p.add_argument(
        "--key-mode",
        choices=("company_invoice", "invoice_num"),
        default="company_invoice",
        help="company_invoice: CONCAT(company_code,'|',invoice#); invoice_num: invoice number only",
    )
    p.add_argument(
        "--invoice-json-path",
        default="$.INVOICE_NUM",
        help="JSON path in raw_source_row for invoice number (Oracle default: $.INVOICE_NUM)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print SQL only; do not execute")
    args = p.parse_args()

    if not args.project:
        print("Set --project or GCP_PROJECT / GOOGLE_CLOUD_PROJECT", file=sys.stderr)
        return 2
    if args.source_dataset == args.target_dataset:
        print("source-dataset and target-dataset must differ.", file=sys.stderr)
        return 2

    stmts = build_sql(
        project=args.project,
        source=args.source_dataset,
        target=args.target_dataset,
        key_mode=args.key_mode,
        json_path=args.invoice_json_path,
    )

    if args.dry_run:
        for i, sql in enumerate(stmts, 1):
            print(f"-- Statement {i}\n{sql};\n")
        return 0

    from google.cloud import bigquery

    client = bigquery.Client(project=args.project)
    src_ref = bigquery.DatasetReference(args.project, args.source_dataset)
    src_ds = client.get_dataset(src_ref)
    tgt_ref = bigquery.DatasetReference(args.project, args.target_dataset)
    tgt_ds = bigquery.Dataset(tgt_ref)
    tgt_ds.location = src_ds.location
    client.create_dataset(tgt_ds, exists_ok=True)
    print(f"Dataset {args.project}.{args.target_dataset} (location={src_ds.location})")

    for i, sql in enumerate(stmts, 1):
        print(f"Running statement {i}/{len(stmts)}…")
        job = client.query(sql)
        job.result()
        if job.num_dml_affected_rows is not None:
            print(f"  DML rows: {job.num_dml_affected_rows}")
        else:
            print(f"  Job {job.job_id} done; bytes processed: {job.total_bytes_processed or 0}")

    print("Done. Point BQ_DATASET to the target dataset when using AnkReg.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
