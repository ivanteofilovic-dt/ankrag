"""BigQuery dataset init and GL load."""

from __future__ import annotations

from pathlib import Path

from google.cloud import bigquery

from ankrag.config import require_settings


def run_schema_sql(sql_path: Path | None = None) -> None:
    """Apply sql/bigquery/schema.sql with PROJECT/DATASET substituted."""
    settings = require_settings()
    root = Path(__file__).resolve().parents[2]
    path = sql_path or root / "sql" / "bigquery" / "schema.sql"
    sql = path.read_text()
    sql = sql.replace("PROJECT", settings.gcp_project)
    sql = sql.replace("DATASET", settings.bq_dataset)

    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    # Split on semicolons outside strings is fragile; run statement-by-statement for CREATE.
    statements = _split_sql_statements(sql)
    for stmt in statements:
        stripped = stmt.strip()
        if not stripped or stripped.startswith("--"):
            continue
        job = client.query(stripped)
        job.result()


def _split_sql_statements(sql: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(sql):
        c = sql[i]
        if c == "'" and not in_double:
            in_single = not in_single
            buf.append(c)
        elif c == '"' and not in_single:
            in_double = not in_double
            buf.append(c)
        elif c == ";" and not in_single and not in_double:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    if buf:
        parts.append("".join(buf))
    return [p for p in parts if p.strip()]


def load_gl_csv_to_bigquery(
    gcs_uri: str,
    table_id: str = "gl_lines",
    *,
    write_disposition: str = bigquery.WriteDisposition.WRITE_APPEND,
    autodetect: bool = False,
    schema_file: Path | None = None,
) -> str:
    """
    Load CSV (or wildcard gs://bucket/prefix/*.csv) into gl_lines.

    For production, prefer an explicit schema JSON next to this repo (see sql/bigquery/gl_load_schema.json).
    """
    settings = require_settings()
    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    full_table = f"{settings.gcp_project}.{settings.bq_dataset}.{table_id}"
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=autodetect,
        write_disposition=write_disposition,
    )
    if schema_file and schema_file.exists():
        job_config.schema = client.schema_from_json(str(schema_file))
        job_config.autodetect = False

    job = client.load_table_from_uri(gcs_uri, full_table, job_config=job_config)
    job.result()
    return full_table
