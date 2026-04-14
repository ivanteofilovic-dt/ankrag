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
        stripped = _strip_leading_line_comments(stmt)
        if not stripped:
            continue
        job = client.query(stripped)
        job.result()


def _split_sql_statements(sql: str) -> list[str]:
    """Split on semicolons outside strings and outside ``--`` line comments."""
    parts: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    in_line_comment = False
    i = 0
    while i < len(sql):
        c = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""

        if in_line_comment:
            buf.append(c)
            if c in "\n\r":
                in_line_comment = False
            i += 1
            continue

        if c == "'" and not in_double:
            in_single = not in_single
            buf.append(c)
        elif c == '"' and not in_single:
            in_double = not in_double
            buf.append(c)
        elif not in_single and not in_double and c == "-" and nxt == "-":
            in_line_comment = True
            buf.extend(["-", "-"])
            i += 2
            continue
        elif c == ";" and not in_single and not in_double:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    if buf:
        parts.append("".join(buf))
    return [p for p in parts if p.strip()]


def _strip_leading_line_comments(sql: str) -> str:
    """Drop leading ``--`` comment lines and blank lines so blocks are not skipped as 'comment-only'."""
    lines = sql.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("--"):
            i += 1
            continue
        break
    return "\n".join(lines[i:]).strip()


def load_gl_csv_to_bigquery(
    gcs_uri: str,
    table_id: str = "gl_lines",
    *,
    write_disposition: str = bigquery.WriteDisposition.WRITE_APPEND,
    autodetect: bool = False,
    schema_file: Path | None = None,
) -> str:
    """
    Load comma-separated CSV (or wildcard ``gs://bucket/prefix/*.csv``) into ``gl_lines``.

    Tab-separated Oracle / subledger exports (``GL_YYYYMM.txt``) are not loaded here; use
    ``ankrag.ingest.gl_oracle.load_oracle_gl_tsv_to_bigquery`` or ``ankrag load-gl --oracle-export``.

    For production, prefer an explicit schema JSON (see ``sql/bigquery/gl_load_schema.json``).
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
