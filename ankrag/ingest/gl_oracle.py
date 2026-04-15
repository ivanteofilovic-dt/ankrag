"""Map Oracle / subledger GL text exports (tab-separated) into ``gl_lines`` load format.

Historical files such as ``GL_202603.txt`` are tab-separated text with this header row
(in order)::

    ENTITY, GL_SOURCE_NAME, GL_CATEGORY, JOURNAL_NUMBER, BOOKING_DATE, PERIOD, ACCOUNT,
    HFM_ACCOUNT, HFM_DSCRIPTIONS, DEPARTMENT, PRODUCT, WORK_ORDER, IC, PROJECT, SYSTEM,
    RESERVE, INVOICE_NUM, SUPPLIER_NUMBER, SUPPLIER_CUSTMER_NAME, GL_LINE_DESCRIPTION,
    PO_NUMBER, NET_ACCOUNTED, TRANSACTION_TYPE_NAME, GL_TAX, SUBLEDGER_TAX_CODE,
    EMPLOYEE_NAME

``BOOKING_DATE`` uses English month abbreviations and 12-hour clock, e.g.
``Mar 31 2026 12:00 AM``. Exports are often Windows-1252 (Nordic); when decoding fails as
UTF-8, :func:`iter_oracle_gl_rows` falls back to cp1252.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterator, Sequence

from google.cloud import bigquery
from google.cloud import storage

from ankrag.config import require_settings

# Minimum columns required to build join keys, amounts, and dates.
_ORACLE_GL_REQUIRED_FIELDS = frozenset(
    {
        "ENTITY",
        "JOURNAL_NUMBER",
        "BOOKING_DATE",
        "PERIOD",
        "ACCOUNT",
        "NET_ACCOUNTED",
        "DEPARTMENT",
        "PRODUCT",
        "GL_LINE_DESCRIPTION",
        "HFM_ACCOUNT",
        "HFM_DSCRIPTIONS",
        "INVOICE_NUM",
        "IC",
        "PROJECT",
        "SYSTEM",
        "RESERVE",
    }
)


def _decode_oracle_gl_bytes(data: bytes, *, encoding: str | None) -> str:
    if encoding is not None:
        return data.decode(encoding)
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252")


def _parse_booking_date(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%b %d %Y %I:%M %p", "%b %d %Y %I:%M:%S %p"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _period_bounds(period: str) -> tuple[date | None, date | None]:
    """PERIOD is YYYYMM (e.g. 202603)."""
    p = (period or "").strip()
    if not re.fullmatch(r"\d{6}", p):
        return None, None
    y, m = int(p[:4]), int(p[4:6])
    if m < 1 or m > 12:
        return None, None
    start = date(y, m, 1)
    if m == 12:
        end = date(y, 12, 31)
    else:
        end = date(y, m + 1, 1) - timedelta(days=1)
    return start, end


def _parse_amount(s: str) -> Decimal | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return Decimal(s.replace(",", ""))
    except InvalidOperation:
        return None


def _row_fingerprint(row: dict[str, str]) -> str:
    parts = [
        (row.get(k) or "").strip()
        for k in (
            "ENTITY",
            "JOURNAL_NUMBER",
            "ACCOUNT",
            "BOOKING_DATE",
            "NET_ACCOUNTED",
            "DEPARTMENT",
            "PRODUCT",
            "GL_LINE_DESCRIPTION",
            "HFM_ACCOUNT",
        )
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def compute_join_key(row: dict[str, str]) -> str:
    """
    Stable key per GL line for this export shape.

    When ``INVOICE_NUM`` is present, it is embedded so keys align with AP-style
    invoice identifiers; a short content hash disambiguates multiple lines on
    the same invoice with identical metadata.
    """
    entity = (row.get("ENTITY") or "").strip() or "UNK"
    inv = (row.get("INVOICE_NUM") or "").strip()
    jrnl = (row.get("JOURNAL_NUMBER") or "").strip() or "0"
    fp = _row_fingerprint(row)
    if inv:
        return f"{entity}|{inv}|{fp}"
    return f"{entity}|JRNL|{jrnl}|{fp}"


def _gl_line_id(row: dict[str, str], fp: str) -> str:
    jrnl = (row.get("JOURNAL_NUMBER") or "").strip() or "0"
    acct = (row.get("ACCOUNT") or "").strip() or "0"
    return f"{jrnl}-{acct}-{fp[:8]}"


def _description(row: dict[str, str]) -> str:
    gl_desc = (row.get("GL_LINE_DESCRIPTION") or "").strip()
    hfm = (row.get("HFM_DSCRIPTIONS") or "").strip()
    if hfm and gl_desc:
        return f"{hfm} | {gl_desc}"
    return gl_desc or hfm or ""


def oracle_gl_row_to_load_tuple(row: dict[str, str]) -> tuple:
    fp = _row_fingerprint(row)
    posting = _parse_booking_date(row.get("BOOKING_DATE", ""))
    p_start, p_end = _period_bounds(row.get("PERIOD", ""))
    amount = _parse_amount(row.get("NET_ACCOUNTED", ""))
    desc = _description(row)
    raw = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    return (
        compute_join_key(row),
        _gl_line_id(row, fp),
        posting.isoformat() if posting else "",
        (row.get("ENTITY") or "").strip(),
        (row.get("ACCOUNT") or "").strip(),
        (row.get("DEPARTMENT") or "").strip(),
        (row.get("PRODUCT") or "").strip(),
        (row.get("IC") or "").strip(),
        (row.get("PROJECT") or "").strip(),
        (row.get("SYSTEM") or "").strip(),
        (row.get("RESERVE") or "").strip(),
        str(amount) if amount is not None else "",
        "",
        p_start.isoformat() if p_start else "",
        p_end.isoformat() if p_end else "",
        desc,
        raw,
    )


_GL_HEADER = [
    "join_key",
    "gl_line_id",
    "posting_date",
    "company_code",
    "account",
    "cost_center",
    "product_code",
    "ic",
    "project",
    "gl_system",
    "reserve",
    "amount",
    "currency",
    "periodization_start",
    "periodization_end",
    "description",
    "raw_source_row",
]


def iter_oracle_gl_rows(path: Path, *, encoding: str | None = None) -> Iterator[dict[str, str]]:
    data = path.read_bytes()
    text = _decode_oracle_gl_bytes(data, encoding=encoding)
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames is None:
        return
    names = [((n or "").strip()) for n in reader.fieldnames]
    missing = _ORACLE_GL_REQUIRED_FIELDS - set(names)
    if missing:
        raise ValueError(
            "GL export is missing expected columns (wrong delimiter or format?): "
            + ", ".join(sorted(missing))
        )
    for raw in reader:
        yield {k: (v if v is not None else "") for k, v in raw.items()}


def oracle_gl_tsv_to_csv_bytes(path: Path, *, encoding: str | None = None) -> bytes:
    """Build UTF-8 CSV matching ``sql/bigquery/gl_load_schema.json`` column order."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_GL_HEADER)
    for row in iter_oracle_gl_rows(path, encoding=encoding):
        writer.writerow(oracle_gl_row_to_load_tuple(row))
    return buf.getvalue().encode("utf-8")


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri}")
    rest = uri[5:]
    if "/" not in rest:
        raise ValueError(f"Invalid gs:// URI (missing object path): {uri}")
    bucket, blob = rest.split("/", 1)
    return bucket, blob


def load_oracle_gl_tsv_paths_to_bigquery(
    paths: Sequence[Path],
    table_id: str = "gl_lines",
    *,
    write_disposition: str = bigquery.WriteDisposition.WRITE_APPEND,
    schema_file: Path | None = None,
    encoding: str | None = None,
) -> str:
    """
    Transform each tab-separated GL file and load into ``gl_lines``.

    The first load uses ``write_disposition``; further files always append so multi-month
    folders do not overwrite earlier months.
    """
    resolved = sorted((p.resolve() for p in paths if p.is_file()), key=lambda p: p.name)
    if not resolved:
        raise ValueError("No GL export files to load (expected paths to existing files)")
    settings = require_settings()
    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    full_table = f"{settings.gcp_project}.{settings.bq_dataset}.{table_id}"
    root = Path(__file__).resolve().parents[2]
    schema_path = schema_file or root / "sql" / "bigquery" / "gl_load_schema.json"
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=write_disposition,
        autodetect=False,
    )
    if schema_path.exists():
        job_config.schema = client.schema_from_json(str(schema_path))
    for i, path in enumerate(resolved):
        data = oracle_gl_tsv_to_csv_bytes(path, encoding=encoding)
        job = client.load_table_from_file(io.BytesIO(data), full_table, job_config=job_config)
        job.result()
        if i == 0:
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    return full_table


def load_oracle_gl_tsv_to_bigquery(
    source: Path | str,
    table_id: str = "gl_lines",
    *,
    write_disposition: str = bigquery.WriteDisposition.WRITE_APPEND,
    schema_file: Path | None = None,
    encoding: str | None = None,
    directory_glob: str = "GL_*.txt",
    recursive: bool = False,
) -> str:
    """
    Transform a tab-separated GL export (e.g. ``GL_202603.txt``) and load into ``gl_lines``.

    ``source`` may be a local file, a local directory of ``GL_*.txt`` files, or
    ``gs://bucket/object.txt`` (single object; downloaded to a temp file). For many objects
    under GCS, download them locally or run ``load_oracle_gl_tsv_paths_to_bigquery`` on paths.

    ``encoding`` defaults to UTF-8 (with BOM allowed); if that fails, Windows-1252 is used.
    Pass an explicit encoding (e.g. ``"cp1252"``) to force one codec.
    """
    cleanup: Path | None = None
    try:
        if isinstance(source, str) and source.startswith("gs://"):
            bucket, blob = _parse_gs_uri(source)
            st = storage.Client(project=settings.gcp_project)
            fd, name = tempfile.mkstemp(suffix=".txt", prefix="gl_oracle_")
            os.close(fd)
            cleanup = Path(name)
            st.bucket(bucket).blob(blob).download_to_filename(str(cleanup))
            path = cleanup
        else:
            path = Path(source)
        if path.is_dir():
            it = path.rglob(directory_glob) if recursive else path.glob(directory_glob)
            paths = [p for p in it if p.is_file()]
            if not paths:
                raise ValueError(
                    f"No files matching {directory_glob!r} under {path} "
                    f"({'recursive' if recursive else 'top level only'})"
                )
            return load_oracle_gl_tsv_paths_to_bigquery(
                paths,
                table_id=table_id,
                write_disposition=write_disposition,
                schema_file=schema_file,
                encoding=encoding,
            )
        if not path.is_file():
            raise FileNotFoundError(f"Not a file or directory: {path}")
        return load_oracle_gl_tsv_paths_to_bigquery(
            [path],
            table_id=table_id,
            write_disposition=write_disposition,
            schema_file=schema_file,
            encoding=encoding,
        )
    finally:
        if cleanup is not None:
            cleanup.unlink(missing_ok=True)
