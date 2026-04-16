"""Map Oracle / subledger GL text exports (tab-separated) into ``gl_lines`` load format.

By default, loads only rows that have supplier metadata (non-empty ``INVOICE_NUM`` and at
least one of ``SUPPLIER_NUMBER`` / ``SUPPLIER_CUSTMER_NAME``). Rows whose
``GL_LINE_DESCRIPTION`` contains ``ankreg`` (any case) still load, but account and coding
dimensions are cleared unless disabled — those lines are not treated as reliable coding
ground truth.

For ``join_key_mode`` ``invoice`` or ``entity_invoice``, Oracle detail rows are rolled up
to **one output row per join key** (invoice number, or entity+invoice), summing amounts and
merging metadata so ``gl_lines`` aligns with “one GL row per invoice” for linking to
``invoice_extractions``. Use ``aggregate_by_invoice=False`` to keep one row per subledger
line. ``fingerprint`` mode is always one row per source line.

Join keys default to trimmed ``INVOICE_NUM`` so ``invoice_extractions`` can use the same
value from ``invoice_number`` (see ``--join-key-mode`` / ``compute_join_key``).

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

``NET_ACCOUNTED`` may contain more than nine decimal places; BigQuery ``NUMERIC`` allows at
most scale 9, so amounts are rounded half-up to nine fractional digits before load. The
full precision remains in ``raw_source_row``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import OrderedDict
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
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


# BigQuery NUMERIC(38, 9): at most nine digits after the decimal point.
_BQ_NUMERIC_QUANTUM = Decimal("0.000000001")


def _format_amount_for_bigquery_csv(raw_net_accounted: str) -> str:
    """
    Return a CSV field loadable as BigQuery NUMERIC, or empty string for NULL.

    Values with more than nine fractional digits are rounded half-up; scientific notation
    is never emitted (``format(..., "f")``).
    """
    amount = _parse_amount(raw_net_accounted)
    if amount is None:
        return ""
    try:
        q = amount.quantize(_BQ_NUMERIC_QUANTUM, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return ""
    return format(q, "f")


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


def normalize_invoice_join_value(s: str | None) -> str:
    """Trim whitespace for matching GL ``INVOICE_NUM`` to extracted ``invoice_number``."""
    return (s or "").strip()


def row_has_supplier_columns(row: dict[str, str]) -> bool:
    """True when the row carries AP-style supplier metadata (invoice + supplier id or name)."""
    inv = normalize_invoice_join_value(row.get("INVOICE_NUM"))
    sup = normalize_invoice_join_value(row.get("SUPPLIER_NUMBER"))
    name = normalize_invoice_join_value(row.get("SUPPLIER_CUSTMER_NAME"))
    return bool(inv and (sup or name))


def gl_line_description_has_ankreg(row: dict[str, str]) -> bool:
    return "ankreg" in (row.get("GL_LINE_DESCRIPTION") or "").casefold()


def compute_join_key(row: dict[str, str], *, mode: str = "fingerprint") -> str:
    """
    Join key for linking ``gl_lines`` to ``invoice_extractions``.

    * ``invoice`` — trimmed ``INVOICE_NUM`` only (matches extracted invoice number).
    * ``entity_invoice`` — ``ENTITY|INVOICE_NUM`` to reduce cross-company collisions.
    * ``fingerprint`` — legacy per-line key (entity, invoice, content hash).
    """
    entity = (row.get("ENTITY") or "").strip() or "UNK"
    inv = normalize_invoice_join_value(row.get("INVOICE_NUM"))
    if mode == "fingerprint":
        jrnl = (row.get("JOURNAL_NUMBER") or "").strip() or "0"
        fp = _row_fingerprint(row)
        if inv:
            return f"{entity}|{inv}|{fp}"
        return f"{entity}|JRNL|{jrnl}|{fp}"
    if mode == "entity_invoice":
        return f"{entity}|{inv}" if inv else ""
    return inv


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


def oracle_gl_row_to_load_tuple(
    row: dict[str, str],
    *,
    join_key_mode: str = "invoice",
    null_coding_when_ankreg: bool = True,
) -> tuple:
    fp = _row_fingerprint(row)
    posting = _parse_booking_date(row.get("BOOKING_DATE", ""))
    p_start, p_end = _period_bounds(row.get("PERIOD", ""))
    desc = _description(row)
    raw = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    strip_coding = bool(null_coding_when_ankreg) and gl_line_description_has_ankreg(row)
    acct = (row.get("ACCOUNT") or "").strip()
    dept = (row.get("DEPARTMENT") or "").strip()
    prod = (row.get("PRODUCT") or "").strip()
    ic = (row.get("IC") or "").strip()
    proj = (row.get("PROJECT") or "").strip()
    system = (row.get("SYSTEM") or "").strip()
    reserve = (row.get("RESERVE") or "").strip()
    if strip_coding:
        acct = dept = prod = ic = proj = system = reserve = ""
    return (
        compute_join_key(row, mode=join_key_mode),
        _gl_line_id(row, fp),
        posting.isoformat() if posting else "",
        (row.get("ENTITY") or "").strip(),
        acct,
        dept,
        prod,
        ic,
        proj,
        system,
        reserve,
        _format_amount_for_bigquery_csv(row.get("NET_ACCOUNTED", "")),
        "",
        p_start.isoformat() if p_start else "",
        p_end.isoformat() if p_end else "",
        desc,
        raw,
    )


_RAW_AGG_JSON_MAX_CHARS = 950_000


def _format_decimal_amount_for_bq(amount: Decimal) -> str:
    try:
        q = amount.quantize(_BQ_NUMERIC_QUANTUM, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return ""
    return format(q, "f")


def _row_coding_fields(
    row: dict[str, str], *, null_coding_when_ankreg: bool
) -> tuple[str, str, str, str, str, str, str]:
    strip_coding = bool(null_coding_when_ankreg) and gl_line_description_has_ankreg(row)
    acct = (row.get("ACCOUNT") or "").strip()
    dept = (row.get("DEPARTMENT") or "").strip()
    prod = (row.get("PRODUCT") or "").strip()
    ic = (row.get("IC") or "").strip()
    proj = (row.get("PROJECT") or "").strip()
    system = (row.get("SYSTEM") or "").strip()
    reserve = (row.get("RESERVE") or "").strip()
    if strip_coding:
        acct = dept = prod = ic = proj = system = reserve = ""
    return acct, dept, prod, ic, proj, system, reserve


def _raw_json_for_invoice_agg(rows: list[dict[str, str]]) -> str:
    """Serialize source rows for audit; shrink if JSON would exceed BigQuery practical limits."""
    n = len(rows)
    lines = list(rows)
    while True:
        payload = {"aggregated_invoice_gl": True, "line_count": n, "lines": lines}
        s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(s) <= _RAW_AGG_JSON_MAX_CHARS:
            return s
        if len(lines) <= 1:
            return json.dumps(
                {
                    "aggregated_invoice_gl": True,
                    "line_count": n,
                    "truncated": True,
                    "fieldnames": sorted(lines[0].keys()) if lines else [],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        lines = lines[: max(1, len(lines) // 2)]


def _aggregate_oracle_rows_to_load_tuple(
    rows: list[dict[str, str]],
    *,
    join_key: str,
    join_key_mode: str,
    null_coding_when_ankreg: bool,
) -> tuple:
    if not rows:
        raise ValueError("aggregate requires at least one row")
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            _parse_booking_date(r.get("BOOKING_DATE", "")) or date.min,
            (r.get("JOURNAL_NUMBER") or "").strip(),
            (r.get("ACCOUNT") or "").strip(),
        ),
    )
    for r in sorted_rows:
        if compute_join_key(r, mode=join_key_mode) != join_key:
            raise ValueError("aggregate group contains mismatched join keys")

    total = Decimal("0")
    any_amount = False
    posting_dates: list[date] = []
    period_starts: list[date] = []
    period_ends: list[date] = []
    entities: set[str] = set()
    codings: list[tuple[str, str, str, str, str, str, str]] = []
    desc_parts: list[str] = []
    seen_desc: set[str] = set()

    for row in sorted_rows:
        amt = _parse_amount(row.get("NET_ACCOUNTED", ""))
        if amt is not None:
            total += amt
            any_amount = True
        pd = _parse_booking_date(row.get("BOOKING_DATE", ""))
        if pd is not None:
            posting_dates.append(pd)
        ps, pe = _period_bounds(row.get("PERIOD", ""))
        if ps is not None:
            period_starts.append(ps)
        if pe is not None:
            period_ends.append(pe)
        ent = (row.get("ENTITY") or "").strip()
        if ent:
            entities.add(ent)
        codings.append(_row_coding_fields(row, null_coding_when_ankreg=null_coding_when_ankreg))
        d = _description(row)
        if d and d not in seen_desc:
            seen_desc.add(d)
            desc_parts.append(d)

    def _unanimous(values: set[str]) -> str:
        if len(values) == 1:
            return next(iter(values))
        return ""

    company = _unanimous(entities)
    dims: list[str] = []
    for i in range(7):
        col = {t[i] for t in codings if t[i]}
        dims.append(_unanimous(col) if col else "")

    desc_joined = " | ".join(desc_parts)
    if len(desc_joined) > 16000:
        desc_joined = desc_joined[:15997] + "..."

    posting_min = min(posting_dates) if posting_dates else None
    p_start = min(period_starts) if period_starts else None
    p_end = max(period_ends) if period_ends else None

    h = hashlib.sha256(join_key.encode("utf-8")).hexdigest()[:14]
    gl_line_id = f"AGG-{h}"
    amount_str = _format_decimal_amount_for_bq(total) if any_amount else ""
    raw = _raw_json_for_invoice_agg(sorted_rows)

    return (
        join_key,
        gl_line_id,
        posting_min.isoformat() if posting_min else "",
        company,
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        dims[4],
        dims[5],
        dims[6],
        amount_str,
        "",
        p_start.isoformat() if p_start else "",
        p_end.isoformat() if p_end else "",
        desc_joined,
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


_GL_JOIN_KEY_MODES = frozenset({"invoice", "entity_invoice", "fingerprint"})


def oracle_gl_tsv_to_csv_bytes(
    path: Path,
    *,
    encoding: str | None = None,
    max_rows: int | None = None,
    join_key_mode: str = "invoice",
    supplier_rows_only: bool = True,
    null_coding_when_ankreg: bool = True,
    aggregate_by_invoice: bool = True,
) -> tuple[bytes, int, int]:
    """
    Build UTF-8 CSV matching ``sql/bigquery/gl_load_schema.json`` column order.

    Returns ``(csv_bytes, rows_written, detail_rows_consumed)`` where ``rows_written`` is the number
    of data rows in the CSV (after rollup, one per join key) and ``detail_rows_consumed`` is how
    many matching subledger source rows were read (equals ``rows_written`` when not aggregating).

    With ``aggregate_by_invoice`` and ``join_key_mode`` ``invoice`` or ``entity_invoice``, each
    distinct join key yields one CSV row (sums ``NET_ACCOUNTED``, merges descriptions, etc.).
    ``max_rows`` then caps how many **matching subledger detail rows** are read before flushing
    groups (the last invoice in the cap may be incomplete).

    With ``join_key_mode`` ``fingerprint`` or ``aggregate_by_invoice=False``, each matching
    source row is one CSV row; ``max_rows`` caps those rows directly.
    """
    if join_key_mode not in _GL_JOIN_KEY_MODES:
        raise ValueError(
            f"join_key_mode must be one of {sorted(_GL_JOIN_KEY_MODES)}, got {join_key_mode!r}"
        )
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_GL_HEADER)
    use_aggregate = aggregate_by_invoice and join_key_mode in ("invoice", "entity_invoice")

    if not use_aggregate:
        n = 0
        for row in iter_oracle_gl_rows(path, encoding=encoding):
            if supplier_rows_only and not row_has_supplier_columns(row):
                continue
            jk = compute_join_key(row, mode=join_key_mode)
            if not jk:
                continue
            writer.writerow(
                oracle_gl_row_to_load_tuple(
                    row,
                    join_key_mode=join_key_mode,
                    null_coding_when_ankreg=null_coding_when_ankreg,
                )
            )
            n += 1
            if max_rows is not None and n >= max_rows:
                break
        return buf.getvalue().encode("utf-8"), n, n

    groups: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    consumed = 0
    for row in iter_oracle_gl_rows(path, encoding=encoding):
        if supplier_rows_only and not row_has_supplier_columns(row):
            continue
        jk = compute_join_key(row, mode=join_key_mode)
        if not jk:
            continue
        groups.setdefault(jk, []).append(row)
        consumed += 1
        if max_rows is not None and consumed >= max_rows:
            break
    for jk, grouped in groups.items():
        writer.writerow(
            _aggregate_oracle_rows_to_load_tuple(
                grouped,
                join_key=jk,
                join_key_mode=join_key_mode,
                null_coding_when_ankreg=null_coding_when_ankreg,
            )
        )
    return buf.getvalue().encode("utf-8"), len(groups), consumed


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
    max_rows: int | None = None,
    join_key_mode: str = "invoice",
    supplier_rows_only: bool = True,
    null_coding_when_ankreg: bool = True,
    aggregate_by_invoice: bool = True,
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
    remaining = max_rows
    total_written = 0
    for i, path in enumerate(resolved):
        data, out_rows, detail_used = oracle_gl_tsv_to_csv_bytes(
            path,
            encoding=encoding,
            max_rows=remaining,
            join_key_mode=join_key_mode,
            supplier_rows_only=supplier_rows_only,
            null_coding_when_ankreg=null_coding_when_ankreg,
            aggregate_by_invoice=aggregate_by_invoice,
        )
        if out_rows == 0:
            continue
        total_written += out_rows
        job = client.load_table_from_file(io.BytesIO(data), full_table, job_config=job_config)
        job.result()
        if i == 0:
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        if remaining is not None:
            remaining -= detail_used
            if remaining <= 0:
                break
    if total_written == 0:
        raise ValueError(
            "No GL rows matched filters (supplier columns required by default; "
            "check source or pass supplier_rows_only=False)."
        )
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
    max_rows: int | None = None,
    join_key_mode: str = "invoice",
    supplier_rows_only: bool = True,
    null_coding_when_ankreg: bool = True,
    aggregate_by_invoice: bool = True,
) -> str:
    """
    Transform a tab-separated GL export (e.g. ``GL_202603.txt``) and load into ``gl_lines``.

    ``source`` may be a local file, a local directory of ``GL_*.txt`` files, or
    ``gs://bucket/object.txt`` (single object; downloaded to a temp file). For many objects
    under GCS, download them locally or run ``load_oracle_gl_tsv_paths_to_bigquery`` on paths.

    ``encoding`` defaults to UTF-8 (with BOM allowed); if that fails, Windows-1252 is used.
    Pass an explicit encoding (e.g. ``"cp1252"``) to force one codec.
    """
    settings = require_settings()
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
                max_rows=max_rows,
                join_key_mode=join_key_mode,
                supplier_rows_only=supplier_rows_only,
                null_coding_when_ankreg=null_coding_when_ankreg,
                aggregate_by_invoice=aggregate_by_invoice,
            )
        if not path.is_file():
            raise FileNotFoundError(f"Not a file or directory: {path}")
        return load_oracle_gl_tsv_paths_to_bigquery(
            [path],
            table_id=table_id,
            write_disposition=write_disposition,
            schema_file=schema_file,
            encoding=encoding,
            max_rows=max_rows,
            join_key_mode=join_key_mode,
            supplier_rows_only=supplier_rows_only,
            null_coding_when_ankreg=null_coding_when_ankreg,
            aggregate_by_invoice=aggregate_by_invoice,
        )
    finally:
        if cleanup is not None:
            cleanup.unlink(missing_ok=True)
