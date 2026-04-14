"""Typer CLI for AnkReg pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ankrag import __version__

app = typer.Typer(no_args_is_help=True, help="AnkReg: historical RAG for invoice coding on GCP")


@app.callback()
def _main() -> None:
    pass


@app.command("version")
def cmd_version() -> None:
    typer.echo(__version__)


@app.command("init-bq")
def cmd_init_bq() -> None:
    from ankrag.ingest.bq import run_schema_sql

    run_schema_sql()
    typer.echo("BigQuery schema applied.")


@app.command("upload")
def cmd_upload(
    local_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    prefix: str = typer.Option("historical/invoices", help="GCS prefix under bucket"),
) -> None:
    from ankrag.ingest.gcs import upload_tree

    pairs = upload_tree(local_dir, prefix)
    typer.echo(f"Uploaded {len(pairs)} files.")


@app.command("upload-file")
def cmd_upload_file(
    local_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    gcs_blob: str = typer.Argument(..., help="Object name under bucket, e.g. batch-jobs/input/run.jsonl"),
) -> None:
    from ankrag.ingest.gcs import upload_file

    uri = upload_file(local_path, gcs_blob)
    typer.echo(uri)


@app.command("load-gl")
def cmd_load_gl(
    source: str = typer.Argument(
        ...,
        help="Comma CSV: gs://bucket/path.csv (or wildcard). Tab GL export: use --oracle-export with local path or gs://.../GL_YYYYMM.txt",
    ),
    oracle_export: bool = typer.Option(
        False,
        "--oracle-export",
        help="Source is tab-separated GL_*.txt (ENTITY, JOURNAL_NUMBER, BOOKING_DATE, …) transformed into gl_lines",
    ),
    autodetect: bool = typer.Option(
        False,
        help="Let BigQuery infer CSV schema (not recommended for production); ignored with --oracle-export",
    ),
) -> None:
    from pathlib import Path

    from ankrag.ingest.bq import load_gl_csv_to_bigquery
    from ankrag.ingest.gl_oracle import load_oracle_gl_tsv_to_bigquery

    schema = Path(__file__).resolve().parents[1] / "sql" / "bigquery" / "gl_load_schema.json"
    if oracle_export:
        p = Path(source)
        loc: Path | str = source if source.startswith("gs://") else p
        table = load_oracle_gl_tsv_to_bigquery(loc, schema_file=schema if schema.exists() else None)
    else:
        table = load_gl_csv_to_bigquery(
            source,
            autodetect=autodetect,
            schema_file=schema if schema.exists() and not autodetect else None,
        )
    typer.echo(f"Loaded into {table}")


@app.command("build-batch-jsonl")
def cmd_build_batch_jsonl(
    manifest_out: Path = typer.Option(
        Path("out/batch_manifest.jsonl"),
        help="Writes keys + gcs_uri + document_id for later import",
    ),
    jsonl_out: Path = typer.Option(Path("out/batch_input.jsonl")),
    prefix: str = typer.Option(
        "historical/invoices",
        "--prefix",
        help="GCS prefix listing PDFs",
    ),
) -> None:
    from google.cloud import storage

    from ankrag.config import require_settings
    from ankrag.extract.batch_jsonl import build_batch_jsonl_for_pdfs, write_local_jsonl
    from ankrag.extract.pipeline import write_manifest

    s = require_settings()
    client = storage.Client(project=s.gcp_project)
    items: list[tuple[str, str, str]] = []
    for blob in client.list_blobs(s.gcs_bucket, prefix=prefix.rstrip("/") + "/"):
        if not blob.name.lower().endswith(".pdf"):
            continue
        uri = f"gs://{s.gcs_bucket}/{blob.name}"
        doc_id = Path(blob.name).stem
        bk = doc_id
        items.append((bk, uri, doc_id))
    if not items:
        raise typer.BadParameter(f"No PDFs under gs://{s.gcs_bucket}/{prefix}")
    write_local_jsonl(build_batch_jsonl_for_pdfs(items), jsonl_out)
    write_manifest(manifest_out, items)
    typer.echo(f"Wrote {jsonl_out} ({len(items)} requests) and {manifest_out}")
    typer.echo("Upload the JSONL to GCS, then: ankrag submit-batch gs://.../batch_input.jsonl --dest-prefix gs://.../out/")


@app.command("submit-batch")
def cmd_submit_batch(
    jsonl_gcs_uri: str = typer.Argument(..., help="gs://bucket/path/batch_input.jsonl"),
    dest_prefix: str = typer.Option(
        ...,
        "--dest-prefix",
        help="gs://bucket/prefix/ for batch output",
    ),
    wait: bool = typer.Option(False, help="Poll until the job completes"),
) -> None:
    from ankrag.extract.batch_job import submit_gemini_batch_job, wait_for_batch_job

    name = submit_gemini_batch_job(jsonl_gcs_uri, dest_prefix)
    typer.echo(name)
    if wait:
        info = wait_for_batch_job(name)
        typer.echo(json.dumps(info, default=str))


@app.command("import-batch-results")
def cmd_import_batch_results(
    jsonl_files: list[Path] = typer.Argument(..., exists=True, dir_okay=False),
    manifest: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="batch_manifest.jsonl from build-batch-jsonl",
    ),
    model_id: str = typer.Option("", help="Defaults to GEMINI_BATCH_MODEL from env"),
) -> None:
    from ankrag.config import get_settings
    from ankrag.extract.pipeline import import_batch_prediction_jsonl, read_manifest

    mid = model_id or get_settings().gemini_batch_model
    gcs_map = read_manifest(manifest) if manifest else None
    total_ok = total_err = 0
    for p in jsonl_files:
        ok, err = import_batch_prediction_jsonl(p, model_id=mid, gcs_by_key=gcs_map)
        total_ok += ok
        total_err += err
        typer.echo(f"{p}: ok={ok} err={err}")
    typer.echo(f"TOTAL ok={total_ok} err={total_err}")


@app.command("embed-backfill")
def cmd_embed_backfill(
    limit: int | None = typer.Option(None),
    dry_run: bool = typer.Option(False),
) -> None:
    from ankrag.embeddings.embed import backfill_embeddings_from_extractions

    n = backfill_embeddings_from_extractions(limit=limit, dry_run=dry_run)
    typer.echo(f"Processed {n} embedding rows.")


@app.command("export-vector-jsonl")
def cmd_export_vector_jsonl(
    out: Path = typer.Option(Path("out/vector_datapoints.jsonl")),
) -> None:
    from google.cloud import bigquery

    from ankrag.config import require_settings
    from ankrag.embeddings.vector_export import write_matching_engine_jsonl

    s = require_settings()
    client = bigquery.Client(project=s.gcp_project, location=s.bq_location)
    sql = f"""
    SELECT join_key, invoice_line_id, embedding
    FROM `{s.gcp_project}.{s.bq_dataset}.invoice_line_embeddings`
    """
    rows = []
    for r in client.query(sql).result():
        rows.append(
            {
                "id": r["invoice_line_id"],
                "embedding": list(r["embedding"]),
                "join_key": r["join_key"],
            }
        )
    write_matching_engine_jsonl(rows, out)
    typer.echo(f"Wrote {len(rows)} datapoints to {out}")


@app.command("similar")
def cmd_similar(
    gcs_uri: str | None = typer.Option(None, help="gs://.../invoice.pdf"),
    local_pdf: Path | None = typer.Option(None, exists=True, dir_okay=False),
    top_k: int | None = typer.Option(None, help="Override RAG_TOP_K / settings.rag_top_k"),
    quiet: bool = typer.Option(False, help="JSON on stdout only; skip neighbor INFO lines on stderr"),
    include_embed_text: bool = typer.Option(False, help="Include full retrieval embed_text in JSON (can be long)"),
) -> None:
    """Extract invoice + embedding search only; no coding suggestion (no second Gemini call)."""
    import logging

    from ankrag.rag.suggest import similar_invoices_for_gcs_pdf, similar_invoices_for_local_pdf

    if bool(gcs_uri) == bool(local_pdf):
        raise typer.BadParameter("Provide exactly one of --gcs-uri or --local-pdf")
    if not quiet:
        logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    if gcs_uri:
        payload = similar_invoices_for_gcs_pdf(
            gcs_uri,
            top_k=top_k,
            log_neighbors=not quiet,
            include_embed_text=include_embed_text,
        )
    else:
        assert local_pdf is not None
        payload = similar_invoices_for_local_pdf(
            local_pdf,
            top_k=top_k,
            log_neighbors=not quiet,
            include_embed_text=include_embed_text,
        )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


@app.command("suggest")
def cmd_suggest(
    gcs_uri: str | None = typer.Option(None, help="gs://.../invoice.pdf"),
    local_pdf: Path | None = typer.Option(None, exists=True, dir_okay=False),
    no_persist: bool = typer.Option(False),
) -> None:
    from ankrag.rag.suggest import suggest_coding_for_gcs_pdf, suggest_coding_for_local_pdf

    if bool(gcs_uri) == bool(local_pdf):
        raise typer.BadParameter("Provide exactly one of --gcs-uri or --local-pdf")
    if gcs_uri:
        sug, hits, meta = suggest_coding_for_gcs_pdf(gcs_uri, persist=not no_persist)
    else:
        assert local_pdf is not None
        sug, hits, meta = suggest_coding_for_local_pdf(local_pdf, persist=not no_persist)
    typer.echo(sug.model_dump_json(indent=2))
    typer.echo(json.dumps({"neighbors": len(hits), **meta}, default=str, indent=2))


@app.command("eval-heldout")
def cmd_eval_heldout(
    n: int = typer.Option(20),
    seed: int = typer.Option(42),
    persist: bool = typer.Option(False),
) -> None:
    from ankrag.eval.run_eval import run_heldout_eval

    r = run_heldout_eval(sample_size=n, seed=seed, persist_suggestions=persist)
    typer.echo(json.dumps(r, default=str, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
