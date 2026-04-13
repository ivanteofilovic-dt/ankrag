"""Upload local files to GCS."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from google.cloud import storage

from ankrag.config import require_settings


def _content_type(path: Path) -> str:
    t, _ = mimetypes.guess_type(str(path))
    return t or "application/octet-stream"


def upload_tree(
    local_dir: Path,
    gcs_prefix: str,
    *,
    bucket: str | None = None,
    project: str | None = None,
) -> list[tuple[str, str]]:
    """
    Upload all files under local_dir to gs://bucket/{gcs_prefix}/...

    Returns list of (local_path, gs_uri).
    """
    settings = require_settings()
    b = bucket or settings.gcs_bucket
    client = storage.Client(project=project or settings.gcp_project)
    bucket_ref = client.bucket(b)
    base = local_dir.resolve()
    uploaded: list[tuple[str, str]] = []

    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        blob_name = f"{gcs_prefix.strip('/')}/{rel}"
        blob = bucket_ref.blob(blob_name)
        blob.upload_from_filename(str(path), content_type=_content_type(path))
        uploaded.append((str(path), f"gs://{b}/{blob_name}"))

    return uploaded


def upload_file(local_path: Path, gcs_blob_name: str, *, bucket: str | None = None, project: str | None = None) -> str:
    """Upload a single file; returns gs:// URI."""
    settings = require_settings()
    b = bucket or settings.gcs_bucket
    client = storage.Client(project=project or settings.gcp_project)
    blob = client.bucket(b).blob(gcs_blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{b}/{gcs_blob_name}"


def download_blobs_matching(prefix: str, dest_dir: Path, *, bucket: str | None = None) -> list[Path]:
    """Download all blobs under prefix (no leading slash) into dest_dir."""
    settings = require_settings()
    b = bucket or settings.gcs_bucket
    client = storage.Client(project=settings.gcp_project)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for blob in client.list_blobs(b, prefix=prefix.rstrip("/") + "/"):
        if blob.name.endswith("/"):
            continue
        target = dest_dir / blob.name.split("/")[-1]
        blob.download_to_filename(str(target))
        out.append(target)
    return out
