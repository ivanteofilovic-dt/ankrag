"""Submit and poll Vertex AI Gemini batch jobs via google-genai."""

from __future__ import annotations

import time
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import JobState

from ankrag.config import require_settings


def _vertex_client() -> genai.Client:
    s = require_settings()
    return genai.Client(
        vertexai=True,
        project=s.gcp_project,
        location=s.gcp_region,
    )


def submit_gemini_batch_job(
    input_gcs_jsonl_uri: str,
    output_gcs_prefix: str,
    *,
    display_name: str = "ankreg-invoice-extract",
) -> str:
    """
    Submit a batch job. input must be gs://.../file.jsonl in Vertex batch format.

    Returns the batch job resource name (for polling).
    """
    s = require_settings()
    client = _vertex_client()
    model = s.gemini_batch_model
    if not model.startswith("publishers/"):
        model = f"publishers/google/models/{model}"

    dest = output_gcs_prefix.rstrip("/")
    if not dest.endswith("/"):
        dest += "/"

    job = client.batches.create(
        model=model,
        src=input_gcs_jsonl_uri,
        config=types.CreateBatchJobConfig(
            dest=dest,
            display_name=display_name,
        ),
    )
    if not job.name:
        raise RuntimeError("Batch job created without name")
    return job.name


def wait_for_batch_job(
    job_name: str,
    *,
    poll_seconds: float = 30.0,
    timeout_seconds: float = 86400.0,
) -> dict[str, Any]:
    """Poll until job completes or times out. Returns final job as dict-like summary."""
    client = _vertex_client()
    start = time.monotonic()
    while True:
        job = client.batches.get(name=job_name)
        state = job.state
        if state in (JobState.JOB_STATE_SUCCEEDED, JobState.JOB_STATE_PARTIALLY_SUCCEEDED):
            return {
                "name": job.name,
                "state": str(state),
                "dest": job.dest,
                "error": job.error,
            }
        if state in (
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_EXPIRED,
        ):
            raise RuntimeError(f"Batch job ended in state {state}: {job.error}")
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError(f"Batch job {job_name} not finished within {timeout_seconds}s")
        time.sleep(poll_seconds)
