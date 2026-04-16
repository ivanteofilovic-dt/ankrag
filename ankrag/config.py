"""GCP and model configuration (environment-driven)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gcp_project: str = ""
    gcp_region: str = "europe-west1"
    gcs_bucket: str = ""
    bq_dataset: str = "ankreg"
    bq_location: str = "EU"

    gemini_model: str = "gemini-2.0-flash-001"
    gemini_batch_model: str = "gemini-2.0-flash-001"
    embedding_model: str = "text-embedding-004"

    # Optional: Vertex AI Vector Search (Matching Engine). If unset, retrieval uses BigQuery ML.DISTANCE.
    matching_engine_index_endpoint: str = ""
    matching_engine_deployed_index_id: str = ""

    # Confidence routing (plan §6)
    confidence_high_threshold: float = 0.85
    confidence_low_threshold: float = 0.5

    rag_top_k: int = 8
    # Per extraction line: number of nearest historical lines (clamped to [3, 5] at use sites).
    rag_neighbors_per_line: int = 5

    @property
    def publisher_gemini_resource(self) -> str:
        """Full Vertex publisher model resource name."""
        mid = self.gemini_model
        if mid.startswith("publishers/"):
            return mid
        return f"publishers/google/models/{mid}"

    @property
    def bq_dataset_full(self) -> str:
        if not self.gcp_project:
            raise ValueError("GCP_PROJECT is required for BigQuery operations")
        return f"{self.gcp_project}.{self.bq_dataset}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def require_settings() -> Settings:
    s = get_settings()
    if not s.gcp_project:
        raise ValueError("GCP_PROJECT must be set")
    if not s.gcs_bucket:
        raise ValueError("GCS_BUCKET must be set")
    return s
