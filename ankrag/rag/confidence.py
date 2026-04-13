"""Combine model confidence with retrieval signals (plan §6)."""

from __future__ import annotations

from typing import Any

from ankrag.rag.models import CodingSuggestion
from ankrag.rag.retrieve import NeighborHit


def distance_to_similarity(distance: float) -> float:
    """Map BigQuery cosine distance to a rough [0,1] score (higher is better)."""
    # Cosine distance in [0, 2] typically; clamp similarity estimate.
    sim = max(0.0, 1.0 - float(distance))
    return min(1.0, sim)


def neighbor_account_agreement(neighbor_rows: list[dict[str, Any]]) -> float:
    """1.0 if all non-null accounts match; lower if mixed."""
    accounts = [str(r.get("account")) for r in neighbor_rows if r.get("account")]
    if not accounts:
        return 0.3
    uniq = set(accounts)
    if len(uniq) == 1:
        return 1.0
    return max(0.2, 1.0 / len(uniq))


def blend_confidence(
    model_confidence: float,
    hits: list[NeighborHit],
    neighbor_rows: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    if not hits:
        final = min(model_confidence, 0.25)
        return final, {"reason": "no_neighbors", "model": model_confidence}

    top_sim = distance_to_similarity(hits[0].distance)
    mean_sim = sum(distance_to_similarity(h.distance) for h in hits[:5]) / min(5, len(hits))
    agree = neighbor_account_agreement(neighbor_rows)
    combined = (
        0.45 * model_confidence + 0.25 * top_sim + 0.15 * mean_sim + 0.15 * agree
    )
    final = max(0.0, min(1.0, combined))
    meta = {
        "model": model_confidence,
        "top_similarity": top_sim,
        "mean_top5_similarity": mean_sim,
        "neighbor_account_agreement": agree,
    }
    return final, meta


def apply_confidence_policy(
    suggestion: CodingSuggestion,
    hits: list[NeighborHit],
    neighbor_rows: list[dict[str, Any]],
) -> tuple[CodingSuggestion, float, dict[str, Any]]:
    final, meta = blend_confidence(suggestion.confidence, hits, neighbor_rows)
    updated = suggestion.model_copy(update={"confidence": final})
    return updated, final, meta
