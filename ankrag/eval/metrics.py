"""Offline metrics and confidence routing buckets."""

from __future__ import annotations

from ankrag.config import get_settings


def account_match_rate(predicted: list[str | None], actual: list[str | None]) -> float:
    pairs = [(p, a) for p, a in zip(predicted, actual, strict=True) if a is not None]
    if not pairs:
        return 0.0
    ok = sum(1 for p, a in pairs if p is not None and str(p).strip() == str(a).strip())
    return ok / len(pairs)


def route_bucket(confidence: float) -> str:
    s = get_settings()
    if confidence >= s.confidence_high_threshold:
        return "auto_post"
    if confidence < s.confidence_low_threshold:
        return "human_review"
    return "discretionary"
