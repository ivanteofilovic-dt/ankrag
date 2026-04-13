"""Structured RAG output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JournalLine(BaseModel):
    account: str | None = None
    cost_center: str | None = None
    product_code: str | None = None
    debit: str | None = None
    credit: str | None = None
    currency: str | None = None
    periodization_start: str | None = None
    periodization_end: str | None = None
    memo: str | None = None


class CodingSuggestion(BaseModel):
    journal_lines: list[JournalLine] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""

    @classmethod
    def from_model_json(cls, data: dict[str, Any]) -> CodingSuggestion:
        return cls.model_validate(data)
