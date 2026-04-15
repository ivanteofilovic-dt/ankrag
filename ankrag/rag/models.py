"""Structured RAG output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class JournalLine(BaseModel):
    account: str | None = None
    cost_center: str | None = None
    product_code: str | None = None
    ic: str | None = None
    project: str | None = None
    gl_system: str | None = None
    reserve: str | None = None
    debit: str | None = None
    credit: str | None = None

    @field_validator("debit", "credit", mode="before")
    @classmethod
    def _coerce_amount_to_str(cls, v: object) -> str | None:
        if v is None:
            return None
        # bool subclasses int; reject accidental booleans before numeric coercion.
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            return v
        return str(v)
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
