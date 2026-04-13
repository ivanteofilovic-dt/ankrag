"""Structured extraction JSON expected from Gemini (batch or online)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InvoiceLineItem(BaseModel):
    line_index: int = Field(ge=0)
    description: str | None = None
    amount: str | None = None
    join_key: str = Field(description="Must match GL join_key for this line")


class InvoiceExtractionResult(BaseModel):
    document_id: str
    supplier: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    currency: str | None = None
    periodization_hint: str | None = None
    lines: list[InvoiceLineItem] = Field(default_factory=list)

    @classmethod
    def from_model_dict(cls, data: dict[str, Any]) -> InvoiceExtractionResult:
        return cls.model_validate(data)
