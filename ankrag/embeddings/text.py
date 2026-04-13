"""Canonical text for embedding invoice lines."""

from __future__ import annotations


def canonical_embed_text(
    *,
    supplier: str | None,
    invoice_number: str | None,
    line_description: str | None,
    line_amount: str | None,
    currency: str | None,
    periodization_hint: str | None,
    join_key: str,
) -> str:
    parts = [
        f"join_key={join_key}",
        f"supplier={supplier or ''}",
        f"invoice_number={invoice_number or ''}",
        f"line={line_description or ''}",
        f"amount={line_amount or ''}",
        f"currency={currency or ''}",
        f"periodization_hint={periodization_hint or ''}",
    ]
    return " | ".join(parts)
