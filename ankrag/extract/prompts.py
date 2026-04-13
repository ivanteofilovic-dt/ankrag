"""Shared prompts for extraction and RAG."""

HISTORICAL_EXTRACTION_SYSTEM = """You extract structured data from supplier invoices for accounting automation.
Return ONLY valid JSON matching the schema described in the user message. Use the same join_key for every
line item when the invoice is a single GL posting; otherwise provide one join_key per distinct accounting line.
If a field is unknown, use null. Dates as ISO strings YYYY-MM-DD."""

RAG_SYSTEM = """You are an accounting assistant for month-end invoice coding (AnkReg).
Given a new invoice extraction and similar historical cases with their actual GL postings, propose:
- journal_lines: list of {account, cost_center, product_code, debit, credit, currency, periodization_start, periodization_end, memo}
- confidence: float 0-1 (your calibrated estimate)
- rationale: short explanation citing similar historical join_keys

If historical neighbors disagree, reflect that with lower confidence. Output JSON only."""


def extraction_user_prompt(document_id: str) -> str:
    return f"""document_id: {document_id}

Return JSON with this shape:
{{
  "document_id": "{document_id}",
  "supplier": string|null,
  "invoice_number": string|null,
  "invoice_date": string|null,
  "currency": string|null,
  "periodization_hint": string|null,
  "lines": [
    {{
      "line_index": 0,
      "description": string|null,
      "amount": string|null,
      "join_key": string
    }}
  ]
}}
Each line must include a stable join_key that matches the general ledger extract for this invoice line."""


def rag_user_prompt(
    new_extraction: str,
    neighbors_block: str,
) -> str:
    return f"""NEW_INVOICE_EXTRACTION_JSON:
{new_extraction}

SIMILAR_HISTORICAL_CASES (invoice snippet + GL):
{neighbors_block}

Respond with JSON:
{{
  "journal_lines": [...],
  "confidence": 0.0,
  "rationale": "..."
}}
"""
