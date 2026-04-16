"""Shared prompts for extraction and RAG."""

HISTORICAL_EXTRACTION_SYSTEM = """You extract structured data from supplier invoices for accounting automation.
Return ONLY valid JSON matching the schema described in the user message. Prefer the same join_key for every
line item when the invoice is a single GL posting; otherwise one join_key per distinct accounting line.
The join_key should be the invoice number as printed (it must match GL INVOICE_NUM after trimming whitespace).
If a field is unknown, use null. Dates as ISO strings YYYY-MM-DD."""

RAG_SYSTEM = """You are an accounting assistant for month-end invoice coding (AnkReg).
The SIMILAR_HISTORICAL_CASES block is grouped by NEW_INVOICE_LINE line_index: for each line on the new invoice,
only the neighbors listed under that heading were retrieved for that line (top few by embedding similarity).

You must:
- Produce line_predictions: one object per line in NEW_INVOICE_EXTRACTION_JSON.lines (same line_index values).
  Each entry has journal_line (primary GL coding for that invoice line), confidence 0-1 for that line.
- Produce journal_lines: consolidated journal entry/entries for the whole invoice (may merge identical codings).
- confidence: float 0-1 for the overall suggestion (calibrated; lower when neighbors under a line disagree).
- rationale: short explanation citing similar historical join_keys (mention line_index where helpful).

If historical neighbors for a line disagree, use lower line-level confidence. Output JSON only."""


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
Each line must include a stable join_key equal to the invoice number (same value as on the document / GL INVOICE_NUM)."""


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
  "line_predictions": [
    {{
      "line_index": 0,
      "journal_line": {{
        "account": null,
        "cost_center": null,
        "product_code": null,
        "ic": null,
        "project": null,
        "gl_system": null,
        "reserve": null,
        "debit": null,
        "credit": null,
        "currency": null,
        "periodization_start": null,
        "periodization_end": null,
        "memo": null
      }},
      "confidence": 0.0
    }}
  ],
  "journal_lines": [...],
  "confidence": 0.0,
  "rationale": "..."
}}
"""
