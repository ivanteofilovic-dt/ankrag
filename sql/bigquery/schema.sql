-- AnkReg BigQuery schema. Replace DATASET with your dataset id (default: ankreg).
-- Run: bq query --use_legacy_sql=false --project_id=$GCP_PROJECT < sql/bigquery/schema.sql
-- Or use: ankrag init-bq

CREATE SCHEMA IF NOT EXISTS `PROJECT.DATASET`
OPTIONS (location = 'EU');

-- Historical GL lines; must include join_key aligning with invoice_extractions.
CREATE TABLE IF NOT EXISTS `PROJECT.DATASET.gl_lines` (
  join_key STRING NOT NULL OPTIONS (description = 'Stable key linking invoice line to GL'),
  gl_line_id STRING OPTIONS (description = 'Optional surrogate from source system'),
  posting_date DATE,
  company_code STRING,
  account STRING,
  cost_center STRING,
  product_code STRING,
  amount NUMERIC,
  currency STRING,
  periodization_start DATE,
  periodization_end DATE,
  description STRING,
  raw_source_row STRING OPTIONS (description = 'Original CSV/row JSON for audit'),
  ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY posting_date
CLUSTER BY join_key, company_code;

CREATE TABLE IF NOT EXISTS `PROJECT.DATASET.invoice_documents` (
  document_id STRING NOT NULL OPTIONS (description = 'Internal id; can match join_key prefix or hash'),
  gcs_uri STRING NOT NULL,
  content_hash STRING,
  source_filename STRING,
  ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- One row per invoice line (or whole-invoice with line_index=0).
CREATE TABLE IF NOT EXISTS `PROJECT.DATASET.invoice_extractions` (
  join_key STRING NOT NULL,
  document_id STRING NOT NULL,
  line_index INT64 NOT NULL OPTIONS (description = '0 for header-only / single-invoice row'),
  supplier STRING,
  invoice_number STRING,
  invoice_date DATE,
  currency STRING,
  line_description STRING,
  line_amount NUMERIC,
  periodization_hint STRING,
  extraction_json STRING NOT NULL OPTIONS (description = 'Full model JSON response'),
  model_id STRING,
  extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
CLUSTER BY join_key, document_id;

-- Denormalized view for training / RAG context.
CREATE OR REPLACE VIEW `PROJECT.DATASET.invoice_gl_training_view` AS
SELECT
  e.join_key,
  e.document_id,
  e.line_index,
  e.supplier,
  e.invoice_number,
  e.invoice_date,
  e.line_description,
  e.line_amount,
  e.currency,
  e.periodization_hint,
  g.account,
  g.cost_center,
  g.product_code,
  g.amount AS gl_amount,
  g.currency AS gl_currency,
  g.periodization_start,
  g.periodization_end,
  g.posting_date,
  g.company_code,
  g.description AS gl_description
FROM `PROJECT.DATASET.invoice_extractions` AS e
LEFT JOIN `PROJECT.DATASET.gl_lines` AS g
  ON e.join_key = g.join_key;

CREATE TABLE IF NOT EXISTS `PROJECT.DATASET.extraction_errors` (
  document_id STRING,
  gcs_uri STRING,
  error_message STRING,
  batch_key STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Embeddings for retrieval (BigQuery cosine when Matching Engine is not configured).
CREATE TABLE IF NOT EXISTS `PROJECT.DATASET.invoice_line_embeddings` (
  join_key STRING NOT NULL,
  invoice_line_id STRING NOT NULL OPTIONS (description = 'join_key + line_index as STRING'),
  document_id STRING NOT NULL,
  line_index INT64 NOT NULL,
  embedding ARRAY<FLOAT64> NOT NULL,
  embed_text STRING,
  embedding_model STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
CLUSTER BY join_key, document_id;

CREATE TABLE IF NOT EXISTS `PROJECT.DATASET.rag_suggestions` (
  suggestion_id STRING NOT NULL,
  document_id STRING,
  gcs_uri STRING,
  join_keys_suggested STRING OPTIONS (description = 'JSON array of neighbor join_keys'),
  model_output_json STRING NOT NULL,
  confidence FLOAT64,
  confidence_components STRING OPTIONS (description = 'JSON object; blended confidence breakdown'),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);
