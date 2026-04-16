export interface HealthResponse {
  ok: boolean;
  gcp_project_set: boolean;
  gcs_bucket_set: boolean;
}

export interface ConfigResponse {
  gcp_project: string | null;
  gcp_region: string;
  bq_dataset: string;
  gemini_model: string;
  embedding_model: string;
  rag_top_k: number;
  rag_neighbors_per_line: number;
  confidence_high_threshold: number;
  confidence_low_threshold: number;
  vector_search_backend: string;
}

export interface StatsResponse {
  configured: boolean;
  counts: Record<string, number | null | undefined>;
  error?: string | null;
}

export interface InvoiceLine {
  line_index: number;
  description?: string | null;
  amount?: string | null;
  join_key: string;
}

export interface Extraction {
  document_id: string;
  supplier?: string | null;
  invoice_number?: string | null;
  invoice_date?: string | null;
  currency?: string | null;
  periodization_hint?: string | null;
  lines: InvoiceLine[];
}

export interface JournalLine {
  account?: string | null;
  cost_center?: string | null;
  product_code?: string | null;
  ic?: string | null;
  project?: string | null;
  gl_system?: string | null;
  reserve?: string | null;
  debit?: string | null;
  credit?: string | null;
  currency?: string | null;
  periodization_start?: string | null;
  periodization_end?: string | null;
  memo?: string | null;
}

export interface LineCodingPrediction {
  line_index: number;
  journal_line: JournalLine;
  confidence: number;
}

export interface CodingSuggestion {
  journal_lines: JournalLine[];
  confidence: number;
  rationale: string;
  line_predictions?: LineCodingPrediction[];
}

export interface TrainingSnippet {
  join_key?: string;
  supplier?: string;
  invoice_number?: string;
  invoice_date?: string;
  line_description?: string;
  line_amount?: string;
  currency?: string;
  account?: string;
  cost_center?: string;
  posting_date?: string;
  [key: string]: unknown;
}

export interface NeighborRecord {
  rank: number;
  join_key: string;
  invoice_line_id: string;
  document_id: string;
  line_index: number;
  cosine_distance: number | null;
  similarity: number | null;
  training: TrainingSnippet | null;
  /** Query invoice line this neighbor was retrieved for (per-line embedding search). */
  query_line_index?: number | null;
}

export interface AnalyzeResponse {
  extraction: Extraction;
  suggestion: CodingSuggestion;
  neighbors: NeighborRecord[];
  confidence_meta: Record<string, unknown>;
  final_confidence: number;
  status: string;
}

export interface SuggestionListItem {
  suggestion_id: string;
  document_id: string | null;
  gcs_uri: string | null;
  confidence: number;
  status: string;
  created_at: string;
  rationale_preview: string;
  journal_lines_preview: JournalLine[];
}

export interface SuggestionsListResponse {
  items: SuggestionListItem[];
}

export interface SuggestionDetailResponse {
  suggestion_id: string;
  document_id: string | null;
  gcs_uri: string | null;
  created_at: string;
  suggestion: CodingSuggestion;
  final_confidence: number;
  confidence_meta: Record<string, unknown>;
  status: string;
  neighbors: NeighborRecord[];
  extraction: Extraction | null;
}
