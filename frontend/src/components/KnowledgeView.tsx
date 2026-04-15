import { Database } from "lucide-react";
import { useEffect, useState } from "react";
import { getConfig, getStats } from "../api/client";
import type { ConfigResponse, StatsResponse } from "../types";

export function KnowledgeView() {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const [c, s] = await Promise.all([getConfig(), getStats()]);
        setConfig(c);
        setStats(s);
      } catch {
        /* handled by empty UI */
      }
    })();
  }, []);

  const emb = stats?.counts?.invoice_line_embeddings;
  const ext = stats?.counts?.invoice_extractions;

  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="max-w-lg rounded-3xl border border-slate-200 bg-white p-8 shadow-xl">
        <Database size={64} className="mx-auto mb-6 text-indigo-600" />
        <h3 className="text-2xl font-bold text-slate-900">RAG knowledge base</h3>
        <p className="mt-2 text-sm text-slate-500">
          Historical invoice lines and GL postings live in BigQuery (<code className="rounded bg-slate-100 px-1">invoice_gl_training_view</code>
          ). Embeddings are stored per line for cosine retrieval.
        </p>
        <div className="mt-6 space-y-3 text-left">
          <div className="flex justify-between rounded-lg border border-slate-100 bg-slate-50 p-3">
            <span className="text-xs font-bold text-slate-400">EMBEDDING ROWS</span>
            <span className="font-mono text-xs font-bold text-slate-700">
              {emb != null ? emb.toLocaleString() : "—"}
            </span>
          </div>
          <div className="flex justify-between rounded-lg border border-slate-100 bg-slate-50 p-3">
            <span className="text-xs font-bold text-slate-400">EXTRACTION ROWS</span>
            <span className="font-mono text-xs font-bold text-slate-700">
              {ext != null ? ext.toLocaleString() : "—"}
            </span>
          </div>
          <div className="flex justify-between rounded-lg border border-slate-100 bg-slate-50 p-3">
            <span className="text-xs font-bold text-slate-400">RETRIEVAL BACKEND</span>
            <span className="font-mono text-xs font-bold text-slate-700">{config?.vector_search_backend ?? "—"}</span>
          </div>
          <div className="flex justify-between rounded-lg border border-slate-100 bg-slate-50 p-3">
            <span className="text-xs font-bold text-slate-400">EMBEDDING MODEL</span>
            <span className="font-mono text-xs font-bold text-slate-700">{config?.embedding_model ?? "—"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
