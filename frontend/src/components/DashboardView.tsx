import {
  AlertCircle,
  BrainCircuit,
  CheckCircle2,
  Clock,
  Database,
  History,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { useEffect, useState } from "react";
import { getConfig, getHealth, getStats } from "../api/client";
import type { ConfigResponse, HealthResponse, StatsResponse } from "../types";

function n(v: number | null | undefined): string {
  if (v == null) return "—";
  return v.toLocaleString();
}

export function DashboardView() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [h, c, st] = await Promise.all([getHealth(), getConfig(), getStats()]);
        if (!cancelled) {
          setHealth(h);
          setConfig(c);
          setStats(st);
          setErr(null);
        }
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : "Failed to load dashboard");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const emb = stats?.counts?.invoice_line_embeddings;
  const sug = stats?.counts?.rag_suggestions;
  const ext = stats?.counts?.invoice_extractions;
  const gl = stats?.counts?.gl_lines;

  const automationRate =
    emb != null && sug != null && emb > 0 ? Math.min(99, Math.round((sug / emb) * 100)) : null;

  const cards = [
    {
      label: "Embedding rows",
      value: n(emb),
      icon: Database,
      color: "text-indigo-600",
      hint: "invoice_line_embeddings",
    },
    {
      label: "RAG suggestions stored",
      value: n(sug),
      icon: Zap,
      color: "text-amber-600",
      hint: "rag_suggestions",
    },
    {
      label: "Invoice extractions",
      value: n(ext),
      icon: ShieldCheck,
      color: "text-emerald-600",
      hint: "invoice_extractions",
    },
    {
      label: "GL lines",
      value: n(gl),
      icon: AlertCircle,
      color: "text-rose-600",
      hint: "gl_lines",
    },
  ];

  const barHeights = [45, 60, 55, 80, 75, 90, 85, 95, 88, 92].map((v, i) => ({
    v,
    i,
    label: emb != null ? `${((emb + i * 137) % 997) % 100}` : String(i + 1),
  }));

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h2 className="text-2xl font-bold text-slate-800">Pipeline overview</h2>
        <button
          type="button"
          className="flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-white transition-colors hover:bg-indigo-700"
        >
          <History size={18} />
          <span>Export (CLI)</span>
        </button>
      </div>

      {err && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          {err} — start the API with <code className="rounded bg-white px-1">ankrag serve</code> and ensure env is
          configured.
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        {cards.map((item) => (
          <div key={item.label} className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-4 flex items-start justify-between">
              <div className={`rounded-lg bg-slate-50 p-2 ${item.color}`}>
                <item.icon size={24} />
              </div>
              <span className="rounded bg-slate-100 px-2 py-1 text-[10px] font-medium text-slate-500">{item.hint}</span>
            </div>
            <p className="text-sm text-slate-500">{item.label}</p>
            <h3 className="text-2xl font-bold text-slate-900">{item.value}</h3>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="rounded-xl border border-slate-200 bg-white p-6 lg:col-span-2">
          <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold">
            <BrainCircuit size={20} className="text-indigo-600" />
            Activity sketch
          </h3>
          <p className="mb-4 text-sm text-slate-500">
            Placeholder bars until we expose time-series from BigQuery. Suggestion rate vs embeddings:{" "}
            {automationRate != null ? `${automationRate}% (rough)` : "—"}
          </p>
          <div className="flex h-64 items-end gap-2 px-4">
            {barHeights.map(({ v, i, label }) => (
              <div key={i} className="flex flex-1 flex-col items-center gap-2">
                <div
                  className="w-full cursor-default rounded-t-sm bg-indigo-500 opacity-80 transition-all hover:opacity-100"
                  style={{ height: `${v}%` }}
                  title={`Bucket ${label}`}
                />
                <span className="mt-2 rotate-45 text-[10px] text-slate-400">{label}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="relative overflow-hidden rounded-xl bg-slate-900 p-6 text-white shadow-xl">
          <div className="relative z-10 space-y-4">
            <h3 className="text-lg font-semibold">Runtime</h3>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400">API</span>
              <span className="flex items-center gap-1 text-emerald-400">
                <CheckCircle2 size={14} /> {health?.ok ? "Up" : "Unknown"}
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400">GCP project</span>
              <span className="text-emerald-400">{health?.gcp_project_set ? "Set" : "Missing"}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400">GCS bucket</span>
              <span className={health?.gcs_bucket_set ? "text-emerald-400" : "text-amber-400"}>
                {health?.gcs_bucket_set ? "Set" : "Missing"}
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400">Retrieval</span>
              <span className="text-indigo-300">{config?.vector_search_backend ?? "—"}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400">Gemini</span>
              <span className="truncate font-mono text-xs text-slate-300">{config?.gemini_model ?? "—"}</span>
            </div>
            <div className="border-t border-slate-800 pt-6">
              <p className="mb-2 text-xs text-slate-500">Embedding model</p>
              <p className="font-mono text-sm text-indigo-300">{config?.embedding_model ?? "—"}</p>
            </div>
          </div>
          <div className="absolute -bottom-10 -right-10 opacity-10">
            <Database size={200} />
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 text-sm text-slate-500">
        <Clock size={16} />
        BigQuery dataset: <span className="font-mono text-slate-700">{config?.bq_dataset ?? "—"}</span> · top_k:{" "}
        {config?.rag_top_k ?? "—"}
      </div>
    </div>
  );
}
