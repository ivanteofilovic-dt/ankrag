import { ChevronRight, FileUp, Filter, Search } from "lucide-react";
import { useCallback, useEffect, useState, type ChangeEvent } from "react";
import { analyzePdf, getSuggestion, listSuggestions } from "../api/client";
import { confidenceBarClass, statusBadgeClass } from "../lib/statusStyles";
import type { ReviewModalState } from "./ReviewModal";
import type { SuggestionListItem } from "../types";

export function InvoicesView({ onOpenReview }: { onOpenReview: (s: ReviewModalState) => void }) {
  const [items, setItems] = useState<SuggestionListItem[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [listErr, setListErr] = useState<string | null>(null);
  const [uploadErr, setUploadErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setListErr(null);
    try {
      const res = await listSuggestions(100);
      setItems(res.items);
    } catch (e) {
      setListErr(e instanceof Error ? e.message : "Failed to load suggestions");
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const filtered = items.filter(
    (inv) =>
      (inv.document_id ?? "").toLowerCase().includes(searchQuery.toLowerCase()) ||
      inv.suggestion_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      inv.rationale_preview.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  const pendingReview = items.filter((i) => i.status === "Needs Review" || i.status === "Anomaly Flagged").length;

  async function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setUploading(true);
    setUploadErr(null);
    try {
      const data = await analyzePdf(file, true);
      onOpenReview({ kind: "analyze", data });
      await load();
    } catch (err) {
      setUploadErr(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function openStored(id: string) {
    try {
      const data = await getSuggestion(id);
      onOpenReview({ kind: "stored", data });
    } catch (e) {
      setListErr(e instanceof Error ? e.message : "Failed to open suggestion");
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-sm text-slate-500">
            {pendingReview > 0 ? (
              <>
                <span className="font-semibold text-amber-700">{pendingReview}</span> suggestions need attention
                (heuristic from confidence).
              </>
            ) : (
              "Upload a PDF to run extraction + RAG, or open a stored suggestion."
            )}
          </p>
        </div>
        <label className="inline-flex cursor-pointer items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-md transition-colors hover:bg-indigo-700 disabled:opacity-60">
          <FileUp size={18} />
          {uploading ? "Analyzing…" : "Upload PDF"}
          <input type="file" accept="application/pdf" className="hidden" disabled={uploading} onChange={onFileChange} />
        </label>
      </div>

      {uploadErr && (
        <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-2 text-sm text-rose-800">{uploadErr}</div>
      )}

      <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 bg-slate-50/50 p-4">
          <div className="relative min-w-[280px] flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-[18px] -translate-y-1/2 text-slate-400" />
            <input
              type="search"
              placeholder="Search document id, suggestion id, rationale…"
              className="w-full rounded-lg border border-slate-200 py-2 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              className="flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-white"
            >
              <Filter size={16} /> Filter
            </button>
            <button
              type="button"
              onClick={() => void load()}
              className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-600 hover:bg-white"
            >
              Refresh
            </button>
          </div>
        </div>

        {listErr && (
          <div className="border-b border-amber-100 bg-amber-50 px-6 py-3 text-sm text-amber-900">{listErr}</div>
        )}

        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-slate-50 text-xs font-semibold uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-6 py-4">Document / id</th>
                <th className="px-6 py-4">AI preview</th>
                <th className="px-6 py-4">Confidence</th>
                <th className="px-6 py-4">Status</th>
                <th className="px-6 py-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-slate-500">
                    Loading suggestions…
                  </td>
                </tr>
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-slate-500">
                    No rows yet. Run <code className="rounded bg-slate-100 px-1">ankrag suggest --local-pdf …</code> or
                    upload a PDF here.
                  </td>
                </tr>
              ) : (
                filtered.map((inv) => (
                  <tr key={inv.suggestion_id} className="group transition-colors hover:bg-slate-50">
                    <td className="px-6 py-4">
                      <div className="font-semibold text-slate-900">{inv.document_id ?? "—"}</div>
                      <div className="text-xs text-slate-500">{inv.suggestion_id}</div>
                      <div className="text-xs text-slate-400">{inv.created_at}</div>
                    </td>
                    <td className="max-w-md px-6 py-4 text-sm">
                      <div className="line-clamp-2 text-slate-600">{inv.rationale_preview || "—"}</div>
                      {inv.journal_lines_preview[0] && (
                        <div className="mt-1 text-xs text-slate-500">
                          {inv.journal_lines_preview[0].account ?? "—"} ·{" "}
                          {inv.journal_lines_preview[0].cost_center ?? "—"}
                        </div>
                      )}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-100">
                          <div
                            className={`h-full rounded-full ${confidenceBarClass(inv.confidence)}`}
                            style={{ width: `${Math.min(100, inv.confidence * 100)}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium">{(inv.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`rounded-full border px-2 py-1 text-[11px] font-medium ${statusBadgeClass(inv.status)}`}
                      >
                        {inv.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button
                        type="button"
                        onClick={() => void openStored(inv.suggestion_id)}
                        className="inline-flex items-center gap-1 rounded-lg p-2 text-sm font-medium text-indigo-600 transition-colors hover:bg-indigo-50"
                      >
                        Details <ChevronRight size={16} />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
