import { BarChart3 } from "lucide-react";

export function AnalyticsPlaceholder() {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white py-24 text-center">
      <BarChart3 className="mb-4 text-slate-300" size={48} />
      <h2 className="text-lg font-semibold text-slate-800">Historical insights</h2>
      <p className="mt-2 max-w-md text-sm text-slate-500">
        Use <code className="rounded bg-slate-100 px-1">ankrag eval-heldout</code> and BigQuery for offline metrics. A
        charting view can plug into the same datasets later.
      </p>
    </div>
  );
}
