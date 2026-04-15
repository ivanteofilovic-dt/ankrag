export function statusBadgeClass(status: string): string {
  switch (status) {
    case "Auto-Posted":
      return "bg-emerald-100 text-emerald-700 border-emerald-200";
    case "Needs Review":
      return "bg-amber-100 text-amber-700 border-amber-200";
    case "Anomaly Flagged":
      return "bg-rose-100 text-rose-700 border-rose-200";
    default:
      return "bg-slate-100 text-slate-700 border-slate-200";
  }
}

export function confidenceBarClass(confidence: number): string {
  if (confidence > 0.8) return "bg-emerald-500";
  if (confidence > 0.5) return "bg-amber-500";
  return "bg-rose-500";
}
