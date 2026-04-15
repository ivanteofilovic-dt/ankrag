import type {
  AnalyzeResponse,
  ConfigResponse,
  HealthResponse,
  StatsResponse,
  SuggestionDetailResponse,
  SuggestionsListResponse,
} from "../types";

async function parseJson<T>(res: Response): Promise<T> {
  const text = await res.text();
  if (!res.ok) {
    let detail = text;
    try {
      const j = JSON.parse(text) as { detail?: string };
      if (typeof j.detail === "string") detail = j.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail || res.statusText);
  }
  return text ? (JSON.parse(text) as T) : ({} as T);
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch("/api/health");
  return parseJson<HealthResponse>(res);
}

export async function getConfig(): Promise<ConfigResponse> {
  const res = await fetch("/api/config");
  return parseJson<ConfigResponse>(res);
}

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch("/api/stats");
  return parseJson<StatsResponse>(res);
}

export async function listSuggestions(limit = 50): Promise<SuggestionsListResponse> {
  const res = await fetch(`/api/suggestions?limit=${limit}`);
  return parseJson<SuggestionsListResponse>(res);
}

export async function getSuggestion(id: string): Promise<SuggestionDetailResponse> {
  const res = await fetch(`/api/suggestions/${encodeURIComponent(id)}`);
  return parseJson<SuggestionDetailResponse>(res);
}

export async function analyzePdf(file: File, persist = true): Promise<AnalyzeResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const url = persist ? "/api/analyze" : "/api/analyze?persist=false";
  const res = await fetch(url, {
    method: "POST",
    body: fd,
  });
  return parseJson<AnalyzeResponse>(res);
}
