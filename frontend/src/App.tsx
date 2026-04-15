import {
  BarChart3,
  BrainCircuit,
  Database,
  FileText,
  LayoutDashboard,
  Settings,
} from "lucide-react";
import { useState } from "react";
import { AnalyticsPlaceholder } from "./components/AnalyticsPlaceholder";
import { DashboardView } from "./components/DashboardView";
import { InvoicesView } from "./components/InvoicesView";
import { KnowledgeView } from "./components/KnowledgeView";
import { ReviewModal, type ReviewModalState } from "./components/ReviewModal";

type TabId = "dashboard" | "invoices" | "analytics" | "knowledge";

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("dashboard");
  const [review, setReview] = useState<ReviewModalState | null>(null);

  const nav: { id: TabId; label: string; icon: typeof LayoutDashboard; badge?: string }[] = [
    { id: "dashboard", label: "MEC overview", icon: LayoutDashboard },
    { id: "invoices", label: "Invoice control", icon: FileText },
    { id: "analytics", label: "Historical insights", icon: BarChart3 },
    { id: "knowledge", label: "RAG pipeline", icon: Database },
  ];

  return (
    <div className="flex min-h-screen bg-[#F8FAFC] font-sans text-slate-900">
      <aside className="fixed flex h-full w-64 flex-col border-r border-slate-200 bg-white p-6">
        <div className="mb-12 flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-lg shadow-indigo-200">
            <BrainCircuit size={24} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">AnkReg</h1>
            <span className="-mt-1 block text-xs font-medium text-indigo-600">Invoice coding</span>
          </div>
        </div>

        <nav className="flex-1 space-y-1">
          {nav.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setActiveTab(item.id)}
              className={`flex w-full items-center justify-between rounded-xl p-3 transition-all ${
                activeTab === item.id
                  ? "bg-indigo-50 font-semibold text-indigo-700"
                  : "text-slate-500 hover:bg-slate-50"
              }`}
            >
              <span className="flex items-center gap-3">
                <item.icon size={20} />
                {item.label}
              </span>
              {item.badge && (
                <span className="rounded-full bg-rose-500 px-2 py-0.5 text-[10px] text-white">{item.badge}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="border-t border-slate-100 pt-6">
          <div className="flex items-center gap-3 rounded-xl bg-slate-50 p-4">
            <div className="flex size-8 items-center justify-center rounded-full bg-indigo-100 text-xs font-bold uppercase text-indigo-700">
              IT
            </div>
            <div>
              <p className="text-xs font-bold text-slate-900">Local session</p>
              <p className="text-[10px] font-medium text-slate-500">ankrag serve + Vite</p>
            </div>
          </div>
        </div>
      </aside>

      <main className="ml-64 flex-1 p-8">
        <header className="mb-8 flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight text-slate-900">
              {activeTab === "dashboard" && "Welcome"}
              {activeTab === "invoices" && "Invoice control center"}
              {activeTab === "knowledge" && "Knowledge base"}
              {activeTab === "analytics" && "Analytics"}
            </h1>
            <p className="mt-1 text-slate-500">
              {activeTab === "dashboard" && "Live counts from BigQuery and API health."}
              {activeTab === "invoices" && "Upload PDFs or review persisted RAG suggestions."}
              {activeTab === "knowledge" && "How embeddings and GL training data connect."}
              {activeTab === "analytics" && "Offline evaluation and reporting."}
            </p>
          </div>
          <div className="flex gap-4">
            <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2 shadow-sm">
              <div className="size-2 animate-pulse rounded-full bg-emerald-500" />
              <span className="text-sm font-medium text-slate-600">API</span>
            </div>
            <button
              type="button"
              className="rounded-xl bg-slate-900 p-2.5 text-white shadow-md transition-all hover:bg-slate-800"
              aria-label="Settings"
            >
              <Settings size={20} />
            </button>
          </div>
        </header>

        {activeTab === "dashboard" && <DashboardView />}
        {activeTab === "invoices" && <InvoicesView onOpenReview={setReview} />}
        {activeTab === "knowledge" && <KnowledgeView />}
        {activeTab === "analytics" && <AnalyticsPlaceholder />}

        <ReviewModal state={review} onClose={() => setReview(null)} />
      </main>
    </div>
  );
}
