"use client";

import React, { useState } from "react";
import {
  ArrowDown,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Database,
  FileSearch,
  Filter,
  Flame,
  GitBranch,
  Layers,
  Play,
  RotateCcw,
  Scale,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";

const SAMPLE_ROUTING_QUERIES = [
  {
    query: "0-6-methylguanine-DNA methyltransferase promoter methylation glioblastoma",
    expectedRoute: "Lexical Fast Path",
    score_margin: 0.42,
    max_idf: 7.8,
    reason: "Distinctive biomedical terminology with high maximum IDF and strong top-1 score separation.",
  },
  {
    query: "Does exercise improve health in sick people?",
    expectedRoute: "Fallback to Fixed Hybrid",
    score_margin: 0.08,
    max_idf: 3.2,
    reason: "Generic keywords with low IDF and minimal score margin; requires dense semantic expansion to prevent recall collapse.",
  },
  {
    query: "8-bit scalar quantization vector memory reduction",
    expectedRoute: "Lexical Fast Path",
    score_margin: 0.35,
    max_idf: 6.4,
    reason: "Exact technical keywords present in arXiv title and abstract; lexical probe sufficient.",
  },
  {
    query: "financial investment strategies for retirement",
    expectedRoute: "Fallback to Fixed Hybrid",
    score_margin: 0.12,
    max_idf: 3.8,
    reason: "Ambiguous query matching thousands of general finance documents; lexical probe is uncertain.",
  },
];

export default function RoutingExplainerPage() {
  const [testQuery, setTestQuery] = useState(SAMPLE_ROUTING_QUERIES[0].query);
  const [activeThreshold, setActiveThreshold] = useState<{
    margin: number;
    idf: number;
    dataset: string;
  }>({
    margin: 0.20,
    idf: 5.0,
    dataset: "SciFact",
  });

  // Calculate simulated routing features
  const words = testQuery.toLowerCase().trim().split(/\s+/).filter(Boolean);
  // Heuristic mock IDF: longer/uncommon words get higher IDF
  const maxWordLen = Math.max(...words.map((w) => w.length), 0);
  const simulatedMaxIdf = Math.min(9.5, Math.max(2.0, maxWordLen * 0.75 + (words.length > 5 ? 1.0 : 0.0)));
  const hasSpecificTerms = words.some((w) =>
    ["quantization", "methyltransferase", "glioblastoma", "interleukin", "translocation", "hnsw", "reciprocal"].includes(w)
  );
  const simulatedMargin = hasSpecificTerms ? 0.38 : (simulatedMaxIdf > 5.5 ? 0.26 : 0.11);

  const passesMargin = simulatedMargin >= activeThreshold.margin;
  const passesIdf = simulatedMaxIdf >= activeThreshold.idf;
  const decisionIsLexical = passesMargin && passesIdf;

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 space-y-10">
      {/* Header */}
      <div className="border-b border-neutral-800 pb-6">
        <div className="flex items-center gap-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-2">
          <FileSearch className="h-4 w-4" />
          <span>Algorithmic Methodology</span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          Query-Adaptive Quality-Aware Routing
        </h1>
        <p className="mt-2 text-sm text-neutral-400 max-w-3xl leading-relaxed">
          How RAGScaleBench eliminates up to 46% of dense vector search invocations without sacrificing retrieval accuracy via calibrated two-stage lexical gating.
        </p>
      </div>

      {/* Decision Architecture Flowchart */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-6">
        <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-indigo-400" />
          Two-Stage Gating Architecture
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          {/* Stage 1 */}
          <div className="rounded-lg bg-neutral-950 border border-neutral-800 p-4 space-y-2">
            <div className="flex items-center justify-between text-indigo-400 font-bold uppercase text-[10px] tracking-wider">
              <span>Stage 1: Lexical Probe</span>
              <span className="font-mono">~1.9 ms</span>
            </div>
            <h3 className="font-semibold text-white text-sm">
              Fast BM25 Keyword Search
            </h3>
            <p className="text-neutral-400 text-xs leading-relaxed">
              Executes inverted index search across the corpus to retrieve candidate scores and calculate term statistics.
            </p>
            <div className="pt-2 border-t border-neutral-900 text-neutral-300 font-mono text-[11px]">
              Extract: Δ_score = (s₁ - s₂)<br />
              Extract: max(IDF(qᵢ))
            </div>
          </div>

          {/* Gating Decision */}
          <div className="rounded-lg bg-neutral-950 border border-neutral-800 p-4 space-y-2">
            <div className="flex items-center justify-between text-emerald-400 font-bold uppercase text-[10px] tracking-wider">
              <span>Gating Condition</span>
              <span className="font-mono">&lt; 0.1 ms</span>
            </div>
            <h3 className="font-semibold text-white text-sm">
              Threshold Comparison
            </h3>
            <p className="text-neutral-400 text-xs leading-relaxed">
              Evaluates query discriminability against calibrated out-of-sample thresholds:
            </p>
            <div className="bg-neutral-900 p-2 rounded border border-neutral-800 text-[11px] font-mono text-indigo-300">
              IF (Δ_score ≥ τ_margin) AND (max(IDF) ≥ τ_idf):<br />
              &nbsp;&nbsp;ROUTE TO FAST PATH<br />
              ELSE:<br />
              &nbsp;&nbsp;FALLBACK TO HYBRID
            </div>
          </div>

          {/* Outcomes */}
          <div className="rounded-lg bg-neutral-950 border border-neutral-800 p-4 space-y-3">
            <div className="flex items-center justify-between text-amber-400 font-bold uppercase text-[10px] tracking-wider">
              <span>Outcome Pathways</span>
            </div>
            <div className="rounded bg-emerald-950/40 border border-emerald-800/60 p-2 text-[11px]">
              <span className="font-bold text-emerald-300 block">Fast Path (46% of queries)</span>
              <span className="text-neutral-300">Return BM25 top-k. Skip neural embedding and Qdrant search entirely.</span>
            </div>
            <div className="rounded bg-indigo-950/40 border border-indigo-800/60 p-2 text-[11px]">
              <span className="font-bold text-indigo-300 block">Quality Fallback (54% of queries)</span>
              <span className="text-neutral-300">Execute dense search + Reciprocal Rank Fusion (k=60). Protect recall.</span>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Routing Simulator */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-6">
        <div className="flex items-center justify-between border-b border-neutral-800 pb-4">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-emerald-400" />
              Live Query Routing Simulator
            </h2>
            <p className="text-xs text-neutral-400 mt-0.5">
              Type any query or select an example below to simulate lexical feature extraction and threshold gating.
            </p>
          </div>

          {/* Dataset Threshold Switcher */}
          <div className="flex items-center gap-1 text-xs">
            <span className="text-neutral-400 mr-1 text-[11px]">Calibration Set:</span>
            {[
              { dataset: "SciFact", margin: 0.20, idf: 5.0 },
              { dataset: "NFCorpus", margin: 0.25, idf: 6.0 },
              { dataset: "FiQA", margin: 0.20, idf: 4.0 },
            ].map((cfg) => (
              <button
                key={cfg.dataset}
                onClick={() => setActiveThreshold(cfg)}
                className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
                  activeThreshold.dataset === cfg.dataset
                    ? "bg-indigo-600 text-white font-bold"
                    : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"
                }`}
              >
                {cfg.dataset}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-xs font-semibold text-neutral-300 uppercase tracking-wider block mb-1.5">
              Input Query for Feature Extraction:
            </label>
            <input
              type="text"
              value={testQuery}
              onChange={(e) => setTestQuery(e.target.value)}
              className="w-full rounded-lg border border-neutral-800 bg-neutral-950 px-3.5 py-2.5 text-sm text-white placeholder-neutral-500 focus:border-indigo-500 focus:outline-none font-mono"
            />
          </div>

          {/* Quick Select Buttons */}
          <div className="flex flex-wrap gap-2">
            {SAMPLE_ROUTING_QUERIES.map((sq, idx) => (
              <button
                key={idx}
                onClick={() => setTestQuery(sq.query)}
                className="text-left rounded border border-neutral-800 bg-neutral-950 px-2.5 py-1.5 text-xs text-neutral-300 hover:border-neutral-700 hover:bg-neutral-900 transition-colors"
              >
                {sq.expectedRoute.includes("Fast") ? "⚡" : "🔄"} {sq.query.slice(0, 45)}...
              </button>
            ))}
          </div>

          {/* Extracted Features & Decision Card */}
          <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-5 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
            <div className="rounded bg-neutral-900 p-3 border border-neutral-800">
              <span className="text-[10px] uppercase text-neutral-500 block">
                Top-1 vs Top-2 Score Margin
              </span>
              <div className="flex items-baseline justify-between mt-1">
                <span className="text-xl font-bold text-white">
                  Δ = {simulatedMargin.toFixed(2)}
                </span>
                <span
                  className={`text-[10px] font-sans font-bold px-1.5 py-0.5 rounded ${
                    passesMargin
                      ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                      : "bg-rose-950 text-rose-400 border border-rose-800"
                  }`}
                >
                  {passesMargin ? `≥ ${activeThreshold.margin}` : `< ${activeThreshold.margin}`}
                </span>
              </div>
            </div>

            <div className="rounded bg-neutral-900 p-3 border border-neutral-800">
              <span className="text-[10px] uppercase text-neutral-500 block">
                Maximum Term IDF
              </span>
              <div className="flex items-baseline justify-between mt-1">
                <span className="text-xl font-bold text-white">
                  IDF = {simulatedMaxIdf.toFixed(2)}
                </span>
                <span
                  className={`text-[10px] font-sans font-bold px-1.5 py-0.5 rounded ${
                    passesIdf
                      ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                      : "bg-rose-950 text-rose-400 border border-rose-800"
                  }`}
                >
                  {passesIdf ? `≥ ${activeThreshold.idf}` : `< ${activeThreshold.idf}`}
                </span>
              </div>
            </div>

            <div
              className={`rounded p-3 border flex flex-col justify-center ${
                decisionIsLexical
                  ? "bg-emerald-950/40 border-emerald-800/80 text-emerald-300"
                  : "bg-indigo-950/40 border-indigo-800/80 text-indigo-300"
              }`}
            >
              <span className="text-[10px] uppercase opacity-75 font-sans font-semibold">
                Router Decision
              </span>
              <span className="text-sm font-bold font-sans mt-0.5">
                {decisionIsLexical
                  ? "⚡ Route to Lexical Fast Path"
                  : "🔄 Fallback to Fixed Hybrid"}
              </span>
              <span className="text-[10px] opacity-75 mt-0.5">
                {decisionIsLexical
                  ? "Saves ~38 ms of dense embedding & search"
                  : "Preserves recall on ambiguous query"}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Ablation Study Table */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Scale className="h-4 w-4 text-indigo-400" />
              Ablation Study: Why Both Score Margin and IDF Are Essential
            </h2>
            <p className="text-xs text-neutral-400 mt-0.5">
              Empirical sweep testing individual feature removal on SciFact test set (300 queries).
            </p>
          </div>
        </div>

        <div className="overflow-x-auto rounded-lg border border-neutral-800 bg-neutral-950 font-mono text-xs">
          <table className="w-full text-left text-neutral-300">
            <thead className="bg-neutral-900 text-[11px] uppercase tracking-wider text-neutral-400 border-b border-neutral-800">
              <tr>
                <th className="px-4 py-3">Router Feature Configuration</th>
                <th className="px-4 py-3">nDCG@10</th>
                <th className="px-4 py-3">Δ vs Full Router</th>
                <th className="px-4 py-3">Dense Calls Saved</th>
                <th className="px-4 py-3">Audit Observation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800">
              <tr className="hover:bg-neutral-900/40 bg-indigo-950/20">
                <td className="px-4 py-3 font-sans font-semibold text-white">
                  Full Adaptive Router (Margin + Max IDF)
                </td>
                <td className="px-4 py-3 font-bold text-emerald-400">0.6861</td>
                <td className="px-4 py-3 text-neutral-400">0.0000 (Reference)</td>
                <td className="px-4 py-3 text-emerald-400 font-bold">46.0% (138 queries)</td>
                <td className="px-4 py-3 font-sans text-[11px] text-emerald-300">
                  Optimal trade-off: non-inferior to hybrid.
                </td>
              </tr>
              <tr className="hover:bg-neutral-900/40">
                <td className="px-4 py-3 font-sans font-semibold text-neutral-200">
                  Ablation A: Score Margin Only (No IDF)
                </td>
                <td className="px-4 py-3 text-rose-300 font-bold">0.6770</td>
                <td className="px-4 py-3 text-rose-400 font-bold">-0.0091</td>
                <td className="px-4 py-3 text-neutral-300">51.3% (154 queries)</td>
                <td className="px-4 py-3 font-sans text-[11px] text-neutral-400">
                  Common stopwords trigger false score separation.
                </td>
              </tr>
              <tr className="hover:bg-neutral-900/40">
                <td className="px-4 py-3 font-sans font-semibold text-neutral-200">
                  Ablation B: Max IDF Only (No Score Margin)
                </td>
                <td className="px-4 py-3 text-rose-300 font-bold">0.6719</td>
                <td className="px-4 py-3 text-rose-400 font-bold">-0.0142</td>
                <td className="px-4 py-3 text-neutral-300">42.0% (126 queries)</td>
                <td className="px-4 py-3 font-sans text-[11px] text-neutral-400">
                  Rare words in irrelevant context pass gating.
                </td>
              </tr>
              <tr className="hover:bg-neutral-900/40">
                <td className="px-4 py-3 font-sans font-semibold text-neutral-200">
                  Fixed Hybrid (Always Both)
                </td>
                <td className="px-4 py-3 text-neutral-200">0.6851</td>
                <td className="px-4 py-3 text-neutral-400">-0.0010</td>
                <td className="px-4 py-3 text-rose-400">0.0% (0 queries saved)</td>
                <td className="px-4 py-3 font-sans text-[11px] text-neutral-400">
                  Full dense search executed on 100% of queries.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
