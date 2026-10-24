"use client";

import React, { useState } from "react";
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  CheckCircle2,
  Database,
  FileCheck2,
  Flame,
  GitBranch,
  HelpCircle,
  Info,
  Scale,
  ShieldAlert,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";

const RETRIEVAL_POLICIES = [
  {
    dataset: "SciFact (Biomedical Claims)",
    test_queries: 300,
    rows: [
      {
        policy: "Lexical BM25",
        ndcg10: 0.6653,
        recall10: 0.7711,
        latency_ms: 1.9,
        dense_calls: 0,
        status: "Baseline Lexical",
      },
      {
        policy: "Dense Exact (Brute Force)",
        ndcg10: 0.6406,
        recall10: 0.7544,
        latency_ms: 4.8,
        dense_calls: 300,
        status: "Baseline Dense",
      },
      {
        policy: "Fixed Hybrid (RRF k=60)",
        ndcg10: 0.6851,
        recall10: 0.7956,
        latency_ms: 43.7,
        dense_calls: 300,
        status: "Target Quality Baseline",
      },
      {
        policy: "Query-Adaptive Quality-Aware Router",
        ndcg10: 0.6861,
        recall10: 0.7967,
        latency_ms: 34.0,
        dense_calls: 162,
        status: "Non-Inferior & 46.0% Savings",
        highlight: true,
      },
    ],
    bootstrap: {
      comparison: "Adaptive Router vs Fixed Hybrid",
      delta_ndcg: "+0.0010",
      ci_95: "[-0.0175, +0.0194]",
      p_value: "0.9160 (Not statistically different)",
      tie_rate: "90.0% (270 / 300 identical)",
      non_inferiority: "Passes exploratory margin δ = 0.025 (lower bound -0.0175 > -0.025)",
    },
  },
  {
    dataset: "NFCorpus (Clinical Medicine)",
    test_queries: 323,
    rows: [
      {
        policy: "Lexical BM25",
        ndcg10: 0.3218,
        recall10: 0.2415,
        latency_ms: 2.1,
        dense_calls: 0,
        status: "Baseline Lexical",
      },
      {
        policy: "Dense Exact (Brute Force)",
        ndcg10: 0.3129,
        recall10: 0.2301,
        latency_ms: 4.5,
        dense_calls: 323,
        status: "Baseline Dense",
      },
      {
        policy: "Fixed Hybrid (RRF k=60)",
        ndcg10: 0.3448,
        recall10: 0.2612,
        latency_ms: 44.1,
        dense_calls: 323,
        status: "Highest Quality",
        highlight: true,
      },
      {
        policy: "Query-Adaptive Quality-Aware Router",
        ndcg10: 0.3340,
        recall10: 0.2504,
        latency_ms: 38.2,
        dense_calls: 263,
        status: "18.6% Savings, Slight Recall Dip",
      },
    ],
    bootstrap: {
      comparison: "Adaptive Router vs Fixed Hybrid",
      delta_ndcg: "-0.0108",
      ci_95: "[-0.0221, -0.0004]",
      p_value: "0.0432 (Statistically lower)",
      tie_rate: "83.6% (270 / 323 identical)",
      non_inferiority: "Inconclusive (lower bound -0.0221 touches boundary of δ = 0.025)",
    },
  },
  {
    dataset: "FiQA-2018 (Financial QA)",
    test_queries: 648,
    rows: [
      {
        policy: "Lexical BM25",
        ndcg10: 0.2361,
        recall10: 0.3204,
        latency_ms: 121.8,
        dense_calls: 0,
        status: "Severe Lexical Dilution",
      },
      {
        policy: "Dense Exact (Brute Force)",
        ndcg10: 0.3532,
        recall10: 0.4411,
        latency_ms: 5.3,
        dense_calls: 648,
        status: "Optimal Quality & Fastest (5.3 ms)",
        highlight: true,
      },
      {
        policy: "Fixed Hybrid (RRF k=60)",
        ndcg10: 0.3300,
        recall10: 0.4112,
        latency_ms: 128.3,
        dense_calls: 648,
        status: "Degraded by -0.0232 nDCG (p=0.0296)",
      },
      {
        policy: "Query-Adaptive Quality-Aware Router",
        ndcg10: 0.3216,
        recall10: 0.4006,
        latency_ms: 127.2,
        dense_calls: 582,
        status: "Degraded by -0.0316 nDCG (p=0.0034)",
      },
    ],
    bootstrap: {
      comparison: "Dense Exact vs Adaptive Router",
      delta_ndcg: "+0.0316 (Dense beats Adaptive)",
      ci_95: "[+0.0104, +0.0531]",
      p_value: "0.0034 (Statistically superior)",
      tie_rate: "N/A (Dense-Only optimal)",
      non_inferiority: "Violated: Hybrid fusion should NOT be applied on this domain",
    },
  },
];

export default function RetrievalPage() {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const activeDataset = RETRIEVAL_POLICIES[selectedIdx];

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 space-y-10">
      {/* Header */}
      <div className="border-b border-neutral-800 pb-6">
        <div className="flex items-center gap-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-2">
          <GitBranch className="h-4 w-4" />
          <span>Cross-Domain Retrieval Evaluation</span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          Retrieval Configuration Benchmarks & Statistical Significance
        </h1>
        <p className="mt-2 text-sm text-neutral-400 max-w-3xl leading-relaxed">
          Comparing Lexical BM25, Dense Exact, Fixed Reciprocal Rank Fusion, and Query-Adaptive Quality-Aware Routing. Rigorously audits statistical confidence intervals, non-inferiority margins, and domain limitations.
        </p>
      </div>

      {/* Exploratory Non-Inferiority Disclosure Card */}
      <div className="rounded-xl border border-amber-900/60 bg-amber-950/20 p-5 backdrop-blur-sm space-y-2">
        <div className="flex items-center gap-2 text-amber-300 font-bold text-sm">
          <Scale className="h-4 w-4 text-amber-400" />
          <span>Priority 1 Audit: Non-Inferiority Margin (δ = 0.025) Reclassified as Exploratory</span>
        </div>
        <p className="text-xs text-neutral-300 leading-relaxed">
          The 0.025 nDCG@10 non-inferiority margin was chosen retrospectively to match acceptable empirical variance rather than pre-registered in a formal confirmatory study protocol. In accordance with statistical integrity principles, this analysis is explicitly labelled <strong>exploratory post-hoc non-inferiority testing</strong>. While the Adaptive Router achieves 0.6861 nDCG@10 (vs 0.6851 Fixed Hybrid, 95% CI [-0.0175, +0.0194]) on SciFact, claim generalizing to all corpora is empirically refuted by FiQA.
        </p>
      </div>

      {/* Dataset Selector Tabs */}
      <div className="flex flex-wrap gap-2 border-b border-neutral-800 pb-2">
        {RETRIEVAL_POLICIES.map((ds, idx) => (
          <button
            key={idx}
            onClick={() => setSelectedIdx(idx)}
            className={`rounded-lg px-4 py-2 text-xs font-semibold transition-colors flex items-center gap-2 ${
              selectedIdx === idx
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "bg-neutral-900 text-neutral-400 hover:bg-neutral-800 hover:text-white"
            }`}
          >
            <span>{ds.dataset}</span>
            <span className="text-[10px] opacity-75 font-mono">
              ({ds.test_queries} test queries)
            </span>
          </button>
        ))}
      </div>

      {/* Results Table */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight">
              {activeDataset.dataset} — Policy Performance Matrix
            </h2>
            <p className="text-xs text-neutral-400 mt-0.5">
              Evaluated on {activeDataset.test_queries} locked test queries with zero train/dev query overlap.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto rounded-lg border border-neutral-800 bg-neutral-950">
          <table className="w-full text-left text-xs text-neutral-300">
            <thead className="bg-neutral-900/80 text-[11px] uppercase tracking-wider text-neutral-400 border-b border-neutral-800 font-mono">
              <tr>
                <th className="px-4 py-3">Retrieval Policy</th>
                <th className="px-4 py-3">nDCG@10</th>
                <th className="px-4 py-3">Recall@10</th>
                <th className="px-4 py-3">p50 Latency</th>
                <th className="px-4 py-3">Dense Search Calls</th>
                <th className="px-4 py-3">Audit Operational Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800 font-mono text-xs">
              {activeDataset.rows.map((row, rIdx) => (
                <tr
                  key={rIdx}
                  className={`hover:bg-neutral-800/40 transition-colors ${
                    row.highlight ? "bg-indigo-950/20" : ""
                  }`}
                >
                  <td className="px-4 py-3.5 font-sans font-semibold text-white flex items-center gap-2">
                    {row.highlight && (
                      <span className="h-2 w-2 rounded-full bg-emerald-400" />
                    )}
                    {row.policy}
                  </td>
                  <td className="px-4 py-3.5 font-bold text-emerald-400 text-sm">
                    {row.ndcg10.toFixed(4)}
                  </td>
                  <td className="px-4 py-3.5 text-neutral-200">
                    {row.recall10.toFixed(4)}
                  </td>
                  <td className="px-4 py-3.5 text-neutral-300">
                    {row.latency_ms.toFixed(1)} ms
                  </td>
                  <td className="px-4 py-3.5 text-indigo-300">
                    {row.dense_calls} / {activeDataset.test_queries}
                  </td>
                  <td className="px-4 py-3.5 font-sans text-neutral-300 text-[11px]">
                    <span
                      className={`inline-block rounded px-2 py-0.5 font-medium ${
                        row.status.includes("Optimal") || row.status.includes("Non-Inferior")
                          ? "bg-emerald-950/70 border border-emerald-800/60 text-emerald-300"
                          : row.status.includes("Degraded")
                          ? "bg-rose-950/70 border border-rose-800/60 text-rose-300"
                          : "bg-neutral-800 text-neutral-300"
                      }`}
                    >
                      {row.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Paired Bootstrap Card */}
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-5 space-y-3">
          <h3 className="text-xs font-semibold uppercase text-neutral-300 tracking-wider flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
            Paired Bootstrap Significance Testing (B = 10,000 Resamples)
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-mono">
            <div className="rounded bg-neutral-900/60 border border-neutral-800/80 p-3">
              <span className="text-[10px] uppercase text-neutral-500 block">
                Mean Difference
              </span>
              <span className="text-sm font-bold text-white mt-1 block">
                {activeDataset.bootstrap.delta_ndcg}
              </span>
            </div>
            <div className="rounded bg-neutral-900/60 border border-neutral-800/80 p-3">
              <span className="text-[10px] uppercase text-neutral-500 block">
                95% Confidence Interval
              </span>
              <span className="text-sm font-bold text-indigo-300 mt-1 block">
                {activeDataset.bootstrap.ci_95}
              </span>
            </div>
            <div className="rounded bg-neutral-900/60 border border-neutral-800/80 p-3">
              <span className="text-[10px] uppercase text-neutral-500 block">
                Paired p-value
              </span>
              <span className="text-sm font-bold text-emerald-400 mt-1 block">
                {activeDataset.bootstrap.p_value}
              </span>
            </div>
            <div className="rounded bg-neutral-900/60 border border-neutral-800/80 p-3">
              <span className="text-[10px] uppercase text-neutral-500 block">
                Decision Tie Rate
              </span>
              <span className="text-sm font-bold text-neutral-300 mt-1 block">
                {activeDataset.bootstrap.tie_rate}
              </span>
            </div>
          </div>
          <p className="text-[11px] text-neutral-400 font-sans leading-relaxed">
            <strong>Statistical Conclusion:</strong> {activeDataset.bootstrap.non_inferiority}.
          </p>
        </div>
      </div>

      {/* Forensic Deep-Dive: FiQA Dense Superiority Case */}
      {selectedIdx === 2 && (
        <div className="rounded-xl border border-rose-900/60 bg-rose-950/20 p-6 space-y-4">
          <div className="flex items-center gap-2 text-rose-300 font-bold text-base">
            <ShieldAlert className="h-5 w-5 text-rose-400" />
            <span>Forensic Analysis: Why Dense-Only Retrieval Dominates FiQA</span>
          </div>
          <div className="space-y-2 text-xs text-neutral-300 leading-relaxed font-sans">
            <p>
              On the FiQA-2018 financial corpus (57,638 documents), <strong>Dense Exact retrieval achieves 0.3532 nDCG@10</strong> in <strong>5.3 ms</strong>. In contrast, Fixed Hybrid achieves only <strong>0.3300 nDCG@10</strong> (-0.0232 degradation, paired bootstrap p=0.0296) and takes <strong>128.3 ms</strong>.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3 font-mono text-xs">
              <div className="rounded bg-neutral-950 p-3 border border-neutral-800">
                <span className="text-neutral-400 block font-sans font-semibold mb-1">
                  Cause 1: Semantic Dilution via Short Financial Postings
                </span>
                <span className="text-neutral-300">
                  Financial forum queries (e.g. "is it legal to invest in pre-IPO stock?") share common terms with thousands of unrelated stock posts. Pure Python BM25 scores dilute dense top-10 precision when fused via RRF.
                </span>
              </div>
              <div className="rounded bg-neutral-950 p-3 border border-neutral-800">
                <span className="text-neutral-400 block font-sans font-semibold mb-1">
                  Cause 2: High Lexical Probe Overhead on 57k Documents
                </span>
                <span className="text-neutral-300">
                  Executing BM25 scanning over 57,638 documents in pure Python incurs 121.8 ms overhead, whereas Qdrant dense vector search completes in only 5.3 ms (a 24.2x speed advantage for dense).
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
