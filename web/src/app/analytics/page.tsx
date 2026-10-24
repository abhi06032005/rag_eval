"use client";

import React from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Clock,
  Cpu,
  Database,
  Flame,
  HardDrive,
  Info,
  Server,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";

export default function AnalyticsPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 space-y-10">
      {/* Header */}
      <div className="border-b border-neutral-800 pb-6">
        <div className="flex items-center gap-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-2">
          <Activity className="h-4 w-4" />
          <span>Systems & Hardware Telemetry</span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          Latency Distributions & Resource Economics
        </h1>
        <p className="mt-2 text-sm text-neutral-400 max-w-3xl leading-relaxed">
          Comprehensive profiling of end-to-end pipeline latency, tail regression at the 99th percentile, Qdrant index RSS memory footprints, and vector compute savings.
        </p>
      </div>

      {/* Latency Percentiles Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Median p50 Card */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-3">
          <div className="flex items-center justify-between text-neutral-400">
            <span className="text-xs font-semibold uppercase tracking-wider">
              Median Response (p50)
            </span>
            <span className="flex items-center text-xs font-bold text-emerald-400">
              <ArrowDown className="h-4 w-4 mr-0.5" />
              -22.3% Faster
            </span>
          </div>
          <div className="flex items-baseline gap-3">
            <span className="text-4xl font-extrabold text-white font-mono">
              34.00
            </span>
            <span className="text-xs text-neutral-400 font-mono">
              ms (Adaptive Router)
            </span>
          </div>
          <div className="text-xs text-neutral-400 font-mono border-t border-neutral-800/80 pt-2">
            vs Fixed Hybrid: <span className="text-neutral-200">43.74 ms</span>
          </div>
          <p className="text-[11px] text-neutral-400 leading-relaxed font-sans">
            46% of queries follow the ultra-fast lexical fast path (~1.9 ms), reducing median latency substantially across the entire corpus.
          </p>
        </div>

        {/* 95th Percentile p95 Card */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-3">
          <div className="flex items-center justify-between text-neutral-400">
            <span className="text-xs font-semibold uppercase tracking-wider">
              95th Percentile (p95)
            </span>
            <span className="flex items-center text-xs font-bold text-emerald-400">
              <ArrowDown className="h-4 w-4 mr-0.5" />
              -3.9% Parity
            </span>
          </div>
          <div className="flex items-baseline gap-3">
            <span className="text-4xl font-extrabold text-white font-mono">
              69.50
            </span>
            <span className="text-xs text-neutral-400 font-mono">
              ms (Adaptive Router)
            </span>
          </div>
          <div className="text-xs text-neutral-400 font-mono border-t border-neutral-800/80 pt-2">
            vs Fixed Hybrid: <span className="text-neutral-200">72.31 ms</span>
          </div>
          <p className="text-[11px] text-neutral-400 leading-relaxed font-sans">
            Maintains competitive SLA boundaries up through the 95th percentile with negligible variation from full hybrid search.
          </p>
        </div>

        {/* 99th Percentile p99 Card (Tail Regression) */}
        <div className="rounded-xl border border-rose-900/60 bg-rose-950/20 p-6 space-y-3">
          <div className="flex items-center justify-between text-rose-300">
            <span className="text-xs font-semibold uppercase tracking-wider">
              Tail Latency (p99)
            </span>
            <span className="flex items-center text-xs font-bold text-rose-400">
              <ArrowUp className="h-4 w-4 mr-0.5" />
              +14.3% Inflation
            </span>
          </div>
          <div className="flex items-baseline gap-3">
            <span className="text-4xl font-extrabold text-rose-200 font-mono">
              101.23
            </span>
            <span className="text-xs text-rose-300 font-mono">
              ms (Adaptive Router)
            </span>
          </div>
          <div className="text-xs text-rose-300 font-mono border-t border-rose-800/60 pt-2">
            vs Fixed Hybrid: <span className="text-rose-100">88.55 ms</span>
          </div>
          <p className="text-[11px] text-rose-300/80 leading-relaxed font-sans">
            <strong>Documented Architectural Trade-off:</strong> Complex queries triggering fallback accumulate probe + feature extraction + full dense search.
          </p>
        </div>
      </div>

      {/* Forensic Tail Latency Breakdown Diagram */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-5">
        <div className="flex items-center justify-between border-b border-neutral-800 pb-4">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-400" />
              Forensic Analysis of p99 Tail Latency Regression (+14.3%)
            </h2>
            <p className="text-xs text-neutral-400 mt-0.5">
              Deconstructing why adaptive routing is slower on the 1% tail of difficult queries.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
          <div className="rounded-lg bg-neutral-950 border border-neutral-800 p-4 space-y-3 font-mono">
            <div className="font-sans font-bold text-white text-sm">
              Fixed Hybrid Pipeline (Predictable Overhead)
            </div>
            <div className="space-y-1.5 text-neutral-300">
              <div className="flex justify-between p-2 rounded bg-neutral-900/60">
                <span>1. Lexical BM25 Probe:</span>
                <span>~2.0 ms</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-neutral-900/60">
                <span>2. Dense BAAI/bge Embedding:</span>
                <span>~38.5 ms</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-neutral-900/60">
                <span>3. Qdrant HNSW Search (k=10):</span>
                <span>~3.2 ms</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-neutral-800 font-bold text-white">
                <span>Total Fixed Pipeline Latency:</span>
                <span>~43.7 ms (p99: 88.55 ms)</span>
              </div>
            </div>
          </div>

          <div className="rounded-lg bg-neutral-950 border border-rose-950 p-4 space-y-3 font-mono">
            <div className="font-sans font-bold text-rose-300 text-sm">
              Adaptive Router on Fallback Queries (Compound Overhead)
            </div>
            <div className="space-y-1.5 text-neutral-300">
              <div className="flex justify-between p-2 rounded bg-neutral-900/60 text-amber-300">
                <span>1. Initial Lexical Probe:</span>
                <span>~2.0 ms (Paid upfront)</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-neutral-900/60 text-amber-300">
                <span>2. Feature Extraction & Gating:</span>
                <span>~0.8 ms</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-neutral-900/60 text-rose-300">
                <span>3. Dense Embedding (Triggered):</span>
                <span>~38.5 ms</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-neutral-900/60 text-rose-300">
                <span>4. Qdrant HNSW + Fusion:</span>
                <span>~3.5 ms</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-rose-950/80 font-bold text-rose-200 border border-rose-800/80">
                <span>Total Fallback Latency:</span>
                <span>~44.8 ms (p99: 101.23 ms)</span>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-lg bg-neutral-950 border border-neutral-800 p-4 text-xs text-neutral-300 leading-relaxed font-sans">
          <strong>Production Takeaway:</strong> Adaptive routing provides substantial compute and median latency savings for the vast majority of queries, but systems with strict hard real-time latency deadlines (SLA &lt; 90 ms at p99) must account for the compound fallback penalty or execute dense retrieval speculatively.
        </div>
      </div>

      {/* Vector Database Memory Footprint Audit */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-5">
        <div className="flex items-center justify-between border-b border-neutral-800 pb-4">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <HardDrive className="h-4 w-4 text-emerald-400" />
              Qdrant Vector Index Quantization Audit (57,638 FiQA Embeddings)
            </h2>
            <p className="text-xs text-neutral-400 mt-0.5">
              Empirical telemetry measured on 384-dimensional dense embeddings stored on disk and in RAM across 57,638 passages.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto rounded-lg border border-neutral-800 bg-neutral-950 font-mono text-xs">
          <table className="w-full text-left text-neutral-300">
            <thead className="bg-neutral-900 text-[11px] uppercase tracking-wider text-neutral-400 border-b border-neutral-800">
              <tr>
                <th className="px-4 py-3">Quantization Mode</th>
                <th className="px-4 py-3">Vector Storage Format</th>
                <th className="px-4 py-3">Qdrant RSS RAM</th>
                <th className="px-4 py-3">Memory Reduction</th>
                <th className="px-4 py-3">Recall@10 Retention</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800">
              <tr className="hover:bg-neutral-900/50">
                <td className="px-4 py-3 font-sans font-semibold text-white">
                  Baseline (Uncompressed)
                </td>
                <td className="px-4 py-3">32-bit Float (FP32)</td>
                <td className="px-4 py-3">87.2 MB</td>
                <td className="px-4 py-3 text-neutral-500">1.0x (Reference)</td>
                <td className="px-4 py-3 text-emerald-400 font-bold">100.0%</td>
              </tr>
              <tr className="hover:bg-neutral-900/50">
                <td className="px-4 py-3 font-sans font-semibold text-indigo-300">
                  Scalar Quantization (SQ8)
                </td>
                <td className="px-4 py-3">8-bit Signed Int (INT8)</td>
                <td className="px-4 py-3 text-emerald-400 font-bold">29.4 MB</td>
                <td className="px-4 py-3 text-emerald-400 font-bold">3.0x Reduction (-66.3%)</td>
                <td className="px-4 py-3 text-emerald-400 font-bold">99.4% (Preserved)</td>
              </tr>
              <tr className="hover:bg-neutral-900/50">
                <td className="px-4 py-3 font-sans font-semibold text-amber-300">
                  Binary Quantization (BQ1)
                </td>
                <td className="px-4 py-3">1-bit Bitpacked Signs</td>
                <td className="px-4 py-3 text-emerald-400 font-bold">14.1 MB</td>
                <td className="px-4 py-3 text-emerald-400 font-bold">6.2x Reduction (-83.8%)</td>
                <td className="px-4 py-3 text-amber-400 font-semibold">91.8% (Minor Dip)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
