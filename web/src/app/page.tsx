import React from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Database,
  FileCheck2,
  FileText,
  Flame,
  GitBranch,
  Layers,
  MessageSquare,
  ShieldAlert,
  ShieldCheck,
  TrendingUp,
  Zap,
} from "lucide-react";

export default function HomePage() {
  return (
    <div className="relative overflow-hidden">
      {/* Background radial glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-gradient-to-b from-indigo-900/20 via-emerald-950/10 to-transparent blur-3xl pointer-events-none -z-10" />

      {/* Hero Section */}
      <section className="mx-auto max-w-7xl px-4 pt-16 pb-12 sm:px-6 lg:pt-24 lg:pb-16 text-center">
        <div className="inline-flex items-center gap-2 rounded-full border border-indigo-500/30 bg-indigo-950/40 px-3 py-1 text-xs font-medium text-indigo-300 mb-6 backdrop-blur-sm">
          <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
          <span>Priority 1 Scientific Audit Complete • Zero Train/Test Leakage</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white max-w-4xl mx-auto leading-tight">
          Scientific RAG Benchmarking &{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-teal-300 to-emerald-400">
            Query-Adaptive Retrieval
          </span>
        </h1>

        <p className="mt-6 text-base sm:text-lg text-neutral-300 max-w-2xl mx-auto leading-relaxed">
          An empirical systems platform evaluating lexical, dense HNSW, fixed reciprocal rank fusion, and calibrated query-adaptive routing across scientific, biomedical, and financial corpora with real LLM answer generation.
        </p>

        {/* CTA Buttons */}
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/playground"
            className="flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-600/30 hover:bg-indigo-500 transition-colors"
          >
            <MessageSquare className="h-4 w-4" />
            <span>Launch QA Playground</span>
            <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            href="/retrieval"
            className="flex items-center gap-2 rounded-lg border border-neutral-700 bg-neutral-900 px-5 py-2.5 text-sm font-semibold text-neutral-200 hover:bg-neutral-800 transition-colors"
          >
            <GitBranch className="h-4 w-4" />
            <span>View Benchmark Matrix</span>
          </Link>
          <Link
            href="/reproducibility"
            className="flex items-center gap-2 rounded-lg border border-neutral-800 bg-neutral-950 px-5 py-2.5 text-sm font-semibold text-neutral-400 hover:text-white hover:bg-neutral-900 transition-colors"
          >
            <FileCheck2 className="h-4 w-4" />
            <span>Integrity Audit Report</span>
          </Link>
        </div>
      </section>

      {/* Metric Highlight Spotlight Cards */}
      <section className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 backdrop-blur-sm hover:border-neutral-700 transition-colors">
            <div className="flex items-center justify-between text-neutral-400 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">SciFact Retrieval</span>
              <TrendingUp className="h-4 w-4 text-emerald-400" />
            </div>
            <div className="text-3xl font-extrabold text-white">0.6861</div>
            <div className="text-xs text-emerald-400 font-medium mt-1">
              nDCG@10 (Adaptive Router)
            </div>
            <p className="text-[11px] text-neutral-400 mt-2">
              Statistically non-inferior to Fixed Hybrid (0.6851, p=0.916) while eliminating 46.0% of dense search calls.
            </p>
          </div>

          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 backdrop-blur-sm hover:border-neutral-700 transition-colors">
            <div className="flex items-center justify-between text-neutral-400 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">Dense Superiority (FiQA)</span>
              <ShieldAlert className="h-4 w-4 text-amber-400" />
            </div>
            <div className="text-3xl font-extrabold text-white">0.3532</div>
            <div className="text-xs text-amber-400 font-medium mt-1">
              nDCG@10 (Dense Exact in 5.3 ms)
            </div>
            <p className="text-[11px] text-neutral-400 mt-2">
              Explicitly reported: Dense beats Hybrid (0.3300) and Adaptive (0.3216, p=0.0034) by 24x lower latency.
            </p>
          </div>

          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 backdrop-blur-sm hover:border-neutral-700 transition-colors">
            <div className="flex items-center justify-between text-neutral-400 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">Median Latency</span>
              <Zap className="h-4 w-4 text-indigo-400" />
            </div>
            <div className="text-3xl font-extrabold text-white">-22.3%</div>
            <div className="text-xs text-indigo-400 font-medium mt-1">
              p50 Latency (34.00 vs 43.74 ms)
            </div>
            <p className="text-[11px] text-neutral-400 mt-2">
              Lexical fast path executes in 1.9 ms, saving median response time across 138 evaluated queries.
            </p>
          </div>

          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 backdrop-blur-sm hover:border-neutral-700 transition-colors">
            <div className="flex items-center justify-between text-neutral-400 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">Tail Regression Disclosed</span>
              <Activity className="h-4 w-4 text-rose-400" />
            </div>
            <div className="text-3xl font-extrabold text-white">+14.3%</div>
            <div className="text-xs text-rose-400 font-medium mt-1">
              p99 Latency (101.23 vs 88.55 ms)
            </div>
            <p className="text-[11px] text-neutral-400 mt-2">
              Queries triggering fallback pay lexical probe + routing feature extraction + full dense search.
            </p>
          </div>
        </div>
      </section>

      {/* Architecture & Core Principles */}
      <section className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
        <div className="rounded-2xl border border-neutral-800 bg-neutral-900/40 p-8">
          <div className="max-w-2xl mb-8">
            <h2 className="text-2xl font-bold text-white tracking-tight">
              Rigorous Scientific Principles & Systems Engineering
            </h2>
            <p className="text-sm text-neutral-400 mt-2">
              Built to address pervasive methodological flaws in contemporary RAG literature: data leakage, post-hoc metric cherry-picking, and silent mock fallbacks.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                <CheckCircle2 className="h-5 w-5" />
              </div>
              <h3 className="font-semibold text-white text-base">
                Zero-Leakage Calibrations
              </h3>
              <p className="text-xs text-neutral-400 leading-relaxed">
                All routing thresholds (<code className="text-indigo-300 font-mono">τ_margin</code>, <code className="text-indigo-300 font-mono">τ_idf</code>) are tuned strictly on training/dev splits. Test splits have 0 query overlap (SciFact: 809 train / 300 test; NFCorpus: 324 dev / 323 test; FiQA: 500 dev / 648 test).
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <ShieldCheck className="h-5 w-5" />
              </div>
              <h3 className="font-semibold text-white text-base">
                No Silent LLM Fallback
              </h3>
              <p className="text-xs text-neutral-400 leading-relaxed">
                Real generative runtime mode never fakes LLM answers via regex or heuristic text extraction. When local Ollama or hosted OpenAI endpoints are offline, the service explicitly returns HTTP 503 errors and logs the condition.
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
                <Cpu className="h-5 w-5" />
              </div>
              <h3 className="font-semibold text-white text-base">
                Forensic Failure Reporting
              </h3>
              <p className="text-xs text-neutral-400 leading-relaxed">
                Negative findings are emphasized rather than hidden: lexical BM25 evaluation over FiQA's 57,638 financial documents dilutes dense semantic signals, proving that hybrid retrieval is domain-dependent rather than globally superior.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Cross-Domain Benchmark Summary Table */}
      <section className="mx-auto max-w-7xl px-4 py-8 sm:px-6 mb-16">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              Verified Cross-Domain Performance Matrix
            </h2>
            <p className="text-xs text-neutral-400">
              Evaluated using BAAI/bge-small-en-v1.5 and official BEIR qrels (test sets).
            </p>
          </div>
          <Link
            href="/retrieval"
            className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
          >
            <span>Full Statistical Comparisons</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>

        <div className="overflow-x-auto rounded-xl border border-neutral-800 bg-neutral-900/50">
          <table className="w-full text-left text-xs text-neutral-300">
            <thead className="bg-neutral-950/80 text-[11px] uppercase tracking-wider text-neutral-400 border-b border-neutral-800">
              <tr>
                <th className="px-4 py-3">Corpus / Domain</th>
                <th className="px-4 py-3">Corpus Size</th>
                <th className="px-4 py-3">Test Queries</th>
                <th className="px-4 py-3">BM25 nDCG@10</th>
                <th className="px-4 py-3">Dense Exact nDCG@10</th>
                <th className="px-4 py-3">Fixed Hybrid nDCG@10</th>
                <th className="px-4 py-3">Adaptive Router nDCG@10</th>
                <th className="px-4 py-3">Dense Invocations Saved</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800/60 font-mono text-xs">
              <tr className="hover:bg-neutral-800/30 transition-colors">
                <td className="px-4 py-3 font-sans font-semibold text-white">
                  SciFact (Biomedical Claims)
                </td>
                <td className="px-4 py-3">5,183 docs</td>
                <td className="px-4 py-3">300</td>
                <td className="px-4 py-3">0.6653</td>
                <td className="px-4 py-3">0.6406</td>
                <td className="px-4 py-3 text-neutral-200">0.6851</td>
                <td className="px-4 py-3 font-bold text-emerald-400">0.6861</td>
                <td className="px-4 py-3 text-emerald-400 font-bold">46.0% (138 saved)</td>
              </tr>
              <tr className="hover:bg-neutral-800/30 transition-colors">
                <td className="px-4 py-3 font-sans font-semibold text-white">
                  NFCorpus (Clinical Nutrition)
                </td>
                <td className="px-4 py-3">3,633 docs</td>
                <td className="px-4 py-3">323</td>
                <td className="px-4 py-3">0.3218</td>
                <td className="px-4 py-3">0.3129</td>
                <td className="px-4 py-3 font-bold text-emerald-400">0.3448</td>
                <td className="px-4 py-3 text-neutral-200">0.3340</td>
                <td className="px-4 py-3 text-indigo-400 font-semibold">18.6% (60 saved)</td>
              </tr>
              <tr className="hover:bg-neutral-800/30 transition-colors bg-amber-950/10">
                <td className="px-4 py-3 font-sans font-semibold text-amber-300">
                  FiQA (Financial QA)
                </td>
                <td className="px-4 py-3">57,638 docs</td>
                <td className="px-4 py-3">648</td>
                <td className="px-4 py-3">0.2361</td>
                <td className="px-4 py-3 font-bold text-emerald-400">0.3532 (5.3 ms)</td>
                <td className="px-4 py-3 text-rose-300">0.3300 (Degraded)</td>
                <td className="px-4 py-3 text-rose-300">0.3216 (Degraded)</td>
                <td className="px-4 py-3 text-neutral-400">10.2% (66 saved)</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="mt-3 flex items-center gap-2 text-[11px] text-neutral-400">
          <ShieldAlert className="h-3.5 w-3.5 text-amber-400" />
          <span>
            <strong>Scientific Disclosure:</strong> FiQA marks a documented failure mode for hybrid fusion: dense-only retrieval strictly outperforms hybrid and adaptive policies by +0.0232 nDCG@10 (p=0.0296) and is ~24x faster.
          </span>
        </div>
      </section>
    </div>
  );
}
