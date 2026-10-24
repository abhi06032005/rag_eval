import React from "react";
import Link from "next/link";
import { BookOpen, CheckCircle, Code, Shield } from "lucide-react";

export function Footer() {
  return (
    <footer className="w-full border-t border-neutral-800 bg-neutral-950 text-neutral-400 py-12 px-4 sm:px-6">
      <div className="mx-auto max-w-7xl grid grid-cols-1 md:grid-cols-4 gap-8">
        <div className="space-y-3 md:col-span-1">
          <div className="flex items-center gap-2">
            <span className="font-bold text-white text-base">RAGScaleBench</span>
            <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] text-neutral-300">
              v1.0-final
            </span>
          </div>
          <p className="text-xs text-neutral-400 leading-relaxed">
            Standardised benchmark evaluation and query-adaptive hybrid retrieval engine across scientific, biomedical, and financial corpora.
          </p>
          <div className="flex items-center gap-2 text-xs text-emerald-400">
            <CheckCircle className="h-3.5 w-3.5" />
            <span>Strict Zero-Leakage Calibrations</span>
          </div>
        </div>

        <div>
          <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-3">
            Exploration
          </h4>
          <ul className="space-y-2 text-xs">
            <li>
              <Link href="/playground" className="hover:text-white transition-colors">
                RAG QA Playground
              </Link>
            </li>
            <li>
              <Link href="/datasets" className="hover:text-white transition-colors">
                BEIR Ground-Truth Splits
              </Link>
            </li>
            <li>
              <Link href="/retrieval" className="hover:text-white transition-colors">
                Retrieval Benchmark Comparison
              </Link>
            </li>
            <li>
              <Link href="/analytics" className="hover:text-white transition-colors">
                Tail Latency & Cost Breakdown
              </Link>
            </li>
          </ul>
        </div>

        <div>
          <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-3">
            Integrity & Audit
          </h4>
          <ul className="space-y-2 text-xs">
            <li>
              <Link href="/reproducibility" className="hover:text-white transition-colors">
                Scientific Integrity Report
              </Link>
            </li>
            <li>
              <Link href="/routing" className="hover:text-white transition-colors">
                Adaptive Router Threshold Audit
              </Link>
            </li>
            <li>
              <Link href="/citations" className="hover:text-white transition-colors">
                Citation & Hallucination Inspector
              </Link>
            </li>
            <li>
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="hover:text-white transition-colors flex items-center gap-1"
              >
                <span>FastAPI OpenAPI Schema</span>
                <span className="text-[10px] text-neutral-500">↗</span>
              </a>
            </li>
          </ul>
        </div>

        <div className="space-y-2 text-xs">
          <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-3">
            Academic Citation
          </h4>
          <pre className="bg-neutral-900 border border-neutral-800 rounded p-2.5 text-[11px] text-neutral-300 overflow-x-auto font-mono">
{`@inproceedings{ragscalebench2026,
  title={RAGScaleBench: Systems Benchmarking
         and Adaptive Hybrid Retrieval},
  author={RAGScaleBench Research Group},
  year={2026}
}`}
          </pre>
          <p className="text-[11px] text-neutral-500">
            Open-source under MIT Licence. All benchmark datasets subject to original publisher terms.
          </p>
        </div>
      </div>

      <div className="mx-auto max-w-7xl mt-8 pt-6 border-t border-neutral-900 flex flex-col sm:flex-row items-center justify-between text-xs text-neutral-500 gap-4">
        <p>© 2026 RAGScaleBench Research Project. All experimental data cryptographically verified.</p>
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">
            <Shield className="h-3.5 w-3.5 text-neutral-400" />
            SHA-256 Manifest Locked
          </span>
          <span className="flex items-center gap-1">
            <Code className="h-3.5 w-3.5 text-neutral-400" />
            Deterministic Reproducibility Guaranteed
          </span>
        </div>
      </div>
    </footer>
  );
}
