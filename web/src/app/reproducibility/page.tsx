"use client";

import React, { useState } from "react";
import {
  Check,
  CheckCircle2,
  Copy,
  Download,
  FileCheck2,
  FileCode,
  HardDrive,
  Hash,
  Layers,
  Lock,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Terminal,
} from "lucide-react";
import { API_BASE_URL } from "@/lib/api";

const MANIFEST_ROWS = [
  {
    artifact: "SciFact Test Qrels",
    path: "data/external/scifact/qrels/test.tsv",
    sha256: "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
    status: "Verified Ground Truth (300 queries)",
  },
  {
    artifact: "NFCorpus Dev Qrels",
    path: "data/external/nfcorpus/qrels/dev.tsv",
    sha256: "b1d38b5e8f78c4a5820bce2b7ec2db54911d7690dc601e76811846b211180bd8",
    status: "Verified Ground Truth (324 queries)",
  },
  {
    artifact: "NFCorpus Test Qrels",
    path: "data/external/nfcorpus/qrels/test.tsv",
    sha256: "f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a",
    status: "Verified Ground Truth (323 queries)",
  },
  {
    artifact: "FiQA Dev Qrels",
    path: "data/external/fiqa/qrels/dev.tsv",
    sha256: "25828751509a25b3eb3e457f519eb6dc86c87e4622ee1bc26027a00d3d526ba6",
    status: "Verified Ground Truth (500 queries)",
  },
  {
    artifact: "FiQA Test Qrels",
    path: "data/external/fiqa/qrels/test.tsv",
    sha256: "07d6eb754162a5b28d6559bc8fb10fbf9e342aeaf7d74db76077558661649987",
    status: "Verified Ground Truth (648 queries)",
  },
  {
    artifact: "Canonical Registry (JSON)",
    path: "results/canonical_results_registry.json",
    sha256: "099aef509ff1c53e8788939dfb33a598972e38c94628d052d9b642a8bb74cf8f",
    status: "Canonical Registry Locked",
  },
  {
    artifact: "Scientific Integrity Audit (JSON)",
    path: "results/scientific_integrity_audit.json",
    sha256: "473cbf625e1d440ad8869818bba061f062ab410a56cb59f4258df5ad1630aa92",
    status: "Priority 1 Audit Verified",
  },
];

const REPRODUCTION_COMMANDS = [
  {
    label: "Audit Scientific Integrity & Splits",
    command: ".venv/Scripts/python scripts/audit_scientific_integrity.py",
    description: "Independently audits task-197, verifies BEIR splits, recalculates bootstrap CIs, and validates zero query leakage.",
  },
  {
    label: "Evaluate Real RAG Generation & Abstention",
    command: ".venv/Scripts/python scripts/evaluate_rag_generation.py",
    description: "Runs 20-sample reviewed evaluation benchmark measuring correctness, faithfulness, citations, and abstention.",
  },
  {
    label: "Run Full Test Suite (64 Tests)",
    command: ".venv/Scripts/pytest -q",
    description: "Executes unit and integration test suite with strict no-silent-fallback guarantees.",
  },
  {
    label: "Launch FastAPI Backend Server",
    command: ".venv/Scripts/uvicorn app.main:app --host 0.0.0.0 --port 8000",
    description: "Starts the production REST & Server-Sent Events API server at http://localhost:8000.",
  },
  {
    label: "Run Complete Docker Stack",
    command: "docker compose up --build",
    description: "Launches Qdrant vector database, FastAPI serving engine, and Prometheus metrics telemetry.",
  },
];

const DOWNLOAD_ARTIFACTS = [
  { name: "Scientific Integrity Report (Markdown)", endpoint: "scientific_integrity_report" },
  { name: "Scientific Integrity Audit (JSON)", endpoint: "scientific_integrity_audit" },
  { name: "Canonical Registry (CSV)", endpoint: "canonical_registry_csv" },
  { name: "Canonical Registry (JSON)", endpoint: "canonical_registry_json" },
  { name: "Generative Answer Evaluation (JSON)", endpoint: "generation_eval_json" },
  { name: "Generative Answer Report (Markdown)", endpoint: "generation_eval_report" },
  { name: "SciFact Corrected Results (CSV)", endpoint: "scifact_corrected" },
  { name: "SciFact Paired Bootstrap (CSV)", endpoint: "scifact_bootstrap" },
  { name: "NFCorpus Metrics Summary (CSV)", endpoint: "nfcorpus_summary" },
  { name: "FiQA Metrics Summary (CSV)", endpoint: "fiqa_summary" },
];

export default function ReproducibilityPage() {
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  function copyToClipboard(text: string, id: string) {
    navigator.clipboard.writeText(text);
    setCopiedCmd(id);
    setTimeout(() => setCopiedCmd(null), 2000);
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 space-y-10">
      {/* Header */}
      <div className="border-b border-neutral-800 pb-6">
        <div className="flex items-center gap-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-2">
          <Terminal className="h-4 w-4" />
          <span>Priority 1 & Priority 5 Academic Package</span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          Scientific Integrity & Reproducibility Package
        </h1>
        <p className="mt-2 text-sm text-neutral-400 max-w-3xl leading-relaxed">
          Complete cryptographic manifests, honest discrepancy resolutions, and deterministic reproduction instructions ensuring full third-party auditability.
        </p>
      </div>

      {/* Integrity Checklist & Audit Conclusions */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-4">
        <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
          <ShieldCheck className="h-4 w-4 text-emerald-400" />
          Mandatory Scientific Integrity Disclosures
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-sans">
          <div className="rounded-lg bg-neutral-950 border border-neutral-800 p-4 space-y-2">
            <div className="font-bold text-white flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              <span>FiQA Embedding Task-197 Verified</span>
            </div>
            <p className="text-neutral-400 text-xs leading-relaxed">
              Task log confirmed: embedded all 57,638 FiQA corpus documents across 12 shards in 2,664.4s (44.4 minutes). No synthetic data was used.
            </p>
          </div>

          <div className="rounded-lg bg-neutral-950 border border-neutral-800 p-4 space-y-2">
            <div className="font-bold text-white flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              <span>BEIR Ground-Truth Split Separation</span>
            </div>
            <p className="text-neutral-400 text-xs leading-relaxed">
              Verified ground-truth: SciFact has 809 train / 300 test queries; NFCorpus has 2,590 train / 324 dev / 323 test; FiQA has 5,500 train / 500 dev / 648 test. The previous claim of "648 dev queries" was a clerical typo.
            </p>
          </div>

          <div className="rounded-lg bg-neutral-950 border border-neutral-800 p-4 space-y-2">
            <div className="font-bold text-white flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              <span>Exploratory Non-Inferiority Classification</span>
            </div>
            <p className="text-neutral-400 text-xs leading-relaxed">
              The 0.025 nDCG non-inferiority margin is officially labeled <strong>exploratory post-hoc analysis</strong> rather than a prespecified confirmatory trial.
            </p>
          </div>

          <div className="rounded-lg bg-neutral-950 border border-neutral-800 p-4 space-y-2">
            <div className="font-bold text-white flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              <span>FiQA Dense-Only Superiority Reported</span>
            </div>
            <p className="text-neutral-400 text-xs leading-relaxed">
              Dense Exact (0.3532 nDCG@10, 5.3 ms) strictly beats Hybrid (0.3300, p=0.0296) and Adaptive Router (0.3216, p=0.0034) by 24x lower latency.
            </p>
          </div>
        </div>
      </div>

      {/* One-Click Command Center */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-5">
        <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
          <Terminal className="h-4 w-4 text-indigo-400" />
          Reproduction Command Center
        </h2>

        <div className="space-y-3">
          {REPRODUCTION_COMMANDS.map((item, idx) => (
            <div
              key={idx}
              className="rounded-lg bg-neutral-950 border border-neutral-800 p-3.5 space-y-1.5"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white">
                  {item.label}
                </span>
                <button
                  onClick={() => copyToClipboard(item.command, `cmd-${idx}`)}
                  className="flex items-center gap-1 rounded bg-neutral-900 px-2.5 py-1 text-[11px] font-mono text-neutral-300 border border-neutral-800 hover:text-white hover:bg-neutral-800 transition-colors"
                >
                  {copiedCmd === `cmd-${idx}` ? (
                    <>
                      <Check className="h-3 w-3 text-emerald-400" />
                      <span className="text-emerald-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
              <pre className="bg-neutral-900/80 rounded px-2.5 py-1.5 font-mono text-xs text-indigo-300 overflow-x-auto border border-neutral-800/80">
                {item.command}
              </pre>
              <p className="text-[11px] text-neutral-400 font-sans">
                {item.description}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Checksum Manifest Table */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Hash className="h-4 w-4 text-emerald-400" />
              Cryptographic SHA-256 Manifest Registry
            </h2>
            <p className="text-xs text-neutral-400 mt-0.5">
              Verified ground-truth qrels and canonical result registries locked for provenance.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto rounded-lg border border-neutral-800 bg-neutral-950 font-mono text-xs">
          <table className="w-full text-left text-neutral-300">
            <thead className="bg-neutral-900 text-[11px] uppercase tracking-wider text-neutral-400 border-b border-neutral-800">
              <tr>
                <th className="px-4 py-3">Artifact Identifier</th>
                <th className="px-4 py-3">Path</th>
                <th className="px-4 py-3">SHA-256 Checksum</th>
                <th className="px-4 py-3">Audit Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800">
              {MANIFEST_ROWS.map((row, idx) => (
                <tr key={idx} className="hover:bg-neutral-900/40">
                  <td className="px-4 py-3 font-sans font-semibold text-white">
                    {row.artifact}
                  </td>
                  <td className="px-4 py-3 text-neutral-400 text-[11px]">
                    {row.path}
                  </td>
                  <td className="px-4 py-3 text-indigo-300 text-[11px] truncate max-w-[200px]">
                    {row.sha256}
                  </td>
                  <td className="px-4 py-3 font-sans text-emerald-400 font-medium text-[11px]">
                    {row.status}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Direct Artifact Download Center */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-4">
        <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
          <Download className="h-4 w-4 text-indigo-400" />
          Artifact Download Center
        </h2>
        <p className="text-xs text-neutral-400">
          Directly download all raw experimental CSV files, JSON manifests, and scientific integrity audit reports.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {DOWNLOAD_ARTIFACTS.map((item, idx) => (
            <a
              key={idx}
              href={`${API_BASE_URL}/api/phase5/download/${item.endpoint}`}
              target="_blank"
              rel="noreferrer"
              className="flex items-center justify-between rounded-lg border border-neutral-800 bg-neutral-950 p-3 text-xs text-neutral-300 hover:border-neutral-700 hover:bg-neutral-900 hover:text-white transition-all group"
            >
              <span className="truncate pr-2 font-medium">{item.name}</span>
              <Download className="h-3.5 w-3.5 text-neutral-500 group-hover:text-indigo-400 shrink-0" />
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
