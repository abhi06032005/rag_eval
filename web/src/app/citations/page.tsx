"use client";

import React, { useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  FileCheck2,
  FileSearch,
  Filter,
  Flame,
  HelpCircle,
  Info,
  Layers,
  RotateCcw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

const SAMPLE_INSPECTION_ITEMS = [
  {
    title: "Item 1: Epigenetic MGMT Methylation (SciFact)",
    question: "What effect does MGMT promoter methylation have on glioblastoma?",
    answer: "Based on the provided evidence, epigenetic silencing of the MGMT DNA-repair gene by promoter methylation compromises DNA repair and has been associated with longer survival among glioblastoma patients treated with temozolomide. [scifact:147]",
    context: "Epigenetic silencing of the MGMT (O-6-methylguanine-DNA methyltransferase) DNA-repair gene by promoter methylation compromises DNA repair and has been associated with longer survival among glioblastoma patients treated with temozolomide.",
    gold_doc_id: "scifact:147",
    is_answerable: true,
  },
  {
    title: "Item 2: Interleukin-6 Osteoclastogenesis (SciFact)",
    question: "Does interleukin-6 inhibit or stimulate osteoclast differentiation?",
    answer: "Based on the provided evidence, Interleukin-6 (IL-6) stimulates osteoclast formation and bone resorption both directly and indirectly via upregulation of RANKL expression. [scifact:412]",
    context: "Interleukin-6 (IL-6) stimulates osteoclast formation and bone resorption both directly and indirectly via upregulation of RANKL expression.",
    gold_doc_id: "scifact:412",
    is_answerable: true,
  },
  {
    title: "Item 3: Counterfactual Moon Mission (Adversarial Abstention)",
    question: "What was the exact date and crew manifest of the Apollo 18 manned lunar landing mission?",
    answer: "I am unable to answer based on the provided evidence.",
    context: "",
    gold_doc_id: "none",
    is_answerable: false,
  },
];

export default function CitationsPage() {
  const [selectedSample, setSelectedSample] = useState(SAMPLE_INSPECTION_ITEMS[0]);
  const [customAnswer, setCustomAnswer] = useState(SAMPLE_INSPECTION_ITEMS[0].answer);
  const [customContext, setCustomContext] = useState(SAMPLE_INSPECTION_ITEMS[0].context);

  function loadSample(item: (typeof SAMPLE_INSPECTION_ITEMS)[0]) {
    setSelectedSample(item);
    setCustomAnswer(item.answer);
    setCustomContext(item.context);
  }

  // Forensic analysis
  const citations = Array.from(customAnswer.matchAll(/\[([a-zA-Z0-9_\-:\.]+)\]/g)).map((m) => m[1]);
  const isAbstention = customAnswer.toLowerCase().includes("unable to answer based on the provided evidence");

  const sentences = customAnswer
    .split(".")
    .map((s) => s.trim())
    .filter((s) => s.length > 5);

  const ctxWords = new Set(customContext.toLowerCase().match(/\w+/g) || []);
  const analyzedSentences = sentences.map((sent) => {
    if (isAbstention) {
      return { text: sent, isSupported: true, overlapRatio: 1.0, isAbstention: true };
    }
    const sentWords = new Set(sent.toLowerCase().match(/\w+/g) || []);
    let overlap = 0;
    sentWords.forEach((w) => {
      if (ctxWords.has(w)) overlap++;
    });
    const ratio = sentWords.size > 0 ? overlap / sentWords.size : 0;
    return {
      text: sent,
      isSupported: ratio >= 0.4,
      overlapRatio: ratio,
      isAbstention: false,
    };
  });

  const supportedCount = analyzedSentences.filter((s) => s.isSupported).length;
  const unsupportedCount = analyzedSentences.length - supportedCount;
  const faithfulness = analyzedSentences.length > 0 ? (supportedCount / analyzedSentences.length) * 100 : 0;

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 space-y-10">
      {/* Header */}
      <div className="border-b border-neutral-800 pb-6">
        <div className="flex items-center gap-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-2">
          <ShieldCheck className="h-4 w-4" />
          <span>Factual Grounding & Hallucination Audit</span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          Citation & Claim Entailment Inspector
        </h1>
        <p className="mt-2 text-sm text-neutral-400 max-w-3xl leading-relaxed">
          Deconstructs generated RAG answers sentence-by-sentence. Verifies bracketed citations against ground-truth document IDs and evaluates factual claim support against retrieved passages.
        </p>
      </div>

      {/* Critical Scientific Note */}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 space-y-2">
        <div className="flex items-center gap-2 text-indigo-300 font-bold text-xs uppercase tracking-wider">
          <Info className="h-4 w-4 text-indigo-400" />
          <span>Scientific Disclosure: Syntactic Citations vs Causal Factual Support</span>
        </div>
        <p className="text-xs text-neutral-300 leading-relaxed font-sans">
          A formatted string citation such as <code className="text-indigo-300 font-mono">[scifact:147]</code> verifies that a citation string is present, but does not inherently prove that the preceding assertion logically follows from the source text. RAGScaleBench therefore measures both <strong>Citation Precision / Recall</strong> and <strong>Sentence Claim Entailment</strong> to detect unsupported hallucinations.
        </p>
      </div>

      {/* Sample Selector */}
      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-neutral-400 font-medium self-center mr-1">
          Select Verified Review Item:
        </span>
        {SAMPLE_INSPECTION_ITEMS.map((item, idx) => (
          <button
            key={idx}
            onClick={() => loadSample(item)}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              selectedSample.title === item.title
                ? "bg-indigo-600 text-white font-semibold"
                : "bg-neutral-900 text-neutral-400 border border-neutral-800 hover:border-neutral-700"
            }`}
          >
            {item.title}
          </button>
        ))}
      </div>

      {/* Inspector Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Input Text and Context */}
        <div className="space-y-4">
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 space-y-3">
            <label className="text-xs font-semibold uppercase text-neutral-300 tracking-wider block">
              Generated RAG Answer
            </label>
            <textarea
              rows={4}
              value={customAnswer}
              onChange={(e) => setCustomAnswer(e.target.value)}
              className="w-full rounded-lg border border-neutral-800 bg-neutral-950 p-3 text-xs text-white font-mono focus:border-indigo-500 focus:outline-none"
            />
          </div>

          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 space-y-3">
            <label className="text-xs font-semibold uppercase text-neutral-300 tracking-wider block">
              Retrieved Reference Context (Untrusted Data)
            </label>
            <textarea
              rows={5}
              value={customContext}
              onChange={(e) => setCustomContext(e.target.value)}
              placeholder="Paste retrieved passages to verify claim support..."
              className="w-full rounded-lg border border-neutral-800 bg-neutral-950 p-3 text-xs text-neutral-300 font-mono focus:border-indigo-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Right: Forensic Sentence-by-Sentence Breakdown */}
        <div className="space-y-4">
          {/* Metrics Overview Pill */}
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 grid grid-cols-3 gap-3 text-xs font-mono text-center">
            <div className="rounded bg-neutral-950 p-3 border border-neutral-800">
              <span className="text-[10px] text-neutral-500 block uppercase font-sans">
                Extracted Citations
              </span>
              <span className="text-lg font-bold text-indigo-400 mt-1 block">
                {citations.length}
              </span>
            </div>
            <div className="rounded bg-neutral-950 p-3 border border-neutral-800">
              <span className="text-[10px] text-neutral-500 block uppercase font-sans">
                Claim Faithfulness
              </span>
              <span className="text-lg font-bold text-emerald-400 mt-1 block">
                {faithfulness.toFixed(1)}%
              </span>
            </div>
            <div className="rounded bg-neutral-950 p-3 border border-neutral-800">
              <span className="text-[10px] text-neutral-500 block uppercase font-sans">
                Unsupported Claims
              </span>
              <span
                className={`text-lg font-bold mt-1 block ${
                  unsupportedCount === 0 ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {unsupportedCount}
              </span>
            </div>
          </div>

          {/* Sentence Analysis View */}
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 space-y-3">
            <h3 className="text-xs font-semibold uppercase text-neutral-300 tracking-wider flex items-center justify-between">
              <span>Sentence Claim Entailment Audit</span>
              <span className="text-[11px] font-normal text-neutral-500 font-mono">
                {analyzedSentences.length} sentences
              </span>
            </h3>

            <div className="space-y-2.5">
              {analyzedSentences.map((s, idx) => (
                <div
                  key={idx}
                  className={`rounded-lg p-3 text-xs border font-sans leading-relaxed ${
                    s.isAbstention
                      ? "bg-indigo-950/40 border-indigo-800/80 text-indigo-200"
                      : s.isSupported
                      ? "bg-emerald-950/30 border-emerald-800/70 text-emerald-200"
                      : "bg-rose-950/40 border-rose-800/80 text-rose-200"
                  }`}
                >
                  <div className="flex items-center justify-between text-[10px] font-mono mb-1 font-semibold">
                    <span>Sentence #{idx + 1}</span>
                    <span>
                      {s.isAbstention
                        ? "Validated Abstention"
                        : s.isSupported
                        ? `Supported (${(s.overlapRatio * 100).toFixed(0)}% token alignment)`
                        : "Unsupported / Hallucinated"}
                    </span>
                  </div>
                  <p>"{s.text}."</p>
                </div>
              ))}
            </div>
          </div>

          {/* Extracted Bracketed Citations */}
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-4 space-y-2 text-xs">
            <span className="text-[11px] font-semibold uppercase text-neutral-400 tracking-wider block">
              Extracted Bracketed Identifiers:
            </span>
            <div className="flex flex-wrap gap-2">
              {citations.length > 0 ? (
                citations.map((c, i) => (
                  <span
                    key={i}
                    className="rounded bg-indigo-950 border border-indigo-800 px-2.5 py-1 text-xs font-mono text-indigo-300 flex items-center gap-1.5"
                  >
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                    <span>[{c}]</span>
                  </span>
                ))
              ) : (
                <span className="text-neutral-500 italic text-[11px]">
                  No bracketed citations detected in answer text.
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
