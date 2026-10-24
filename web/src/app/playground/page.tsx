"use client";

import React, { useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Cpu,
  Database,
  ExternalLink,
  Flame,
  Info,
  Layers,
  MessageSquare,
  Play,
  RotateCcw,
  Send,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import { API_BASE_URL, AskResponse, askQuestion } from "@/lib/api";

const PRELOADED_QUERIES = [
  {
    category: "Biomedical (SciFact)",
    question:
      "What effect does 0-6-methylguanine-DNA methyltransferase (MGMT) have on glioblastoma treatment?",
  },
  {
    category: "Systems Engineering (arXiv)",
    question:
      "What is the memory footprint reduction mechanism of 8-bit scalar vector quantization?",
  },
  {
    category: "Clinical Nutrition (NFCorpus)",
    question:
      "Does green tea consumption improve lipid profiles in dyslipidemic adults?",
  },
  {
    category: "Adversarial Abstention Test",
    question:
      "What was the exact date and crew manifest of the Apollo 18 manned lunar landing mission?",
  },
];

export default function PlaygroundPage() {
  const [question, setQuestion] = useState(PRELOADED_QUERIES[0].question);
  const [backend, setBackend] = useState<"mock" | "ollama" | "openai">("mock");
  const [modelName, setModelName] = useState("");
  const [k, setK] = useState(5);
  const [hnswEf, setHnswEf] = useState(64);
  const [temperature, setTemperature] = useState(0.0);
  const [useStreaming, setUseStreaming] = useState(true);

  const [loading, setLoading] = useState(false);
  const [streamingTokens, setStreamingTokens] = useState<string[]>([]);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedSources, setExpandedSources] = useState(false);

  async function handleExecute() {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setStreamingTokens([]);
    setResult(null);

    if (useStreaming) {
      try {
        const response = await fetch(`${API_BASE_URL}/ask/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: question.trim(),
            k,
            hnsw_ef: hnswEf,
            temperature,
            max_tokens: 256,
            generator_backend: backend,
            model: modelName || undefined,
            stream: true,
          }),
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(
            errData.detail || `Server returned HTTP ${response.status}`
          );
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        if (reader) {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split("\n\n");
            buffer = parts.pop() || "";

            for (const part of parts) {
              if (part.startsWith("data: ")) {
                try {
                  const data = JSON.parse(part.slice(6));
                  if (data.event === "token") {
                    setStreamingTokens((prev) => [...prev, data.token]);
                  } else if (data.event === "done") {
                    setResult(data);
                  } else if (data.event === "error") {
                    setError(data.error);
                  }
                } catch {}
              }
            }
          }
        }
      } catch (err: any) {
        setError(err.message || String(err));
      } finally {
        setLoading(false);
      }
    } else {
      // Standard POST /ask
      try {
        const resp = await askQuestion({
          question: question.trim(),
          k,
          hnsw_ef: hnswEf,
          temperature,
          max_tokens: 256,
          generator_backend: backend,
          model: modelName || undefined,
        });
        setResult(resp);
      } catch (err: any) {
        setError(err.message || String(err));
      } finally {
        setLoading(false);
      }
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      {/* Page Header */}
      <div className="mb-8 border-b border-neutral-800 pb-6">
        <div className="flex items-center gap-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-2">
          <MessageSquare className="h-4 w-4" />
          <span>Interactive Generative RAG Playground</span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          Evidence-Grounded Question Answering
        </h1>
        <p className="mt-2 text-sm text-neutral-400 max-w-3xl leading-relaxed">
          Test real-time embedding, nearest-neighbor vector retrieval, injection-resistant context assembly, and LLM answer generation with explicit citation links and abstention telemetry.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Query Input & Controls */}
        <div className="lg:col-span-5 space-y-6">
          {/* Query Box */}
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase text-neutral-300 tracking-wider">
                Research Question
              </label>
              <button
                onClick={() => setQuestion("")}
                className="text-[11px] text-neutral-500 hover:text-neutral-300 flex items-center gap-1 transition-colors"
              >
                <RotateCcw className="h-3 w-3" />
                Clear
              </button>
            </div>

            <textarea
              rows={4}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter scientific question or claim to verify..."
              className="w-full rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2.5 text-sm text-white placeholder-neutral-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />

            {/* Preloaded Sample Queries */}
            <div>
              <span className="text-[11px] text-neutral-400 block mb-2 font-medium">
                Select Preloaded Test Query:
              </span>
              <div className="space-y-1.5">
                {PRELOADED_QUERIES.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => setQuestion(q.question)}
                    className="w-full text-left rounded border border-neutral-800/80 bg-neutral-950/70 p-2 text-xs hover:border-neutral-700 hover:bg-neutral-800/50 transition-colors group"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-neutral-300 group-hover:text-indigo-300 transition-colors">
                        {q.category}
                      </span>
                      <ArrowRight className="h-3 w-3 text-neutral-600 group-hover:text-neutral-400" />
                    </div>
                    <p className="text-[11px] text-neutral-400 truncate mt-0.5">
                      {q.question}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Model & Generation Parameters */}
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-5 space-y-4">
            <h3 className="text-xs font-semibold uppercase text-neutral-300 tracking-wider flex items-center gap-2">
              <Cpu className="h-4 w-4 text-indigo-400" />
              Runtime Model Provider
            </h3>

            {/* Provider Tabs */}
            <div className="grid grid-cols-3 gap-2">
              <button
                type="button"
                onClick={() => setBackend("mock")}
                className={`rounded-lg border px-3 py-2 text-xs font-medium transition-colors text-center ${
                  backend === "mock"
                    ? "border-indigo-500 bg-indigo-950/40 text-indigo-200 font-semibold"
                    : "border-neutral-800 bg-neutral-950 text-neutral-400 hover:border-neutral-700"
                }`}
              >
                Mock Fixture
                <span className="block text-[10px] text-neutral-500 mt-0.5">
                  Offline Deterministic
                </span>
              </button>

              <button
                type="button"
                onClick={() => setBackend("ollama")}
                className={`rounded-lg border px-3 py-2 text-xs font-medium transition-colors text-center ${
                  backend === "ollama"
                    ? "border-emerald-500 bg-emerald-950/40 text-emerald-200 font-semibold"
                    : "border-neutral-800 bg-neutral-950 text-neutral-400 hover:border-neutral-700"
                }`}
              >
                Local Ollama
                <span className="block text-[10px] text-neutral-500 mt-0.5">
                  Llama 3.2 3B
                </span>
              </button>

              <button
                type="button"
                onClick={() => setBackend("openai")}
                className={`rounded-lg border px-3 py-2 text-xs font-medium transition-colors text-center ${
                  backend === "openai"
                    ? "border-indigo-500 bg-indigo-950/40 text-indigo-200 font-semibold"
                    : "border-neutral-800 bg-neutral-950 text-neutral-400 hover:border-neutral-700"
                }`}
              >
                OpenAI API
                <span className="block text-[10px] text-neutral-500 mt-0.5">
                  GPT-4o-mini / vLLM
                </span>
              </button>
            </div>

            {backend !== "mock" && (
              <div className="rounded-lg bg-neutral-950 p-3 border border-neutral-800 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-neutral-400">Custom Model Identifier:</span>
                  <input
                    type="text"
                    placeholder={
                      backend === "ollama" ? "llama3.2:3b" : "gpt-4o-mini"
                    }
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    className="rounded border border-neutral-800 bg-neutral-900 px-2 py-1 text-xs text-white placeholder-neutral-600 focus:outline-none"
                  />
                </div>
                <div className="flex items-center gap-1.5 text-[11px] text-amber-400">
                  <Info className="h-3.5 w-3.5 shrink-0" />
                  <span>
                    Strict mode active: Silent mock fallback is disabled. If daemon is offline, HTTP 503 is returned.
                  </span>
                </div>
              </div>
            )}

            {/* Sliders: Top-K and HNSW ef */}
            <div className="space-y-3 pt-2 border-t border-neutral-800 text-xs">
              <div>
                <div className="flex justify-between text-neutral-400 mb-1">
                  <span>Retrieved Passages (top-k)</span>
                  <span className="font-mono text-white">{k}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={20}
                  value={k}
                  onChange={(e) => setK(Number(e.target.value))}
                  className="w-full accent-indigo-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-neutral-400 mb-1">
                  <span>HNSW ef_search Parameter</span>
                  <span className="font-mono text-white">{hnswEf}</span>
                </div>
                <input
                  type="range"
                  min={16}
                  max={256}
                  step={16}
                  value={hnswEf}
                  onChange={(e) => setHnswEf(Number(e.target.value))}
                  className="w-full accent-indigo-500"
                />
              </div>

              <div className="flex items-center justify-between pt-2">
                <span className="text-neutral-400">Stream Answer Tokens (SSE)</span>
                <input
                  type="checkbox"
                  checked={useStreaming}
                  onChange={(e) => setUseStreaming(e.target.checked)}
                  className="h-4 w-4 rounded border-neutral-800 bg-neutral-950 text-indigo-600 focus:ring-indigo-500"
                />
              </div>
            </div>

            {/* Execute Button */}
            <button
              onClick={handleExecute}
              disabled={loading || !question.trim()}
              className="w-full flex items-center justify-center gap-2 rounded-lg bg-indigo-600 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-600/25 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {loading ? (
                <>
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  <span>Retrieving & Generating...</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-white" />
                  <span>Execute RAG Pipeline</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Execution Results & Telemetry */}
        <div className="lg:col-span-7 space-y-6">
          {/* Error Banner */}
          {error && (
            <div className="rounded-xl border border-rose-800/80 bg-rose-950/40 p-4 text-xs text-rose-300 flex items-start gap-3">
              <ShieldAlert className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-rose-200">
                  Generation Backend Unavailable
                </h4>
                <p className="mt-1 text-rose-300/90 leading-relaxed font-mono text-[11px]">
                  {error}
                </p>
                <p className="mt-2 text-[10px] text-rose-400">
                  Per scientific integrity standards, RAGScaleBench refuses to emit heuristic mock text when a user requests an LLM. Switch to "Mock Fixture" to test deterministic offline generation.
                </p>
              </div>
            </div>
          )}

          {/* Answer Card */}
          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-neutral-800/80 pb-3">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-indigo-400" />
                <h3 className="text-sm font-bold text-white tracking-tight">
                  Generated Answer
                </h3>
              </div>

              {/* Provider Badge */}
              <div className="flex items-center gap-2">
                {result?.generator_info?.is_mock ? (
                  <span className="rounded-full bg-amber-950/70 border border-amber-800/60 px-2.5 py-0.5 text-[10px] font-semibold text-amber-300">
                    Mock Fixture (Leading Sentence)
                  </span>
                ) : result?.generator_info ? (
                  <span className="rounded-full bg-emerald-950/70 border border-emerald-800/60 px-2.5 py-0.5 text-[10px] font-semibold text-emerald-300">
                    Real LLM ({result.generator_info.backend})
                  </span>
                ) : null}

                {result?.abstained && (
                  <span className="rounded-full bg-rose-950/80 border border-rose-800 px-2.5 py-0.5 text-[10px] font-bold text-rose-300">
                    Abstained (Insufficient Evidence)
                  </span>
                )}
              </div>
            </div>

            {/* Answer Content */}
            <div className="min-h-[140px] rounded-lg bg-neutral-950 border border-neutral-800/70 p-4 text-sm text-neutral-100 leading-relaxed">
              {loading && streamingTokens.length === 0 ? (
                <div className="flex items-center gap-2 text-neutral-500 italic">
                  <span className="h-3 w-3 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
                  <span>Searching Qdrant index and invoking generator...</span>
                </div>
              ) : streamingTokens.length > 0 ? (
                <div>
                  <p className="whitespace-pre-wrap">
                    {streamingTokens.join("")}
                  </p>
                </div>
              ) : result ? (
                <div>
                  <p className="whitespace-pre-wrap">{result.answer}</p>
                </div>
              ) : (
                <div className="text-neutral-500 italic flex flex-col items-center justify-center py-8">
                  <MessageSquare className="h-8 w-8 text-neutral-700 mb-2" />
                  <span>Click "Execute RAG Pipeline" to retrieve evidence and generate answer.</span>
                </div>
              )}
            </div>

            {/* Citation Tags */}
            {result?.sources && result.sources.length > 0 && (
              <div className="pt-2">
                <span className="text-[11px] font-semibold uppercase text-neutral-400 tracking-wider mr-2">
                  Cited Source Passages:
                </span>
                <div className="inline-flex flex-wrap gap-1.5 mt-1">
                  {result.sources.map((src, i) => (
                    <span
                      key={i}
                      className="rounded bg-indigo-950/60 border border-indigo-800/60 px-2 py-0.5 text-[11px] font-mono text-indigo-300"
                    >
                      [{src}]
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Timing Waterfall Card */}
          {result?.timings_ms && (
            <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-semibold uppercase text-neutral-300 tracking-wider flex items-center gap-2">
                  <Clock className="h-4 w-4 text-emerald-400" />
                  Latency Breakdown (Total: {result.timings_ms.total.toFixed(2)} ms)
                </h4>
                <span className="text-[11px] font-mono text-neutral-400">
                  Tokens: {result.generator_info?.token_count || 0}
                </span>
              </div>

              <div className="space-y-2 text-xs">
                {/* Embedding Bar */}
                <div>
                  <div className="flex justify-between text-[11px] text-neutral-400 mb-1">
                    <span>1. Query Embedding (BAAI/bge-small-en-v1.5)</span>
                    <span className="font-mono text-white">
                      {result.timings_ms.embedding.toFixed(2)} ms
                    </span>
                  </div>
                  <div className="h-2 w-full bg-neutral-950 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-indigo-500 rounded-full"
                      style={{
                        width: `${Math.min(
                          100,
                          (result.timings_ms.embedding /
                            Math.max(result.timings_ms.total, 0.01)) *
                            100
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Retrieval Bar */}
                <div>
                  <div className="flex justify-between text-[11px] text-neutral-400 mb-1">
                    <span>2. Vector Search (Qdrant HNSW ef={hnswEf})</span>
                    <span className="font-mono text-white">
                      {result.timings_ms.retrieval.toFixed(2)} ms
                    </span>
                  </div>
                  <div className="h-2 w-full bg-neutral-950 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full"
                      style={{
                        width: `${Math.min(
                          100,
                          (result.timings_ms.retrieval /
                            Math.max(result.timings_ms.total, 0.01)) *
                            100
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Context Assembly */}
                <div>
                  <div className="flex justify-between text-[11px] text-neutral-400 mb-1">
                    <span>3. Delimited Context Assembly & Sanitization</span>
                    <span className="font-mono text-white">
                      {result.timings_ms.context_assembly.toFixed(2)} ms
                    </span>
                  </div>
                  <div className="h-2 w-full bg-neutral-950 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-teal-500 rounded-full"
                      style={{
                        width: `${Math.min(
                          100,
                          (result.timings_ms.context_assembly /
                            Math.max(result.timings_ms.total, 0.01)) *
                            100
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Generation */}
                <div>
                  <div className="flex justify-between text-[11px] text-neutral-400 mb-1">
                    <span>4. LLM Generation ({backend})</span>
                    <span className="font-mono text-white">
                      {result.timings_ms.generation.toFixed(2)} ms
                    </span>
                  </div>
                  <div className="h-2 w-full bg-neutral-950 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-amber-500 rounded-full"
                      style={{
                        width: `${Math.min(
                          100,
                          (result.timings_ms.generation /
                            Math.max(result.timings_ms.total, 0.01)) *
                            100
                        )}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Retrieved Evidence Passages Accordion */}
          {result?.source_details && result.source_details.length > 0 && (
            <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5">
              <button
                onClick={() => setExpandedSources(!expandedSources)}
                className="w-full flex items-center justify-between text-xs font-semibold uppercase text-neutral-300 tracking-wider"
              >
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-indigo-400" />
                  <span>
                    Retrieved Evidence Passages ({result.source_details.length})
                  </span>
                </div>
                {expandedSources ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </button>

              {expandedSources && (
                <div className="mt-4 space-y-3">
                  {result.source_details.map((hit, idx) => (
                    <div
                      key={idx}
                      className="rounded-lg border border-neutral-800 bg-neutral-950 p-3 space-y-1.5 text-xs"
                    >
                      <div className="flex items-center justify-between font-mono text-[11px]">
                        <span className="text-indigo-400 font-semibold">
                          #{idx + 1} [{hit.doc_id}:{hit.chunk_id}]
                        </span>
                        <span className="text-emerald-400 font-bold">
                          Cosine Score: {hit.score.toFixed(4)}
                        </span>
                      </div>
                      {hit.title && (
                        <div className="text-[11px] text-neutral-300 font-medium italic">
                          Title: {hit.title}
                        </div>
                      )}
                      <p className="text-neutral-400 text-xs leading-relaxed font-sans bg-neutral-900/40 p-2 rounded border border-neutral-800/60">
                        "{hit.text}"
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
