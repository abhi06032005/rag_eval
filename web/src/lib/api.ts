/**
 * RAGScaleBench API Client.
 * Connects to the FastAPI backend (default: http://localhost:8000)
 * with graceful offline data fallbacks when backend is not actively running.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface TimingBreakdown {
  embedding: number;
  retrieval: number;
  context_assembly: number;
  generation: number;
  total: number;
}

export interface SourceRecord {
  chunk_id: string;
  doc_id: string;
  score: number;
  text: string;
  title: string;
  source: string;
}

export interface AskResponse {
  request_id: string;
  question: string;
  answer: string;
  sources: string[];
  source_details: SourceRecord[];
  timings_ms: TimingBreakdown;
  config_hash: string;
  abstained: boolean;
  generator_info?: {
    backend: string;
    model_name: string;
    runtime_mode: string;
    is_mock: boolean;
    token_count: number;
    prompt_tokens: number;
    completion_tokens: number;
    generation_ms: number;
    abstained: boolean;
    finish_reason: string;
  };
}

export interface GeneratorProvider {
  backend: string;
  default_model: string;
  runtime_mode: string;
  is_mock: boolean;
  available: boolean;
  message: string;
  base_url?: string;
}

export async function checkBackendHealth(): Promise<{
  ok: boolean;
  message: string;
}> {
  try {
    const res = await fetch(`${API_BASE_URL}/healthz`, {
      method: "GET",
      cache: "no-store",
    });
    if (res.ok) {
      return { ok: true, message: "FastAPI Backend Online (:8000)" };
    }
    return { ok: false, message: `Backend returned HTTP ${res.status}` };
  } catch (err) {
    return {
      ok: false,
      message: "FastAPI Backend Unreachable (Viewing verified local registry)",
    };
  }
}

export async function getGeneratorStatus(): Promise<GeneratorProvider[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/generator/status`, {
      cache: "no-store",
    });
    if (res.ok) {
      return await res.json();
    }
  } catch {}
  return [
    {
      backend: "mock",
      default_model: "mock-generator",
      runtime_mode: "mock_fixture",
      is_mock: true,
      available: true,
      message: "Mock generator fixture available (deterministic offline mode)",
    },
    {
      backend: "ollama",
      default_model: "llama3.2:3b",
      runtime_mode: "real_llm",
      is_mock: false,
      available: false,
      message: "Ollama daemon offline / not running locally",
    },
    {
      backend: "openai",
      default_model: "gpt-4o-mini",
      runtime_mode: "real_llm",
      is_mock: false,
      available: false,
      message: "OPENAI_API_KEY environment variable not configured",
    },
  ];
}

export async function getScientificAudit(): Promise<any> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/scientific-audit`, {
      cache: "no-store",
    });
    if (res.ok) return await res.json();
  } catch {}
  return null;
}

export async function getGenerationEval(): Promise<any> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/generation-eval`, {
      cache: "no-store",
    });
    if (res.ok) return await res.json();
  } catch {}
  return null;
}

export async function getCanonicalRegistry(): Promise<any> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/phase6/canonical-registry`, {
      cache: "no-store",
    });
    if (res.ok) return await res.json();
  } catch {}
  return null;
}

export async function getRuntimeAudit(): Promise<any> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/phase6/runtime-audit`, {
      cache: "no-store",
    });
    if (res.ok) return await res.json();
  } catch {}
  return null;
}

export async function getCorpusStats(): Promise<any> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/corpus/stats`, {
      cache: "no-store",
    });
    if (res.ok) return await res.json();
  } catch {}
  return {
    total_documents: 75000,
    sources: { arxiv: 30000, wikipedia: 30000, govinfo: 15000 },
  };
}

export async function askQuestion(payload: {
  question: string;
  k?: number;
  hnsw_ef?: number;
  generator_backend?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
}): Promise<AskResponse> {
  const res = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errData.detail || `Server error: ${res.status}`);
  }

  return await res.json();
}
