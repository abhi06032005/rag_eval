"""
API Schemas for Retrieve and Ask Endpoints.
"""

from pydantic import BaseModel, Field


class TimingBreakdown(BaseModel):
    embedding: float = 0.0
    retrieval: float = 0.0
    context_assembly: float = 0.0
    generation: float = 0.0
    total: float = 0.0


class SourceRecord(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str
    title: str = ""
    source: str = ""


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=10, ge=1, le=100)
    score_threshold: float | None = None
    hnsw_ef: int = Field(default=64, ge=1, le=512)
    exact: bool = False


class RetrieveResponse(BaseModel):
    request_id: str
    query: str
    hits: list[SourceRecord]
    timings_ms: TimingBreakdown
    config_hash: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=10, ge=1, le=50)
    score_threshold: float | None = None
    hnsw_ef: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    generator_backend: str | None = Field(default=None, description="mock, ollama, openai, vllm, api")
    model: str | None = Field(default=None, description="Specific LLM model identifier")
    stream: bool = Field(default=False, description="Stream answer tokens via SSE")


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    sources: list[str]
    source_details: list[SourceRecord]
    timings_ms: TimingBreakdown
    config_hash: str
    abstained: bool = False
    generator_info: dict | None = None

