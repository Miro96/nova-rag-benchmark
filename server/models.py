"""Pydantic models for the leaderboard API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class ServerInfo(BaseModel):
    name: str
    git_url: str = ""
    git_user: str = ""
    version: str = ""


class IngestMetrics(BaseModel):
    total_files: int = 0
    total_sec: float = 0
    files_per_sec: float = 0
    index_size_mb: float = 0
    ram_peak_mb: float = 0


class LatencyMetrics(BaseModel):
    p50_ms: float = 0
    p95_ms: float = 0
    p99_ms: float = 0
    mean_ms: float = 0


class RetrievalMetrics(BaseModel):
    total_queries: int = 0
    total_hits: int = 0
    hit_at_1: float = 0
    hit_at_3: float = 0
    hit_at_5: float = 0
    hit_at_10: float = 0
    symbol_hit_at_5: float = 0
    chunk_hit_at_5: float = 0
    mrr: float = 0
    latency: LatencyMetrics = LatencyMetrics()
    tokens: dict = {}


class EfficiencyMetrics(BaseModel):
    avg_tool_calls: float = 0
    avg_total_llm_tokens: float = 0


class BenchmarkSubmission(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    bench_version: str = ""
    dataset_version: str = ""
    server: ServerInfo
    environment: dict = {}
    repos: list[str] = []
    ingest: IngestMetrics = IngestMetrics()
    retrieval: RetrievalMetrics = RetrievalMetrics()
    efficiency: EfficiencyMetrics = EfficiencyMetrics()
    composite_score: float = 0
    by_difficulty: dict = {}
    by_type: dict = {}
    query_details: list[dict] = []
    replicates: list[dict] = []
    iqr: dict = {}
    by_repo: dict = {}
    baseline: dict = {}
    ab_comparison: dict = {}
    startup_ms: float = 0

    @model_validator(mode="after")
    def validate_query_details_is_list(self) -> BenchmarkSubmission:
        """Reject malformed query_details (e.g., non-list)."""
        if not isinstance(self.query_details, list):
            raise ValueError(
                "query_details must be a list, "
                f"got {type(self.query_details).__name__}"
            )
        return self


class LeaderboardEntry(BaseModel):
    rank: int
    id: str
    server_name: str
    git_url: str
    git_user: str
    hit_at_1: float
    hit_at_5: float
    symbol_hit_at_5: float
    chunk_hit_at_5: float = 0
    mrr: float
    query_latency_p50_ms: float
    query_latency_p95_ms: float
    ingest_total_sec: float
    ram_peak_mb: float
    composite_score: float
    avg_response_tokens: float = 0
    p95_response_tokens: float = 0
    total_response_tokens: float = 0
    avg_llm_tokens: float = 0
    submitted_at: str
