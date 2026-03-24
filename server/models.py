"""Pydantic models for the leaderboard API."""

from __future__ import annotations

from pydantic import BaseModel


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
    mrr: float = 0
    latency: LatencyMetrics = LatencyMetrics()


class EfficiencyMetrics(BaseModel):
    avg_tool_calls: float = 0


class BenchmarkSubmission(BaseModel):
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


class LeaderboardEntry(BaseModel):
    rank: int
    id: str
    server_name: str
    git_url: str
    git_user: str
    hit_at_1: float
    hit_at_5: float
    symbol_hit_at_5: float
    mrr: float
    query_latency_p50_ms: float
    query_latency_p95_ms: float
    ingest_total_sec: float
    ram_peak_mb: float
    composite_score: float
    submitted_at: str
