"""Metrics for evaluating RAG code search quality and performance."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath


@dataclass
class QueryResult:
    """Result of a single benchmark query."""
    query_id: str
    query_text: str
    query_type: str
    difficulty: str
    expected_files: list[str]
    expected_symbols: list[str]
    returned_files: list[str]
    returned_symbols: list[str]
    latency_ms: float
    tool_calls: int = 1
    found_file: bool = False
    found_symbol: bool = False


@dataclass
class BenchmarkMetrics:
    """All computed metrics from a benchmark run."""
    # Retrieval quality
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    symbol_hit_at_5: float = 0.0
    mrr: float = 0.0

    # Latency
    query_latency_p50_ms: float = 0.0
    query_latency_p95_ms: float = 0.0
    query_latency_p99_ms: float = 0.0
    query_latency_mean_ms: float = 0.0

    # Ingest
    ingest_total_sec: float = 0.0
    ingest_files_per_sec: float = 0.0
    ingest_total_files: int = 0

    # Resources
    index_size_mb: float = 0.0
    ram_peak_mb: float = 0.0

    # Efficiency (A/B)
    avg_tool_calls: float = 0.0
    total_queries: int = 0
    total_hits: int = 0

    # Composite
    composite_score: float = 0.0

    # Breakdowns
    by_difficulty: dict[str, dict] = field(default_factory=dict)
    by_type: dict[str, dict] = field(default_factory=dict)
    by_repo: dict[str, dict] = field(default_factory=dict)


def normalize_path(path: str) -> str:
    """Normalize a file path for comparison."""
    p = PurePosixPath(path.replace("\\", "/"))
    # Remove leading ./ or /
    parts = [part for part in p.parts if part not in (".", "/")]
    return "/".join(parts).lower()


def file_matches(returned: str, expected: str) -> bool:
    """Check if a returned file path matches the expected path.

    Match is suffix-based on path-component boundaries: returned must equal
    expected or end with "/" + expected. Empty paths never match.
    """
    if not returned or not expected:
        return False
    r = normalize_path(returned)
    e = normalize_path(expected)
    if not r or not e:
        return False
    return r == e or r.endswith("/" + e)


_TOKEN_SPLIT_RE = re.compile(r"[._\-/\s:]+")
_CAMEL_TOKEN_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)


def tokenize_symbol(name: str) -> set[str]:
    """Tokenize a symbol name into lowercase tokens.

    Splits on common separators (`.`, `_`, `-`, `/`, whitespace, `:`) and on
    camelCase/PascalCase boundaries. Returns the set of lowercased tokens so
    we can compare on whole-token boundaries instead of raw substring.
    """
    if not name:
        return set()
    tokens: set[str] = set()
    for chunk in _TOKEN_SPLIT_RE.split(name):
        if not chunk:
            continue
        for piece in _CAMEL_TOKEN_RE.findall(chunk):
            if piece:
                tokens.add(piece.lower())
    return tokens


def symbol_matches(returned_symbols: list[str], expected: str) -> bool:
    """Check if any returned symbol matches the expected symbol.

    Match is token-boundary aware: every token of `expected` (after splitting
    on separators and camelCase) must appear as a whole token of a returned
    symbol. This prevents false positives like "get" matching "target" or
    "forget" while still allowing "get" to match "getUser".
    """
    expected_tokens = tokenize_symbol(expected)
    if not expected_tokens:
        return False
    for sym in returned_symbols:
        if not sym:
            continue
        sym_tokens = tokenize_symbol(sym)
        if expected_tokens.issubset(sym_tokens):
            return True
    return False


def compute_hit_at_k(results: list[QueryResult], k: int) -> float:
    """Compute Hit@K: fraction of queries where the correct file is in top-K.

    Returns a float in [0, 1]. By construction (top-K is a prefix of top-K'
    for k <= k'), Hit@K is monotonically non-decreasing in k.
    """
    if not results:
        return 0.0
    if k <= 0:
        return 0.0
    hits = 0
    for r in results:
        if not r.expected_files:
            continue
        top_k_files = r.returned_files[:k]
        for expected in r.expected_files:
            if any(file_matches(ret, expected) for ret in top_k_files):
                hits += 1
                break
    return hits / len(results)


def compute_symbol_hit_at_k(results: list[QueryResult], k: int) -> float:
    """Compute Symbol Hit@K averaged over queries that have expected symbols.

    Returns a float in [0, 1]. Queries without expected symbols are excluded
    from the denominator.
    """
    if not results or k <= 0:
        return 0.0
    hits = 0
    total_with_symbols = 0
    for r in results:
        if not r.expected_symbols:
            continue
        total_with_symbols += 1
        top_k_symbols = r.returned_symbols[:k]
        for expected in r.expected_symbols:
            if symbol_matches(top_k_symbols, expected):
                hits += 1
                break
    return hits / total_with_symbols if total_with_symbols else 0.0


def compute_mrr(results: list[QueryResult]) -> float:
    """Compute Mean Reciprocal Rank over queries that have expected files.

    Returns a float in [0, 1]. For each query the reciprocal of the rank of
    the first matching returned file is used (0 if no match). The mean is
    over queries with at least one expected file so the value lies in (0, 1]
    whenever any query produced a hit.
    """
    if not results:
        return 0.0
    rr_sum = 0.0
    total = 0
    for r in results:
        if not r.expected_files:
            continue
        total += 1
        for rank, ret_file in enumerate(r.returned_files, 1):
            if any(file_matches(ret_file, exp) for exp in r.expected_files):
                rr_sum += 1.0 / rank
                break
    return rr_sum / total if total else 0.0


def compute_percentile(values: list[float], p: float) -> float:
    """Compute percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return sorted_vals[lower]
    frac = idx - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


def compute_metrics(
    results: list[QueryResult],
    ingest_total_sec: float = 0.0,
    ingest_total_files: int = 0,
    index_size_mb: float = 0.0,
    ram_peak_mb: float = 0.0,
) -> BenchmarkMetrics:
    """Compute all metrics from query results."""
    latencies = [r.latency_ms for r in results]

    metrics = BenchmarkMetrics(
        hit_at_1=compute_hit_at_k(results, 1),
        hit_at_3=compute_hit_at_k(results, 3),
        hit_at_5=compute_hit_at_k(results, 5),
        hit_at_10=compute_hit_at_k(results, 10),
        symbol_hit_at_5=compute_symbol_hit_at_k(results, 5),
        mrr=compute_mrr(results),
        query_latency_p50_ms=compute_percentile(latencies, 50),
        query_latency_p95_ms=compute_percentile(latencies, 95),
        query_latency_p99_ms=compute_percentile(latencies, 99),
        query_latency_mean_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        ingest_total_sec=ingest_total_sec,
        ingest_files_per_sec=ingest_total_files / ingest_total_sec if ingest_total_sec > 0 else 0.0,
        ingest_total_files=ingest_total_files,
        index_size_mb=index_size_mb,
        ram_peak_mb=ram_peak_mb,
        avg_tool_calls=sum(r.tool_calls for r in results) / len(results) if results else 0.0,
        total_queries=len(results),
        total_hits=sum(1 for r in results if any(
            any(file_matches(ret, exp) for ret in r.returned_files[:5])
            for exp in r.expected_files
        )),
    )

    # Breakdowns
    for group_key, group_fn in [
        ("by_difficulty", lambda r: r.difficulty),
        ("by_type", lambda r: r.query_type),
    ]:
        groups: dict[str, list[QueryResult]] = {}
        for r in results:
            key = group_fn(r)
            groups.setdefault(key, []).append(r)

        breakdown = {}
        for key, group in groups.items():
            group_latencies = [r.latency_ms for r in group]
            breakdown[key] = {
                "count": len(group),
                "hit_at_5": compute_hit_at_k(group, 5),
                "symbol_hit_at_5": compute_symbol_hit_at_k(group, 5),
                "mrr": compute_mrr(group),
                "latency_p50_ms": compute_percentile(group_latencies, 50),
            }
        setattr(metrics, group_key, breakdown)

    # Composite score
    latency_score = 1.0 / (1.0 + metrics.query_latency_p95_ms / 1000.0)
    resource_score = 1.0 / (1.0 + (metrics.ram_peak_mb + metrics.index_size_mb) / 1000.0)
    tool_score = 1.0 / (1.0 + metrics.avg_tool_calls)

    metrics.composite_score = (
        0.30 * metrics.hit_at_5
        + 0.15 * metrics.symbol_hit_at_5
        + 0.15 * metrics.mrr
        + 0.15 * tool_score
        + 0.15 * latency_score
        + 0.10 * resource_score
    )

    return metrics
