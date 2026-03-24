"""Benchmark runner — orchestrates the full pipeline."""

from __future__ import annotations

import logging
import platform
import time
import uuid
from pathlib import Path

import psutil

from rag_bench import __version__
from rag_bench.adapter import RAGAdapter
from rag_bench.datasets.loader import (
    Query,
    clone_repo,
    get_repo_files,
    load_queries,
    load_repos,
)
from rag_bench.mcp_client import MCPClient, create_client
from rag_bench.metrics import (
    BenchmarkMetrics,
    QueryResult,
    compute_metrics,
    file_matches,
    symbol_matches,
)
from rag_bench.report import print_results_table

logger = logging.getLogger(__name__)


async def run_benchmark(
    server_config: dict,
    repo_filter: str | None = None,
    ab_baseline: bool = False,
    top_k: int = 10,
) -> dict:
    """Run the full benchmark pipeline."""
    run_id = uuid.uuid4().hex[:12]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("=== rag-bench run %s ===", run_id)
    logger.info("Server: %s", server_config.get("name", "custom"))

    # Load repos and queries
    repos = load_repos()
    if repo_filter:
        repos = [r for r in repos if r.name == repo_filter]
    queries = load_queries(repo_filter)

    if not queries:
        raise RuntimeError(f"No queries found{f' for repo {repo_filter}' if repo_filter else ''}")

    # Clone repos
    repo_dirs: dict[str, Path] = {}
    for repo in repos:
        repo_dirs[repo.name] = clone_repo(repo)

    # Start MCP server
    client = create_client(server_config)
    async with client:
        adapter = RAGAdapter(client, server_config)
        detected = await adapter.detect_tools()
        logger.info("Detected tools: %s", detected)

        # === INGEST PHASE ===
        logger.info("=== INGEST PHASE ===")
        process = client._process
        ram_before = _get_process_memory(process.pid) if process else 0

        ingest_start = time.perf_counter()
        total_files = 0

        for repo in repos:
            repo_dir = repo_dirs[repo.name]
            files = get_repo_files(repo_dir)
            logger.info("Ingesting %s (%d files)...", repo.name, len(files))

            # Try directory ingest first
            try:
                await adapter.ingest_directory(str(repo_dir))
                total_files += len(files)
                logger.info("Ingested %s via directory ingest", repo.name)
            except (RuntimeError, Exception) as e:
                logger.info("Directory ingest failed (%s), trying file-by-file...", e)
                for f in files:
                    try:
                        await adapter.ingest_file(str(f))
                        total_files += 1
                    except Exception as fe:
                        logger.debug("Failed to ingest %s: %s", f, fe)

        ingest_sec = time.perf_counter() - ingest_start
        ram_after = _get_process_memory(process.pid) if process else 0
        ram_peak = max(ram_after, ram_before)

        # Measure index size
        index_size_mb = _estimate_index_size(server_config)

        logger.info(
            "Ingest done: %d files in %.1fs (%.1f files/s)",
            total_files, ingest_sec, total_files / ingest_sec if ingest_sec > 0 else 0,
        )

        # === WARMUP ===
        logger.info("=== WARMUP (5 queries) ===")
        for q in queries[:5]:
            try:
                await adapter.query(q.query, top_k=top_k)
            except Exception as e:
                logger.debug("Warmup query failed: %s", e)

        # === BENCHMARK PHASE ===
        logger.info("=== BENCHMARK PHASE (%d queries) ===", len(queries))
        query_results = []

        for i, q in enumerate(queries):
            try:
                raw_result = await adapter.query_raw(q.query, top_k=top_k)
                search_results = adapter._parse_search_results(raw_result)

                returned_files = [r.file_path for r in search_results]
                returned_symbols = [r.symbol for r in search_results if r.symbol]

                qr = QueryResult(
                    query_id=q.id,
                    query_text=q.query,
                    query_type=q.type,
                    difficulty=q.difficulty,
                    expected_files=q.expected_files,
                    expected_symbols=q.expected_symbols,
                    returned_files=returned_files,
                    returned_symbols=returned_symbols,
                    latency_ms=raw_result.latency_ms,
                )

                # Check hits
                qr.found_file = any(
                    any(file_matches(ret, exp) for ret in returned_files[:5])
                    for exp in q.expected_files
                )
                qr.found_symbol = any(
                    symbol_matches(returned_symbols[:5], exp)
                    for exp in q.expected_symbols
                ) if q.expected_symbols else False

                query_results.append(qr)

                status = "✓" if qr.found_file else "✗"
                logger.info(
                    "  [%d/%d] %s %s (%.0fms)",
                    i + 1, len(queries), status, q.id, raw_result.latency_ms,
                )

            except Exception as e:
                logger.warning("  [%d/%d] ERROR %s: %s", i + 1, len(queries), q.id, e)
                query_results.append(QueryResult(
                    query_id=q.id,
                    query_text=q.query,
                    query_type=q.type,
                    difficulty=q.difficulty,
                    expected_files=q.expected_files,
                    expected_symbols=q.expected_symbols,
                    returned_files=[],
                    returned_symbols=[],
                    latency_ms=0,
                ))

    # === COMPUTE METRICS ===
    metrics = compute_metrics(
        query_results,
        ingest_total_sec=ingest_sec,
        ingest_total_files=total_files,
        index_size_mb=index_size_mb,
        ram_peak_mb=ram_peak,
    )

    # Print results
    print_results_table(server_config.get("name", "custom"), metrics)

    # Build result JSON
    result = _build_result_json(
        run_id, server_config, metrics, query_results, repos,
    )
    return result


def _get_process_memory(pid: int) -> float:
    """Get process memory in MB."""
    try:
        proc = psutil.Process(pid)
        mem = proc.memory_info()
        return mem.rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def _estimate_index_size(server_config: dict) -> float:
    """Try to estimate index size on disk."""
    # This is server-specific; return 0 if we can't detect
    return 0.0


def _build_result_json(
    run_id: str,
    server_config: dict,
    metrics: BenchmarkMetrics,
    query_results: list[QueryResult],
    repos,
) -> dict:
    return {
        "run_id": run_id,
        "bench_version": __version__,
        "dataset_version": "v1",
        "server": {
            "name": server_config.get("name", "custom"),
            "git_url": server_config.get("git_url", ""),
            "git_user": server_config.get("git_user", ""),
            "version": server_config.get("version", ""),
        },
        "environment": {
            "os": platform.system().lower(),
            "arch": platform.machine(),
            "python": platform.python_version(),
            "cpu": platform.processor() or "unknown",
        },
        "repos": [r.name for r in repos],
        "ingest": {
            "total_files": metrics.ingest_total_files,
            "total_sec": round(metrics.ingest_total_sec, 2),
            "files_per_sec": round(metrics.ingest_files_per_sec, 1),
            "index_size_mb": round(metrics.index_size_mb, 1),
            "ram_peak_mb": round(metrics.ram_peak_mb, 1),
        },
        "retrieval": {
            "total_queries": metrics.total_queries,
            "total_hits": metrics.total_hits,
            "hit_at_1": round(metrics.hit_at_1, 4),
            "hit_at_3": round(metrics.hit_at_3, 4),
            "hit_at_5": round(metrics.hit_at_5, 4),
            "hit_at_10": round(metrics.hit_at_10, 4),
            "symbol_hit_at_5": round(metrics.symbol_hit_at_5, 4),
            "mrr": round(metrics.mrr, 4),
            "latency": {
                "p50_ms": round(metrics.query_latency_p50_ms, 1),
                "p95_ms": round(metrics.query_latency_p95_ms, 1),
                "p99_ms": round(metrics.query_latency_p99_ms, 1),
                "mean_ms": round(metrics.query_latency_mean_ms, 1),
            },
        },
        "efficiency": {
            "avg_tool_calls": round(metrics.avg_tool_calls, 2),
        },
        "composite_score": round(metrics.composite_score, 4),
        "by_difficulty": metrics.by_difficulty,
        "by_type": metrics.by_type,
        "query_details": [
            {
                "id": qr.query_id,
                "type": qr.query_type,
                "difficulty": qr.difficulty,
                "found_file": qr.found_file,
                "found_symbol": qr.found_symbol,
                "latency_ms": round(qr.latency_ms, 1),
                "returned_files": qr.returned_files[:5],
            }
            for qr in query_results
        ],
    }
