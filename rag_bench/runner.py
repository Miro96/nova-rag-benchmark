"""Benchmark runner — orchestrates the full pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import time
import uuid
from pathlib import Path

from rag_bench import __version__
from rag_bench.adapter import RAGAdapter
from rag_bench.base_client import BaseClient
from rag_bench.datasets.loader import (
    Query,
    clone_repo,
    get_repo_files,
    load_queries,
    load_repos,
    load_warmup_queries,
)
from rag_bench.mcp_client import create_client as create_mcp_client
from rag_bench.cli_client import CLIClient
from rag_bench.inprocess_client import InProcessClient
from rag_bench.metrics import (
    BenchmarkMetrics,
    MemorySampler,
    QueryResult,
    compute_iqr,
    compute_metrics,
    directory_size_mb,
    file_matches,
    symbol_matches,
)
from rag_bench.report import print_results_table

logger = logging.getLogger(__name__)


def _nova_rag_data_dir() -> Path:
    """Resolve the on-disk index directory used by nova-rag-style servers."""
    return Path(os.getenv("NOVA_RAG_DATA_DIR", str(Path.home() / ".nova-rag")))


_SUPPORTED_TRANSPORTS = ("mcp", "cli", "inprocess")
_LEGACY_TRANSPORT_ALIASES = {"stdio": "mcp", "sse": "mcp"}


def _create_client(server_config: dict) -> BaseClient:
    """Create the appropriate client based on the preset's ``transport`` field.

    Defaults to ``"mcp"`` for backward compatibility with presets that
    don't specify a transport.
    """
    transport = (server_config.get("transport") or "mcp").lower()
    # Map legacy names
    transport = _LEGACY_TRANSPORT_ALIASES.get(transport, transport)

    if transport == "mcp":
        return create_mcp_client(server_config)

    if transport == "cli":
        command = server_config.get("command", "")
        if not command:
            raise RuntimeError(
                "CLI transport requires a 'command' field in the preset"
            )
        args = server_config.get("args", [])
        env = server_config.get("env")
        return CLIClient(command=command, args=args, env=env)

    if transport == "inprocess":
        module_path = server_config.get("module", "")
        class_name = server_config.get("class", "")
        if not module_path or not class_name:
            raise RuntimeError(
                "InProcess transport requires 'module' and 'class' fields in the preset"
            )
        return InProcessClient(module_path=module_path, class_name=class_name)

    raise RuntimeError(
        f"Unknown transport type '{transport}'. "
        f"Supported transports: {', '.join(_SUPPORTED_TRANSPORTS)}"
    )


def clean_nova_rag_index(data_dir: Path | None = None) -> Path:
    """Delete the nova-rag index directory if it exists.

    Returns the path that was (or would have been) cleaned. Wiping the
    whole directory clears state from previous runs across different repos
    so a ``--clean-index`` run starts cold.
    """
    target = data_dir or _nova_rag_data_dir()
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    return target


async def run_benchmark(
    server_config: dict,
    repo_filter: str | None = None,
    ab_baseline: bool = False,
    top_k: int = 10,
    replicates: int = 3,
    clean_index: bool = False,
) -> dict:
    """Run the full benchmark pipeline."""
    run_id = str(uuid.uuid4())
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("=== rag-bench run %s ===", run_id)
    logger.info("Server: %s", server_config.get("name", "custom"))
    logger.info("Replicates: %d  clean_index=%s", replicates, clean_index)

    # Load repos and queries
    repos = load_repos()
    if repo_filter:
        repos = [r for r in repos if r.name == repo_filter]
    queries = load_queries(repo_filter)

    if not queries:
        raise RuntimeError(f"No queries found{f' for repo {repo_filter}' if repo_filter else ''}")

    warmup_queries = load_warmup_queries()

    # Clone repos
    repo_dirs: dict[str, Path] = {}
    for repo in repos:
        try:
            repo_dirs[repo.name] = clone_repo(repo)
        except Exception as e:
            logger.error("Failed to clone %s: %s", repo.name, e)
            raise RuntimeError(
                f"Git clone failed for '{repo.name}' ({repo.git_url}): {e}"
            ) from e

    # Optionally wipe pre-existing index state for a fresh ingest measurement
    if clean_index:
        cleaned = clean_nova_rag_index()
        logger.info("Cleaned index dir at %s", cleaned)

    # Create the appropriate client based on transport type
    client = _create_client(server_config)

    # Start MCP server (if MCP transport)
    sampler: MemorySampler | None = None
    ram_peak = 0.0
    ingest_sec = 0.0
    total_files = 0
    index_size_mb = 0.0
    startup_ms = 0.0
    detected_tools: dict[str, str | None] = {}

    replicate_results: list[list[QueryResult]] = []
    replicate_metrics: list[BenchmarkMetrics] = []

    try:
        if hasattr(client, '__aenter__'):
            # MCPClient uses async context manager (start/stop lifecycle)
            async with client:
                startup_t0 = time.perf_counter()
                adapter = RAGAdapter(client, server_config)
                detected = await adapter.detect_tools()
                detected_tools = dict(detected)
                startup_ms = (time.perf_counter() - startup_t0) * 1000.0
                logger.info("Detected tools: %s", detected)
                logger.info("Startup (init + tools/list): %.1f ms", startup_ms)

                # === INGEST PHASE ===
                logger.info("=== INGEST PHASE ===")
                process = client._process
                sampler = MemorySampler(process.pid if process else None)
                sampler.start()

                ingest_start = time.perf_counter()

                for repo in repos:
                    repo_dir = repo_dirs[repo.name]
                    files = get_repo_files(repo_dir)
                    logger.info("Ingesting %s (%d files)...", repo.name, len(files))

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

                index_size_mb = _estimate_index_size(server_config, repo_dirs)

                logger.info(
                    "Ingest done: %d files in %.1fs (%.1f files/s)",
                    total_files, ingest_sec,
                    total_files / ingest_sec if ingest_sec > 0 else 0,
                )

                # === WARMUP ===
                logger.info("=== WARMUP (%d queries) ===", len(warmup_queries))
                for w in warmup_queries:
                    for repo in repos:
                        repo_dir = repo_dirs[repo.name]
                        try:
                            await adapter.query(w.query, top_k=top_k, path=str(repo_dir))
                        except Exception as e:
                            logger.debug("Warmup query failed: %s", e)

                # === BENCHMARK PHASE (N replicates) ===
                for rep in range(replicates):
                    logger.info(
                        "=== REPLICATE %d/%d (%d queries) ===",
                        rep + 1, replicates, len(queries),
                    )
                    results = await _run_query_pass(
                        adapter, client, queries, repo_dirs, top_k=top_k, replicate=rep,
                    )
                    replicate_results.append(results)
                    replicate_metrics.append(
                        compute_metrics(
                            results,
                            ingest_total_sec=ingest_sec,
                            ingest_total_files=total_files,
                            index_size_mb=index_size_mb,
                            ram_peak_mb=0.0,
                        )
                    )
        else:
            # CLI / InProcess — no async context manager needed
            startup_t0 = time.perf_counter()
            adapter = RAGAdapter(client, server_config)
            detected = await adapter.detect_tools()
            detected_tools = dict(detected)
            startup_ms = (time.perf_counter() - startup_t0) * 1000.0
            logger.info("Detected tools: %s", detected)
            logger.info("Startup (init + tools/list): %.1f ms", startup_ms)

            # === INGEST PHASE ===
            logger.info("=== INGEST PHASE ===")
            sampler = MemorySampler(None)
            sampler.start()

            ingest_start = time.perf_counter()

            for repo in repos:
                repo_dir = repo_dirs[repo.name]
                files = get_repo_files(repo_dir)
                logger.info("Ingesting %s (%d files)...", repo.name, len(files))

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

            index_size_mb = _estimate_index_size(server_config, repo_dirs)

            logger.info(
                "Ingest done: %d files in %.1fs (%.1f files/s)",
                total_files, ingest_sec,
                total_files / ingest_sec if ingest_sec > 0 else 0,
            )

            # === WARMUP ===
            logger.info("=== WARMUP (%d queries) ===", len(warmup_queries))
            for w in warmup_queries:
                for repo in repos:
                    repo_dir = repo_dirs[repo.name]
                    try:
                        await adapter.query(w.query, top_k=top_k, path=str(repo_dir))
                    except Exception as e:
                        logger.debug("Warmup query failed: %s", e)

            # === BENCHMARK PHASE (N replicates) ===
            for rep in range(replicates):
                logger.info(
                    "=== REPLICATE %d/%d (%d queries) ===",
                    rep + 1, replicates, len(queries),
                )
                results = await _run_query_pass(
                    adapter, client, queries, repo_dirs, top_k=top_k, replicate=rep,
                )
                replicate_results.append(results)
                replicate_metrics.append(
                    compute_metrics(
                        results,
                        ingest_total_sec=ingest_sec,
                        ingest_total_files=total_files,
                        index_size_mb=index_size_mb,
                        ram_peak_mb=0.0,
                    )
                )
    finally:
        if sampler is not None:
            ram_peak = sampler.stop()

    # === A/B BASELINE (grep/glob agent, optional) ===
    baseline_result: dict | None = None
    if ab_baseline:
        logger.info("=== A/B BASELINE ===")
        try:
            baseline_result = await _run_baseline_pass(
                queries, repo_dirs, server_config=server_config,
            )
            logger.info("Baseline complete: Hit@5=%.4f",
                        baseline_result.get("hit_at_5", 0))
        except Exception as e:
            logger.warning(
                "A/B baseline failed (DeepSeek API may be unavailable): %s", e,
            )
            logger.warning("Continuing with RAG-only metrics.")
            baseline_result = None

    # === COMPUTE FINAL METRICS (median over replicates) ===
    final_query_results = replicate_results[-1] if replicate_results else []
    final_metrics = compute_metrics(
        final_query_results,
        ingest_total_sec=ingest_sec,
        ingest_total_files=total_files,
        index_size_mb=index_size_mb,
        ram_peak_mb=ram_peak,
    )

    # Replace top-level (and breakdown) numeric values with the median across
    # replicates. The single-replicate object above gives us the structural
    # template; we then overwrite values metric-by-metric.
    median_metrics = _median_metrics(replicate_metrics, fallback=final_metrics)
    final_metrics = _apply_median(final_metrics, median_metrics)
    final_metrics.ram_peak_mb = ram_peak

    # Print results
    print_results_table(server_config.get("name", "custom"), final_metrics)

    # Build result JSON
    server_info = getattr(client, "server_info", None) or {}
    result = _build_result_json(
        run_id=run_id,
        server_config=server_config,
        server_info=server_info,
        metrics=final_metrics,
        query_results=final_query_results,
        repos=repos,
        replicate_metrics=replicate_metrics,
        startup_ms=startup_ms,
        detected_tools=detected_tools,
        baseline_result=baseline_result,
    )
    return result


async def _run_query_pass(
    adapter: RAGAdapter,
    client,
    queries: list[Query],
    repo_dirs: dict[str, Path],
    top_k: int,
    replicate: int,
    query_timeout: float = 120.0,
) -> list[QueryResult]:
    """Run all queries once and return per-query results.

    Each query is guarded by ``query_timeout`` (default 120s). When a query
    exceeds the timeout it is marked with ``error="timeout"`` and a
    ``latency_ms`` of ``query_timeout * 1000`` so it does not silently
    vanish from the output.
    """
    results: list[QueryResult] = []
    for i, q in enumerate(queries):
        repo_dir = repo_dirs.get(q.repo)
        path_arg = str(repo_dir) if repo_dir else None
        calls_before = client.call_count
        try:
            raw_result = await asyncio.wait_for(
                adapter.query_raw(q.query, top_k=top_k, path=path_arg),
                timeout=query_timeout,
            )
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
                tool_calls=client.call_count - calls_before,
                repo=q.repo,
            )

            qr.found_file = any(
                any(file_matches(ret, exp) for ret in returned_files[:5])
                for exp in q.expected_files
            )
            qr.found_symbol = any(
                symbol_matches(returned_symbols[:5], exp)
                for exp in q.expected_symbols
            ) if q.expected_symbols else False

            results.append(qr)

            status = "✓" if qr.found_file else "✗"
            logger.info(
                "  rep%d [%d/%d] %s %s (%.0fms)",
                replicate + 1, i + 1, len(queries), status, q.id, raw_result.latency_ms,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "  rep%d [%d/%d] TIMEOUT %s (%.0fs limit)",
                replicate + 1, i + 1, len(queries), q.id, query_timeout,
            )
            results.append(QueryResult(
                query_id=q.id,
                query_text=q.query,
                query_type=q.type,
                difficulty=q.difficulty,
                expected_files=q.expected_files,
                expected_symbols=q.expected_symbols,
                returned_files=[],
                returned_symbols=[],
                latency_ms=query_timeout * 1000,
                tool_calls=client.call_count - calls_before,
                repo=q.repo,
                error="timeout",
            ))
        except (ConnectionError, BrokenPipeError, ProcessLookupError) as e:
            # MCP server crash — mark as server_error, subsequent queries
            # will also fail, but we'll still produce partial results.
            logger.warning(
                "  rep%d [%d/%d] SERVER_CRASH %s: %s",
                replicate + 1, i + 1, len(queries), q.id, e,
            )
            results.append(QueryResult(
                query_id=q.id,
                query_text=q.query,
                query_type=q.type,
                difficulty=q.difficulty,
                expected_files=q.expected_files,
                expected_symbols=q.expected_symbols,
                returned_files=[],
                returned_symbols=[],
                latency_ms=0,
                tool_calls=client.call_count - calls_before,
                repo=q.repo,
                error="server_error",
            ))
        except Exception as e:
            logger.warning(
                "  rep%d [%d/%d] ERROR %s: %s",
                replicate + 1, i + 1, len(queries), q.id, e,
            )
            results.append(QueryResult(
                query_id=q.id,
                query_text=q.query,
                query_type=q.type,
                difficulty=q.difficulty,
                expected_files=q.expected_files,
                expected_symbols=q.expected_symbols,
                returned_files=[],
                returned_symbols=[],
                latency_ms=0,
                tool_calls=client.call_count - calls_before,
                repo=q.repo,
                error=str(e)[:200],
            ))
    return results


_MEDIAN_FIELDS: tuple[str, ...] = (
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_10",
    "symbol_hit_at_5",
    "mrr",
    "query_latency_p50_ms",
    "query_latency_p95_ms",
    "query_latency_p99_ms",
    "query_latency_mean_ms",
    "avg_tool_calls",
    "composite_score",
)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _median_metrics(
    reps: list[BenchmarkMetrics],
    fallback: BenchmarkMetrics,
) -> dict[str, float]:
    if not reps:
        return {f: getattr(fallback, f) for f in _MEDIAN_FIELDS}
    return {
        f: _median([getattr(r, f) for r in reps]) for f in _MEDIAN_FIELDS
    }


def _apply_median(
    target: BenchmarkMetrics, median: dict[str, float]
) -> BenchmarkMetrics:
    for f, v in median.items():
        setattr(target, f, v)
    return target


def _replicate_summary(reps: list[BenchmarkMetrics]) -> list[dict]:
    summary = []
    for i, m in enumerate(reps):
        summary.append({
            "index": i,
            "hit_at_1": round(m.hit_at_1, 4),
            "hit_at_3": round(m.hit_at_3, 4),
            "hit_at_5": round(m.hit_at_5, 4),
            "hit_at_10": round(m.hit_at_10, 4),
            "symbol_hit_at_5": round(m.symbol_hit_at_5, 4),
            "mrr": round(m.mrr, 4),
            "latency_p50_ms": round(m.query_latency_p50_ms, 1),
            "latency_p95_ms": round(m.query_latency_p95_ms, 1),
            "latency_p99_ms": round(m.query_latency_p99_ms, 1),
            "latency_mean_ms": round(m.query_latency_mean_ms, 1),
            "avg_tool_calls": round(m.avg_tool_calls, 2),
            "composite_score": round(m.composite_score, 4),
        })
    return summary


def _replicate_iqr(reps: list[BenchmarkMetrics]) -> dict[str, float]:
    return {
        "hit_at_5": round(compute_iqr([r.hit_at_5 for r in reps]), 4),
        "symbol_hit_at_5": round(compute_iqr([r.symbol_hit_at_5 for r in reps]), 4),
        "mrr": round(compute_iqr([r.mrr for r in reps]), 4),
        "latency_p50_ms": round(compute_iqr([r.query_latency_p50_ms for r in reps]), 1),
        "latency_p95_ms": round(compute_iqr([r.query_latency_p95_ms for r in reps]), 1),
        "latency_mean_ms": round(compute_iqr([r.query_latency_mean_ms for r in reps]), 1),
        "composite_score": round(compute_iqr([r.composite_score for r in reps]), 4),
    }


def _estimate_index_size(
    server_config: dict,
    repo_dirs: dict[str, Path] | None = None,
) -> float:
    """Estimate the on-disk size of the server's index in MB.

    Resolution order (first non-zero result wins):
      1. ``server_config['index_dir']`` — explicit override.
      2. ``server_config['index_dirs']`` — list of paths to sum.
      3. Per-repo ``<repo>/.nova-rag`` (if a repo's project keeps state in tree).
      4. ``$NOVA_RAG_DATA_DIR`` if set.
      5. ``~/.nova-rag/`` — nova-rag's default cache location.

    Returns 0.0 if no index can be found on disk.
    """
    explicit = server_config.get("index_dir")
    if explicit:
        size = directory_size_mb(explicit)
        if size > 0:
            return size

    paths = server_config.get("index_dirs") or []
    if paths:
        total = sum(directory_size_mb(p) for p in paths)
        if total > 0:
            return total

    if repo_dirs:
        per_repo = sum(
            directory_size_mb(Path(d) / ".nova-rag") for d in repo_dirs.values()
        )
        if per_repo > 0:
            return per_repo

    env_dir = os.getenv("NOVA_RAG_DATA_DIR")
    if env_dir:
        size = directory_size_mb(env_dir)
        if size > 0:
            return size

    return directory_size_mb(Path.home() / ".nova-rag")


def _build_result_json(
    run_id: str,
    server_config: dict,
    metrics: BenchmarkMetrics,
    query_results: list[QueryResult],
    repos,
    replicate_metrics: list[BenchmarkMetrics],
    startup_ms: float,
    detected_tools: dict[str, str | None],
    baseline_result: dict | None = None,
    server_info: dict[str, str] | None = None,
) -> dict:
    # Resolve server name/version: prefer the MCP initialize handshake,
    # fall back to the preset config.
    si = server_info or {}
    server_name = si.get("name") or server_config.get("name", "custom")
    server_version = si.get("version") or server_config.get("version", "")
    result = {
        "run_id": run_id,
        "bench_version": __version__,
        "dataset_version": "v1",
        "server": {
            "name": server_name,
            "git_url": server_config.get("git_url", ""),
            "git_user": server_config.get("git_user", ""),
            "version": server_version,
            "detected_tools": detected_tools,
        },
        "environment": {
            "os": platform.system().lower(),
            "arch": platform.machine(),
            "python": platform.python_version(),
            "cpu": platform.processor() or "unknown",
            "startup_ms": round(startup_ms, 1),
        },
        "startup_ms": round(startup_ms, 1),
        "repos": [r.name for r in repos],
        "replicates": _replicate_summary(replicate_metrics),
        "iqr": _replicate_iqr(replicate_metrics),
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
        "by_repo": metrics.by_repo,
        "query_details": [
            _query_detail(qr) for qr in query_results
        ],
    }

    # Include A/B baseline results if available
    if baseline_result is not None:
        result["baseline"] = baseline_result

        # Build RAG metrics dict for delta computation
        rag_metrics = {
            "retrieval": {
                "hit_at_5": metrics.hit_at_5,
                "symbol_hit_at_5": metrics.symbol_hit_at_5,
                "mrr": metrics.mrr,
                "latency": {
                    "p50_ms": metrics.query_latency_p50_ms,
                    "p95_ms": metrics.query_latency_p95_ms,
                    "mean_ms": metrics.query_latency_mean_ms,
                },
            },
            "efficiency": {
                "avg_tool_calls": metrics.avg_tool_calls,
            },
            "composite_score": metrics.composite_score,
        }
        result["ab_comparison"] = _compute_ab_deltas(rag_metrics, baseline_result)
    else:
        result["baseline"] = None
        result["ab_comparison"] = None

    return result


def _query_detail(qr: QueryResult) -> dict:
    """Build a query_details entry, including error markers when present."""
    entry = {
        "id": qr.query_id,
        "type": qr.query_type,
        "difficulty": qr.difficulty,
        "repo": qr.repo,
        "found_file": qr.found_file,
        "found_symbol": qr.found_symbol,
        "latency_ms": round(qr.latency_ms, 1),
        "tool_calls": qr.tool_calls,
        "returned_files": qr.returned_files[:5],
        "returned_symbols": [s for s in qr.returned_symbols[:5] if s],
    }
    if qr.error:
        entry["error"] = qr.error
    return entry


async def _run_baseline_pass(
    queries: list[Query],
    repo_dirs: dict[str, Path],
    server_config: dict | None = None,
) -> dict | None:
    """Run the grep/glob baseline agent against all queries.

    When ``server_config`` contains a ``deepseek`` section with an API key
    the baseline uses the DeepSeek-powered agent; otherwise it falls back
    to the local grep/glob strategy from ``baseline.py``.

    Returns a dict with baseline retrieval metrics or ``None`` on failure.
    """
    from rag_bench.baseline import run_baseline_search
    from rag_bench.metrics import (
        QueryResult,
        compute_metrics,
    )

    baseline_results: list[QueryResult] = []
    deepseek_used = False

    # Check for DeepSeek API key
    deepseek_cfg = (server_config or {}).get("deepseek") or {}
    api_key = deepseek_cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY")

    if api_key:
        try:
            logger.info("Using DeepSeek-powered baseline agent")
            baseline_results = await _run_deepseek_baseline(
                queries, repo_dirs, api_key, deepseek_cfg,
            )
            deepseek_used = True
        except Exception as e:
            logger.warning(
                "DeepSeek baseline failed, falling back to local grep/glob: %s", e,
            )

    if not deepseek_used:
        logger.info("Using local grep/glob baseline agent")
        for q in queries:
            repo_dir = repo_dirs.get(q.repo)
            if not repo_dir:
                baseline_results.append(QueryResult(
                    query_id=q.id,
                    query_text=q.query,
                    query_type=q.type,
                    difficulty=q.difficulty,
                    expected_files=q.expected_files,
                    expected_symbols=q.expected_symbols,
                    returned_files=[],
                    returned_symbols=[],
                    latency_ms=0,
                    tool_calls=0,
                    repo=q.repo,
                    error="no_repo_dir",
                ))
                continue

            try:
                br = run_baseline_search(
                    query=q.query,
                    query_type=q.type,
                    repo_dir=repo_dir,
                    expected_symbols=q.expected_symbols,
                )
                qr = QueryResult(
                    query_id=q.id,
                    query_text=q.query,
                    query_type=q.type,
                    difficulty=q.difficulty,
                    expected_files=q.expected_files,
                    expected_symbols=q.expected_symbols,
                    returned_files=br.found_files,
                    returned_symbols=br.found_symbols,
                    latency_ms=br.total_time_ms,
                    tool_calls=br.tool_calls,
                    repo=q.repo,
                )
                qr.found_file = any(
                    any(file_matches(ret, exp) for ret in br.found_files[:5])
                    for exp in q.expected_files
                )
                qr.found_symbol = any(
                    symbol_matches(br.found_symbols[:5], exp)
                    for exp in q.expected_symbols
                ) if q.expected_symbols else False
                baseline_results.append(qr)
            except Exception as e:
                logger.debug("Baseline query %s failed: %s", q.id, e)
                baseline_results.append(QueryResult(
                    query_id=q.id,
                    query_text=q.query,
                    query_type=q.type,
                    difficulty=q.difficulty,
                    expected_files=q.expected_files,
                    expected_symbols=q.expected_symbols,
                    returned_files=[],
                    returned_symbols=[],
                    latency_ms=0,
                    tool_calls=0,
                    repo=q.repo,
                    error=str(e)[:200],
                ))

    metrics = compute_metrics(baseline_results, ingest_total_sec=0, ingest_total_files=0)
    return {
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
        "method": "deepseek" if deepseek_used else "grep_glob",
    }


async def _run_deepseek_baseline(
    queries: list[Query],
    repo_dirs: dict[str, Path],
    api_key: str,
    deepseek_cfg: dict,
) -> list[QueryResult]:
    """Run the DeepSeek-powered baseline agent for each query.

    For each query the agent receives a file listing of the target repo and
    can call grep / glob / read_file tools (via DeepSeek function calling) to
    find matching files — without any RAG or MCP involvement.

    Returns a list of ``QueryResult`` objects, one per query.
    """
    from rag_bench.baseline import DeepSeekBaselineAgent

    model = deepseek_cfg.get("model", "deepseek-v4-flash")
    base_url = deepseek_cfg.get("base_url", "https://api.deepseek.com/v1")
    max_iterations = deepseek_cfg.get("max_iterations", 5)
    timeout = deepseek_cfg.get("timeout", 120.0)

    agent = DeepSeekBaselineAgent(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_iterations=max_iterations,
        timeout=timeout,
    )

    results: list[QueryResult] = []

    for q in queries:
        repo_dir = repo_dirs.get(q.repo)
        if not repo_dir:
            results.append(QueryResult(
                query_id=q.id,
                query_text=q.query,
                query_type=q.type,
                difficulty=q.difficulty,
                expected_files=q.expected_files,
                expected_symbols=q.expected_symbols,
                returned_files=[],
                returned_symbols=[],
                latency_ms=0,
                tool_calls=0,
                repo=q.repo,
                error="no_repo_dir",
            ))
            continue

        try:
            sr = await agent.search(query=q.query, repo_dir=repo_dir)

            qr = QueryResult(
                query_id=q.id,
                query_text=q.query,
                query_type=q.type,
                difficulty=q.difficulty,
                expected_files=q.expected_files,
                expected_symbols=q.expected_symbols,
                returned_files=sr["found_files"],
                returned_symbols=sr["found_symbols"],
                latency_ms=sr["total_time_ms"],
                tool_calls=sr["tool_calls"],
                repo=q.repo,
            )

            qr.found_file = any(
                any(file_matches(ret, exp) for ret in sr["found_files"][:5])
                for exp in q.expected_files
            )
            qr.found_symbol = any(
                symbol_matches(sr["found_symbols"][:5], exp)
                for exp in q.expected_symbols
            ) if q.expected_symbols else False

            results.append(qr)
            logger.info(
                "  baseline %s %s (%.0fms, %d tool_calls)",
                "✓" if qr.found_file else "✗",
                q.id,
                sr["total_time_ms"],
                sr["tool_calls"],
            )

        except Exception as e:
            logger.warning("Baseline query %s failed: %s", q.id, e)
            results.append(QueryResult(
                query_id=q.id,
                query_text=q.query,
                query_type=q.type,
                difficulty=q.difficulty,
                expected_files=q.expected_files,
                expected_symbols=q.expected_symbols,
                returned_files=[],
                returned_symbols=[],
                latency_ms=0,
                tool_calls=0,
                repo=q.repo,
                error=str(e)[:200],
            ))

    return results


def _compute_ab_deltas(
    rag_metrics: dict,
    baseline_metrics: dict | None,
) -> dict | None:
    """Compute per-metric deltas between RAG and baseline.

    Deltas are ``baseline - rag`` for latency / tool_calls (so a positive
    value means RAG is faster / more efficient), and ``rag - baseline`` for
    quality metrics (so positive means RAG is better).

    Returns ``None`` when *baseline_metrics* is ``None``.
    """
    if baseline_metrics is None:
        return None

    b_ret = baseline_metrics.get("retrieval", {})
    b_eff = baseline_metrics.get("efficiency", {})
    b_lat = b_ret.get("latency", {})

    r_ret = rag_metrics.get("retrieval", {})
    r_eff = rag_metrics.get("efficiency", {})
    r_lat = r_ret.get("latency", {})

    def _d(val: float | None, precision: int = 4) -> float | None:
        return round(val, precision) if val is not None else None

    # Quality: rag - baseline (positive = RAG better)
    hit_at_5_delta = _d(r_ret.get("hit_at_5", 0) - b_ret.get("hit_at_5", 0))
    symbol_hit_at_5_delta = _d(r_ret.get("symbol_hit_at_5", 0) - b_ret.get("symbol_hit_at_5", 0))
    mrr_delta = _d(r_ret.get("mrr", 0) - b_ret.get("mrr", 0))

    # Latency: baseline - rag (positive = RAG faster)
    latency_p50_delta = _d(b_lat.get("p50_ms", 0) - r_lat.get("p50_ms", 0), 1)
    latency_p95_delta = _d(b_lat.get("p95_ms", 0) - r_lat.get("p95_ms", 0), 1)
    latency_mean_delta = _d(b_lat.get("mean_ms", 0) - r_lat.get("mean_ms", 0), 1)

    # Efficiency: baseline tool_calls - rag tool_calls (positive = RAG more efficient)
    tool_calls_delta = _d(b_eff.get("avg_tool_calls", 0) - r_eff.get("avg_tool_calls", 0), 2)

    # Composite: rag - baseline
    composite_score_delta = _d(
        rag_metrics.get("composite_score", 0) - baseline_metrics.get("composite_score", 0),
    )

    return {
        "hit_at_5_delta": hit_at_5_delta,
        "symbol_hit_at_5_delta": symbol_hit_at_5_delta,
        "mrr_delta": mrr_delta,
        "latency_p50_delta": latency_p50_delta,
        "latency_p95_delta": latency_p95_delta,
        "latency_mean_delta": latency_mean_delta,
        "tool_calls_delta": tool_calls_delta,
        "composite_score_delta": composite_score_delta,
    }
