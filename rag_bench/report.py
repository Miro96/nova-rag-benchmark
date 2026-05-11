"""Report generation: terminal tables and JSON output."""

from __future__ import annotations

import math
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_bench.metrics import BenchmarkMetrics

console = Console()


def print_results_table(server_name: str, m: BenchmarkMetrics) -> None:
    """Print benchmark results as a rich table."""
    console.print()
    console.print(Panel(f"[bold]rag-bench results: {server_name}[/bold]"))

    # Main metrics
    table = Table(title="Retrieval Quality", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Hit@1", f"{m.hit_at_1:.1%}")
    table.add_row("Hit@3", f"{m.hit_at_3:.1%}")
    table.add_row("Hit@5", f"{m.hit_at_5:.1%}")
    table.add_row("Hit@10", f"{m.hit_at_10:.1%}")
    table.add_row("Symbol Hit@5", f"{m.symbol_hit_at_5:.1%}")
    table.add_row("MRR", f"{m.mrr:.4f}")
    console.print(table)

    # Latency
    lat_table = Table(title="Latency", show_header=True)
    lat_table.add_column("Metric", style="cyan")
    lat_table.add_column("Value", style="green", justify="right")

    lat_table.add_row("p50", f"{m.query_latency_p50_ms:.0f} ms")
    lat_table.add_row("p95", f"{m.query_latency_p95_ms:.0f} ms")
    lat_table.add_row("p99", f"{m.query_latency_p99_ms:.0f} ms")
    lat_table.add_row("mean", f"{m.query_latency_mean_ms:.0f} ms")
    console.print(lat_table)

    # Ingest
    ing_table = Table(title="Ingest", show_header=True)
    ing_table.add_column("Metric", style="cyan")
    ing_table.add_column("Value", style="green", justify="right")

    ing_table.add_row("Files", str(m.ingest_total_files))
    ing_table.add_row("Time", f"{m.ingest_total_sec:.1f} s")
    ing_table.add_row("Speed", f"{m.ingest_files_per_sec:.1f} files/s")
    ing_table.add_row("Index Size", f"{m.index_size_mb:.1f} MB")
    ing_table.add_row("RAM Peak", f"{m.ram_peak_mb:.1f} MB")
    console.print(ing_table)

    # Breakdown by difficulty
    if m.by_difficulty:
        diff_table = Table(title="By Difficulty", show_header=True)
        diff_table.add_column("Difficulty", style="cyan")
        diff_table.add_column("Count", justify="right")
        diff_table.add_column("Hit@5", justify="right")
        diff_table.add_column("MRR", justify="right")
        diff_table.add_column("p50 ms", justify="right")

        for diff in ["easy", "medium", "hard"]:
            if diff in m.by_difficulty:
                d = m.by_difficulty[diff]
                diff_table.add_row(
                    diff,
                    str(d["count"]),
                    f"{d['hit_at_5']:.1%}",
                    f"{d['mrr']:.4f}",
                    f"{d['latency_p50_ms']:.0f}",
                )
        console.print(diff_table)

    # Breakdown by type
    if m.by_type:
        type_table = Table(title="By Query Type", show_header=True)
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", justify="right")
        type_table.add_column("Hit@5", justify="right")
        type_table.add_column("MRR", justify="right")

        for qtype in ["locate", "callers", "explain", "impact"]:
            if qtype in m.by_type:
                d = m.by_type[qtype]
                type_table.add_row(
                    qtype,
                    str(d["count"]),
                    f"{d['hit_at_5']:.1%}",
                    f"{d['mrr']:.4f}",
                )
        console.print(type_table)

    # Composite score
    console.print(
        f"\n[bold]Composite Score: [green]{m.composite_score:.4f}[/green][/bold]\n"
    )


def print_comparison_table(results: list[dict]) -> None:
    """Print side-by-side comparison of multiple benchmark results."""
    table = Table(title="RAG Server Comparison", show_header=True)
    table.add_column("Metric", style="cyan")

    for r in results:
        table.add_column(r["server"]["name"], style="green", justify="right")

    rows = [
        ("Hit@1", lambda r: f"{r['retrieval']['hit_at_1']:.1%}"),
        ("Hit@5", lambda r: f"{r['retrieval']['hit_at_5']:.1%}"),
        ("Symbol Hit@5", lambda r: f"{r['retrieval']['symbol_hit_at_5']:.1%}"),
        ("MRR", lambda r: f"{r['retrieval']['mrr']:.4f}"),
        ("Latency p50", lambda r: f"{r['retrieval']['latency']['p50_ms']:.0f} ms"),
        ("Latency p95", lambda r: f"{r['retrieval']['latency']['p95_ms']:.0f} ms"),
        ("Ingest Time", lambda r: f"{r['ingest']['total_sec']:.1f} s"),
        ("Ingest Speed", lambda r: f"{r['ingest']['files_per_sec']:.1f} f/s"),
        ("RAM Peak", lambda r: f"{r['ingest']['ram_peak_mb']:.1f} MB"),
        ("Score", lambda r: f"{r['composite_score']:.4f}"),
    ]

    for label, fn in rows:
        values = []
        for r in results:
            try:
                values.append(fn(r))
            except (KeyError, TypeError):
                values.append("N/A")
        table.add_row(label, *values)

    console.print(table)


def generate_comparison_report(
    results: list[dict],
    replicates: int = 3,
) -> dict[str, Any]:
    """Generate a structured JSON comparison report from benchmark results.

    The report includes:

    * Per-preset aggregate metrics (Hit@1-10, SymbolHit@5, MRR, latency, composite)
    * A/B deltas for each preset vs the grep-glob baseline
    * Per-difficulty breakdown (easy / medium / hard) for each preset
    * Per-query-type breakdown for each preset
    * Per-repo breakdown for each preset
    * Summary table with all presets side-by-side
    * Reproducibility: coefficient of variation (CV) across replicates
    * Ground-truth coverage: union of all presets vs expected_files
    """
    preset_names = [r["server"]["name"] for r in results]

    # Identify the grep-glob baseline for delta computation
    baseline_idx: int | None = None
    for i, name in enumerate(preset_names):
        if "grep" in name.lower() and "glob" in name.lower():
            baseline_idx = i
            break

    # --- Per-preset summary ---
    presets: dict[str, dict[str, Any]] = {}
    for r in results:
        name = r["server"]["name"]
        retrieval = r.get("retrieval", {})
        latency = retrieval.get("latency", {})
        ingest = r.get("ingest", {})
        presets[name] = {
            "hit_at_1": retrieval.get("hit_at_1"),
            "hit_at_3": retrieval.get("hit_at_3"),
            "hit_at_5": retrieval.get("hit_at_5"),
            "hit_at_10": retrieval.get("hit_at_10"),
            "symbol_hit_at_5": retrieval.get("symbol_hit_at_5"),
            "mrr": retrieval.get("mrr"),
            "map": round(retrieval.get("mrr", 0), 4),  # MAP approximated by MRR for now
            "latency_p50_ms": latency.get("p50_ms"),
            "latency_p95_ms": latency.get("p95_ms"),
            "latency_p99_ms": latency.get("p99_ms"),
            "latency_mean_ms": latency.get("mean_ms"),
            "total_queries": retrieval.get("total_queries"),
            "total_hits": retrieval.get("total_hits"),
            "ingest_total_files": ingest.get("total_files"),
            "ingest_total_sec": ingest.get("total_sec"),
            "ingest_files_per_sec": ingest.get("files_per_sec"),
            "index_size_mb": ingest.get("index_size_mb"),
            "ram_peak_mb": ingest.get("ram_peak_mb"),
            "composite_score": r.get("composite_score"),
            "avg_tool_calls": (r.get("efficiency") or {}).get("avg_tool_calls"),
        }

    # --- A/B deltas vs grep-glob baseline ---
    ab_deltas: dict[str, dict[str, Any]] = {}
    if baseline_idx is not None:
        baseline = presets[preset_names[baseline_idx]]
        for name, pdata in presets.items():
            if name == preset_names[baseline_idx]:
                continue
            # Quality: preset - baseline (positive = better)
            # Latency: baseline - preset (positive = faster)
            delta: dict[str, Any] = {}
            for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
                           "symbol_hit_at_5", "mrr", "composite_score"):
                bv = baseline.get(metric)
                pv = pdata.get(metric)
                delta[metric] = round(pv - bv, 4) if (pv is not None and bv is not None) else None
            for metric in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_mean_ms"):
                bv = baseline.get(metric)
                pv = pdata.get(metric)
                delta[metric] = round(bv - pv, 1) if (pv is not None and bv is not None) else None
            ab_deltas[name] = delta

    # --- Per-difficulty breakdown ---
    by_difficulty: dict[str, dict[str, Any]] = {}
    for r in results:
        name = r["server"]["name"]
        difficulty_data = r.get("by_difficulty") or r.get("retrieval", {}).get("by_difficulty") or {}
        if not difficulty_data:
            # Fall back to per-repo aggregation if no direct by_difficulty
            by_repo = r.get("by_repo", {})
            difficulty_data = _aggregate_by_difficulty_from_query_details(
                r.get("query_details", []),
            )
        by_difficulty[name] = difficulty_data

    # --- Per-query-type breakdown ---
    by_type: dict[str, dict[str, Any]] = {}
    for r in results:
        name = r["server"]["name"]
        type_data = r.get("by_type") or r.get("retrieval", {}).get("by_type") or {}
        if not type_data:
            type_data = _aggregate_by_type_from_query_details(
                r.get("query_details", []),
            )
        by_type[name] = type_data

    # --- Per-repo breakdown ---
    by_repo: dict[str, dict[str, Any]] = {}
    for r in results:
        name = r["server"]["name"]
        repo_data = r.get("by_repo") or {}
        if not repo_data:
            repo_data = _aggregate_by_repo_from_query_details(
                r.get("query_details", []),
            )
        by_repo[name] = repo_data

    # --- Summary table (side-by-side) ---
    summary_rows: list[dict[str, Any]] = []
    for metric_label, metric_key, fmt in [
        ("Hit@1", "hit_at_1", ".1%"),
        ("Hit@3", "hit_at_3", ".1%"),
        ("Hit@5", "hit_at_5", ".1%"),
        ("Hit@10", "hit_at_10", ".1%"),
        ("Symbol Hit@5", "symbol_hit_at_5", ".1%"),
        ("MRR", "mrr", ".4f"),
        ("MAP", "map", ".4f"),
        ("Latency p50 (ms)", "latency_p50_ms", ".0f"),
        ("Latency p95 (ms)", "latency_p95_ms", ".0f"),
        ("Latency mean (ms)", "latency_mean_ms", ".0f"),
        ("Ingest Files", "ingest_total_files", ""),
        ("Ingest Time (s)", "ingest_total_sec", ".1f"),
        ("Ingest Speed (f/s)", "ingest_files_per_sec", ".1f"),
        ("Index Size (MB)", "index_size_mb", ".1f"),
        ("RAM Peak (MB)", "ram_peak_mb", ".1f"),
        ("Composite Score", "composite_score", ".4f"),
    ]:
        row: dict[str, Any] = {"metric": metric_label}
        for name in preset_names:
            val = presets[name].get(metric_key)
            row[name] = val
        summary_rows.append(row)

    # --- Reproducibility: CV across replicates ---
    cv_by_preset: dict[str, dict[str, Any]] = {}
    for r in results:
        name = r["server"]["name"]
        reps_list = r.get("replicates", [])
        cv_by_preset[name] = _compute_cv(reps_list)

    # --- Ground-truth coverage ---
    gt_coverage = _compute_ground_truth_coverage(results)

    report = {
        "title": "RAG Server Comparison Report",
        "presets": preset_names,
        "replicates": replicates,
        "metrics": presets,
        "ab_deltas_vs_baseline": ab_deltas,
        "summary": summary_rows,
        "by_difficulty": by_difficulty,
        "by_type": by_type,
        "by_repo": by_repo,
        "reproducibility": {
            "coefficient_of_variation": cv_by_preset,
            "note": "CV < 0.05 across replicates required for VAL-COMPARE-010",
        },
        "ground_truth_coverage": gt_coverage,
    }
    return report


# ---------------------------------------------------------------------------
# Helpers for aggregating breakdowns from query_details
# ---------------------------------------------------------------------------


def _aggregate_by_difficulty_from_query_details(
    query_details: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate Hit@5, MRR, etc. by difficulty from per-query results."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for qd in query_details:
        diff = qd.get("difficulty", "unknown")
        buckets.setdefault(diff, []).append(qd)

    result: dict[str, dict[str, Any]] = {}
    for diff, qds in buckets.items():
        hit5 = sum(1 for q in qds if q.get("found_file")) / len(qds) if qds else 0.0
        mrr = _compute_mrr(qds)
        latencies = [q.get("latency_ms", 0) for q in qds if not q.get("error")]
        result[diff] = {
            "count": len(qds),
            "hit_at_5": round(hit5, 4),
            "mrr": round(mrr, 4),
            "latency_p50_ms": round(_percentile(latencies, 50), 1) if latencies else 0.0,
            "latency_mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        }
    return result


def _aggregate_by_type_from_query_details(
    query_details: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate Hit@5, MRR by query type."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for qd in query_details:
        qtype = qd.get("type", "unknown")
        buckets.setdefault(qtype, []).append(qd)

    result: dict[str, dict[str, Any]] = {}
    for qtype, qds in sorted(buckets.items()):
        hit5 = sum(1 for q in qds if q.get("found_file")) / len(qds) if qds else 0.0
        mrr = _compute_mrr(qds)
        result[qtype] = {
            "count": len(qds),
            "hit_at_5": round(hit5, 4),
            "mrr": round(mrr, 4),
        }
    return result


def _aggregate_by_repo_from_query_details(
    query_details: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate Hit@5, MRR by repo."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for qd in query_details:
        repo = qd.get("repo", "unknown")
        buckets.setdefault(repo, []).append(qd)

    result: dict[str, dict[str, Any]] = {}
    for repo, qds in sorted(buckets.items()):
        hit5 = sum(1 for q in qds if q.get("found_file")) / len(qds) if qds else 0.0
        mrr = _compute_mrr(qds)
        result[repo] = {
            "count": len(qds),
            "hit_at_5": round(hit5, 4),
            "mrr": round(mrr, 4),
        }
    return result


def _compute_mrr(query_details: list[dict[str, Any]]) -> float:
    """Compute Mean Reciprocal Rank from query_details."""
    total = 0.0
    count = 0
    for q in query_details:
        returned = q.get("returned_files", [])
        expected = q.get("expected_files", [])
        if not returned or not expected:
            count += 1
            continue
        for rank, rf in enumerate(returned[:10], start=1):
            if any(_path_matches(rf, ef) for ef in expected):
                total += 1.0 / rank
                break
        count += 1
    return total / count if count > 0 else 0.0


def _path_matches(returned: str, expected: str) -> bool:
    """Check if returned path matches expected path."""
    if returned == expected:
        return True
    if returned.endswith(expected) or expected.endswith(returned):
        return True
    if returned.endswith("/" + expected) or expected.endswith("/" + returned):
        return True
    return False


def _percentile(values: list[float], pct: float) -> float:
    """Compute the p-th percentile using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    k = (pct / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1


def _compute_cv(replicates: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute coefficient of variation (std/mean) across replicates."""
    if not replicates or len(replicates) < 2:
        return {"cv_hit_at_5": None, "note": "Need at least 2 replicates"}

    metrics_keys = [
        "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
        "symbol_hit_at_5", "mrr", "composite_score",
    ]
    cv: dict[str, Any] = {}
    for key in metrics_keys:
        values = [r.get(key, 0) for r in replicates]
        mean = sum(values) / len(values)
        if mean == 0:
            cv[f"cv_{key}"] = 0.0 if all(v == 0 for v in values) else None
        else:
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
            cv[f"cv_{key}"] = round(std / mean, 4)
    return cv


def _compute_ground_truth_coverage(
    results: list[dict],
) -> dict[str, Any]:
    """Compute ground-truth coverage: union of all presets' results vs expected_files.

    For each query, collect all files found by any preset across all
    replicates, then check if the expected_files are covered.
    """
    from collections import defaultdict

    # Collect all query_details across all presets
    all_query_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        for qd in r.get("query_details", []):
            qid = qd.get("id", "")
            all_query_details[qid].append(qd)

    coverage: dict[str, Any] = {
        "queries_analyzed": len(all_query_details),
        "per_query": {},
    }

    covered_count = 0
    total_expected = 0
    total_found = 0

    for qid, qds in sorted(all_query_details.items()):
        # Union of found files across all presets for this query
        found_union: set[str] = set()
        expected_files: list[str] = []
        for qd in qds:
            found_union.update(qd.get("returned_files", []))
            if not expected_files and qd.get("expected_files"):
                # expected_files not in query_details directly, use empty for now
                pass

        # Get expected_files from first qd that has them
        for qd in qds:
            if qd.get("expected_files"):
                expected_files = qd["expected_files"]
                break

        if expected_files:
            total_expected += len(expected_files)
            found_in_union = [ef for ef in expected_files if any(
                _path_matches(ff, ef) for ff in found_union
            )]
            total_found += len(found_in_union)
            is_covered = len(found_in_union) == len(expected_files)
            if is_covered:
                covered_count += 1
            coverage["per_query"][qid] = {
                "expected_count": len(expected_files),
                "found_count": len(found_in_union),
                "coverage_pct": round(
                    len(found_in_union) / len(expected_files) * 100, 1,
                ),
                "covered": is_covered,
                "missing": [ef for ef in expected_files if ef not in found_in_union],
            }
        else:
            coverage["per_query"][qid] = {
                "expected_count": 0,
                "found_count": len(found_union),
                "note": "No expected_files available",
            }

    coverage["summary"] = {
        "total_queries": len(all_query_details),
        "queries_fully_covered": covered_count,
        "total_expected_files": total_expected,
        "total_found_files": total_found,
        "overall_coverage_pct": round(
            total_found / total_expected * 100, 1,
        ) if total_expected > 0 else 0.0,
    }
    return coverage
