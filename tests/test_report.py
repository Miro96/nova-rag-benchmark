"""Tests for terminal output in rag_bench/report.py.

Covers VAL-REPORT-001 through VAL-REPORT-004.
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from rag_bench.metrics import BenchmarkMetrics
from rag_bench.report import (
    _fmt_tokens,
    print_comparison_table,
    print_results_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(**overrides) -> BenchmarkMetrics:
    """Build a BenchmarkMetrics with sensible non-zero defaults + overrides."""
    defaults = {
        "hit_at_1": 0.50,
        "hit_at_3": 0.60,
        "hit_at_5": 0.75,
        "hit_at_10": 0.85,
        "symbol_hit_at_5": 0.60,
        "mrr": 0.55,
        "query_latency_p50_ms": 42.0,
        "query_latency_p95_ms": 80.0,
        "query_latency_p99_ms": 120.0,
        "query_latency_mean_ms": 50.0,
        "ingest_total_sec": 10.0,
        "ingest_files_per_sec": 20.0,
        "ingest_total_files": 200,
        "index_size_mb": 15.0,
        "ram_peak_mb": 250.0,
        "avg_tool_calls": 1.5,
        "total_queries": 100,
        "total_hits": 75,
        "composite_score": 0.72,
        # Token fields
        "avg_response_tokens": 1234.5,
        "p50_response_tokens": 1100.0,
        "p95_response_tokens": 1800.0,
        "total_response_tokens": 129000,
        "avg_prompt_tokens": 0.0,
        "avg_completion_tokens": 0.0,
        "avg_total_llm_tokens": 0.0,
        "total_llm_tokens": 0,
    }
    defaults.update(overrides)
    return BenchmarkMetrics(**defaults)


def _capture_print_results_table(server_name: str, metrics: BenchmarkMetrics,
                                  baseline_result: dict | None = None) -> str:
    """Call print_results_table and capture its rich output as a string."""
    buf = StringIO()
    c = Console(file=buf, width=120, force_terminal=False)
    import rag_bench.report as rpt
    old = rpt.console
    rpt.console = c
    try:
        print_results_table(server_name, metrics,
                            baseline_result=baseline_result)
    finally:
        rpt.console = old
    return buf.getvalue()


def _capture_print_comparison_table(results: list[dict]) -> str:
    """Call print_comparison_table and capture its rich output as a string."""
    buf = StringIO()
    c = Console(file=buf, width=120, force_terminal=False)
    import rag_bench.report as rpt
    old = rpt.console
    rpt.console = c
    try:
        print_comparison_table(results)
    finally:
        rpt.console = old
    return buf.getvalue()


def _make_result(name: str, **tokens) -> dict:
    """Build a minimal result dict for print_comparison_table."""
    retrieval: dict = {
        "hit_at_1": 0.5,
        "hit_at_5": 0.65,
        "symbol_hit_at_5": 0.3,
        "mrr": 0.55,
        "latency": {"p50_ms": 20.0, "p95_ms": 35.0},
    }
    if tokens:
        retrieval["tokens"] = tokens
    return {
        "server": {"name": name},
        "retrieval": retrieval,
        "ingest": {"total_sec": 5.0, "files_per_sec": 20.0, "ram_peak_mb": 200.0},
        "composite_score": 0.72,
    }


# ---------------------------------------------------------------------------
# VAL-REPORT-001: results table includes Tokens section
# ---------------------------------------------------------------------------

class TestResultsTableTokensSection:
    """VAL-REPORT-001: print_results_table prints a table titled 'Tokens'
    containing rows for avg, p50, p95, total response tokens."""

    def test_tokens_table_title_present(self):
        """Output contains the 'Tokens' table title."""
        m = _make_metrics()
        out = _capture_print_results_table("test", m)
        assert "Tokens" in out, (
            f"Expected 'Tokens' in output, got:\n{out[:500]}"
        )

    def test_tokens_avg_row_present(self):
        """Tokens table contains 'avg' row with numeric value."""
        m = _make_metrics(avg_response_tokens=1234.5)
        out = _capture_print_results_table("test", m)
        assert "avg" in out
        assert "1234.5" in out

    def test_tokens_p50_row_present(self):
        """Tokens table contains 'p50' row with numeric value."""
        m = _make_metrics(p50_response_tokens=1100.0)
        out = _capture_print_results_table("test", m)
        assert "p50" in out
        assert "1100.0" in out

    def test_tokens_p95_row_present(self):
        """Tokens table contains 'p95' row with numeric value."""
        m = _make_metrics(p95_response_tokens=1800.0)
        out = _capture_print_results_table("test", m)
        assert "p95" in out
        assert "1800.0" in out

    def test_tokens_total_row_present(self):
        """Tokens table contains 'total' row with numeric value."""
        m = _make_metrics(total_response_tokens=129000)
        out = _capture_print_results_table("test", m)
        assert "total" in out
        assert "129000" in out

    def test_metrics_with_only_zero_tokens_still_render_table(self):
        """When all token fields are 0.0/0, the Tokens table still renders."""
        m = _make_metrics(
            avg_response_tokens=0.0,
            p50_response_tokens=0.0,
            p95_response_tokens=0.0,
            total_response_tokens=0,
        )
        out = _capture_print_results_table("test", m)
        assert "Tokens" in out
        assert "0.0" in out


# ---------------------------------------------------------------------------
# VAL-REPORT-002: results table shows LLM tokens for baseline
# ---------------------------------------------------------------------------

class TestResultsTableBaselineLLMTokens:
    """VAL-REPORT-002: print_results_table with baseline LLM token data
    renders rows or a section for avg_prompt_tokens, avg_completion_tokens,
    avg_total_llm_tokens."""

    def test_baseline_llm_section_rendered(self):
        """When baseline_result with efficiency token data is passed,
        the output includes a Baseline LLM Tokens section."""
        m = _make_metrics()
        baseline = {
            "efficiency": {
                "avg_prompt_tokens": 2200.5,
                "avg_completion_tokens": 180.3,
                "avg_total_llm_tokens": 2380.8,
                "total_llm_tokens": 249000,
            },
        }
        out = _capture_print_results_table("test", m,
                                            baseline_result=baseline)
        assert "LLM" in out or "Baseline" in out, (
            f"Expected baseline LLM section in output, got:\n{out[:1000]}"
        )
        assert "2200.5" in out
        assert "180.3" in out
        assert "2380.8" in out

    def test_baseline_llm_total_shown(self):
        """Baseline LLM Tokens section includes total_llm_tokens."""
        m = _make_metrics()
        baseline = {
            "efficiency": {
                "avg_prompt_tokens": 100.0,
                "avg_completion_tokens": 50.0,
                "avg_total_llm_tokens": 150.0,
                "total_llm_tokens": 15000,
            },
        }
        out = _capture_print_results_table("test", m,
                                            baseline_result=baseline)
        assert "15000" in out, (
            f"Expected total_llm_tokens 15000 in output, got:\n{out[:1000]}"
        )

    def test_baseline_without_efficiency_no_crash(self):
        """Baseline dict without 'efficiency' key does not crash."""
        m = _make_metrics()
        baseline: dict = {}
        out = _capture_print_results_table("test", m,
                                            baseline_result=baseline)
        # Should still render normally without crash
        assert "Tokens" in out

    def test_baseline_without_token_fields_no_crash(self):
        """Baseline dict with efficiency but missing token fields does not crash."""
        m = _make_metrics()
        baseline = {"efficiency": {}}
        out = _capture_print_results_table("test", m,
                                            baseline_result=baseline)
        assert "Tokens" in out

    def test_no_baseline_passed_does_not_render_llm_section(self):
        """When no baseline_result is passed, no LLM token section appears."""
        m = _make_metrics()
        out = _capture_print_results_table("test", m)
        # Should NOT contain LLM token labels like avg_prompt_tokens
        assert "prompt_tokens" not in out.lower()


# ---------------------------------------------------------------------------
# VAL-REPORT-003: comparison table includes token rows
# ---------------------------------------------------------------------------

class TestComparisonTableTokenRows:
    """VAL-REPORT-003: print_comparison_table includes rows labeled
    'Avg Tokens' and 'P95 Tokens' showing each preset's value side-by-side."""

    def test_avg_tokens_row_present(self):
        """Output contains 'Avg Tokens' row label."""
        results = [
            _make_result("nova-rag", avg=1234.5, p50=1100.0, p95=1800.0, total=129000),
            _make_result("grep-glob", avg=500.0, p50=450.0, p95=700.0, total=50000),
        ]
        out = _capture_print_comparison_table(results)
        assert "Avg Tokens" in out, (
            f"Expected 'Avg Tokens' in output, got:\n{out[:1000]}"
        )

    def test_p95_tokens_row_present(self):
        """Output contains 'P95 Tokens' row label."""
        results = [
            _make_result("nova-rag", avg=1234.5, p50=1100.0, p95=1800.0, total=129000),
            _make_result("grep-glob", avg=500.0, p50=450.0, p95=700.0, total=50000),
        ]
        out = _capture_print_comparison_table(results)
        assert "P95 Tokens" in out, (
            f"Expected 'P95 Tokens' in output, got:\n{out[:1000]}"
        )

    def test_token_values_appear_side_by_side(self):
        """Both presets' avg token values appear in the same rendered row."""
        results = [
            _make_result("nova-rag", avg=1234.5, p95=1800.0),
            _make_result("grep-glob", avg=500.0, p95=700.0),
        ]
        out = _capture_print_comparison_table(results)
        # Both values should be present
        assert "1234.5" in out
        assert "500.0" in out


# ---------------------------------------------------------------------------
# VAL-REPORT-004: empty/missing token data renders gracefully
# ---------------------------------------------------------------------------

class TestEmptyTokenDataGraceful:
    """VAL-REPORT-004: Calling print_results_table or print_comparison_table
    with token fields missing does not raise and shows 'N/A' or '0'."""

    def test_results_table_with_zero_tokens_no_crash(self):
        """print_results_table with all-zero token fields does not raise."""
        m = _make_metrics(
            avg_response_tokens=0.0,
            p50_response_tokens=0.0,
            p95_response_tokens=0.0,
            total_response_tokens=0,
        )
        try:
            _capture_print_results_table("test", m)
        except Exception as e:
            pytest.fail(f"print_results_table raised on zero tokens: {e}")

    def test_results_table_defaults_no_crash(self):
        """Default BenchmarkMetrics (all fields default) does not raise."""
        m = BenchmarkMetrics()
        try:
            _capture_print_results_table("test", m)
        except Exception as e:
            pytest.fail(f"print_results_table raised on defaults: {e}")

    def test_comparison_table_missing_tokens_shows_na(self):
        """Missing token data in a result dict renders as 'N/A'."""
        results = [
            _make_result("nova-rag", avg=1234.5, p95=1800.0),
            _make_result("legacy"),  # No tokens key
        ]
        out = _capture_print_comparison_table(results)
        assert "N/A" in out, (
            f"Expected 'N/A' for legacy preset, got:\n{out[:1000]}"
        )

    def test_comparison_table_empty_tokens_shows_na(self):
        """Empty tokens dict renders as 'N/A'."""
        results = [
            _make_result("preset-a"),  # No tokens
            _make_result("preset-b"),  # No tokens
        ]
        out = _capture_print_comparison_table(results)
        # Each preset should show N/A for tokens
        assert "N/A" in out

    def test_comparison_table_partial_tokens_no_crash(self):
        """Tokens dict missing individual keys does not crash."""
        results = [
            _make_result("partial", avg=500.0),  # only avg, no p95
        ]
        try:
            _capture_print_comparison_table(results)
        except Exception as e:
            pytest.fail(f"print_comparison_table raised on partial tokens: {e}")

    def test_results_table_with_none_baseline_no_crash(self):
        """print_results_table with baseline_result=None does not crash."""
        m = _make_metrics()
        try:
            _capture_print_results_table("test", m, baseline_result=None)
        except Exception as e:
            pytest.fail(f"print_results_table raised with baseline=None: {e}")

    def test_results_table_baseline_none_token_fields_no_crash(self):
        """Baseline with None values for token fields does not crash."""
        m = _make_metrics()
        baseline = {
            "efficiency": {
                "avg_prompt_tokens": None,
                "avg_completion_tokens": None,
                "avg_total_llm_tokens": None,
                "total_llm_tokens": None,
            },
        }
        try:
            _capture_print_results_table("test", m, baseline_result=baseline)
        except Exception as e:
            pytest.fail(f"print_results_table raised on None token fields: {e}")


# ============================================================================
# Chunk Hit@5 in terminal output
# ============================================================================


class TestResultsTableChunkHitAt5:
    """VAL-RUNNER-003 terminal output: results table includes Chunk Hit@5 row."""

    def test_chunk_hit_at_5_row_present(self):
        """Retrieval Quality table includes 'Chunk Hit@5' row."""
        m = _make_metrics(chunk_hit_at_5=0.35)
        out = _capture_print_results_table("test", m)
        assert "Chunk Hit@5" in out, (
            f"Expected 'Chunk Hit@5' row in output, got:\n{out[:500]}"
        )
        assert "35.0%" in out, (
            f"Expected 35.0% for chunk_hit_at_5=0.35, got:\n{out[:500]}"
        )

    def test_chunk_hit_at_5_zero_renders(self):
        """Chunk Hit@5 row renders even when value is 0.0."""
        m = _make_metrics(chunk_hit_at_5=0.0)
        out = _capture_print_results_table("test", m)
        assert "Chunk Hit@5" in out
        assert "0.0%" in out

    def test_chunk_hit_at_5_one_renders(self):
        """Chunk Hit@5 row renders when value is 1.0."""
        m = _make_metrics(chunk_hit_at_5=1.0)
        out = _capture_print_results_table("test", m)
        assert "Chunk Hit@5" in out
        assert "100.0%" in out


class TestComparisonTableChunkHitAt5:
    """VAL-RUNNER-005 terminal output: comparison table includes Chunk Hit@5 row."""

    def test_chunk_hit_at_5_row_present_in_comparison(self):
        """Comparison table includes 'Chunk Hit@5' row."""
        results = [
            {
                "server": {"name": "nova-rag"},
                "retrieval": {
                    "hit_at_1": 0.5,
                    "hit_at_5": 0.65,
                    "symbol_hit_at_5": 0.3,
                    "chunk_hit_at_5": 0.42,
                    "mrr": 0.55,
                    "latency": {"p50_ms": 20.0, "p95_ms": 35.0},
                },
                "ingest": {"total_sec": 5.0, "files_per_sec": 20.0, "ram_peak_mb": 200.0},
                "composite_score": 0.72,
            },
        ]
        out = _capture_print_comparison_table(results)
        assert "Chunk Hit@5" in out, (
            f"Expected 'Chunk Hit@5' in comparison output, got:\n{out[:500]}"
        )
        assert "42.0%" in out, (
            f"Expected 42.0% for chunk_hit_at_5=0.42, got:\n{out[:500]}"
        )

    def test_chunk_hit_at_5_missing_from_result_uses_zero(self):
        """When chunk_hit_at_5 is missing, comparison shows 0.0%."""
        results = [
            {
                "server": {"name": "legacy"},
                "retrieval": {
                    "hit_at_1": 0.5,
                    "hit_at_5": 0.65,
                    "symbol_hit_at_5": 0.3,
                    # chunk_hit_at_5 intentionally missing
                    "mrr": 0.55,
                    "latency": {"p50_ms": 20.0, "p95_ms": 35.0},
                },
                "ingest": {"total_sec": 5.0, "files_per_sec": 20.0, "ram_peak_mb": 200.0},
                "composite_score": 0.72,
            },
        ]
        out = _capture_print_comparison_table(results)
        assert "Chunk Hit@5" in out
        # Should show 0.0% for missing chunk_hit_at_5
        assert "0.0%" in out


class TestComparisonReportChunkMetrics:
    """VAL-RUNNER-005: generate_comparison_report includes chunk data."""

    def test_chunk_in_per_preset_metrics(self):
        from rag_bench.report import generate_comparison_report

        results = [
            {
                "server": {"name": "test"},
                "retrieval": {
                    "hit_at_1": 0.5, "hit_at_3": 0.6, "hit_at_5": 0.65,
                    "hit_at_10": 0.7, "symbol_hit_at_5": 0.3,
                    "chunk_hit_at_5": 0.42,
                    "mrr": 0.55, "total_queries": 100, "total_hits": 65,
                    "latency": {"p50_ms": 20.0, "p95_ms": 35.0, "p99_ms": 50.0, "mean_ms": 25.0},
                },
                "ingest": {"total_files": 200, "total_sec": 10.0, "files_per_sec": 20.0,
                           "index_size_mb": 15.0, "ram_peak_mb": 200.0},
                "composite_score": 0.72,
                "replicates": [],
                "query_details": [],
            },
        ]
        report = generate_comparison_report(results, replicates=1)
        assert "metrics" in report
        assert "test" in report["metrics"]
        assert "chunk_hit_at_5" in report["metrics"]["test"]
        assert report["metrics"]["test"]["chunk_hit_at_5"] == 0.42

    def test_chunk_in_summary_rows(self):
        from rag_bench.report import generate_comparison_report

        results = [
            {
                "server": {"name": "test"},
                "retrieval": {
                    "hit_at_1": 0.5, "hit_at_3": 0.6, "hit_at_5": 0.65,
                    "hit_at_10": 0.7, "symbol_hit_at_5": 0.3,
                    "chunk_hit_at_5": 0.42,
                    "mrr": 0.55, "total_queries": 100, "total_hits": 65,
                    "latency": {"p50_ms": 20.0, "p95_ms": 35.0, "p99_ms": 50.0, "mean_ms": 25.0},
                },
                "ingest": {"total_files": 200, "total_sec": 10.0, "files_per_sec": 20.0,
                           "index_size_mb": 15.0, "ram_peak_mb": 200.0},
                "composite_score": 0.72,
                "replicates": [],
                "query_details": [],
            },
        ]
        report = generate_comparison_report(results, replicates=1)
        summary_rows = report["summary"]
        chunk_row = [r for r in summary_rows if r["metric"] == "Chunk Hit@5"]
        assert len(chunk_row) == 1, f"Expected 'Chunk Hit@5' in summary rows, got: {[r['metric'] for r in summary_rows]}"
        assert chunk_row[0]["test"] == 0.42

    def test_chunk_default_when_missing(self):
        from rag_bench.report import generate_comparison_report

        results = [
            {
                "server": {"name": "legacy"},
                "retrieval": {
                    "hit_at_1": 0.5, "hit_at_3": 0.6, "hit_at_5": 0.65,
                    "hit_at_10": 0.7, "symbol_hit_at_5": 0.3,
                    # chunk_hit_at_5 intentionally missing
                    "mrr": 0.55, "total_queries": 100, "total_hits": 65,
                    "latency": {"p50_ms": 20.0, "p95_ms": 35.0, "p99_ms": 50.0, "mean_ms": 25.0},
                },
                "ingest": {"total_files": 200, "total_sec": 10.0, "files_per_sec": 20.0,
                           "index_size_mb": 15.0, "ram_peak_mb": 200.0},
                "composite_score": 0.72,
                "replicates": [],
                "query_details": [],
            },
        ]
        report = generate_comparison_report(results, replicates=1)
        assert report["metrics"]["legacy"]["chunk_hit_at_5"] == 0.0
        chunk_row = [r for r in report["summary"] if r["metric"] == "Chunk Hit@5"]
        assert len(chunk_row) == 1
        assert chunk_row[0]["legacy"] == 0.0


# ---------------------------------------------------------------------------
# _fmt_tokens helper tests
# ---------------------------------------------------------------------------

class TestFmtTokens:
    """Unit tests for the _fmt_tokens helper."""

    def test_fmt_valid_float(self):
        assert _fmt_tokens(
            {"retrieval": {"tokens": {"avg": 1234.5}}}, "avg",
        ) == "1234.5"

    def test_fmt_valid_int(self):
        assert _fmt_tokens(
            {"retrieval": {"tokens": {"total": 129000}}}, "total",
        ) == "129000"

    def test_fmt_missing_tokens_dict(self):
        assert _fmt_tokens({"retrieval": {}}, "avg") == "N/A"

    def test_fmt_missing_retrieval(self):
        assert _fmt_tokens({}, "avg") == "N/A"

    def test_fmt_missing_key(self):
        assert _fmt_tokens(
            {"retrieval": {"tokens": {"avg": 500.0}}}, "p95",
        ) == "N/A"

    def test_fmt_none_tokens(self):
        assert _fmt_tokens(
            {"retrieval": {"tokens": None}}, "avg",
        ) == "N/A"
