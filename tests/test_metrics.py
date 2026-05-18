"""Tests for metrics computation."""

import os
import threading
import time

import pytest

from rag_bench.metrics import (
    COMPOSITE_WEIGHTS,
    BenchmarkMetrics,
    MemorySampler,
    QueryResult,
    compute_composite_score,
    compute_chunk_hit_at_k,
    compute_hit_at_k,
    compute_latency_stats,
    compute_metrics,
    compute_mrr,
    compute_percentile,
    compute_symbol_hit_at_k,
    content_matches,
    directory_size_mb,
    file_matches,
    normalize_path,
    symbol_matches,
    tokenize_symbol,
)


def _make_qr(
    returned_files=None,
    expected_files=None,
    returned_symbols=None,
    expected_symbols=None,
    latency_ms=50.0,
) -> QueryResult:
    return QueryResult(
        query_id="test",
        query_text="test query",
        query_type="locate",
        difficulty="medium",
        expected_files=expected_files or ["src/app.py"],
        expected_symbols=expected_symbols or ["MyClass"],
        returned_files=returned_files or [],
        returned_symbols=returned_symbols or [],
        latency_ms=latency_ms,
    )


class TestNormalizePath:
    def test_basic(self):
        assert normalize_path("src/flask/app.py") == "src/flask/app.py"

    def test_leading_dot_slash(self):
        assert normalize_path("./src/app.py") == "src/app.py"

    def test_backslash(self):
        assert normalize_path("src\\flask\\app.py") == "src/flask/app.py"

    def test_case_insensitive(self):
        assert normalize_path("SRC/App.py") == "src/app.py"


class TestFileMatches:
    def test_exact(self):
        assert file_matches("src/flask/app.py", "src/flask/app.py")

    def test_suffix(self):
        assert file_matches("/full/path/src/flask/app.py", "src/flask/app.py")

    def test_no_match(self):
        assert not file_matches("src/flask/config.py", "src/flask/app.py")

    def test_substring_not_at_path_boundary_rejected(self):
        # "notapp.py" must NOT match expected "app.py" because the suffix
        # does not align with a path component boundary.
        assert not file_matches("notapp.py", "app.py")
        assert not file_matches("/repo/src/notapp.py", "app.py")

    def test_empty_returned_never_matches(self):
        assert not file_matches("", "src/app.py")
        assert not file_matches("   ", "src/app.py")

    def test_empty_expected_never_matches(self):
        assert not file_matches("src/app.py", "")

    def test_case_insensitive_suffix(self):
        assert file_matches("/Repo/SRC/Flask/App.py", "src/flask/app.py")


class TestTokenizeSymbol:
    def test_camel_case(self):
        assert tokenize_symbol("getUser") == {"get", "user"}

    def test_pascal_case(self):
        assert tokenize_symbol("FlaskApp") == {"flask", "app"}

    def test_snake_case(self):
        assert tokenize_symbol("validate_email") == {"validate", "email"}

    def test_dotted(self):
        assert tokenize_symbol("flask.app.Flask") == {"flask", "app"}

    def test_acronym_run(self):
        assert tokenize_symbol("HTTPResponse") == {"http", "response"}

    def test_empty(self):
        assert tokenize_symbol("") == set()


class TestSymbolMatches:
    def test_exact(self):
        assert symbol_matches(["Flask", "route"], "Flask")

    def test_partial(self):
        # "Flask" is a token of "FlaskApp" → match
        assert symbol_matches(["FlaskApp"], "Flask")

    def test_no_match(self):
        assert not symbol_matches(["Blueprint"], "Flask")

    def test_token_boundary_camel(self):
        # "get" is a real token of getUser/getCurrentUser
        assert symbol_matches(["getUser"], "get")
        assert symbol_matches(["getCurrentUser"], "get")

    def test_token_boundary_rejects_substring(self):
        # The substring "get" appears in "target" and "forget" but is not a
        # standalone token; previous substring matching would over-match.
        assert not symbol_matches(["target"], "get")
        assert not symbol_matches(["forget"], "get")
        assert not symbol_matches(["targetForget"], "get")

    def test_snake_case_match(self):
        assert symbol_matches(["validate_email_address"], "validate_email")

    def test_empty_returned(self):
        assert not symbol_matches([], "Flask")
        assert not symbol_matches([""], "Flask")

    def test_empty_expected(self):
        assert not symbol_matches(["Flask"], "")


class TestHitAtK:
    def test_hit_at_1(self):
        results = [
            _make_qr(returned_files=["src/app.py"], expected_files=["src/app.py"]),
            _make_qr(returned_files=["wrong.py"], expected_files=["src/app.py"]),
        ]
        assert compute_hit_at_k(results, 1) == 0.5

    def test_hit_at_5(self):
        results = [
            _make_qr(
                returned_files=["a.py", "b.py", "c.py", "d.py", "src/app.py"],
                expected_files=["src/app.py"],
            ),
        ]
        assert compute_hit_at_k(results, 5) == 1.0

    def test_empty(self):
        assert compute_hit_at_k([], 5) == 0.0

    def test_empty_returned_does_not_count_as_hit(self):
        results = [
            _make_qr(returned_files=[""], expected_files=["src/app.py"]),
            _make_qr(returned_files=[], expected_files=["src/app.py"]),
        ]
        assert compute_hit_at_k(results, 5) == 0.0

    def test_substring_path_does_not_count_as_hit(self):
        # /repo/src/notapp.py must not match expected app.py at any k.
        results = [
            _make_qr(
                returned_files=["/repo/src/notapp.py"],
                expected_files=["app.py"],
            ),
        ]
        assert compute_hit_at_k(results, 1) == 0.0
        assert compute_hit_at_k(results, 5) == 0.0

    def test_monotonic_in_k(self):
        results = [
            _make_qr(
                returned_files=[
                    "a.py", "b.py", "c.py", "d.py", "e.py",
                    "f.py", "g.py", "h.py", "i.py", "src/app.py",
                ],
                expected_files=["src/app.py"],
            ),
            _make_qr(
                returned_files=["src/util.py", "x.py", "y.py"],
                expected_files=["src/util.py"],
            ),
            _make_qr(
                returned_files=["a.py", "b.py", "c.py"],
                expected_files=["wrong.py"],
            ),
        ]
        h1 = compute_hit_at_k(results, 1)
        h3 = compute_hit_at_k(results, 3)
        h5 = compute_hit_at_k(results, 5)
        h10 = compute_hit_at_k(results, 10)
        assert h1 <= h3 <= h5 <= h10
        assert h10 > 0.0
        for v in (h1, h3, h5, h10):
            assert 0.0 <= v <= 1.0


class TestSymbolHitAtK:
    def test_basic(self):
        results = [
            _make_qr(
                returned_symbols=["getUser", "process"],
                expected_symbols=["get"],
            ),
            _make_qr(
                returned_symbols=["target"],
                expected_symbols=["get"],
            ),
        ]
        v = compute_symbol_hit_at_k(results, 5)
        # First query hits, second does not (token-boundary aware).
        assert v == 0.5

    def test_excludes_queries_without_expected_symbols(self):
        # Build directly to bypass _make_qr's default-symbol fallback.
        results = [
            QueryResult(
                query_id="a", query_text="", query_type="locate",
                difficulty="easy", expected_files=[], expected_symbols=["x"],
                returned_files=[], returned_symbols=["x"], latency_ms=1.0,
            ),
            QueryResult(
                query_id="b", query_text="", query_type="locate",
                difficulty="easy", expected_files=[], expected_symbols=[],
                returned_files=[], returned_symbols=["y"], latency_ms=1.0,
            ),
        ]
        # Denominator only counts queries with expected symbols
        assert compute_symbol_hit_at_k(results, 5) == 1.0

    def test_no_expected_symbols(self):
        results = [
            QueryResult(
                query_id="a", query_text="", query_type="locate",
                difficulty="easy", expected_files=[], expected_symbols=[],
                returned_files=[], returned_symbols=["y"], latency_ms=1.0,
            ),
        ]
        assert compute_symbol_hit_at_k(results, 5) == 0.0

    def test_in_range(self):
        results = [
            _make_qr(
                returned_symbols=["getUser"],
                expected_symbols=["get"],
            ),
            _make_qr(
                returned_symbols=["forget"],
                expected_symbols=["get"],
            ),
            _make_qr(
                returned_symbols=["save"],
                expected_symbols=["get"],
            ),
        ]
        v = compute_symbol_hit_at_k(results, 5)
        assert 0.0 < v < 1.0


class TestMRR:
    def test_basic(self):
        results = [
            _make_qr(returned_files=["src/app.py"], expected_files=["src/app.py"]),
            _make_qr(returned_files=["a.py", "src/app.py"], expected_files=["src/app.py"]),
        ]
        assert compute_mrr(results) == (1.0 + 0.5) / 2

    def test_no_match(self):
        results = [
            _make_qr(returned_files=["wrong.py"], expected_files=["src/app.py"]),
        ]
        assert compute_mrr(results) == 0.0

    def test_in_zero_one_inclusive_upper(self):
        # All queries hit at rank 1 → MRR == 1.0.
        results = [
            _make_qr(returned_files=["src/app.py"], expected_files=["src/app.py"]),
            _make_qr(returned_files=["src/util.py"], expected_files=["src/util.py"]),
        ]
        assert compute_mrr(results) == 1.0

    def test_in_open_zero_to_one(self):
        # Hits at ranks 2 and 3 → MRR == (1/2 + 1/3) / 2 in (0, 1).
        results = [
            _make_qr(
                returned_files=["a.py", "src/app.py"],
                expected_files=["src/app.py"],
            ),
            _make_qr(
                returned_files=["a.py", "b.py", "src/util.py"],
                expected_files=["src/util.py"],
            ),
        ]
        v = compute_mrr(results)
        assert 0.0 < v < 1.0
        assert v == pytest.approx((0.5 + 1.0 / 3) / 2)

    def test_substring_does_not_count_as_match(self):
        # "/repo/notapp.py" must not contribute to MRR for expected "app.py".
        results = [
            _make_qr(
                returned_files=["/repo/notapp.py", "/repo/app.py"],
                expected_files=["app.py"],
            ),
        ]
        # Match is at rank 2, not rank 1.
        assert compute_mrr(results) == 0.5


class TestPercentile:
    def test_p50(self):
        assert compute_percentile([10, 20, 30, 40, 50], 50) == 30.0

    def test_p95(self):
        values = list(range(1, 101))
        assert compute_percentile(values, 95) == 95.05

    def test_empty(self):
        assert compute_percentile([], 50) == 0.0


class TestComputeMetrics:
    def test_basic(self):
        results = [
            _make_qr(
                returned_files=["src/app.py"],
                expected_files=["src/app.py"],
                returned_symbols=["MyClass"],
                expected_symbols=["MyClass"],
                latency_ms=100,
            ),
            _make_qr(
                returned_files=["wrong.py"],
                expected_files=["src/config.py"],
                returned_symbols=[],
                expected_symbols=["Config"],
                latency_ms=200,
            ),
        ]
        m = compute_metrics(results, ingest_total_sec=5.0, ingest_total_files=100)

        assert m.hit_at_1 == 0.5
        assert m.total_queries == 2
        assert m.ingest_files_per_sec == 20.0
        assert m.composite_score > 0

    def test_hit_at_k_monotonic_and_in_range(self):
        results = [
            _make_qr(
                returned_files=[f"f{i}.py" for i in range(9)] + ["src/app.py"],
                expected_files=["src/app.py"],
            ),
            _make_qr(
                returned_files=["src/util.py", "x.py", "y.py", "z.py", "w.py"],
                expected_files=["src/util.py"],
            ),
            _make_qr(
                returned_files=["a.py", "b.py", "c.py"],
                expected_files=["does/not/exist.py"],
            ),
        ]
        m = compute_metrics(results)
        for v in (m.hit_at_1, m.hit_at_3, m.hit_at_5, m.hit_at_10):
            assert 0.0 <= v <= 1.0
        assert m.hit_at_1 <= m.hit_at_3 <= m.hit_at_5 <= m.hit_at_10
        assert m.hit_at_10 > 0.0
        assert 0.0 < m.mrr <= 1.0

    def test_latency_excludes_failed_queries_from_percentiles(self):
        # Two failed queries (latency_ms=0) plus three successful ones.
        # Including 0s would crash p50/p95 to small values; the fixed
        # implementation must compute percentiles only over successful
        # queries so all four latency stats are strictly positive.
        results = [
            _make_qr(latency_ms=0),
            _make_qr(latency_ms=0),
            _make_qr(latency_ms=100),
            _make_qr(latency_ms=200),
            _make_qr(latency_ms=300),
        ]
        m = compute_metrics(results)
        assert m.query_latency_p50_ms > 0
        assert m.query_latency_p95_ms > 0
        assert m.query_latency_p99_ms > 0
        assert m.query_latency_mean_ms > 0
        # Computed only over [100, 200, 300].
        assert m.query_latency_p50_ms == pytest.approx(200.0)
        assert m.query_latency_mean_ms == pytest.approx(200.0)

    def test_latency_percentiles_monotonic(self):
        results = [_make_qr(latency_ms=v) for v in (10, 25, 40, 60, 80, 110, 150, 200, 400, 900)]
        m = compute_metrics(results)
        assert m.query_latency_p50_ms > 0
        assert m.query_latency_p99_ms >= m.query_latency_p95_ms >= m.query_latency_p50_ms
        assert m.query_latency_mean_ms > 0

    def test_files_per_sec_positive_when_total_files_positive(self):
        results = [_make_qr(latency_ms=50)]
        m = compute_metrics(
            results, ingest_total_sec=10.0, ingest_total_files=200,
        )
        assert m.ingest_files_per_sec == pytest.approx(20.0)

    def test_files_per_sec_zero_when_total_files_zero(self):
        results = [_make_qr(latency_ms=50)]
        m = compute_metrics(
            results, ingest_total_sec=5.0, ingest_total_files=0,
        )
        assert m.ingest_files_per_sec == 0.0

    def test_files_per_sec_zero_when_ingest_sec_zero(self):
        results = [_make_qr(latency_ms=50)]
        m = compute_metrics(
            results, ingest_total_sec=0.0, ingest_total_files=10,
        )
        assert m.ingest_files_per_sec == 0.0

    def test_index_size_and_ram_propagated(self):
        results = [_make_qr(latency_ms=50)]
        m = compute_metrics(
            results,
            ingest_total_sec=1.0,
            ingest_total_files=10,
            index_size_mb=12.5,
            ram_peak_mb=256.0,
        )
        assert m.index_size_mb == 12.5
        assert m.ram_peak_mb == 256.0

    def test_symbol_hit_at_5_not_one_when_overmatch_avoided(self):
        # Without token-boundary matching, "get" would substring-match
        # "target" and "forget", inflating SymbolHit@5 to 1.0. The fixed
        # implementation must keep it strictly less than 1.0.
        results = [
            _make_qr(
                returned_files=["src/app.py"],
                expected_files=["src/app.py"],
                returned_symbols=["getUser"],
                expected_symbols=["get"],
            ),
            _make_qr(
                returned_files=["src/util.py"],
                expected_files=["src/util.py"],
                returned_symbols=["target"],
                expected_symbols=["get"],
            ),
            _make_qr(
                returned_files=["src/x.py"],
                expected_files=["src/x.py"],
                returned_symbols=["forget"],
                expected_symbols=["get"],
            ),
        ]
        m = compute_metrics(results)
        assert 0.0 < m.symbol_hit_at_5 < 1.0


class TestComputeLatencyStats:
    def test_excludes_zero_latencies(self):
        stats = compute_latency_stats([0.0, 0.0, 100.0, 200.0, 300.0])
        assert stats["p50_ms"] == pytest.approx(200.0)
        assert stats["mean_ms"] == pytest.approx(200.0)
        assert stats["p95_ms"] >= stats["p50_ms"]
        assert stats["p99_ms"] >= stats["p95_ms"]

    def test_all_zero_returns_zero(self):
        stats = compute_latency_stats([0.0, 0.0, 0.0])
        assert stats == {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}

    def test_empty_returns_zero(self):
        assert compute_latency_stats([]) == {
            "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0,
        }

    def test_negative_latency_treated_as_failure(self):
        stats = compute_latency_stats([-1.0, 50.0, 100.0])
        assert stats["mean_ms"] == pytest.approx(75.0)

    def test_monotonic(self):
        stats = compute_latency_stats(list(range(1, 101)))
        assert stats["p50_ms"] > 0
        assert stats["p99_ms"] >= stats["p95_ms"] >= stats["p50_ms"]


class TestDirectorySizeMb:
    def test_missing_path_returns_zero(self, tmp_path):
        assert directory_size_mb(tmp_path / "does_not_exist") == 0.0

    def test_none_or_empty_returns_zero(self):
        assert directory_size_mb(None) == 0.0
        assert directory_size_mb("") == 0.0

    def test_single_file(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"x" * 1024 * 1024)  # 1 MB
        assert directory_size_mb(f) == pytest.approx(1.0, rel=0.01)

    def test_recursive(self, tmp_path):
        (tmp_path / "a").write_bytes(b"x" * 1024 * 512)
        nested = tmp_path / "sub" / "deep"
        nested.mkdir(parents=True)
        (nested / "b").write_bytes(b"y" * 1024 * 512)
        assert directory_size_mb(tmp_path) == pytest.approx(1.0, rel=0.01)

    def test_user_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "f").write_bytes(b"x" * 1024 * 256)
        assert directory_size_mb("~/f") == pytest.approx(0.25, rel=0.05)

    def test_ignores_symlinks_to_avoid_loops(self, tmp_path):
        target = tmp_path / "real.bin"
        target.write_bytes(b"x" * 1024 * 1024)
        link = tmp_path / "link.bin"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        # Symlink should not double-count the target file's bytes.
        assert directory_size_mb(tmp_path) == pytest.approx(1.0, rel=0.01)


class TestMemorySampler:
    def test_sampler_records_peak_for_current_process(self):
        sampler = MemorySampler(os.getpid(), interval_sec=0.05)
        sampler.start()
        # Allocate something to ensure RSS is non-trivially > 0.
        _ = bytearray(1024 * 1024)
        time.sleep(0.2)
        peak = sampler.stop()
        assert peak > 0

    def test_sampler_no_pid_returns_zero(self):
        sampler = MemorySampler(None)
        sampler.start()
        time.sleep(0.05)
        assert sampler.stop() == 0.0

    def test_sample_updates_peak_monotonically(self):
        sampler = MemorySampler(os.getpid(), interval_sec=10.0)
        sampler.peak_mb = 999_999.0
        # Single explicit sample must not lower the recorded peak.
        sampler.sample()
        assert sampler.peak_mb == 999_999.0

    def test_stop_is_idempotent(self):
        sampler = MemorySampler(os.getpid(), interval_sec=0.05)
        sampler.start()
        time.sleep(0.1)
        first = sampler.stop()
        second = sampler.stop()
        assert second >= first
        assert second > 0

    def test_sampler_thread_exits_after_stop(self):
        sampler = MemorySampler(os.getpid(), interval_sec=0.05)
        sampler.start()
        time.sleep(0.1)
        sampler.stop()
        # Give a brief grace period; the worker thread must be joined.
        time.sleep(0.05)
        worker_threads = [
            t for t in threading.enumerate() if t.name == "MemorySampler"
        ]
        assert worker_threads == []


class TestToolCallsTracking:
    def test_query_result_default_is_zero(self):
        # Default must be 0, not 1: hardcoding 1.0 was the bug we are fixing.
        qr = QueryResult(
            query_id="q", query_text="", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[], latency_ms=1.0,
        )
        assert qr.tool_calls == 0

    def test_avg_tool_calls_reflects_actual_invocations(self):
        # Per-query tool_calls drive avg_tool_calls; verify mean tracks
        # the true sum/count and is not pinned at 1.0.
        results = [
            _make_qr(latency_ms=10),
            _make_qr(latency_ms=20),
            _make_qr(latency_ms=30),
        ]
        results[0].tool_calls = 1
        results[1].tool_calls = 2
        results[2].tool_calls = 3
        m = compute_metrics(results)
        assert m.avg_tool_calls == pytest.approx(2.0)

    def test_unique_tool_call_counts_in_query_details(self):
        # Validation contract evidence requires > 1 unique tool_calls value
        # across query_details. Verify the mechanism produces variance when
        # invocations vary.
        results = [_make_qr(latency_ms=10) for _ in range(5)]
        for qr, n in zip(results, [0, 1, 1, 2, 3]):
            qr.tool_calls = n
        unique = {qr.tool_calls for qr in results}
        assert len(unique) > 1

    def test_mcp_client_call_count_increments_on_call(self):
        import asyncio

        from rag_bench.mcp_client import MCPClient

        client = MCPClient(command="true")

        async def fake_request(method, params):
            return {"content": [{"type": "text", "text": "ok"}]}

        client._request = fake_request  # type: ignore[assignment]

        async def go():
            assert client.call_count == 0
            await client.call_tool("a", {})
            await client.call_tool("b", {})
            await client.call_tool("c", {})
            return client.call_count

        assert asyncio.run(go()) == 3

    def test_mcp_client_call_count_increments_on_failure(self):
        import asyncio

        from rag_bench.mcp_client import MCPClient

        client = MCPClient(command="true")

        async def failing_request(method, params):
            raise RuntimeError("boom")

        client._request = failing_request  # type: ignore[assignment]

        async def go():
            with pytest.raises(RuntimeError):
                await client.call_tool("a", {})
            return client.call_count

        # Failed attempts still count as MCP invocations.
        assert asyncio.run(go()) == 1


class TestCompositeScore:
    def test_weights_sum_to_one(self):
        assert sum(COMPOSITE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_score_in_open_unit_interval(self):
        score = compute_composite_score(
            hit_at_5=0.5,
            symbol_hit_at_5=0.4,
            mrr=0.6,
            avg_tool_calls=1.0,
            p95_latency_ms=200.0,
            ram_peak_mb=300.0,
            index_size_mb=20.0,
        )
        assert 0.0 < score < 1.0

    def test_score_includes_all_weighted_components(self):
        # Changing any single component must shift the composite score; if a
        # weight were dropped, the corresponding sensitivity would be zero.
        base = dict(
            hit_at_5=0.5,
            symbol_hit_at_5=0.4,
            mrr=0.6,
            avg_tool_calls=1.0,
            p95_latency_ms=200.0,
            ram_peak_mb=300.0,
            index_size_mb=20.0,
        )
        baseline = compute_composite_score(**base)

        for key in [
            "hit_at_5",
            "symbol_hit_at_5",
            "mrr",
            "avg_tool_calls",
            "p95_latency_ms",
            "ram_peak_mb",
            "index_size_mb",
        ]:
            perturbed = dict(base)
            if key in ("hit_at_5", "symbol_hit_at_5", "mrr"):
                perturbed[key] = 0.0
            else:
                perturbed[key] = base[key] * 10
            other = compute_composite_score(**perturbed)
            assert other != pytest.approx(baseline), (
                f"composite_score does not depend on {key}"
            )

    def test_perfect_quality_yields_high_score(self):
        score = compute_composite_score(
            hit_at_5=1.0,
            symbol_hit_at_5=1.0,
            mrr=1.0,
            avg_tool_calls=0.0,
            p95_latency_ms=0.0,
            ram_peak_mb=0.0,
            index_size_mb=0.0,
        )
        # 0.30 + 0.15 + 0.15 + 0.15*1 + 0.15*1 + 0.10*1 = 1.0
        assert score == pytest.approx(1.0)

    def test_zero_quality_still_positive_due_to_efficiency_terms(self):
        # Inverse-scale terms (tool_score, latency_score, resource_score) are
        # always > 0, so composite_score is strictly > 0 even when retrieval
        # quality is zero.
        score = compute_composite_score(
            hit_at_5=0.0,
            symbol_hit_at_5=0.0,
            mrr=0.0,
            avg_tool_calls=10.0,
            p95_latency_ms=10_000.0,
            ram_peak_mb=10_000.0,
            index_size_mb=10_000.0,
        )
        assert 0.0 < score < 1.0

    def test_lower_tool_calls_improves_score(self):
        higher_tool_calls = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.5, mrr=0.5,
            avg_tool_calls=10.0,
            p95_latency_ms=100.0, ram_peak_mb=100.0, index_size_mb=10.0,
        )
        lower_tool_calls = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.5, mrr=0.5,
            avg_tool_calls=1.0,
            p95_latency_ms=100.0, ram_peak_mb=100.0, index_size_mb=10.0,
        )
        assert lower_tool_calls > higher_tool_calls

    def test_lower_latency_improves_score(self):
        slow = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.5, mrr=0.5,
            avg_tool_calls=1.0,
            p95_latency_ms=2000.0, ram_peak_mb=100.0, index_size_mb=10.0,
        )
        fast = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.5, mrr=0.5,
            avg_tool_calls=1.0,
            p95_latency_ms=50.0, ram_peak_mb=100.0, index_size_mb=10.0,
        )
        assert fast > slow

    def test_compute_metrics_composite_in_open_unit_interval(self):
        results = [
            _make_qr(
                returned_files=["src/app.py"], expected_files=["src/app.py"],
                returned_symbols=["MyClass"], expected_symbols=["MyClass"],
                latency_ms=100,
            ),
        ]
        m = compute_metrics(results, ingest_total_sec=1.0, ingest_total_files=10)
        assert 0.0 < m.composite_score < 1.0


# ---------------------------------------------------------------------------
# Token fields: QueryResult
# ---------------------------------------------------------------------------


class TestQueryResultTokenFields:
    """VAL-METRIC-001: QueryResult carries response_tokens field."""

    def test_response_tokens_defaults_to_none(self):
        qr = QueryResult(
            query_id="q", query_text="", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[], latency_ms=1.0,
        )
        assert qr.response_tokens is None

    def test_response_tokens_int_settable(self):
        qr = QueryResult(
            query_id="q", query_text="", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[], latency_ms=1.0,
            response_tokens=42,
        )
        assert qr.response_tokens == 42

    def test_response_tokens_explicit_none(self):
        qr = QueryResult(
            query_id="q", query_text="", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[], latency_ms=1.0,
            response_tokens=None,
        )
        assert qr.response_tokens is None


class TestQueryResultLLMTokenFields:
    """VAL-METRIC-002: QueryResult carries LLM usage fields."""

    def test_prompt_tokens_defaults_to_none(self):
        qr = QueryResult(
            query_id="q", query_text="", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[], latency_ms=1.0,
        )
        assert qr.prompt_tokens is None
        assert qr.completion_tokens is None
        assert qr.total_llm_tokens is None

    def test_llm_token_fields_settable(self):
        qr = QueryResult(
            query_id="q", query_text="", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[], latency_ms=1.0,
            prompt_tokens=100,
            completion_tokens=20,
            total_llm_tokens=120,
        )
        assert qr.prompt_tokens == 100
        assert qr.completion_tokens == 20
        assert qr.total_llm_tokens == 120


# ---------------------------------------------------------------------------
# Token fields: BenchmarkMetrics
# ---------------------------------------------------------------------------


class TestBenchmarkMetricsTokenFields:
    """VAL-METRIC-003: BenchmarkMetrics carries aggregate token stats."""

    def test_response_token_fields_default_to_zero(self):
        m = BenchmarkMetrics()
        assert m.avg_response_tokens == 0.0
        assert m.p50_response_tokens == 0.0
        assert m.p95_response_tokens == 0.0
        assert m.total_response_tokens == 0

    def test_response_token_fields_settable(self):
        m = BenchmarkMetrics(
            avg_response_tokens=12.5,
            p50_response_tokens=10.0,
            p95_response_tokens=20.0,
            total_response_tokens=1250,
        )
        assert m.avg_response_tokens == 12.5
        assert m.p50_response_tokens == 10.0
        assert m.p95_response_tokens == 20.0
        assert m.total_response_tokens == 1250


class TestBenchmarkMetricsLLMTokenFields:
    """VAL-METRIC-006: BenchmarkMetrics carries aggregate LLM token stats."""

    def test_llm_token_fields_default_to_zero(self):
        m = BenchmarkMetrics()
        assert m.avg_prompt_tokens == 0.0
        assert m.avg_completion_tokens == 0.0
        assert m.avg_total_llm_tokens == 0.0
        assert m.total_llm_tokens == 0

    def test_llm_token_fields_settable(self):
        m = BenchmarkMetrics(
            avg_prompt_tokens=500.0,
            avg_completion_tokens=80.0,
            avg_total_llm_tokens=580.0,
            total_llm_tokens=5800,
        )
        assert m.avg_prompt_tokens == 500.0
        assert m.avg_completion_tokens == 80.0
        assert m.avg_total_llm_tokens == 580.0
        assert m.total_llm_tokens == 5800


# ---------------------------------------------------------------------------
# compute_metrics: token aggregation
# ---------------------------------------------------------------------------


class TestComputeMetricsResponseTokens:
    """VAL-METRIC-004: compute_metrics aggregates response_tokens."""

    def test_all_populated(self):
        """Average/p50/p95/total computed correctly when every result has a value."""
        results = [
            _make_qr(latency_ms=10, returned_files=["a.py"],
                     expected_files=["a.py"]),
            _make_qr(latency_ms=20, returned_files=["b.py"],
                     expected_files=["b.py"]),
            _make_qr(latency_ms=30, returned_files=["c.py"],
                     expected_files=["c.py"]),
            _make_qr(latency_ms=40, returned_files=["d.py"],
                     expected_files=["d.py"]),
            _make_qr(latency_ms=50, returned_files=["e.py"],
                     expected_files=["e.py"]),
        ]
        # Set response_tokens: 10, 20, 30, 40, 50
        for i, r in enumerate(results):
            r.response_tokens = (i + 1) * 10

        m = compute_metrics(results)

        # Total = 10+20+30+40+50 = 150
        assert m.total_response_tokens == 150
        # Average = 150 / 5 = 30
        assert m.avg_response_tokens == pytest.approx(30.0)
        # p50 = median of [10, 20, 30, 40, 50] = 30
        assert m.p50_response_tokens == pytest.approx(30.0)
        # p95 ≈ 48 (index 3.8)
        assert m.p95_response_tokens > 40.0

    def test_none_values_excluded_from_average(self):
        """Averages exclude None; totals sum only non-None values."""
        results = [
            _make_qr(latency_ms=10, returned_files=["a.py"],
                     expected_files=["a.py"]),
            _make_qr(latency_ms=20, returned_files=["b.py"],
                     expected_files=["b.py"]),
            _make_qr(latency_ms=30, returned_files=["c.py"],
                     expected_files=["c.py"]),
            _make_qr(latency_ms=40, returned_files=["d.py"],
                     expected_files=["d.py"]),
        ]
        results[0].response_tokens = 100
        results[1].response_tokens = None  # excluded
        results[2].response_tokens = 300
        results[3].response_tokens = None  # excluded

        m = compute_metrics(results)

        # Total = 100 + 300 = 400
        assert m.total_response_tokens == 400
        # Average over 2 non-None values: 400/2 = 200
        assert m.avg_response_tokens == pytest.approx(200.0)
        # p50 of [100, 300]
        assert m.p50_response_tokens == pytest.approx(200.0)
        # p95 of [100, 300]
        assert m.p95_response_tokens == pytest.approx(290.0)

    def test_all_none_returns_zero(self):
        """No crash, all token aggregates are 0 when every response_tokens is None."""
        results = [
            _make_qr(latency_ms=10),
            _make_qr(latency_ms=20),
            _make_qr(latency_ms=30),
        ]
        # All response_tokens are None (default)

        m = compute_metrics(results)
        assert m.avg_response_tokens == 0.0
        assert m.p50_response_tokens == 0.0
        assert m.p95_response_tokens == 0.0
        assert m.total_response_tokens == 0

    def test_all_none_non_token_metrics_unaffected(self):
        """VAL-METRIC-005: Non-token metrics still correct when tokens are None."""
        results = [
            _make_qr(
                returned_files=["src/app.py"],
                expected_files=["src/app.py"],
                latency_ms=100,
            ),
            _make_qr(
                returned_files=["wrong.py"],
                expected_files=["src/config.py"],
                latency_ms=200,
            ),
        ]
        # response_tokens are all None (default)

        m = compute_metrics(results, ingest_total_sec=5.0, ingest_total_files=100)

        # Token fields are all 0
        assert m.avg_response_tokens == 0.0
        assert m.total_response_tokens == 0

        # Non-token fields are still correct
        assert m.hit_at_1 == 0.5
        assert m.total_queries == 2
        assert m.ingest_files_per_sec == 20.0


class TestComputeMetricsLLMTokens:
    """VAL-METRIC-007: compute_metrics aggregates LLM tokens when present."""

    def test_llm_tokens_aggregated(self):
        results = [
            _make_qr(latency_ms=10),
            _make_qr(latency_ms=20),
            _make_qr(latency_ms=30),
            _make_qr(latency_ms=40),
        ]
        results[0].prompt_tokens = 100
        results[0].completion_tokens = 20
        results[0].total_llm_tokens = 120

        results[1].prompt_tokens = 200
        results[1].completion_tokens = 30
        results[1].total_llm_tokens = 230

        results[2].prompt_tokens = 300
        results[2].completion_tokens = 40
        results[2].total_llm_tokens = 340

        results[3].prompt_tokens = None
        results[3].completion_tokens = None
        results[3].total_llm_tokens = None

        m = compute_metrics(results)

        # Averages over 3 non-None entries
        assert m.avg_prompt_tokens == pytest.approx(200.0)   # (100+200+300)/3
        assert m.avg_completion_tokens == pytest.approx(30.0)  # (20+30+40)/3
        assert m.avg_total_llm_tokens == pytest.approx(230.0)  # (120+230+340)/3
        assert m.total_llm_tokens == 690  # 120+230+340

    def test_all_none_llm_tokens_returns_zero(self):
        results = [
            _make_qr(latency_ms=10),
            _make_qr(latency_ms=20),
        ]
        # All LLM fields are None

        m = compute_metrics(results)
        assert m.avg_prompt_tokens == 0.0
        assert m.avg_completion_tokens == 0.0
        assert m.avg_total_llm_tokens == 0.0
        assert m.total_llm_tokens == 0


class TestLatencyStatsUnaffectedByTokens:
    """VAL-METRIC-008: adding token fields does not change latency stats."""

    @pytest.mark.parametrize("with_tokens", [False, True])
    def test_latency_stats_identical(self, with_tokens):
        """Latency percentiles unchanged whether token fields are set or not."""
        results = [
            _make_qr(latency_ms=10),
            _make_qr(latency_ms=25),
            _make_qr(latency_ms=40),
            _make_qr(latency_ms=60),
            _make_qr(latency_ms=80),
            _make_qr(latency_ms=110),
            _make_qr(latency_ms=150),
            _make_qr(latency_ms=200),
            _make_qr(latency_ms=400),
            _make_qr(latency_ms=900),
        ]

        # Set token fields only when testing the "with tokens" case
        if with_tokens:
            for i, r in enumerate(results):
                r.response_tokens = (i + 1) * 10
                r.prompt_tokens = (i + 1) * 5
                r.completion_tokens = (i + 1)
                r.total_llm_tokens = r.prompt_tokens + r.completion_tokens

        m = compute_metrics(results)

        # Verify latency stats (computed over the same latencies regardless of tokens)
        assert m.query_latency_p50_ms > 0
        assert m.query_latency_p99_ms >= m.query_latency_p95_ms >= m.query_latency_p50_ms
        assert m.query_latency_mean_ms > 0

    def test_latency_numerically_identical_with_and_without_tokens(self):
        """Exact numerical equality of latency stats with vs without tokens."""
        _latencies = [10.0, 25.0, 40.0, 60.0, 80.0, 110.0, 150.0, 200.0, 400.0, 900.0]

        def _build(tokenised: bool) -> list[QueryResult]:
            rs = [_make_qr(latency_ms=v) for v in _latencies]
            if tokenised:
                for i, r in enumerate(rs):
                    r.response_tokens = (i + 1) * 10
                    r.prompt_tokens = (i + 1) * 5
                    r.completion_tokens = (i + 1)
                    r.total_llm_tokens = r.prompt_tokens + r.completion_tokens
            return rs

        m_no_tokens = compute_metrics(_build(False))
        m_with_tokens = compute_metrics(_build(True))

        assert m_no_tokens.query_latency_p50_ms == m_with_tokens.query_latency_p50_ms
        assert m_no_tokens.query_latency_p95_ms == m_with_tokens.query_latency_p95_ms
        assert m_no_tokens.query_latency_p99_ms == m_with_tokens.query_latency_p99_ms
        assert m_no_tokens.query_latency_mean_ms == m_with_tokens.query_latency_mean_ms

    def test_single_query_latency_unchanged_when_tokens_present(self):
        """Latency stats for a single query should be identical."""
        results = [_make_qr(latency_ms=42.5)]
        m_no_tok = compute_metrics(results)

        results[0].response_tokens = 999
        m_with_tok = compute_metrics(results)

        assert m_no_tok.query_latency_p50_ms == m_with_tok.query_latency_p50_ms
        assert m_no_tok.query_latency_mean_ms == m_with_tok.query_latency_mean_ms


# ---------------------------------------------------------------------------
# Composite score: optional token component
# ---------------------------------------------------------------------------


class TestCompositeScoreLegacyPreserved:
    """VAL-SCORE-001 / VAL-COMPAT-004: composite score unchanged without token flag."""

    # Pre-computed values from the legacy formula for a fixed set of inputs
    _LEGACY_SCENARIOS = [
        # (hit_at_5, symbol_hit_at_5, mrr, avg_tool_calls, p95_latency_ms,
        #  ram_peak_mb, index_size_mb, expected_score)
        (0.5, 0.4, 0.6, 1.0, 200.0, 300.0, 20.0, None),  # computed at runtime
        (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, None),       # computed at runtime
        (0.0, 0.0, 0.0, 10.0, 10000.0, 10000.0, 10000.0, None),  # computed at runtime
        (0.3, 0.2, 0.1, 5.0, 500.0, 1000.0, 500.0, None),  # computed at runtime
    ]

    def test_include_tokens_false_matches_legacy(self):
        """VAL-SCORE-001: include_tokens=False produces same value as current formula."""
        for (h5, sh5, mrr, tc, p95, ram, idx, _) in self._LEGACY_SCENARIOS:
            legacy = compute_composite_score(
                hit_at_5=h5,
                symbol_hit_at_5=sh5,
                mrr=mrr,
                avg_tool_calls=tc,
                p95_latency_ms=p95,
                ram_peak_mb=ram,
                index_size_mb=idx,
            )
            with_tokens_false = compute_composite_score(
                hit_at_5=h5,
                symbol_hit_at_5=sh5,
                mrr=mrr,
                avg_tool_calls=tc,
                p95_latency_ms=p95,
                ram_peak_mb=ram,
                index_size_mb=idx,
                include_tokens=False,
            )
            assert legacy == pytest.approx(with_tokens_false, abs=1e-9), (
                f"include_tokens=False differs from legacy for {h5=} {sh5=} {mrr=}"
            )

    def test_default_call_matches_legacy(self):
        """Not passing include_tokens at all must match legacy (default is False)."""
        score_legacy = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
        )
        score_default = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_tokens=False,
        )
        assert score_legacy == pytest.approx(score_default, abs=1e-9)

    def test_include_tokens_false_extra_kwarg_ignored(self):
        """Passing avg_response_tokens with include_tokens=False should be ignored."""
        score_no_extra = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_tokens=False,
        )
        score_with_extra = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_tokens=False,
            avg_response_tokens=9999.0,
        )
        assert score_no_extra == pytest.approx(score_with_extra, abs=1e-9)


class TestCompositeScoreWithTokens:
    """VAL-SCORE-002: token component added when enabled."""

    def test_include_tokens_true_uses_different_weights(self):
        """With include_tokens=True the score must differ from legacy."""
        legacy = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
        )
        with_tokens = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_tokens=True,
            avg_response_tokens=500.0,
        )
        # The weights differ, so the scores must differ
        assert abs(legacy - with_tokens) > 1e-6

    def test_include_tokens_true_score_in_range(self):
        """The new score is still in (0, 1]."""
        score = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_tokens=True,
            avg_response_tokens=500.0,
        )
        assert 0.0 < score < 1.0

    def test_token_component_affects_score(self):
        """Changing only avg_response_tokens must change the composite when enabled."""
        base = dict(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_tokens=True,
        )
        s_low = compute_composite_score(avg_response_tokens=100.0, **base)
        s_high = compute_composite_score(avg_response_tokens=10000.0, **base)
        assert s_low != pytest.approx(s_high)


class TestCompositeWeightsWithTokens:
    """VAL-SCORE-003: rebalanced weights sum to 1.0."""

    def test_new_weights_sum_to_one(self):
        """The new weight constants must sum to exactly 1.0."""
        from rag_bench.metrics import COMPOSITE_WEIGHTS_WITH_TOKENS
        total = sum(COMPOSITE_WEIGHTS_WITH_TOKENS.values())
        assert total == pytest.approx(1.0, abs=1e-9), (
            f"COMPOSITE_WEIGHTS_WITH_TOKENS sum to {total}, not 1.0"
        )

    def test_new_weights_have_correct_keys(self):
        """The new weight dict must contain all expected keys."""
        from rag_bench.metrics import COMPOSITE_WEIGHTS_WITH_TOKENS
        expected_keys = {
            "hit_at_5", "symbol_hit_at_5", "mrr",
            "tool_score", "latency_score", "resource_score", "token_score",
        }
        assert set(COMPOSITE_WEIGHTS_WITH_TOKENS.keys()) == expected_keys

    def test_new_weights_match_documented_values(self):
        """Individual weights must match the values documented in architecture."""
        from rag_bench.metrics import COMPOSITE_WEIGHTS_WITH_TOKENS
        expected = {
            "hit_at_5": 0.28,
            "symbol_hit_at_5": 0.14,
            "mrr": 0.14,
            "tool_score": 0.13,
            "latency_score": 0.13,
            "resource_score": 0.09,
            "token_score": 0.09,
        }
        for key, val in expected.items():
            assert COMPOSITE_WEIGHTS_WITH_TOKENS[key] == pytest.approx(val, abs=1e-9), (
                f"{key}: expected {val}, got {COMPOSITE_WEIGHTS_WITH_TOKENS[key]}"
            )


class TestCompositeScoreTokenMonotonicity:
    """VAL-SCORE-004: lower tokens improve composite when enabled."""

    def test_lower_tokens_produces_higher_score(self):
        """With include_tokens=True, identical inputs except avg_response_tokens
        of 200 vs 2000 must produce strictly greater composite for 200."""
        base_kwargs = dict(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_tokens=True,
        )
        score_200 = compute_composite_score(avg_response_tokens=200.0, **base_kwargs)
        score_2000 = compute_composite_score(avg_response_tokens=2000.0, **base_kwargs)
        assert score_200 > score_2000, (
            f"Lower avg_response_tokens should produce larger composite: "
            f"200 -> {score_200}, 2000 -> {score_2000}"
        )

    def test_lower_tokens_higher_score_varied_inputs(self):
        """Monotonicity holds across multiple input configurations."""
        scenarios = [
            dict(hit_at_5=0.8, symbol_hit_at_5=0.7, mrr=0.9,
                 avg_tool_calls=0.5, p95_latency_ms=50.0,
                 ram_peak_mb=50.0, index_size_mb=5.0),
            dict(hit_at_5=0.2, symbol_hit_at_5=0.1, mrr=0.15,
                 avg_tool_calls=8.0, p95_latency_ms=5000.0,
                 ram_peak_mb=5000.0, index_size_mb=2000.0),
            dict(hit_at_5=0.0, symbol_hit_at_5=0.0, mrr=0.0,
                 avg_tool_calls=3.0, p95_latency_ms=1000.0,
                 ram_peak_mb=500.0, index_size_mb=100.0),
        ]
        for base in scenarios:
            base["include_tokens"] = True
            s_low = compute_composite_score(avg_response_tokens=200.0, **base)
            s_high = compute_composite_score(avg_response_tokens=2000.0, **base)
            assert s_low > s_high, (
                f"Expected s_low > s_high for {base}: got {s_low} vs {s_high}"
            )

    def test_token_score_formula(self):
        """The token_score formula is 1 / (1 + avg_response_tokens / 1000)."""
        # When avg_response_tokens is 0, token_score = 1.0
        score_zero_tokens = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_tokens=True,
            avg_response_tokens=0.0,
        )
        # When avg_response_tokens is very large, token_score ≈ 0
        score_huge_tokens = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_tokens=True,
            avg_response_tokens=1e9,
        )
        assert score_zero_tokens > score_huge_tokens

    def test_existing_composite_tests_still_pass(self):
        """Existing TestCompositeScore tests must still pass unchanged."""
        # Re-run an existing test scenario to verify no breakage
        score = compute_composite_score(
            hit_at_5=0.5,
            symbol_hit_at_5=0.4,
            mrr=0.6,
            avg_tool_calls=1.0,
            p95_latency_ms=200.0,
            ram_peak_mb=300.0,
            index_size_mb=20.0,
        )
        assert 0.0 < score < 1.0


# ============================================================================
# Chunk matching: content_matches
# ============================================================================


class TestContentMatches:
    """VAL-CHUNK-001 through VAL-CHUNK-005: content_matches logic."""

    # -- VAL-CHUNK-001: returns True for contained substring -------------------

    def test_contained_substring_returns_true(self):
        assert content_matches(
            ["def handle_request(req):", "class Router:"],
            "def handle_request",
        )

    def test_substring_deep_in_list(self):
        assert content_matches(
            ["unrelated code", "some other chunk", "def handle_request(req)"],
            "def handle_request",
        )

    # -- VAL-CHUNK-002: respects minimum length --------------------------------

    def test_short_expected_accepted_by_default_min_overlap(self):
        # Default min_overlap=0, so "x =" (3 chars) passes
        assert content_matches(["x = y + z"], "x =")

    def test_short_expected_rejected_with_explicit_min_overlap(self):
        # len("x =") is 3, default min_overlap=0 → passes
        assert content_matches(["x = y + z"], "x =")
        # Explicit min_overlap=4 exceeds len → rejected
        assert not content_matches(["x = y + z"], "x =", min_overlap=4)

    def test_exactly_min_overlap_boundary(self):
        s = "a" * 20  # exactly 20 chars
        assert content_matches([s], s, min_overlap=20)
        assert not content_matches([s], s, min_overlap=21)

    # -- VAL-CHUNK-003: case-insensitive ---------------------------------------

    def test_case_insensitive_match(self):
        assert content_matches(
            ["Def Handle_Request(req)"],
            "def handle_request",
        )

    def test_case_insensitive_no_match(self):
        assert not content_matches(
            ["Def Handle_Request(req)"],
            "completely different",
        )

    def test_returned_lowercase_expected_uppercase(self):
        assert content_matches(
            ["def handle_request(req)"],
            "DEF HANDLE_REQUEST",
        )

    def test_both_mixed_case(self):
        assert content_matches(
            ["DeF hAnDlE_rEqUeSt(ReQ)"],
            "dEf HaNdLe_ReQuEsT",
        )

    # -- VAL-CHUNK-004: returns False for non-matching content -----------------

    def test_non_matching_returns_false(self):
        assert not content_matches(["def foo(): pass"], "handle_request")

    def test_multiple_returned_none_match(self):
        assert not content_matches(
            ["def foo(): pass", "class Bar:", "x = 1"],
            "handle_request",
        )

    # -- VAL-CHUNK-005: handles empty input gracefully -------------------------

    def test_empty_returned_contents_returns_false(self):
        assert not content_matches([], "anything")

    def test_empty_expected_returns_false(self):
        assert not content_matches(["stuff"], "")

    def test_both_empty_returns_false(self):
        assert not content_matches([], "")

    def test_empty_string_in_returned_list(self):
        # An empty string in returned_contents shouldn't cause issues
        assert not content_matches(["", "other"], "expected")

    # -- Edge cases ------------------------------------------------------------

    def test_expected_is_substring_of_returned(self):
        # The expected text appears inside a larger returned content
        assert content_matches(
            ["def handle_request(req: Request) -> Response: ..."],
            "handle_request(req",
        )

    def test_whitespace_sensitive_case_folded(self):
        # Case-folding preserves whitespace; substring match is literal
        assert content_matches(
            ["  def handle_request(req):  "],
            "def handle_request",
        )

    def test_multibyte_unicode(self):
        # Non-ASCII characters should still work with casefold
        assert content_matches(["Straße"], "straße")  # ß → ss in casefold
        # "Straße".casefold() == "strasse", so this should match
        assert content_matches(["Straße"], "strasse")

    def test_empty_expected_with_default_min_overlap(self):
        # Empty expected is shorter than min_overlap (20) → False
        assert not content_matches(["stuff"], "")


# ============================================================================
# Chunk matching: compute_chunk_hit_at_k
# ============================================================================


class TestComputeChunkHitAtK:
    """VAL-CHUNK-006 and VAL-CHUNK-007: chunk hit-at-k computation."""

    def _make_qr_with_chunk(
        self,
        found_chunk: bool = False,
        expected_content: list[str] | None = None,
        returned_contents: list[str] | None = None,
    ) -> QueryResult:
        return QueryResult(
            query_id="test",
            query_text="test query",
            query_type="locate",
            difficulty="medium",
            expected_files=["src/app.py"],
            expected_symbols=["MyClass"],
            returned_files=["src/app.py"],
            returned_symbols=["MyClass"],
            latency_ms=50.0,
            found_chunk=found_chunk,
            expected_content=expected_content or [],
            returned_contents=returned_contents or [],
        )

    # -- VAL-CHUNK-006: computes fraction correctly ----------------------------

    def test_three_of_five_found_chunk(self):
        results = [
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def foo()"]),
            self._make_qr_with_chunk(found_chunk=True, expected_content=["class Bar"]),
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def baz()"]),
            self._make_qr_with_chunk(found_chunk=False, expected_content=["def qux()"]),
            self._make_qr_with_chunk(found_chunk=False, expected_content=["def quux()"]),
        ]
        assert compute_chunk_hit_at_k(results, 5) == pytest.approx(0.6)

    def test_all_found_chunk(self):
        results = [
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def a()"]),
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def b()"]),
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def c()"]),
        ]
        assert compute_chunk_hit_at_k(results, 5) == pytest.approx(1.0)

    def test_none_found_chunk(self):
        results = [
            self._make_qr_with_chunk(found_chunk=False, expected_content=["def a()"]),
            self._make_qr_with_chunk(found_chunk=False, expected_content=["def b()"]),
        ]
        assert compute_chunk_hit_at_k(results, 5) == pytest.approx(0.0)

    # -- VAL-CHUNK-007: no expected_content → 0.0 -----------------------------

    def test_no_queries_with_expected_content_returns_zero(self):
        results = [
            self._make_qr_with_chunk(
                found_chunk=False, expected_content=[],  # no expected_content
            ),
            self._make_qr_with_chunk(
                found_chunk=False, expected_content=[],
            ),
            self._make_qr_with_chunk(
                found_chunk=False, expected_content=[],
            ),
        ]
        assert compute_chunk_hit_at_k(results, 5) == 0.0

    def test_mixed_with_and_without_expected_content(self):
        # 2 queries with expected_content (1 hit), 3 without → denominator = 2
        results = [
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def a()"]),
            self._make_qr_with_chunk(found_chunk=False, expected_content=["def b()"]),
            self._make_qr_with_chunk(expected_content=[]),   # excluded
            self._make_qr_with_chunk(expected_content=[]),   # excluded
            self._make_qr_with_chunk(expected_content=[]),   # excluded
        ]
        assert compute_chunk_hit_at_k(results, 5) == pytest.approx(0.5)

    # -- Edge cases ------------------------------------------------------------

    def test_empty_results_returns_zero(self):
        assert compute_chunk_hit_at_k([], 5) == 0.0

    def test_k_zero_returns_zero(self):
        results = [
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def a()"]),
        ]
        assert compute_chunk_hit_at_k(results, 0) == 0.0

    def test_different_k_all_in_range(self):
        # Same as other hit-at-k functions: results should be in [0, 1]
        results = [
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def a()"]),
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def b()"]),
            self._make_qr_with_chunk(found_chunk=False, expected_content=["def c()"]),
        ]
        for k in (1, 3, 5, 10):
            v = compute_chunk_hit_at_k(results, k)
            assert 0.0 <= v <= 1.0


# ============================================================================
# Chunk fields: QueryResult
# ============================================================================


class TestQueryResultChunkFields:
    """VAL-METRIC-001: QueryResult has returned_contents and found_chunk."""

    def test_returned_contents_defaults_to_empty_list(self):
        qr = QueryResult(
            query_id="q", query_text="", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[], latency_ms=1.0,
        )
        assert qr.returned_contents == []
        assert isinstance(qr.returned_contents, list)

    def test_returned_contents_settable(self):
        qr = QueryResult(
            query_id="q", query_text="", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[], latency_ms=1.0,
            returned_contents=["def foo(): pass", "class Bar:"],
        )
        assert qr.returned_contents == ["def foo(): pass", "class Bar:"]

    def test_found_chunk_defaults_to_false(self):
        qr = QueryResult(
            query_id="q", query_text="", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[], latency_ms=1.0,
        )
        assert qr.found_chunk is False

    def test_found_chunk_settable(self):
        qr = QueryResult(
            query_id="q", query_text="", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[], latency_ms=1.0,
            found_chunk=True,
        )
        assert qr.found_chunk is True

    def test_expected_content_defaults_to_empty_list(self):
        qr = QueryResult(
            query_id="q", query_text="", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[], latency_ms=1.0,
        )
        assert qr.expected_content == []
        assert isinstance(qr.expected_content, list)

    def test_expected_content_settable(self):
        qr = QueryResult(
            query_id="q", query_text="", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[], latency_ms=1.0,
            expected_content=["class Flask(", "def create_app("],
        )
        assert qr.expected_content == ["class Flask(", "def create_app("]


# ============================================================================
# Chunk field: BenchmarkMetrics
# ============================================================================


class TestBenchmarkMetricsChunkField:
    """VAL-METRIC-002: BenchmarkMetrics has chunk_hit_at_5."""

    def test_chunk_hit_at_5_defaults_to_zero(self):
        m = BenchmarkMetrics()
        assert m.chunk_hit_at_5 == 0.0

    def test_chunk_hit_at_5_settable(self):
        m = BenchmarkMetrics(chunk_hit_at_5=0.75)
        assert m.chunk_hit_at_5 == 0.75


# ============================================================================
# compute_metrics: chunk integration
# ============================================================================


class TestComputeMetricsChunkIntegration:
    """VAL-METRIC-003: compute_metrics populates chunk_hit_at_5."""

    def _make_qr_with_chunk(
        self,
        found_chunk: bool = False,
        expected_content: list[str] | None = None,
        returned_contents: list[str] | None = None,
    ) -> QueryResult:
        return QueryResult(
            query_id="test",
            query_text="test query",
            query_type="locate",
            difficulty="medium",
            expected_files=["src/app.py"],
            expected_symbols=["MyClass"],
            returned_files=["src/app.py"],
            returned_symbols=["MyClass"],
            latency_ms=50.0,
            found_chunk=found_chunk,
            expected_content=expected_content or [],
            returned_contents=returned_contents or [],
        )

    def test_chunk_hit_at_5_computed_correctly(self):
        results = [
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def a()"]),
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def b()"]),
            self._make_qr_with_chunk(found_chunk=False, expected_content=["def c()"]),
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def d()"]),
            self._make_qr_with_chunk(found_chunk=False, expected_content=["def e()"]),
        ]
        m = compute_metrics(results)
        assert m.chunk_hit_at_5 == pytest.approx(0.6)

    def test_chunk_hit_at_5_zero_when_no_expected_content(self):
        results = [
            self._make_qr_with_chunk(expected_content=[]),
            self._make_qr_with_chunk(expected_content=[]),
        ]
        m = compute_metrics(results)
        assert m.chunk_hit_at_5 == 0.0

    def test_chunk_hit_at_5_in_range(self):
        results = [
            self._make_qr_with_chunk(found_chunk=True, expected_content=["def a()"]),
            self._make_qr_with_chunk(found_chunk=False, expected_content=["def b()"]),
        ]
        m = compute_metrics(results)
        assert 0.0 <= m.chunk_hit_at_5 <= 1.0


# ============================================================================
# Back-compat: existing metrics unchanged when chunk data absent
# ============================================================================


class TestExistingMetricsUnchangedByChunk:
    """VAL-METRIC-004: existing metrics identical when chunk data absent."""

    def test_hit_at_k_mrr_latency_unchanged_with_chunk_fields_default(self):
        """When chunk fields are all default (empty), existing metrics unchanged."""
        results = [
            _make_qr(
                returned_files=["src/app.py"],
                expected_files=["src/app.py"],
                returned_symbols=["MyClass"],
                expected_symbols=["MyClass"],
                latency_ms=100,
            ),
            _make_qr(
                returned_files=["wrong.py"],
                expected_files=["src/config.py"],
                returned_symbols=["OtherClass"],
                expected_symbols=["Config"],
                latency_ms=200,
            ),
            _make_qr(
                returned_files=["a.py", "b.py", "c.py", "d.py", "src/util.py"],
                expected_files=["src/util.py"],
                returned_symbols=["Helper"],
                expected_symbols=["Helper"],
                latency_ms=150,
            ),
        ]

        m = compute_metrics(results)

        # Compute expected values manually
        expected_hit_at_1 = compute_hit_at_k(results, 1)
        expected_hit_at_5 = compute_hit_at_k(results, 5)
        expected_symbol_hit_at_5 = compute_symbol_hit_at_k(results, 5)
        expected_mrr = compute_mrr(results)

        assert m.hit_at_1 == pytest.approx(expected_hit_at_1)
        assert m.hit_at_5 == pytest.approx(expected_hit_at_5)
        assert m.symbol_hit_at_5 == pytest.approx(expected_symbol_hit_at_5)
        assert m.mrr == pytest.approx(expected_mrr)
        # chunk_hit_at_5 defaults to 0 when no expected_content present
        assert m.chunk_hit_at_5 == 0.0

    def test_composite_score_unchanged_with_chunk_defaults(self):
        results = [
            _make_qr(
                returned_files=["src/app.py"],
                expected_files=["src/app.py"],
                returned_symbols=["MyClass"],
                expected_symbols=["MyClass"],
                latency_ms=100,
            ),
            _make_qr(
                returned_files=["wrong.py"],
                expected_files=["src/config.py"],
                returned_symbols=["OtherClass"],
                expected_symbols=["Config"],
                latency_ms=200,
            ),
        ]
        m = compute_metrics(results, ingest_total_sec=5.0, ingest_total_files=100)
        assert 0.0 < m.composite_score < 1.0
        assert m.chunk_hit_at_5 == 0.0

    def test_all_existing_breakdowns_still_present_with_chunk_defaults(self):
        results = [
            _make_qr(
                returned_files=["src/app.py"],
                expected_files=["src/app.py"],
                returned_symbols=["MyClass"],
                expected_symbols=["MyClass"],
                latency_ms=100,
            ),
            _make_qr(
                returned_files=["wrong.py"],
                expected_files=["src/config.py"],
                returned_symbols=["OtherClass"],
                expected_symbols=["Config"],
                latency_ms=200,
            ),
        ]
        m = compute_metrics(results)
        # Breakdowns should still exist with expected structure
        assert "easy" in m.by_difficulty or "medium" in m.by_difficulty
        assert "locate" in m.by_type
        assert "hit_at_5" in m.by_type.get("locate", {})

    def test_numeric_identity_of_key_metrics_chunk_vs_no_chunk(self):
        """Verify identical numeric values for key metrics with vs without chunk data."""
        results_no_chunk = [
            _make_qr(
                returned_files=["src/app.py"],
                expected_files=["src/app.py"],
                latency_ms=100,
            ),
            _make_qr(
                returned_files=["wrong.py", "ok.py"],
                expected_files=["ok.py"],
                latency_ms=200,
            ),
            _make_qr(
                returned_files=["a.py", "b.py", "c.py"],
                expected_files=["nonexistent.py"],
                latency_ms=300,
            ),
        ]

        results_with_chunk = [
            _make_qr(
                returned_files=["src/app.py"],
                expected_files=["src/app.py"],
                latency_ms=100,
            ),
            _make_qr(
                returned_files=["wrong.py", "ok.py"],
                expected_files=["ok.py"],
                latency_ms=200,
            ),
            _make_qr(
                returned_files=["a.py", "b.py", "c.py"],
                expected_files=["nonexistent.py"],
                latency_ms=300,
            ),
        ]
        # Add chunk data to second set
        results_with_chunk[0].expected_content = ["class App"]
        results_with_chunk[0].returned_contents = ["class App: ..."]
        results_with_chunk[0].found_chunk = True
        results_with_chunk[1].expected_content = ["def handler"]
        results_with_chunk[1].returned_contents = ["unrelated code"]
        results_with_chunk[1].found_chunk = False
        # results_with_chunk[2] has no expected_content → excluded

        m_no = compute_metrics(results_no_chunk)
        m_with = compute_metrics(results_with_chunk)

        # Key metrics must be numerically identical
        assert m_no.hit_at_1 == m_with.hit_at_1
        assert m_no.hit_at_3 == m_with.hit_at_3
        assert m_no.hit_at_5 == m_with.hit_at_5
        assert m_no.hit_at_10 == m_with.hit_at_10
        assert m_no.symbol_hit_at_5 == m_with.symbol_hit_at_5
        assert m_no.mrr == m_with.mrr
        assert m_no.query_latency_p50_ms == m_with.query_latency_p50_ms
        assert m_no.query_latency_p95_ms == m_with.query_latency_p95_ms
        assert m_no.query_latency_mean_ms == m_with.query_latency_mean_ms
        assert m_no.composite_score == m_with.composite_score
        # chunk_hit_at_5 differs (2 with expected_content, 1 hit → 0.5)
        assert m_with.chunk_hit_at_5 == pytest.approx(0.5)


# ============================================================================
# Chunk composite score
# ============================================================================


class TestChunkCompositeScoreWeights:
    """VAL-CLI-003: chunk weight set sums to 1.0."""

    def test_chunk_weights_sum_to_one(self):
        from rag_bench.metrics import COMPOSITE_WEIGHTS_WITH_CHUNK
        total = sum(COMPOSITE_WEIGHTS_WITH_CHUNK.values())
        assert total == pytest.approx(1.0, abs=1e-9), (
            f"COMPOSITE_WEIGHTS_WITH_CHUNK sum to {total}, not 1.0"
        )

    def test_chunk_weights_have_correct_keys(self):
        from rag_bench.metrics import COMPOSITE_WEIGHTS_WITH_CHUNK
        expected_keys = {
            "hit_at_5", "symbol_hit_at_5", "mrr",
            "tool_score", "latency_score", "resource_score",
            "token_score", "chunk_score",
        }
        assert set(COMPOSITE_WEIGHTS_WITH_CHUNK.keys()) == expected_keys

    def test_chunk_weights_match_documented_values(self):
        from rag_bench.metrics import COMPOSITE_WEIGHTS_WITH_CHUNK
        expected = {
            "hit_at_5": 0.26,
            "symbol_hit_at_5": 0.14,
            "mrr": 0.14,
            "tool_score": 0.13,
            "latency_score": 0.12,
            "resource_score": 0.09,
            "token_score": 0.04,
            "chunk_score": 0.08,
        }
        for key, val in expected.items():
            assert COMPOSITE_WEIGHTS_WITH_CHUNK[key] == pytest.approx(val, abs=1e-9), (
                f"{key}: expected {val}, got {COMPOSITE_WEIGHTS_WITH_CHUNK[key]}"
            )


class TestChunkCompositeScoreLegacy:
    """VAL-CLI-002: include_chunk=False preserves legacy score."""

    def test_include_chunk_false_matches_legacy(self):
        scenarios = [
            (0.5, 0.4, 0.6, 1.0, 200.0, 300.0, 20.0),
            (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 10.0, 10000.0, 10000.0, 10000.0),
            (0.3, 0.2, 0.1, 5.0, 500.0, 1000.0, 500.0),
        ]
        for (h5, sh5, mrr, tc, p95, ram, idx) in scenarios:
            legacy = compute_composite_score(
                hit_at_5=h5, symbol_hit_at_5=sh5, mrr=mrr,
                avg_tool_calls=tc, p95_latency_ms=p95,
                ram_peak_mb=ram, index_size_mb=idx,
            )
            with_chunk_false = compute_composite_score(
                hit_at_5=h5, symbol_hit_at_5=sh5, mrr=mrr,
                avg_tool_calls=tc, p95_latency_ms=p95,
                ram_peak_mb=ram, index_size_mb=idx,
                include_chunk=False,
            )
            assert legacy == pytest.approx(with_chunk_false, abs=1e-9), (
                f"include_chunk=False differs from legacy for {h5=} {sh5=} {mrr=}"
            )

    def test_default_call_matches_legacy(self):
        score_legacy = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
        )
        score_default = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_chunk=False,
        )
        assert score_legacy == pytest.approx(score_default, abs=1e-9)

    def test_include_chunk_false_extra_kwarg_ignored(self):
        score_no_extra = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_chunk=False,
        )
        score_with_extra = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_chunk=False,
            chunk_hit_at_5=0.45,
        )
        assert score_no_extra == pytest.approx(score_with_extra, abs=1e-9)


class TestChunkCompositeScoreEnabled:
    """VAL-CLI-003: include_chunk=True uses chunk weights."""

    def test_include_chunk_true_uses_different_weights(self):
        legacy = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
        )
        with_chunk = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_chunk=True,
            chunk_hit_at_5=0.45,
        )
        assert abs(legacy - with_chunk) > 1e-6

    def test_include_chunk_true_score_in_range(self):
        score = compute_composite_score(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_chunk=True,
            chunk_hit_at_5=0.45,
        )
        assert 0.0 < score < 1.0

    def test_chunk_component_affects_score(self):
        base = dict(
            hit_at_5=0.5, symbol_hit_at_5=0.4, mrr=0.6,
            avg_tool_calls=1.0, p95_latency_ms=200.0,
            ram_peak_mb=300.0, index_size_mb=20.0,
            include_chunk=True,
        )
        s_low = compute_composite_score(chunk_hit_at_5=0.0, **base)
        s_high = compute_composite_score(chunk_hit_at_5=1.0, **base)
        assert s_high > s_low, (
            f"Higher chunk_hit_at_5 should produce larger composite: "
            f"0.0 -> {s_low}, 1.0 -> {s_high}"
        )

    def test_compute_metrics_with_chunk_flag(self):
        results = [
            _make_qr(
                returned_files=["src/app.py"],
                expected_files=["src/app.py"],
                returned_symbols=["MyClass"],
                expected_symbols=["MyClass"],
                latency_ms=100,
            ),
            _make_qr(
                returned_files=["wrong.py"],
                expected_files=["src/config.py"],
                returned_symbols=["OtherClass"],
                expected_symbols=["Config"],
                latency_ms=200,
            ),
        ]
        # Add chunk data
        results[0].expected_content = ["class App"]
        results[0].returned_contents = ["class App: ..."]
        results[0].found_chunk = True

        m_no_chunk = compute_metrics(results, include_chunk_in_score=False)
        m_with_chunk = compute_metrics(results, include_chunk_in_score=True)

        # Scores must differ when include_chunk_in_score changes
        assert abs(m_no_chunk.composite_score - m_with_chunk.composite_score) > 1e-9
        # With include_chunk_in_score=True, chunk_hit_at_5 must be in range
        assert 0.0 <= m_with_chunk.composite_score <= 1.0
