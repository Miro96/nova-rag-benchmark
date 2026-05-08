"""Tests for metrics computation."""

import os
import threading
import time

import pytest

from rag_bench.metrics import (
    MemorySampler,
    QueryResult,
    compute_hit_at_k,
    compute_latency_stats,
    compute_metrics,
    compute_mrr,
    compute_percentile,
    compute_symbol_hit_at_k,
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
