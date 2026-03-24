"""Tests for metrics computation."""

from rag_bench.metrics import (
    QueryResult,
    compute_hit_at_k,
    compute_metrics,
    compute_mrr,
    compute_percentile,
    compute_symbol_hit_at_k,
    file_matches,
    normalize_path,
    symbol_matches,
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


class TestSymbolMatches:
    def test_exact(self):
        assert symbol_matches(["Flask", "route"], "Flask")

    def test_partial(self):
        assert symbol_matches(["FlaskApp"], "Flask")

    def test_no_match(self):
        assert not symbol_matches(["Blueprint"], "Flask")


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
