"""Tests for dataset loading and runner utilities."""

import os
import tempfile
from pathlib import Path

import pytest

from rag_bench.datasets.loader import PUBLIC_REPOS, load_queries, load_repos

def _public_queries():
    """Shipped dataset only — private overlay sets must not break the suite."""
    return [q for q in load_queries() if q.repo in PUBLIC_REPOS]

from rag_bench.runner import _estimate_index_size


class TestEstimateIndexSize:
    """Tests for _estimate_index_size — index size estimation from preset config."""

    def test_index_dir_explicit_override_wins(self, tmp_path):
        """Step 1: explicit index_dir takes precedence."""
        index_dir = tmp_path / "my_index"
        index_dir.mkdir()
        (index_dir / "data.bin").write_bytes(b"x" * 1024 * 512)  # ~0.5 MB

        config = {"index_dir": str(index_dir)}
        size = _estimate_index_size(config)
        assert size > 0.0, f"Expected > 0 for explicit index_dir, got {size}"

    def test_index_dirs_resolved_per_repo(self, tmp_path, monkeypatch):
        """Step 2: index_dirs entries are resolved inside each repo directory.

        Uses monkeypatch to ensure ~/.nova-rag fallback is empty so the test
        only measures the per-repo index directories.
        """
        repo_a = tmp_path / "repo_a"
        repo_a.mkdir()
        cocoindex_dir = repo_a / ".cocoindex"
        cocoindex_dir.mkdir()
        (cocoindex_dir / "index.db").write_bytes(b"x" * 1024 * 1024)  # ~1 MB

        repo_b = tmp_path / "repo_b"
        repo_b.mkdir()
        cocoindex_dir_b = repo_b / ".cocoindex"
        cocoindex_dir_b.mkdir()
        (cocoindex_dir_b / "index.db").write_bytes(b"x" * 1024 * 512)  # ~0.5 MB

        # Prevent fallback to ~/.nova-rag
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.delenv("NOVA_RAG_DATA_DIR", raising=False)

        config = {"index_dirs": [".cocoindex"]}
        repo_dirs = {"a": repo_a, "b": repo_b}

        size = _estimate_index_size(config, repo_dirs)
        assert size > 0.0, f"index_size_mb should be > 0 for .cocoindex dirs, got {size}"
        # Should be roughly ~1.5 MB
        assert size >= 1.0, f"Expected at least 1.0 MB, got {size}"

    def test_index_dirs_with_nova_rag(self, tmp_path, monkeypatch):
        """index_dirs=['.nova-rag'] finds .nova-rag inside repo dirs."""
        repo = tmp_path / "repo"
        repo.mkdir()
        nova_dir = repo / ".nova-rag"
        nova_dir.mkdir()
        (nova_dir / "data").write_bytes(b"x" * 1024 * 512)

        # Prevent fallback to ~/.nova-rag
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.delenv("NOVA_RAG_DATA_DIR", raising=False)

        config = {"index_dirs": [".nova-rag"]}
        repo_dirs = {"test": repo}

        size = _estimate_index_size(config, repo_dirs)
        assert size > 0.0
        assert size < 1.0, f"Should be ~0.5 MB from repo/.nova-rag, not fallback, got {size}"

    def test_index_dirs_ignored_when_no_repo_dirs(self, tmp_path):
        """When no repo_dirs are provided, index_dirs cannot be resolved and returns 0."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".cocoindex").mkdir()

        config = {"index_dirs": [".cocoindex"]}
        # No repo_dirs → step 2 skipped, falls through to env var / ~/.nova-rag
        size = _estimate_index_size(config, None)
        # Will likely be 0 (no env var set, ~/.nova-rag may not exist)
        # But at minimum, doesn't crash
        assert size >= 0.0

    def test_index_dirs_empty_when_index_missing(self, tmp_path):
        """When index_dirs are declared but the directories don't exist, returns 0 from this step."""
        repo = tmp_path / "repo"
        repo.mkdir()
        # No .cocoindex directory created

        config = {"index_dirs": [".cocoindex"]}
        repo_dirs = {"test": repo}

        size = _estimate_index_size(config, repo_dirs)
        # Falls through to env var / ~/.nova-rag fallback
        assert size >= 0.0, f"Expected >= 0, got {size}"

    def test_index_dir_overrides_index_dirs(self, tmp_path, monkeypatch):
        """Step 1 (index_dir) takes precedence over step 2 (index_dirs)."""
        # Create an explicit index
        explicit_dir = tmp_path / "explicit_index"
        explicit_dir.mkdir()
        (explicit_dir / "data.bin").write_bytes(b"x" * 1024 * 512)  # ~0.5 MB

        # Create a per-repo index that would be found via index_dirs
        repo = tmp_path / "repo"
        repo.mkdir()
        per_repo_dir = repo / ".cocoindex"
        per_repo_dir.mkdir()
        (per_repo_dir / "big.db").write_bytes(b"x" * 1024 * 1024 * 5)  # ~5 MB

        # Prevent fallback
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.delenv("NOVA_RAG_DATA_DIR", raising=False)

        config = {
            "index_dir": str(explicit_dir),
            "index_dirs": [".cocoindex"],
        }
        repo_dirs = {"test": repo}

        size = _estimate_index_size(config, repo_dirs)
        # Should use explicit_dir (~0.5 MB), not per-repo (~5 MB)
        assert size > 0.0
        assert size < 2.0, f"Should use explicit index_dir (~0.5 MB), not per-repo (~5 MB), got {size}"

    def test_backward_compat_nova_rag_data_dir(self, tmp_path, monkeypatch):
        """NOVA_RAG_DATA_DIR env var is still honored as fallback."""
        nova_dir = tmp_path / "custom_nova"
        nova_dir.mkdir()
        (nova_dir / "data").write_bytes(b"x" * 1024 * 512)

        monkeypatch.setenv("NOVA_RAG_DATA_DIR", str(nova_dir))

        config: dict = {}
        size = _estimate_index_size(config)
        assert size > 0.0, f"NOVA_RAG_DATA_DIR should be honored, got {size}"

    def test_backward_compat_default_nova_rag(self, tmp_path, monkeypatch):
        """~/.nova-rag is the ultimate fallback."""
        home_nova = tmp_path / ".nova-rag"
        home_nova.mkdir()
        (home_nova / "data").write_bytes(b"x" * 1024 * 256)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        config: dict = {}
        size = _estimate_index_size(config)
        assert size > 0.0

    def test_returns_zero_when_nothing_exists(self, tmp_path, monkeypatch):
        """Returns 0.0 when no index exists anywhere."""
        # Ensure NOVA_RAG_DATA_DIR is not set
        monkeypatch.delenv("NOVA_RAG_DATA_DIR", raising=False)

        # Make ~/.nova-rag not exist by patching home
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: empty_home)

        config: dict = {}
        size = _estimate_index_size(config)
        assert size == 0.0

    def test_multiple_index_dir_names(self, tmp_path, monkeypatch):
        """Multiple index_dir names are all scanned."""
        repo = tmp_path / "repo"
        repo.mkdir()
        idx1 = repo / ".index_a"
        idx1.mkdir()
        (idx1 / "a.db").write_bytes(b"x" * 1024 * 512)
        idx2 = repo / ".index_b"
        idx2.mkdir()
        (idx2 / "b.db").write_bytes(b"x" * 1024 * 256)

        # Prevent fallback
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.delenv("NOVA_RAG_DATA_DIR", raising=False)

        config = {"index_dirs": [".index_a", ".index_b"]}
        repo_dirs = {"test": repo}

        size = _estimate_index_size(config, repo_dirs)
        # Should sum both
        assert size >= 0.5, f"Expected >= 0.5 MB (sum of both), got {size}"


class TestDatasetLoader:
    def test_load_repos(self):
        repos = [r for r in load_repos() if r.name in PUBLIC_REPOS]
        assert len(repos) == 4
        names = {r.name for r in repos}
        assert names == {"flask", "fastapi", "express", "django"}

    def test_load_all_queries(self):
        queries = _public_queries()
        assert len(queries) == 135  # 35 per original repo + 30 django

    def test_load_filtered_queries(self):
        queries = load_queries("flask")
        assert len(queries) == 35
        assert all(q.repo == "flask" for q in queries)

    def test_query_fields(self):
        queries = load_queries("flask")
        q = queries[0]
        assert q.id == "flask_001"
        assert q.type in ("locate", "callers", "explain", "impact",
                          "multi_hop", "cross_package", "architecture",
                          "dead_code", "conditional_path", "test_traceability")
        assert q.difficulty in ("easy", "medium", "hard")
        assert len(q.expected_files) > 0
        assert len(q.query) > 0

    def test_query_types_distribution(self):
        queries = _public_queries()
        types = {q.type for q in queries}
        # Original types + 7 new complex query types from M1
        assert types == {"locate", "callers", "explain", "impact",
                         "multi_hop", "cross_package", "architecture",
                         "dead_code", "conditional_path", "test_traceability"}

    def test_difficulty_distribution(self):
        queries = _public_queries()
        difficulties = {q.difficulty for q in queries}
        assert difficulties == {"easy", "medium", "hard"}


class TestQueryDetailTokens:
    """Test that _query_detail includes response_tokens."""

    def test_query_detail_includes_response_tokens(self):
        from rag_bench.metrics import QueryResult
        from rag_bench.runner import _query_detail

        qr = QueryResult(
            query_id="test_001",
            query_text="test query",
            query_type="locate",
            difficulty="easy",
            expected_files=["test.py"],
            expected_symbols=[],
            returned_files=["test.py"],
            returned_symbols=[],
            latency_ms=50.0,
            tool_calls=2,
            repo="flask",
            response_tokens=150,
        )
        detail = _query_detail(qr)
        assert "response_tokens" in detail
        assert detail["response_tokens"] == 150

    def test_query_detail_response_tokens_none_by_default(self):
        from rag_bench.metrics import QueryResult
        from rag_bench.runner import _query_detail

        qr = QueryResult(
            query_id="test_001",
            query_text="test query",
            query_type="locate",
            difficulty="easy",
            expected_files=["test.py"],
            expected_symbols=[],
            returned_files=["test.py"],
            returned_symbols=[],
            latency_ms=50.0,
            tool_calls=2,
            repo="flask",
        )
        detail = _query_detail(qr)
        assert "response_tokens" in detail
        assert detail["response_tokens"] is None

    def test_query_detail_error_preserves_response_tokens(self):
        from rag_bench.metrics import QueryResult
        from rag_bench.runner import _query_detail

        qr = QueryResult(
            query_id="test_001",
            query_text="test query",
            query_type="locate",
            difficulty="easy",
            expected_files=["test.py"],
            expected_symbols=[],
            returned_files=[],
            returned_symbols=[],
            latency_ms=0,
            tool_calls=0,
            repo="flask",
            error="timeout",
            response_tokens=None,
        )
        detail = _query_detail(qr)
        assert "error" in detail
        assert detail["error"] == "timeout"
        assert "response_tokens" in detail
        assert detail["response_tokens"] is None


class TestReplicateSummaryTokens:
    """Test that _replicate_summary includes avg_response_tokens."""

    def test_replicate_summary_includes_avg_response_tokens(self):
        from rag_bench.metrics import BenchmarkMetrics
        from rag_bench.runner import _replicate_summary

        m = BenchmarkMetrics(
            hit_at_5=0.8,
            avg_response_tokens=1234.5,
        )
        summary = _replicate_summary([m])
        assert len(summary) == 1
        assert "avg_response_tokens" in summary[0]
        assert summary[0]["avg_response_tokens"] == 1234.5
        # Pre-existing keys remain
        assert "hit_at_5" in summary[0]
        assert "hit_at_1" in summary[0]


class TestBuildResultJsonTokens:
    """Test that _build_result_json includes all required token fields."""

    def test_retrieval_tokens_block(self):
        from rag_bench.metrics import BenchmarkMetrics, QueryResult
        from rag_bench.runner import _build_result_json

        metrics = BenchmarkMetrics(
            avg_response_tokens=500.0,
            p50_response_tokens=450.0,
            p95_response_tokens=900.0,
            total_response_tokens=15000,
        )
        result = _build_result_json(
            run_id="test-run",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=[],
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
        )
        tokens = result["retrieval"]["tokens"]
        assert tokens["avg"] == 500.0
        assert tokens["p50"] == 450.0
        assert tokens["p95"] == 900.0
        assert tokens["total"] == 15000

    def test_environment_tokenizer_field(self):
        from rag_bench.metrics import BenchmarkMetrics
        from rag_bench.runner import _build_result_json

        metrics = BenchmarkMetrics()
        result = _build_result_json(
            run_id="test-run",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=[],
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
            tokenizer_name="simple",
        )
        assert result["environment"]["tokenizer"] == "simple"

    def test_environment_tokenizer_default(self):
        from rag_bench.metrics import BenchmarkMetrics
        from rag_bench.runner import _build_result_json

        metrics = BenchmarkMetrics()
        result = _build_result_json(
            run_id="test-run",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=[],
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
        )
        assert result["environment"]["tokenizer"] == "tiktoken"

    def test_efficiency_includes_avg_response_tokens(self):
        from rag_bench.metrics import BenchmarkMetrics
        from rag_bench.runner import _build_result_json

        metrics = BenchmarkMetrics(avg_response_tokens=777.7)
        result = _build_result_json(
            run_id="test-run",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=[],
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
        )
        assert result["efficiency"]["avg_response_tokens"] == 777.7

    def test_query_details_include_response_tokens(self):
        from rag_bench.metrics import BenchmarkMetrics, QueryResult
        from rag_bench.runner import _build_result_json

        qr = QueryResult(
            query_id="test_001",
            query_text="test",
            query_type="locate",
            difficulty="easy",
            expected_files=[],
            expected_symbols=[],
            returned_files=[],
            returned_symbols=[],
            latency_ms=50.0,
            tool_calls=1,
            repo="flask",
            response_tokens=200,
        )
        metrics = BenchmarkMetrics(total_queries=1, total_hits=0)
        result = _build_result_json(
            run_id="test-run",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=[qr],
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
        )
        assert len(result["query_details"]) == 1
        assert result["query_details"][0]["response_tokens"] == 200

    def test_back_compat_existing_keys_preserved(self):
        """Pre-existing keys in result JSON retain their shape/type."""
        from rag_bench.metrics import BenchmarkMetrics, QueryResult
        from rag_bench.runner import _build_result_json

        qr = QueryResult(
            query_id="test_001",
            query_text="test query",
            query_type="locate",
            difficulty="easy",
            expected_files=["test.py"],
            expected_symbols=["my_func"],
            returned_files=["test.py"],
            returned_symbols=["my_func"],
            latency_ms=50.0,
            tool_calls=2,
            repo="flask",
            response_tokens=100,
        )
        metrics = BenchmarkMetrics(
            hit_at_1=0.5,
            hit_at_5=0.8,
            mrr=0.6,
            query_latency_p50_ms=45.0,
            query_latency_p95_ms=90.0,
            total_queries=1,
            total_hits=1,
            avg_response_tokens=100.0,
            p50_response_tokens=100.0,
            p95_response_tokens=100.0,
            total_response_tokens=100,
        )
        result = _build_result_json(
            run_id="test-run",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=[qr],
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
        )

        # Pre-existing top-level keys
        for key in ("run_id", "bench_version", "server", "environment",
                     "repos", "replicates", "iqr", "ingest",
                     "retrieval", "efficiency", "composite_score",
                     "by_difficulty", "by_type", "by_repo", "query_details"):
            assert key in result, f"Missing pre-existing key: {key}"

        # Pre-existing retrieval sub-keys
        for key in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
                     "symbol_hit_at_5", "mrr", "latency", "total_queries", "total_hits"):
            assert key in result["retrieval"], f"Missing retrieval key: {key}"

        # Pre-existing latency sub-keys
        for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms"):
            assert key in result["retrieval"]["latency"], f"Missing latency key: {key}"

        # Pre-existing query_detail keys
        qd = result["query_details"][0]
        for key in ("id", "type", "difficulty", "repo", "found_file",
                     "found_symbol", "latency_ms", "tool_calls",
                     "returned_files", "returned_symbols",
                     "expected_files", "expected_symbols"):
            assert key in qd, f"Missing query_detail key: {key}"

        # Types preserved
        assert isinstance(result["composite_score"], float)
        assert isinstance(qd["latency_ms"], float)
        assert isinstance(qd["found_file"], bool)


class TestRunBenchmarkSignature:
    """Test that run_benchmark accepts tokenizer parameters."""

    def test_run_benchmark_has_tokenizer_params(self):
        import inspect
        from rag_bench.runner import run_benchmark

        sig = inspect.signature(run_benchmark)
        params = sig.parameters
        assert "tokenizer_name" in params
        assert "token_encoding" in params
        assert "include_tokens_in_score" in params

        # Defaults preserve back-compat
        assert params["tokenizer_name"].default == "tiktoken"
        assert params["token_encoding"].default is None
        assert params["include_tokens_in_score"].default is False


# ============================================================================
# Chunk runner integration tests
# ============================================================================


class TestQueryDetailChunkFields:
    """VAL-RUNNER-001, VAL-RUNNER-004: _query_detail includes chunk fields."""

    def test_query_detail_includes_returned_contents(self):
        from rag_bench.metrics import QueryResult
        from rag_bench.runner import _query_detail

        qr = QueryResult(
            query_id="test_001",
            query_text="test query",
            query_type="locate",
            difficulty="easy",
            expected_files=["test.py"],
            expected_symbols=[],
            returned_files=["test.py"],
            returned_symbols=[],
            latency_ms=50.0,
            tool_calls=2,
            repo="flask",
            returned_contents=["def foo(): pass", "class Bar:"],
        )
        detail = _query_detail(qr)
        assert "returned_contents" in detail
        assert detail["returned_contents"] == ["def foo(): pass", "class Bar:"]
        assert "found_chunk" in detail
        assert detail["found_chunk"] is False

    def test_query_detail_includes_found_chunk_true(self):
        from rag_bench.metrics import QueryResult
        from rag_bench.runner import _query_detail

        qr = QueryResult(
            query_id="test_001",
            query_text="test query",
            query_type="locate",
            difficulty="easy",
            expected_files=["test.py"],
            expected_symbols=[],
            returned_files=["test.py"],
            returned_symbols=[],
            latency_ms=50.0,
            tool_calls=2,
            repo="flask",
            returned_contents=["def handle_request(req):"],
            found_chunk=True,
            expected_content=["def handle_request"],
        )
        detail = _query_detail(qr)
        assert detail["found_chunk"] is True
        assert "expected_content" in detail
        assert detail["expected_content"] == ["def handle_request"]

    def test_query_detail_returned_contents_capped_at_5(self):
        from rag_bench.metrics import QueryResult
        from rag_bench.runner import _query_detail

        qr = QueryResult(
            query_id="test_001",
            query_text="test query",
            query_type="locate",
            difficulty="easy",
            expected_files=["test.py"],
            expected_symbols=[],
            returned_files=["test.py"],
            returned_symbols=[],
            latency_ms=50.0,
            tool_calls=2,
            repo="flask",
            returned_contents=[f"chunk_{i}" for i in range(10)],
        )
        detail = _query_detail(qr)
        assert len(detail["returned_contents"]) == 5
        assert detail["returned_contents"] == ["chunk_0", "chunk_1", "chunk_2", "chunk_3", "chunk_4"]

    def test_query_detail_error_preserves_chunk_fields(self):
        from rag_bench.metrics import QueryResult
        from rag_bench.runner import _query_detail

        qr = QueryResult(
            query_id="test_001",
            query_text="test query",
            query_type="locate",
            difficulty="easy",
            expected_files=["test.py"],
            expected_symbols=[],
            returned_files=[],
            returned_symbols=[],
            latency_ms=0,
            tool_calls=0,
            repo="flask",
            error="timeout",
            returned_contents=[],
            found_chunk=False,
        )
        detail = _query_detail(qr)
        assert "error" in detail
        assert "returned_contents" in detail
        assert "found_chunk" in detail
        assert detail["found_chunk"] is False


class TestBuildResultJsonChunkFields:
    """VAL-RUNNER-003: _build_result_json includes chunk_hit_at_5."""

    def test_retrieval_includes_chunk_hit_at_5(self):
        from rag_bench.metrics import BenchmarkMetrics, QueryResult
        from rag_bench.runner import _build_result_json

        metrics = BenchmarkMetrics(
            chunk_hit_at_5=0.42,
        )
        result = _build_result_json(
            run_id="test-run",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=[],
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
        )
        assert "chunk_hit_at_5" in result["retrieval"]
        assert result["retrieval"]["chunk_hit_at_5"] == 0.42

    def test_retrieval_chunk_hit_at_5_default_is_zero(self):
        from rag_bench.metrics import BenchmarkMetrics
        from rag_bench.runner import _build_result_json

        metrics = BenchmarkMetrics()
        result = _build_result_json(
            run_id="test-run",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=[],
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
        )
        assert result["retrieval"]["chunk_hit_at_5"] == 0.0

    def test_chunk_hit_at_5_in_retrieval_is_float(self):
        from rag_bench.metrics import BenchmarkMetrics
        from rag_bench.runner import _build_result_json

        metrics = BenchmarkMetrics(chunk_hit_at_5=0.5)
        result = _build_result_json(
            run_id="test-run",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=[],
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
        )
        assert isinstance(result["retrieval"]["chunk_hit_at_5"], float)
        assert 0.0 <= result["retrieval"]["chunk_hit_at_5"] <= 1.0


class TestMedianFieldsChunk:
    """Test that chunk_hit_at_5 is in _MEDIAN_FIELDS."""

    def test_median_fields_includes_chunk_hit_at_5(self):
        from rag_bench.runner import _MEDIAN_FIELDS
        assert "chunk_hit_at_5" in _MEDIAN_FIELDS


class TestReplicateSummaryChunk:
    """Test that _replicate_summary includes chunk_hit_at_5."""

    def test_replicate_summary_includes_chunk_hit_at_5(self):
        from rag_bench.metrics import BenchmarkMetrics
        from rag_bench.runner import _replicate_summary

        m = BenchmarkMetrics(chunk_hit_at_5=0.35)
        summary = _replicate_summary([m])
        assert len(summary) == 1
        assert "chunk_hit_at_5" in summary[0]
        assert summary[0]["chunk_hit_at_5"] == 0.35


class TestReplicateIqrChunk:
    """Test that _replicate_iqr includes chunk_hit_at_5."""

    def test_replicate_iqr_includes_chunk_hit_at_5(self):
        from rag_bench.metrics import BenchmarkMetrics
        from rag_bench.runner import _replicate_iqr

        reps = [
            BenchmarkMetrics(chunk_hit_at_5=0.3),
            BenchmarkMetrics(chunk_hit_at_5=0.5),
            BenchmarkMetrics(chunk_hit_at_5=0.4),
        ]
        iqr = _replicate_iqr(reps)
        assert "chunk_hit_at_5" in iqr
        assert isinstance(iqr["chunk_hit_at_5"], float)


class TestFoundChunkComputation:
    """VAL-RUNNER-002: found_chunk computed via content_matches."""

    def test_found_chunk_true_when_content_matches(self):
        from rag_bench.metrics import content_matches

        returned = ["def handle_request(req: Request) -> Response:"]
        expected = ["def handle_request"]
        assert content_matches(returned, expected[0])

    def test_found_chunk_false_when_no_match(self):
        from rag_bench.metrics import content_matches

        returned = ["def foo(): pass"]
        expected = ["handle_request"]
        assert not content_matches(returned, expected[0])

    def test_found_chunk_false_when_expected_empty(self):
        from rag_bench.metrics import content_matches

        returned = ["some content"]
        assert not content_matches(returned, "")

    def test_found_chunk_false_when_returned_empty(self):
        from rag_bench.metrics import content_matches

        expected = "def handle_request"
        assert not content_matches([], expected)
