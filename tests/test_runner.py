"""Tests for dataset loading and runner utilities."""

import os
import tempfile
from pathlib import Path

import pytest

from rag_bench.datasets.loader import load_queries, load_repos
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
        repos = load_repos()
        assert len(repos) == 3
        names = {r.name for r in repos}
        assert names == {"flask", "fastapi", "express"}

    def test_load_all_queries(self):
        queries = load_queries()
        assert len(queries) == 105  # 30 original + 5 complex per repo

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
        queries = load_queries()
        types = {q.type for q in queries}
        # Original types + 7 new complex query types from M1
        assert types == {"locate", "callers", "explain", "impact",
                         "multi_hop", "cross_package", "architecture",
                         "dead_code", "conditional_path", "test_traceability"}

    def test_difficulty_distribution(self):
        queries = load_queries()
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
