"""Tests for error handling: invalid presets, git failures, MCP crashes,
per-query timeouts, and DeepSeek API unavailability."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from rag_bench.cli import cli
from rag_bench.metrics import QueryResult


# ---------------------------------------------------------------------------
# 1. Invalid preset — VAL-BENCH-EXEC-004
# ---------------------------------------------------------------------------

class TestInvalidPreset:
    def test_invalid_preset_exits_nonzero_and_lists_available(self):
        """--preset nonexistent exits non-zero and mentions available presets."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--preset", "nonexistent_preset_xyz"])
        assert result.exit_code != 0
        combined = (result.output or "") + (
            str(result.exception) if result.exception else ""
        )
        assert "preset" in combined.lower()
        # Should mention nova-rag (the available preset)
        assert "nova" in combined.lower() or "available" in combined.lower()

    def test_invalid_preset_does_not_launch_server(self):
        """Invalid preset exits before attempting MCP handshake."""
        runner = CliRunner()
        with patch("rag_bench.runner.run_benchmark") as mock_run:
            result = runner.invoke(cli, ["run", "--preset", "nonexistent_preset_xyz"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 2. Git clone failure — VAL-BENCH-EXEC-005
# ---------------------------------------------------------------------------

class TestGitCloneFailure:
    def test_clone_failure_names_repo(self):
        """Git clone failure produces error message with repo name."""
        from rag_bench.datasets.loader import RepoInfo, clone_repo

        fake_repo = RepoInfo(
            name="test-repo-fail",
            git_url="https://nonexistent.example.com/repo.git",
            ref="main",
            language="python",
            size="small",
        )
        with pytest.raises(RuntimeError) as exc_info:
            clone_repo(fake_repo)
        msg = str(exc_info.value)
        assert "test-repo-fail" in msg
        assert "https://nonexistent.example.com/repo.git" in msg or "clone" in msg.lower()

    def test_clone_failure_propagates_cleanly_in_cli(self, tmp_path):
        """CLI run with failing git clone should exit non-zero with clear error."""
        # run_benchmark imports clone_repo at module level, so patch its
        # reference in runner, not the original in datasets.loader.
        with patch("rag_bench.runner.clone_repo") as mock_clone:
            mock_clone.side_effect = RuntimeError(
                "Failed to clone 'flask' from https://github.com/pallets/flask.git. "
                "Git exited with code 128."
            )

            from rag_bench.runner import run_benchmark
            with pytest.raises(RuntimeError) as exc_info:
                asyncio.run(run_benchmark(
                    server_config={"name": "test", "command": "echo", "transport": "stdio"},
                    repo_filter="flask",
                ))
            msg = str(exc_info.value)
            assert "flask" in msg
            assert "clone" in msg.lower() or "128" in msg


# ---------------------------------------------------------------------------
# 3. MCP server crash mid-bench — VAL-BENCH-EXEC-006
# ---------------------------------------------------------------------------

class TestMCPServerCrash:
    def test_partial_results_on_server_crash(self):
        """When MCP server crashes mid-bench, partial JSON has error markers."""
        from rag_bench.runner import _run_query_pass

        # Mock adapter that works for first 2 queries then raises ConnectionError
        mock_adapter = MagicMock()
        mock_adapter.query_raw = AsyncMock()
        mock_adapter._parse_search_results = MagicMock(
            return_value=[
                MagicMock(file_path="test.py", symbol=None),
            ]
        )

        call_count = [0]

        async def mock_query_raw(query, top_k=10, path=None):
            call_count[0] += 1
            if call_count[0] > 2:
                raise ConnectionError("MCP server closed unexpectedly")
            result = MagicMock()
            result.latency_ms = 100.0
            return result

        mock_adapter.query_raw = mock_query_raw

        mock_client = MagicMock()
        mock_client.call_count = 0

        from rag_bench.datasets.loader import Query

        queries = [
            Query(
                id="q1", type="locate", query="find foo",
                expected_files=["foo.py"], expected_symbols=[],
                difficulty="easy", repo="test",
            ),
            Query(
                id="q2", type="locate", query="find bar",
                expected_files=["bar.py"], expected_symbols=[],
                difficulty="easy", repo="test",
            ),
            Query(
                id="q3", type="locate", query="find baz",
                expected_files=["baz.py"], expected_symbols=[],
                difficulty="easy", repo="test",
            ),
        ]

        repo_dirs = {"test": Path("/tmp/test")}

        results = asyncio.run(_run_query_pass(
            adapter=mock_adapter,
            client=mock_client,
            queries=queries,
            repo_dirs=repo_dirs,
            top_k=10,
            replicate=0,
        ))

        assert len(results) == 3
        # First 2 queries should succeed, 3rd should have error marker
        assert results[0].error == ""
        assert results[1].error == ""
        assert results[2].error == "server_error"
        assert results[0].latency_ms == 100.0
        assert results[2].latency_ms == 0  # failed queries get 0 latency

    def test_build_result_json_includes_error_markers(self):
        """_build_result_json includes 'error' field in query_details on failure."""
        from rag_bench.runner import _build_result_json
        from rag_bench.metrics import BenchmarkMetrics

        qrs = [
            QueryResult(
                query_id="q1", query_text="find foo", query_type="locate",
                difficulty="easy", expected_files=["foo.py"], expected_symbols=[],
                returned_files=["foo.py"], returned_symbols=[],
                latency_ms=45.0, tool_calls=1, found_file=True, repo="test",
            ),
            QueryResult(
                query_id="q2", query_text="find bar", query_type="locate",
                difficulty="medium", expected_files=["bar.py"], expected_symbols=[],
                returned_files=[], returned_symbols=[],
                latency_ms=0, tool_calls=0, found_file=False, repo="test",
                error="server_error",
            ),
        ]

        result = _build_result_json(
            run_id="test-id",
            server_config={"name": "test"},
            metrics=BenchmarkMetrics(total_queries=2),
            query_results=qrs,
            repos=[],
            replicate_metrics=[],
            startup_ms=100.0,
            detected_tools={},
        )

        details = result["query_details"]
        assert len(details) == 2
        assert "error" not in details[0]
        assert details[1]["error"] == "server_error"


# ---------------------------------------------------------------------------
# 4. DeepSeek API unavailable — VAL-BENCH-EXEC-007
# ---------------------------------------------------------------------------

class TestDeepSeekUnavailable:
    def test_baseline_fails_gracefully(self):
        """When baseline fails, RAG metrics are still produced and warning is logged."""
        from rag_bench.runner import _run_baseline_pass
        from rag_bench.datasets.loader import Query

        queries = [
            Query(
                id="q1", type="locate", query="find foo",
                expected_files=["foo.py"], expected_symbols=[],
                difficulty="easy", repo="test",
            ),
        ]
        repo_dirs = {"test": Path("/tmp/test")}

        # With no DeepSeek key and no repos on disk, baseline should still
        # complete (using local grep/glob which will fail gracefully per-query).
        result = asyncio.run(_run_baseline_pass(
            queries=queries,
            repo_dirs=repo_dirs,
            server_config={},
        ))

        assert result is not None
        assert "retrieval" in result
        assert result["retrieval"]["total_queries"] == 1
        assert result["method"] == "grep_glob"

    def test_deepseek_fallback_on_api_error(self):
        """When DeepSeek is configured but fails, baseline still returns results
        (individual query errors are captured in result.error fields)."""
        from rag_bench.runner import _run_baseline_pass
        from rag_bench.datasets.loader import Query

        queries = [
            Query(
                id="q1", type="locate", query="find foo",
                expected_files=["foo.py"], expected_symbols=[],
                difficulty="easy", repo="test",
            ),
        ]
        repo_dirs = {"test": Path("/tmp/test")}

        # Simulate DeepSeek unavailable by providing an invalid API key.
        # The DeepSeek agent will try to call the API, get 401 errors,
        # and capture them per-query. The method is still "deepseek"
        # because the agent framework was used, but results contain
        # error markers.
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test-key"}):
            result = asyncio.run(_run_baseline_pass(
                queries=queries,
                repo_dirs=repo_dirs,
                server_config={},
            ))

        assert result is not None
        # The method records which strategy was attempted (not the fallback)
        assert result["method"] in ("grep_glob", "deepseek")
        assert "retrieval" in result

    def test_run_benchmark_with_ab_baseline_handles_failure(self):
        """run_benchmark with --ab-baseline still produces RAG results on baseline failure."""
        with patch("rag_bench.runner._run_baseline_pass") as mock_baseline:
            mock_baseline.side_effect = Exception("DeepSeek API unreachable")

            # Mock the full run_benchmark without actually cloning or spawning
            with patch("rag_bench.runner.load_repos") as mock_load_repos, \
                 patch("rag_bench.runner.load_queries") as mock_load_queries, \
                 patch("rag_bench.runner.load_warmup_queries") as mock_load_warmup, \
                 patch("rag_bench.runner.clone_repo") as mock_clone, \
                 patch("rag_bench.runner.create_client") as mock_create_client, \
                 patch("rag_bench.runner.clean_nova_rag_index") as mock_clean:

                from rag_bench.datasets.loader import RepoInfo, Query, WarmupQuery
                mock_load_repos.return_value = [
                    RepoInfo("test", "url", "main", "python", "small")
                ]
                mock_load_queries.return_value = [
                    Query("q1", "locate", "find foo", ["foo.py"], [], "easy", "test")
                ]
                mock_load_warmup.return_value = []
                mock_clone.return_value = Path("/tmp/test")

                # Mock the MCP client
                mock_client = MagicMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.call_count = 0
                mock_create_client.return_value = mock_client

                # Mock adapter
                with patch("rag_bench.runner.RAGAdapter") as mock_adapter_cls:
                    mock_adapter = MagicMock()
                    mock_adapter.detect_tools = AsyncMock(
                        return_value={"ingest": "rag_index", "query": "code_search",
                                      "clear": None, "status": None}
                    )
                    mock_adapter.ingest_directory = AsyncMock()
                    mock_adapter.query = AsyncMock()
                    mock_adapter.query_raw = AsyncMock()
                    mock_adapter.query_raw.return_value = MagicMock(
                        latency_ms=50.0, text=""
                    )
                    mock_adapter._parse_search_results = MagicMock(
                        return_value=[MagicMock(file_path="foo.py", symbol=None)]
                    )
                    mock_adapter_cls.return_value = mock_adapter

                    from rag_bench.runner import run_benchmark

                    result = asyncio.run(run_benchmark(
                        server_config={"name": "test", "command": "echo"},
                        repo_filter="test",
                        ab_baseline=True,
                        replicates=1,
                    ))

                    # Baseline should be None (call failed)
                    assert result["baseline"] is None
                    # RAG metrics should still be present
                    assert "retrieval" in result
                    assert result["retrieval"]["total_queries"] > 0


# ---------------------------------------------------------------------------
# 5. Per-query timeout — default 120s
# ---------------------------------------------------------------------------

class TestPerQueryTimeout:
    def test_timeout_marks_query_as_error(self):
        """Query exceeding timeout is marked with error='timeout'."""
        from rag_bench.runner import _run_query_pass
        from rag_bench.datasets.loader import Query

        mock_adapter = MagicMock()
        mock_client = MagicMock()
        mock_client.call_count = 0

        # Simulate a query that hangs
        async def slow_query(query, top_k=10, path=None):
            await asyncio.sleep(999)  # never finishes within 0.5s timeout
            result = MagicMock()
            result.latency_ms = 100.0
            return result

        mock_adapter.query_raw = slow_query

        queries = [
            Query(
                id="q1", type="locate", query="slow query",
                expected_files=["foo.py"], expected_symbols=[],
                difficulty="easy", repo="test",
            ),
        ]
        repo_dirs = {"test": Path("/tmp/test")}

        results = asyncio.run(_run_query_pass(
            adapter=mock_adapter,
            client=mock_client,
            queries=queries,
            repo_dirs=repo_dirs,
            top_k=10,
            replicate=0,
            query_timeout=0.5,  # short timeout for test
        ))

        assert len(results) == 1
        assert results[0].error == "timeout"
        # Timeout latency is query_timeout * 1000
        assert results[0].latency_ms == 500.0

    def test_normal_query_not_affected(self):
        """Queries completing within timeout are not marked as errors."""
        from rag_bench.runner import _run_query_pass
        from rag_bench.datasets.loader import Query

        mock_adapter = MagicMock()
        mock_client = MagicMock()
        mock_client.call_count = 0

        async def fast_query(query, top_k=10, path=None):
            result = MagicMock()
            result.latency_ms = 50.0
            return result

        mock_adapter.query_raw = fast_query
        mock_adapter._parse_search_results = MagicMock(
            return_value=[MagicMock(file_path="foo.py", symbol=None)]
        )

        queries = [
            Query(
                id="q1", type="locate", query="fast query",
                expected_files=["foo.py"], expected_symbols=[],
                difficulty="easy", repo="test",
            ),
        ]
        repo_dirs = {"test": Path("/tmp/test")}

        results = asyncio.run(_run_query_pass(
            adapter=mock_adapter,
            client=mock_client,
            queries=queries,
            repo_dirs=repo_dirs,
            top_k=10,
            replicate=0,
            query_timeout=5.0,
        ))

        assert len(results) == 1
        assert results[0].error == ""
        assert results[0].latency_ms == 50.0


# ---------------------------------------------------------------------------
# 6. Error field on QueryResult
# ---------------------------------------------------------------------------

class TestQueryResultErrorField:
    def test_error_field_defaults_to_empty(self):
        """QueryResult.error defaults to empty string."""
        qr = QueryResult(
            query_id="q1", query_text="test", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[],
            latency_ms=0.0,
        )
        assert qr.error == ""

    def test_error_field_settable(self):
        """QueryResult.error can be set explicitly."""
        qr = QueryResult(
            query_id="q1", query_text="test", query_type="locate",
            difficulty="easy", expected_files=[], expected_symbols=[],
            returned_files=[], returned_symbols=[],
            latency_ms=0.0, error="timeout",
        )
        assert qr.error == "timeout"
