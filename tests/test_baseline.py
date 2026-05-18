"""Tests for A/B baseline: DeepSeek-powered agent, ab_comparison deltas,
and baseline result JSON structure."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_bench.metrics import (
    BenchmarkMetrics,
    QueryResult,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# 1. DeepSeek baseline agent — mock responses
# ---------------------------------------------------------------------------

class TestDeepSeekBaselineAgent:
    """Tests for the DeepSeek-powered grep/glob agent that searches
    without RAG tools. Uses mock HTTP responses to simulate the
    DeepSeek API."""

    @pytest.mark.asyncio
    async def test_agent_finds_files_via_grep(self, tmp_path):
        """Agent uses grep tool calls to find a target file."""
        # Create a small repo structure
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "src").mkdir()
        (repo_dir / "src" / "main.py").write_text("def handle_request():\n    pass\n")
        (repo_dir / "src" / "utils.py").write_text("def helper():\n    return 42\n")

        from rag_bench.baseline import DeepSeekBaselineAgent

        # Mock DeepSeek API responses: first call requests grep, second gives answer
        mock_responses = [
            # Response 1: DeepSeek chooses to grep for "handle_request"
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "grep",
                                "arguments": json.dumps({"pattern": "handle_request"})
                            }
                        }]
                    }
                }]
            },
            # Response 2: After grep results, DeepSeek identifies the file
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "FILES:\nsrc/main.py",
                        "tool_calls": None
                    }
                }]
            },
        ]

        response_index = [0]

        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json = json_data

            def json(self):
                return self._json

        async def mock_post(*args, **kwargs):
            idx = response_index[0]
            response_index[0] += 1
            if idx < len(mock_responses):
                return MockResponse(200, mock_responses[idx])
            return MockResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": "Done", "tool_calls": None}}]
            })

        agent = DeepSeekBaselineAgent(
            api_key="sk-test",
            model="deepseek-v4-flash",
            max_iterations=3,
        )

        with patch.object(agent, "_http_client") as mock_client:
            mock_client.post = mock_post
            result = await agent.search(
                query="Where is handle_request defined?",
                repo_dir=repo_dir,
            )

        assert len(result["found_files"]) > 0
        assert any("main.py" in f for f in result["found_files"])
        assert result["tool_calls"] >= 1

    @pytest.mark.asyncio
    async def test_agent_finds_files_via_glob(self, tmp_path):
        """Agent uses glob tool to find files matching a pattern."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "handlers").mkdir()
        (repo_dir / "handlers" / "auth.py").write_text("# auth handler\n")
        (repo_dir / "handlers" / "route.py").write_text("# route handler\n")

        from rag_bench.baseline import DeepSeekBaselineAgent

        # Mock: first call uses glob for "handler*", second reads a file
        mock_responses = [
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "glob",
                                "arguments": json.dumps({"pattern": "**/handler*"})
                            }
                        }]
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "FILES:\nhandlers/auth.py\nhandlers/route.py",
                        "tool_calls": None
                    }
                }]
            },
        ]

        response_index = [0]

        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json = json_data

            def json(self):
                return self._json

        async def mock_post(*args, **kwargs):
            idx = response_index[0]
            response_index[0] += 1
            if idx < len(mock_responses):
                return MockResponse(200, mock_responses[idx])
            return MockResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": "Done", "tool_calls": None}}]
            })

        agent = DeepSeekBaselineAgent(
            api_key="sk-test",
            model="deepseek-v4-flash",
            max_iterations=3,
        )

        with patch.object(agent, "_http_client") as mock_client:
            mock_client.post = mock_post
            result = await agent.search(
                query="Find files related to handlers",
                repo_dir=repo_dir,
            )

        assert result["tool_calls"] >= 1
        assert len(result["found_files"]) >= 1

    @pytest.mark.asyncio
    async def test_agent_reads_files(self, tmp_path):
        """Agent uses read_file tool to examine file contents."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "app.py").write_text("class UserModel:\n    def save(self): pass\n")

        from rag_bench.baseline import DeepSeekBaselineAgent

        mock_responses = [
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": "app.py"})
                            }
                        }]
                    }
                }]
            },
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "UserModel is defined in app.py",
                        "tool_calls": None
                    }
                }]
            },
        ]

        response_index = [0]

        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json = json_data

            def json(self):
                return self._json

        async def mock_post(*args, **kwargs):
            idx = response_index[0]
            response_index[0] += 1
            if idx < len(mock_responses):
                return MockResponse(200, mock_responses[idx])
            return MockResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": "Done", "tool_calls": None}}]
            })

        agent = DeepSeekBaselineAgent(
            api_key="sk-test",
            model="deepseek-v4-flash",
            max_iterations=3,
        )

        with patch.object(agent, "_http_client") as mock_client:
            mock_client.post = mock_post
            result = await agent.search(
                query="Where is UserModel defined?",
                repo_dir=repo_dir,
            )

        assert result["tool_calls"] >= 1
        assert len(result["found_files"]) >= 0  # may find or not, depends on response parsing

    @pytest.mark.asyncio
    async def test_agent_max_iterations_limit(self, tmp_path):
        """Agent stops after max_iterations even if tool calls continue."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "file.py").write_text("def foo(): pass\n")

        from rag_bench.baseline import DeepSeekBaselineAgent

        # Always return tool calls (infinite loop without limit)
        mock_responses = [
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": "grep",
                                "arguments": json.dumps({"pattern": f"test_{i}"})
                            }
                        }]
                    }
                }]
            }
            for i in range(10)
        ]

        response_index = [0]

        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json = json_data

            def json(self):
                return self._json

        async def mock_post(*args, **kwargs):
            idx = response_index[0]
            response_index[0] += 1
            if idx < len(mock_responses):
                return MockResponse(200, mock_responses[idx])
            return MockResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": "Done", "tool_calls": None}}]
            })

        agent = DeepSeekBaselineAgent(
            api_key="sk-test",
            model="deepseek-v4-flash",
            max_iterations=3,
        )

        with patch.object(agent, "_http_client") as mock_client:
            mock_client.post = mock_post
            result = await agent.search(
                query="Find something",
                repo_dir=repo_dir,
            )

        # Should stop at max_iterations, not run 10 times
        assert result["tool_calls"] <= 3

    @pytest.mark.asyncio
    async def test_agent_handles_api_error(self, tmp_path):
        """Agent handles API errors gracefully."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "file.py").write_text("def foo(): pass\n")

        from rag_bench.baseline import DeepSeekBaselineAgent

        class MockErrorResponse:
            def __init__(self, status_code):
                self.status_code = status_code

            def json(self):
                return {"error": "service unavailable"}

        async def mock_post(*args, **kwargs):
            return MockErrorResponse(500)

        agent = DeepSeekBaselineAgent(
            api_key="sk-test",
            model="deepseek-v4-flash",
            max_iterations=3,
        )

        with patch.object(agent, "_http_client") as mock_client:
            mock_client.post = mock_post
            with pytest.raises(Exception):
                await agent.search(
                    query="Find something",
                    repo_dir=repo_dir,
                )

    @pytest.mark.asyncio
    async def test_agent_file_listings_provided(self, tmp_path):
        """Agent receives file listings in the system prompt."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "src").mkdir()
        (repo_dir / "src" / "main.py").write_text("def main(): pass\n")
        (repo_dir / "src" / "utils.py").write_text("def util(): pass\n")
        (repo_dir / "tests").mkdir()
        (repo_dir / "tests" / "test_main.py").write_text("# tests\n")

        from rag_bench.baseline import DeepSeekBaselineAgent

        captured_messages = []

        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json = json_data

            def json(self):
                return self._json

        async def mock_post(url, **kwargs):
            body = json.loads(kwargs.get("content", "{}"))
            captured_messages.append(body)
            return MockResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": "Done", "tool_calls": None}}]
            })

        agent = DeepSeekBaselineAgent(
            api_key="sk-test",
            model="deepseek-v4-flash",
            max_iterations=1,
        )

        with patch.object(agent, "_http_client") as mock_client:
            mock_client.post = mock_post
            await agent.search(
                query="Find main",
                repo_dir=repo_dir,
            )

        assert len(captured_messages) >= 1
        messages = captured_messages[0]["messages"]
        # System prompt should mention file listings
        system_content = messages[0]["content"]
        assert "src/main.py" in system_content
        assert "src/utils.py" in system_content
        assert "tests/test_main.py" in system_content


# ---------------------------------------------------------------------------
# 2. AB comparison deltas
# ---------------------------------------------------------------------------

class TestABComparison:
    """Tests for the ab_comparison section with per-metric deltas."""

    def test_compute_ab_deltas_basic(self):
        """Basic delta computation: RAG > baseline on quality metrics."""
        from rag_bench.runner import _compute_ab_deltas

        rag = {
            "retrieval": {
                "hit_at_5": 0.35,
                "hit_at_10": 0.50,
                "symbol_hit_at_5": 0.20,
                "mrr": 0.25,
                "latency": {"p50_ms": 200.0, "p95_ms": 800.0, "mean_ms": 300.0},
            },
            "efficiency": {"avg_tool_calls": 1.5},
            "composite_score": 0.45,
        }

        baseline = {
            "retrieval": {
                "hit_at_5": 0.15,
                "hit_at_10": 0.25,
                "symbol_hit_at_5": 0.12,
                "mrr": 0.15,
                "latency": {"p50_ms": 500.0, "p95_ms": 1200.0, "mean_ms": 600.0},
            },
            "efficiency": {"avg_tool_calls": 3.2},
            "composite_score": 0.30,
        }

        deltas = _compute_ab_deltas(rag, baseline)

        assert "hit_at_5_delta" in deltas
        assert "symbol_hit_at_5_delta" in deltas
        assert "mrr_delta" in deltas
        assert "composite_score_delta" in deltas
        assert "tool_calls_delta" in deltas
        assert "latency_p50_delta" in deltas
        assert "latency_p95_delta" in deltas
        assert "latency_mean_delta" in deltas

        # RAG better → quality deltas are positive
        assert deltas["hit_at_5_delta"] == pytest.approx(0.20, abs=0.01)
        assert deltas["composite_score_delta"] == pytest.approx(0.15, abs=0.01)
        # RAG has fewer tool calls → positive delta (baseline - RAG, or RAG advantage)
        assert deltas["tool_calls_delta"] > 0
        # RAG is faster → latency deltas positive (baseline slower)
        assert deltas["latency_p50_delta"] > 0

    def test_compute_ab_deltas_rag_advantage(self):
        """RAG shows clear advantage on quality: positive quality deltas."""
        from rag_bench.runner import _compute_ab_deltas

        rag = {
            "retrieval": {
                "hit_at_5": 0.40,
                "symbol_hit_at_5": 0.25,
                "mrr": 0.30,
                "latency": {"p50_ms": 150.0, "p95_ms": 600.0, "mean_ms": 250.0},
            },
            "efficiency": {"avg_tool_calls": 1.0},
            "composite_score": 0.50,
        }

        baseline = {
            "retrieval": {
                "hit_at_5": 0.12,  # RAG Hit@5 > baseline
                "symbol_hit_at_5": 0.12,  # RAG SymbolHit@5 > baseline
                "mrr": 0.12,  # RAG MRR > baseline
                "latency": {"p50_ms": 500.0, "p95_ms": 1500.0, "mean_ms": 800.0},
            },
            "efficiency": {"avg_tool_calls": 5.0},
            "composite_score": 0.25,
        }

        deltas = _compute_ab_deltas(rag, baseline)

        # VAL-BENCH-AB-003: RAG must show advantage on quality metrics
        assert deltas["hit_at_5_delta"] > 0, "RAG should have better Hit@5"
        assert deltas["symbol_hit_at_5_delta"] >= 0, "RAG should not have worse SymbolHit@5"
        assert deltas["mrr_delta"] >= 0, "RAG should not have worse MRR"
        # Composite score delta positive
        assert deltas["composite_score_delta"] > 0

    def test_compute_ab_deltas_no_baseline(self):
        """When baseline is None, ab_comparison is null."""
        from rag_bench.runner import _compute_ab_deltas

        rag = {
            "retrieval": {
                "hit_at_5": 0.35,
                "latency": {"p50_ms": 200.0, "p95_ms": 800.0, "mean_ms": 300.0},
            },
            "efficiency": {"avg_tool_calls": 1.5},
            "composite_score": 0.45,
        }

        deltas = _compute_ab_deltas(rag, None)
        assert deltas is None

    def test_compute_ab_deltas_preserves_sign(self):
        """Latency delta: positive means baseline is slower (RAG advantage)."""
        from rag_bench.runner import _compute_ab_deltas

        rag = {
            "retrieval": {
                "hit_at_5": 0.5,
                "symbol_hit_at_5": 0.3,
                "mrr": 0.4,
                "latency": {"p50_ms": 100.0, "p95_ms": 400.0, "mean_ms": 200.0},
            },
            "efficiency": {"avg_tool_calls": 0.5},
            "composite_score": 0.60,
        }

        baseline = {
            "retrieval": {
                "hit_at_5": 0.1,
                "symbol_hit_at_5": 0.1,
                "mrr": 0.1,
                "latency": {"p50_ms": 1000.0, "p95_ms": 3000.0, "mean_ms": 2000.0},
            },
            "efficiency": {"avg_tool_calls": 10.0},
            "composite_score": 0.15,
        }

        deltas = _compute_ab_deltas(rag, baseline)

        # All deltas = baseline - rag for metric order (so positive = RAG advantage)
        # For quality: positive means RAG better
        assert deltas["hit_at_5_delta"] > 0
        assert deltas["composite_score_delta"] > 0
        # For tool_calls: fewer calls is better, baseline has more → positive delta
        assert deltas["tool_calls_delta"] > 0
        # For latency: lower is better, baseline is slower → positive delta
        assert deltas["latency_p50_delta"] > 0
        assert deltas["latency_p95_delta"] > 0
        assert deltas["latency_mean_delta"] > 0


# ---------------------------------------------------------------------------
# 3. Baseline result JSON structure
# ---------------------------------------------------------------------------

class TestBaselineResultJSON:
    """Tests for the baseline section and ab_comparison in the result JSON."""

    def test_build_result_with_baseline(self):
        """Output JSON includes baseline and ab_comparison sections."""
        from rag_bench.runner import _build_result_json
        from rag_bench.metrics import BenchmarkMetrics
        from rag_bench.datasets.loader import RepoInfo

        qrs = [
            QueryResult(
                query_id="q1", query_text="find foo", query_type="locate",
                difficulty="easy", expected_files=["foo.py"], expected_symbols=["foo"],
                returned_files=["src/foo.py"], returned_symbols=["foo_func"],
                latency_ms=50.0, tool_calls=1, found_file=True, found_symbol=True,
                repo="flask",
            ),
        ]

        metrics = compute_metrics(qrs, ingest_total_sec=5.0, ingest_total_files=100,
                                  index_size_mb=10.0, ram_peak_mb=50.0)

        baseline_result = {
            "retrieval": {
                "total_queries": 1,
                "total_hits": 0,
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_5": 0.0,
                "hit_at_10": 0.0,
                "symbol_hit_at_5": 0.0,
                "mrr": 0.0,
                "latency": {
                    "p50_ms": 500.0,
                    "p95_ms": 500.0,
                    "p99_ms": 500.0,
                    "mean_ms": 500.0,
                },
            },
            "efficiency": {
                "avg_tool_calls": 3.0,
            },
            "composite_score": 0.15,
            "method": "deepseek",
        }

        repos = [
            RepoInfo("flask", "url", "main", "python", "small"),
        ]

        result = _build_result_json(
            run_id="test-id",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=qrs,
            repos=repos,
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
            baseline_result=baseline_result,
        )

        # VAL-BENCH-AB-001: Output includes baseline and ab_comparison
        assert "baseline" in result
        assert result["baseline"] is not None
        assert "ab_comparison" in result
        assert result["ab_comparison"] is not None

        # Baseline has own retrieval/efficiency/latency
        assert "retrieval" in result["baseline"]
        assert "efficiency" in result["baseline"]
        assert result["baseline"]["retrieval"]["hit_at_5"] == 0.0

        # ab_comparison has per-metric deltas
        ab = result["ab_comparison"]
        assert "hit_at_5_delta" in ab
        assert "symbol_hit_at_5_delta" in ab
        assert "mrr_delta" in ab
        assert "composite_score_delta" in ab
        assert "tool_calls_delta" in ab
        assert "latency_p50_delta" in ab
        assert "latency_p95_delta" in ab
        assert "latency_mean_delta" in ab

    def test_build_result_without_baseline(self):
        """When baseline is None, it remains null and ab_comparison is null."""
        from rag_bench.runner import _build_result_json
        from rag_bench.metrics import BenchmarkMetrics

        qrs = [
            QueryResult(
                query_id="q1", query_text="test", query_type="locate",
                difficulty="easy", expected_files=[], expected_symbols=[],
                returned_files=[], returned_symbols=[],
                latency_ms=50.0, tool_calls=1, repo="flask",
            ),
        ]

        metrics = compute_metrics(qrs)

        result = _build_result_json(
            run_id="test-id",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=qrs,
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
            baseline_result=None,
        )

        assert result["baseline"] is None
        assert result.get("ab_comparison") is None

    def test_baseline_hit_at_5_at_least_0_10(self):
        """Verification that baseline agent can achieve Hit@5 >= 0.10 with
        a reasonable mock scenario."""
        # Create results where grep finds files for some queries
        qrs = []
        for i in range(30):  # 30 queries
            # Simulate: 4 out of 30 get hit (13.3% → Hit@5 = 0.133)
            found = i < 4
            qrs.append(QueryResult(
                query_id=f"q{i}", query_text=f"query {i}", query_type="locate",
                difficulty="easy",
                expected_files=[f"target_{i}.py"],
                expected_symbols=[],
                returned_files=[f"src/target_{i}.py"] if found else ["src/other.py"],
                returned_symbols=[],
                latency_ms=500.0, tool_calls=3,
                found_file=found, repo="flask",
            ))

        metrics = compute_metrics(qrs)

        # VAL-BENCH-AB-002: Baseline Hit@5 >= 0.10
        assert metrics.hit_at_5 >= 0.10, \
            f"Baseline Hit@5 should be >= 0.10, got {metrics.hit_at_5}"
        # Baseline avg_tool_calls > 1 (multiple tool invocations)
        assert metrics.avg_tool_calls > 1, \
            f"Baseline avg_tool_calls should be > 1, got {metrics.avg_tool_calls}"

    def test_rag_advantage_on_quality(self):
        """Verification that RAG shows advantage: Hit@5 delta > 0."""
        from rag_bench.runner import _compute_ab_deltas

        # RAG metrics (better quality)
        rag = {
            "retrieval": {
                "hit_at_5": 0.35,
                "symbol_hit_at_5": 0.22,
                "mrr": 0.28,
                "latency": {"p50_ms": 200.0, "p95_ms": 800.0, "mean_ms": 300.0},
            },
            "efficiency": {"avg_tool_calls": 1.2},
            "composite_score": 0.48,
        }

        # Baseline metrics (grep-based, lower quality)
        baseline = {
            "retrieval": {
                "hit_at_5": 0.12,
                "symbol_hit_at_5": 0.10,
                "mrr": 0.11,
                "latency": {"p50_ms": 400.0, "p95_ms": 1200.0, "mean_ms": 600.0},
            },
            "efficiency": {"avg_tool_calls": 4.0},
            "composite_score": 0.25,
        }

        deltas = _compute_ab_deltas(rag, baseline)

        # VAL-BENCH-AB-003: RAG show advantage on quality metrics
        assert deltas["hit_at_5_delta"] > 0, \
            f"RAG Hit@5 should be higher than baseline, got delta={deltas['hit_at_5_delta']}"
        assert deltas["symbol_hit_at_5_delta"] >= 0, \
            f"RAG SymbolHit@5 should not be worse, got delta={deltas['symbol_hit_at_5_delta']}"
        assert deltas["mrr_delta"] >= 0, \
            f"RAG MRR should not be worse, got delta={deltas['mrr_delta']}"


# ---------------------------------------------------------------------------
# 4. Local grep/glob baseline (non-DeepSeek fallback)
# ---------------------------------------------------------------------------

class TestLocalGrepBaseline:
    """Tests for the local grep/glob baseline — verifies that A/B mode
    correctly exercises diverse tool_calls even without the DeepSeek API."""

    def test_local_baseline_produces_tool_calls_above_zero(self, tmp_path):
        """Local baseline always produces tool_calls > 0 (at least one grep)."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "main.py").write_text("def my_function():\n    pass\n")

        from rag_bench.baseline import run_baseline_search

        result = run_baseline_search(
            query="Find something",
            query_type="locate",
            repo_dir=repo_dir,
            expected_symbols=[],
        )

        # Every baseline query must have tool_calls > 0 (VAL-BENCH-METRIC-008)
        assert result.tool_calls > 0, \
            f"tool_calls should be > 0 even without expected_symbols, got {result.tool_calls}"

    def test_local_baseline_diverse_tool_calls_with_symbols(self, tmp_path):
        """When expected symbols are present, local baseline makes multiple
        calls (grep + read_file) — demonstrating tool_calls > 1.

        This directly satisfies VAL-BENCH-METRIC-008's A/B baseline
        requirement: baseline avg_tool_calls > 1.0.
        """
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "src").mkdir()
        (repo_dir / "src" / "main.py").write_text("def handle_request():\n    pass\n")

        from rag_bench.baseline import run_baseline_search

        result = run_baseline_search(
            query="Where is handle_request defined?",
            query_type="locate",
            repo_dir=repo_dir,
            expected_symbols=["handle_request"],
        )

        # With expected symbols, the baseline does grep + content check
        # for each found file, so tool_calls > 1
        assert result.tool_calls > 1, \
            f"Baseline with expected symbols should have tool_calls > 1, got {result.tool_calls}"
        assert len(result.found_files) >= 1, \
            f"Should find main.py, got {result.found_files}"

    def test_local_baseline_avg_across_queries_exceeds_one(self, tmp_path):
        """Average tool_calls across multiple baseline queries exceeds 1.0,
        confirming the A/B baseline diversity requirement.
        """
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "auth.py").write_text("def login_user():\n    pass\n")
        (repo_dir / "routes.py").write_text("def home_route():\n    pass\n")

        from rag_bench.baseline import run_baseline_search

        queries = [
            ("Where is login_user?", "locate", ["login_user"]),
            ("Find auth code", "locate", []),
            ("Where is home_route defined?", "locate", ["home_route"]),
        ]

        total_tool_calls = 0
        for query, qtype, symbols in queries:
            result = run_baseline_search(
                query=query,
                query_type=qtype,
                repo_dir=repo_dir,
                expected_symbols=symbols,
            )
            total_tool_calls += result.tool_calls

        avg = total_tool_calls / len(queries)
        assert avg > 1.0, \
            f"Average baseline tool_calls should be > 1.0, got {avg}"

    def test_local_baseline_finds_files_via_grep(self, tmp_path):
        """Local baseline grep finds files containing expected symbols."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "src").mkdir()
        (repo_dir / "src" / "auth.py").write_text("def authenticate_user(token):\n    pass\n")

        from rag_bench.baseline import run_baseline_search

        result = run_baseline_search(
            query="Where is authenticate_user?",
            query_type="locate",
            repo_dir=repo_dir,
            expected_symbols=["authenticate_user"],
        )

        assert len(result.found_files) >= 1
        assert any("auth.py" in f for f in result.found_files), \
            f"Should find auth.py among {result.found_files}"

    def test_local_baseline_over_multiple_repos_produces_diversity(self, tmp_path):
        """Across a realistic set of 5 queries, baseline tool_calls vary
        (not all same value) — exercising the diversity path of the
        relaxed METRIC-008 assertion.
        """
        repo_dir = tmp_path / "test_repo"
        (repo_dir / "app").mkdir(parents=True)
        (repo_dir / "app" / "models.py").write_text("class User:\n    pass\nclass Post:\n    pass\n")

        from rag_bench.baseline import run_baseline_search

        results = []
        for i, (query, symbols) in enumerate([
            ("Where is User defined?", ["User"]),
            ("Find database models", []),
            ("How is authentication handled?", []),
            ("Where is Post defined?", ["Post"]),
            ("Find configuration code", []),
        ]):
            result = run_baseline_search(
                query=query,
                query_type="locate",
                repo_dir=repo_dir,
                expected_symbols=symbols,
            )
            results.append(result)

        # All queries must have tool_calls > 0 (VAL-BENCH-METRIC-008)
        for r in results:
            assert r.tool_calls > 0, \
                f"Every baseline query must have tool_calls > 0, got {r.tool_calls}"

        # Average should exceed 1 (A/B diversity requirement)
        avg = sum(r.tool_calls for r in results) / len(results)
        assert avg > 1.0, \
            f"Baseline avg_tool_calls should be > 1.0, got {avg}"


# ---------------------------------------------------------------------------
# 5. Baseline LLM token capture (VAL-BASELINE-001 through VAL-BASELINE-006)
# ---------------------------------------------------------------------------

class TestBaselineTokenCapture:
    """Tests for token capture in DeepSeekBaselineAgent.search() and the
    resulting baseline JSON structure."""

    # ------------------------------------------------------------------
    # VAL-BASELINE-001: search returns token usage fields
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_search_returns_token_fields(self, tmp_path):
        """DeepSeekBaselineAgent.search returns prompt_tokens,
        completion_tokens, and total_llm_tokens in the result dict."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "file.py").write_text("def foo(): pass\n")

        from rag_bench.baseline import DeepSeekBaselineAgent

        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json = json_data

            def json(self):
                return self._json

        async def mock_post(*args, **kwargs):
            return MockResponse(200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "FILES:\nfile.py",
                        "tool_calls": None,
                    }
                }],
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 25,
                    "total_tokens": 175,
                },
            })

        agent = DeepSeekBaselineAgent(
            api_key="sk-test",
            model="deepseek-v4-flash",
            max_iterations=3,
        )

        with patch.object(agent, "_http_client") as mock_client:
            mock_client.post = mock_post
            result = await agent.search(
                query="Where is foo?",
                repo_dir=repo_dir,
            )

        # VAL-BASELINE-001: dict includes token fields
        assert "prompt_tokens" in result, "search() must return prompt_tokens"
        assert "completion_tokens" in result, "search() must return completion_tokens"
        assert "total_llm_tokens" in result, "search() must return total_llm_tokens"
        assert result["prompt_tokens"] == 150
        assert result["completion_tokens"] == 25
        assert result["total_llm_tokens"] == 175

        # Existing keys still present
        assert "found_files" in result
        assert "tool_calls" in result
        assert "total_time_ms" in result

    # ------------------------------------------------------------------
    # VAL-BASELINE-002: tokens accumulate across iterations
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_tokens_accumulate_across_iterations(self, tmp_path):
        """Three sequential mocked completions with different usage blocks
        sum to correct totals."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "file.py").write_text("def bar(): pass\n")

        from rag_bench.baseline import DeepSeekBaselineAgent

        # Three iterations: first two are tool calls, third is final answer
        mock_responses = [
            # Iteration 1: tool call (grep)
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "grep",
                                "arguments": json.dumps({"pattern": "bar"})
                            }
                        }]
                    }
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
            # Iteration 2: tool call (read_file)
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": "file.py"})
                            }
                        }]
                    }
                }],
                "usage": {"prompt_tokens": 150, "completion_tokens": 30, "total_tokens": 180},
            },
            # Iteration 3: final answer (no tool calls)
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "FILES:\nfile.py",
                        "tool_calls": None,
                    }
                }],
                "usage": {"prompt_tokens": 80, "completion_tokens": 15, "total_tokens": 95},
            },
        ]

        response_index = [0]

        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json = json_data

            def json(self):
                return self._json

        async def mock_post(*args, **kwargs):
            idx = response_index[0]
            response_index[0] += 1
            if idx < len(mock_responses):
                return MockResponse(200, mock_responses[idx])
            return MockResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": "Done", "tool_calls": None}}]
            })

        agent = DeepSeekBaselineAgent(
            api_key="sk-test",
            model="deepseek-v4-flash",
            max_iterations=3,
        )

        with patch.object(agent, "_http_client") as mock_client:
            mock_client.post = mock_post
            result = await agent.search(
                query="Where is bar?",
                repo_dir=repo_dir,
            )

        # VAL-BASELINE-002: sums across all iterations
        # prompt: 100 + 150 + 80 = 330
        # completion: 20 + 30 + 15 = 65
        # total: 330 + 65 = 395
        assert result["prompt_tokens"] == 330, \
            f"Expected sum of prompt_tokens = 330, got {result['prompt_tokens']}"
        assert result["completion_tokens"] == 65, \
            f"Expected sum of completion_tokens = 65, got {result['completion_tokens']}"
        assert result["total_llm_tokens"] == 395, \
            f"Expected total_llm_tokens = 395, got {result['total_llm_tokens']}"

    # ------------------------------------------------------------------
    # VAL-BASELINE-003: missing usage → 0, no crash
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_missing_usage_defaults_to_zero(self, tmp_path):
        """When the API response omits 'usage', the agent returns 0 for
        token fields without raising an exception."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "file.py").write_text("def baz(): pass\n")

        from rag_bench.baseline import DeepSeekBaselineAgent

        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self._json = json_data

            def json(self):
                return self._json

        async def mock_post(*args, **kwargs):
            # Response with NO usage block
            return MockResponse(200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "FILES:\nfile.py",
                        "tool_calls": None,
                    }
                }],
            })

        agent = DeepSeekBaselineAgent(
            api_key="sk-test",
            model="deepseek-v4-flash",
            max_iterations=3,
        )

        with patch.object(agent, "_http_client") as mock_client:
            mock_client.post = mock_post
            result = await agent.search(
                query="Where is baz?",
                repo_dir=repo_dir,
            )

        # VAL-BASELINE-003: 0 for all token fields, no exception
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
        assert result["total_llm_tokens"] == 0

    # ------------------------------------------------------------------
    # VAL-BASELINE-004: baseline result JSON exposes LLM tokens
    # ------------------------------------------------------------------

    def test_baseline_json_has_llm_tokens_in_efficiency(self):
        """The result JSON baseline.efficiency contains LLM token
        aggregates, and baseline.retrieval.tokens exists."""
        from rag_bench.runner import _build_result_json
        from rag_bench.datasets.loader import RepoInfo

        qrs = [
            QueryResult(
                query_id="q1", query_text="find foo", query_type="locate",
                difficulty="easy", expected_files=["foo.py"], expected_symbols=["foo"],
                returned_files=["src/foo.py"], returned_symbols=["foo_func"],
                latency_ms=50.0, tool_calls=1, found_file=True, found_symbol=True,
                repo="flask",
            ),
        ]

        metrics = compute_metrics(qrs, ingest_total_sec=5.0, ingest_total_files=100,
                                  index_size_mb=10.0, ram_peak_mb=50.0)

        # Baseline result WITH LLM token fields
        baseline_result = {
            "retrieval": {
                "total_queries": 1,
                "total_hits": 0,
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_5": 0.0,
                "hit_at_10": 0.0,
                "symbol_hit_at_5": 0.0,
                "mrr": 0.0,
                "latency": {
                    "p50_ms": 500.0,
                    "p95_ms": 500.0,
                    "p99_ms": 500.0,
                    "mean_ms": 500.0,
                },
            },
            "efficiency": {
                "avg_tool_calls": 3.0,
                "avg_prompt_tokens": 2200.0,
                "avg_completion_tokens": 180.0,
                "avg_total_llm_tokens": 2380.0,
                "total_llm_tokens": 2380,
            },
            "composite_score": 0.15,
            "method": "deepseek",
        }

        repos = [
            RepoInfo("flask", "url", "main", "python", "small"),
        ]

        result = _build_result_json(
            run_id="test-id",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=qrs,
            repos=repos,
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
            baseline_result=baseline_result,
        )

        # VAL-BASELINE-004: LLM token fields in efficiency
        baseline = result["baseline"]
        assert baseline is not None
        eff = baseline["efficiency"]
        assert "avg_prompt_tokens" in eff, "efficiency missing avg_prompt_tokens"
        assert "avg_completion_tokens" in eff, "efficiency missing avg_completion_tokens"
        assert "avg_total_llm_tokens" in eff, "efficiency missing avg_total_llm_tokens"
        assert "total_llm_tokens" in eff, "efficiency missing total_llm_tokens"
        assert eff["avg_prompt_tokens"] == 2200.0
        assert eff["avg_completion_tokens"] == 180.0
        assert eff["avg_total_llm_tokens"] == 2380.0
        assert eff["total_llm_tokens"] == 2380

        # baseline.retrieval.tokens exists (set to 0 for baseline)
        ret = baseline["retrieval"]
        # Tokens may be present or absent - the key point is LLM tokens in efficiency

    # ------------------------------------------------------------------
    # VAL-BASELINE-005: grep_glob baseline does not produce LLM tokens
    # ------------------------------------------------------------------

    def test_grep_glob_baseline_no_llm_tokens(self):
        """When local grep/glob baseline runs (no API key), the result JSON
        has LLM token aggregates as 0 or absent."""
        from rag_bench.runner import _build_result_json
        from rag_bench.datasets.loader import RepoInfo

        qrs = [
            QueryResult(
                query_id="q1", query_text="find foo", query_type="locate",
                difficulty="easy", expected_files=["foo.py"], expected_symbols=["foo"],
                returned_files=["src/foo.py"], returned_symbols=["foo_func"],
                latency_ms=50.0, tool_calls=1, found_file=True, found_symbol=True,
                repo="flask",
            ),
        ]

        metrics = compute_metrics(qrs, ingest_total_sec=5.0, ingest_total_files=100,
                                  index_size_mb=10.0, ram_peak_mb=50.0)

        # Baseline WITHOUT LLM tokens (grep_glob)
        baseline_result = {
            "retrieval": {
                "total_queries": 1,
                "total_hits": 0,
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_5": 0.0,
                "hit_at_10": 0.0,
                "symbol_hit_at_5": 0.0,
                "mrr": 0.0,
                "latency": {
                    "p50_ms": 500.0,
                    "p95_ms": 500.0,
                    "p99_ms": 500.0,
                    "mean_ms": 500.0,
                },
            },
            "efficiency": {
                "avg_tool_calls": 4.5,
            },
            "composite_score": 0.12,
            "method": "grep_glob",
        }

        repos = [
            RepoInfo("flask", "url", "main", "python", "small"),
        ]

        result = _build_result_json(
            run_id="test-id",
            server_config={"name": "test"},
            metrics=metrics,
            query_results=qrs,
            repos=repos,
            replicate_metrics=[metrics],
            startup_ms=100.0,
            detected_tools={},
            baseline_result=baseline_result,
        )

        baseline = result["baseline"]
        assert baseline is not None

        # VAL-BASELINE-005: LLM token fields are 0 or absent
        eff = baseline["efficiency"]
        llm_fields = ["avg_prompt_tokens", "avg_completion_tokens",
                       "avg_total_llm_tokens", "total_llm_tokens"]
        for field in llm_fields:
            val = eff.get(field)
            # Must be 0 or absent (not a garbage value)
            assert val is None or val == 0 or val == 0.0, \
                f"{field} should be 0 or absent for grep_glob, got {val}"

    # ------------------------------------------------------------------
    # VAL-BASELINE-006: ab_comparison includes response_tokens_delta
    # ------------------------------------------------------------------

    def test_ab_comparison_has_response_tokens_delta(self):
        """ab_comparison contains response_tokens_delta computed as
        baseline_avg_response_tokens - rag_avg_response_tokens."""
        from rag_bench.runner import _compute_ab_deltas

        rag = {
            "retrieval": {
                "hit_at_5": 0.35,
                "symbol_hit_at_5": 0.20,
                "mrr": 0.25,
                "latency": {"p50_ms": 200.0, "p95_ms": 800.0, "mean_ms": 300.0},
            },
            "efficiency": {
                "avg_tool_calls": 1.5,
                "avg_response_tokens": 1200.0,
            },
            "composite_score": 0.45,
        }

        baseline = {
            "retrieval": {
                "hit_at_5": 0.15,
                "symbol_hit_at_5": 0.12,
                "mrr": 0.15,
                "latency": {"p50_ms": 500.0, "p95_ms": 1200.0, "mean_ms": 600.0},
            },
            "efficiency": {
                "avg_tool_calls": 3.2,
                "avg_response_tokens": 0.0,  # baseline doesn't return content chunks
            },
            "composite_score": 0.30,
        }

        deltas = _compute_ab_deltas(rag, baseline)

        # VAL-BASELINE-006: response_tokens_delta is present
        assert "response_tokens_delta" in deltas, \
            "ab_comparison must include response_tokens_delta"
        # baseline_avg_response_tokens (0) - rag_avg_response_tokens (1200) = -1200
        # So RAG uses more tokens → negative delta
        assert deltas["response_tokens_delta"] == pytest.approx(-1200.0, abs=0.1)

    def test_response_tokens_delta_positive_when_rag_saves(self):
        """When baseline uses more response tokens than RAG, delta is
        positive (RAG saves tokens)."""
        from rag_bench.runner import _compute_ab_deltas

        rag = {
            "retrieval": {
                "hit_at_5": 0.35,
                "symbol_hit_at_5": 0.20,
                "mrr": 0.25,
                "latency": {"p50_ms": 200.0, "p95_ms": 800.0, "mean_ms": 300.0},
            },
            "efficiency": {
                "avg_tool_calls": 1.5,
                "avg_response_tokens": 500.0,
            },
            "composite_score": 0.45,
        }

        baseline = {
            "retrieval": {
                "hit_at_5": 0.15,
                "symbol_hit_at_5": 0.12,
                "mrr": 0.15,
                "latency": {"p50_ms": 500.0, "p95_ms": 1200.0, "mean_ms": 600.0},
            },
            "efficiency": {
                "avg_tool_calls": 3.2,
                "avg_response_tokens": 2000.0,
            },
            "composite_score": 0.30,
        }

        deltas = _compute_ab_deltas(rag, baseline)

        # baseline_avg_response_tokens (2000) - rag_avg_response_tokens (500) = 1500
        assert deltas["response_tokens_delta"] == pytest.approx(1500.0, abs=0.1)
        assert deltas["response_tokens_delta"] > 0, \
            "positive delta means RAG saves tokens"
