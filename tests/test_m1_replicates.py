"""Tests for M1 reproducibility features: replicates, warmup, clean-index,
search scoping, run metadata, by_repo, and startup_ms."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner

from rag_bench.adapter import RAGAdapter
from rag_bench.cli import cli
from rag_bench.datasets.loader import (
    WARMUP_JSONL,
    WarmupQuery,
    load_queries,
    load_warmup_queries,
)
from rag_bench.mcp_client import CallResult, MCPClient, ToolInfo
from rag_bench.metrics import (
    QueryResult,
    compute_iqr,
    compute_metrics,
)
from rag_bench.runner import (
    _MEDIAN_FIELDS,
    _apply_median,
    _build_result_json,
    _median_metrics,
    _replicate_iqr,
    _replicate_summary,
    clean_nova_rag_index,
)


UUID_V4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z\.\-]+)?$")


class TestWarmupDataset:
    def test_warmup_jsonl_exists_and_nonempty(self):
        assert WARMUP_JSONL.exists()
        assert WARMUP_JSONL.read_text().strip() != ""

    def test_load_warmup_queries(self):
        warmup = load_warmup_queries()
        assert len(warmup) >= 3
        for w in warmup:
            assert isinstance(w, WarmupQuery)
            assert w.id and w.query

    def test_warmup_disjoint_from_benchmark(self):
        warmup_ids = {w.id for w in load_warmup_queries()}
        bench_ids = {q.id for q in load_queries()}
        assert warmup_ids.isdisjoint(bench_ids)
        warmup_texts = {w.query for w in load_warmup_queries()}
        bench_texts = {q.query for q in load_queries()}
        assert warmup_texts.isdisjoint(bench_texts)


class TestCLIOptions:
    def test_clean_index_in_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--clean-index" in result.output

    def test_replicates_in_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--replicates" in result.output


class TestCleanNovaRagIndex:
    def test_removes_existing_dir(self, tmp_path):
        target = tmp_path / "fake-nova-rag"
        (target / "abcd").mkdir(parents=True)
        (target / "abcd" / "faiss.index").write_bytes(b"x" * 100)
        clean_nova_rag_index(target)
        assert not target.exists()

    def test_missing_dir_is_a_noop(self, tmp_path):
        target = tmp_path / "absent"
        clean_nova_rag_index(target)
        assert not target.exists()

    def test_returns_target_path(self, tmp_path):
        target = tmp_path / "x"
        target.mkdir()
        out = clean_nova_rag_index(target)
        assert out == target


class TestComputeIQR:
    def test_basic(self):
        # Q1=2, Q3=4 → IQR=2
        assert compute_iqr([1, 2, 3, 4, 5]) == pytest.approx(2.0)

    def test_constant(self):
        assert compute_iqr([3, 3, 3]) == 0.0

    def test_empty(self):
        assert compute_iqr([]) == 0.0

    def test_single(self):
        assert compute_iqr([42.0]) == 0.0


class TestByRepoBreakdown:
    def test_compute_metrics_groups_by_repo(self):
        results = [
            QueryResult(
                query_id="q1", query_text="", query_type="locate",
                difficulty="easy", expected_files=["src/app.py"],
                expected_symbols=[], returned_files=["src/app.py"],
                returned_symbols=[], latency_ms=50.0, repo="flask",
            ),
            QueryResult(
                query_id="q2", query_text="", query_type="locate",
                difficulty="easy", expected_files=["main.ts"],
                expected_symbols=[], returned_files=["main.ts"],
                returned_symbols=[], latency_ms=70.0, repo="express",
            ),
            QueryResult(
                query_id="q3", query_text="", query_type="locate",
                difficulty="easy", expected_files=["pkg.py"],
                expected_symbols=[], returned_files=["wrong.py"],
                returned_symbols=[], latency_ms=80.0, repo="fastapi",
            ),
        ]
        m = compute_metrics(results)
        assert set(m.by_repo.keys()) == {"flask", "express", "fastapi"}
        assert m.by_repo["flask"]["count"] == 1
        assert m.by_repo["flask"]["hit_at_5"] == 1.0
        assert "latency_p50_ms" in m.by_repo["flask"]
        assert "mrr" in m.by_repo["flask"]

    def test_results_without_repo_are_excluded(self):
        results = [
            QueryResult(
                query_id="q1", query_text="", query_type="locate",
                difficulty="easy", expected_files=["a.py"],
                expected_symbols=[], returned_files=["a.py"],
                returned_symbols=[], latency_ms=10.0, repo="",
            ),
        ]
        m = compute_metrics(results)
        assert m.by_repo == {}


class TestReplicateAggregation:
    def _mk_metrics(self, hit_at_5: float, p95: float, score: float):
        results = [
            QueryResult(
                query_id="q", query_text="", query_type="locate",
                difficulty="easy",
                expected_files=["a.py"] if hit_at_5 > 0 else ["b.py"],
                expected_symbols=[],
                returned_files=["a.py"], returned_symbols=[],
                latency_ms=p95, repo="flask",
            ),
        ]
        m = compute_metrics(results)
        # Force values for deterministic median check
        m.hit_at_5 = hit_at_5
        m.query_latency_p95_ms = p95
        m.composite_score = score
        return m

    def test_median_picks_middle(self):
        reps = [
            self._mk_metrics(0.1, 100, 0.2),
            self._mk_metrics(0.5, 200, 0.5),
            self._mk_metrics(0.9, 300, 0.7),
        ]
        med = _median_metrics(reps, fallback=reps[0])
        assert med["hit_at_5"] == pytest.approx(0.5)
        assert med["query_latency_p95_ms"] == pytest.approx(200)
        assert med["composite_score"] == pytest.approx(0.5)

    def test_replicate_summary_length_and_fields(self):
        reps = [
            self._mk_metrics(0.1, 100, 0.2),
            self._mk_metrics(0.5, 200, 0.5),
            self._mk_metrics(0.9, 300, 0.7),
        ]
        summary = _replicate_summary(reps)
        assert len(summary) == 3
        for entry in summary:
            assert {"hit_at_5", "mrr", "latency_p50_ms", "composite_score"}.issubset(entry)

    def test_replicate_iqr_present(self):
        reps = [
            self._mk_metrics(0.1, 100, 0.2),
            self._mk_metrics(0.5, 200, 0.5),
            self._mk_metrics(0.9, 300, 0.7),
        ]
        iqr = _replicate_iqr(reps)
        assert iqr["hit_at_5"] >= 0.0
        assert iqr["latency_p95_ms"] >= 0.0
        assert iqr["composite_score"] >= 0.0

    def test_apply_median_overwrites_target(self):
        target = self._mk_metrics(0.0, 0.0, 0.0)
        _apply_median(target, {f: 0.42 for f in _MEDIAN_FIELDS})
        for f in _MEDIAN_FIELDS:
            assert getattr(target, f) == pytest.approx(0.42)


class TestBuildResultJson:
    def _stub_metrics(self):
        results = [
            QueryResult(
                query_id="q1", query_text="", query_type="locate",
                difficulty="easy", expected_files=["a.py"],
                expected_symbols=[], returned_files=["a.py"],
                returned_symbols=[], latency_ms=50.0, repo="flask",
            ),
            QueryResult(
                query_id="q2", query_text="", query_type="locate",
                difficulty="easy", expected_files=["b.py"],
                expected_symbols=[], returned_files=["b.py"],
                returned_symbols=[], latency_ms=70.0, repo="fastapi",
            ),
            QueryResult(
                query_id="q3", query_text="", query_type="locate",
                difficulty="easy", expected_files=["c.ts"],
                expected_symbols=[], returned_files=["c.ts"],
                returned_symbols=[], latency_ms=90.0, repo="express",
            ),
        ]
        return compute_metrics(
            results, ingest_total_sec=1.0, ingest_total_files=10,
            index_size_mb=2.0, ram_peak_mb=128.0,
        ), results

    def test_run_id_is_uuid_v4_and_metadata_present(self):
        metrics, results = self._stub_metrics()
        run_id = str(uuid.uuid4())
        out = _build_result_json(
            run_id=run_id,
            server_config={"name": "nova-rag", "version": "0.1.0"},
            metrics=metrics,
            query_results=results,
            repos=[],
            replicate_metrics=[metrics, metrics, metrics],
            startup_ms=42.5,
            detected_tools={"ingest": "rag_index", "query": "code_search"},
        )
        assert UUID_V4_RE.match(out["run_id"])
        assert SEMVER_RE.match(out["bench_version"])
        assert out["server"]["name"] == "nova-rag"
        assert out["environment"]["os"]
        assert out["environment"]["python"]
        assert out["startup_ms"] == 42.5
        assert out["environment"]["startup_ms"] == 42.5

    def test_replicates_array_length_matches(self):
        metrics, results = self._stub_metrics()
        out = _build_result_json(
            run_id=str(uuid.uuid4()),
            server_config={"name": "nova-rag"},
            metrics=metrics,
            query_results=results,
            repos=[],
            replicate_metrics=[metrics, metrics, metrics],
            startup_ms=1.0,
            detected_tools={},
        )
        assert len(out["replicates"]) == 3

    def test_iqr_and_by_repo_present(self):
        metrics, results = self._stub_metrics()
        out = _build_result_json(
            run_id=str(uuid.uuid4()),
            server_config={"name": "nova-rag"},
            metrics=metrics,
            query_results=results,
            repos=[],
            replicate_metrics=[metrics, metrics, metrics],
            startup_ms=1.0,
            detected_tools={},
        )
        assert "iqr" in out
        assert "hit_at_5" in out["iqr"]
        assert set(out["by_repo"].keys()) == {"flask", "fastapi", "express"}

    def test_server_version_from_mcp_handshake(self):
        """server.version comes from the MCP initialize handshake."""
        metrics, results = self._stub_metrics()
        out = _build_result_json(
            run_id=str(uuid.uuid4()),
            server_config={"name": "nova-rag"},
            server_info={"name": "nova-rag", "version": "0.8.2"},
            metrics=metrics,
            query_results=results,
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=1.0,
            detected_tools={},
        )
        assert out["server"]["version"] == "0.8.2"
        assert out["server"]["name"] == "nova-rag"

    def test_server_version_non_empty(self):
        """server.version is non-empty when MCP handshake provides it."""
        metrics, results = self._stub_metrics()
        out = _build_result_json(
            run_id=str(uuid.uuid4()),
            server_config={"name": "test-server"},
            server_info={"name": "test-server", "version": "1.2.3"},
            metrics=metrics,
            query_results=results,
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=1.0,
            detected_tools={},
        )
        assert isinstance(out["server"]["version"], str)
        assert len(out["server"]["version"]) > 0

    def test_server_version_falls_back_to_config(self):
        """When server_info is missing, version falls back to preset config."""
        metrics, results = self._stub_metrics()
        out = _build_result_json(
            run_id=str(uuid.uuid4()),
            server_config={"name": "test-server", "version": "0.1.0"},
            server_info=None,
            metrics=metrics,
            query_results=results,
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=1.0,
            detected_tools={},
        )
        assert out["server"]["version"] == "0.1.0"

    def test_server_name_from_mcp_handshake(self):
        """server.name prefers the MCP handshake over preset config."""
        metrics, results = self._stub_metrics()
        out = _build_result_json(
            run_id=str(uuid.uuid4()),
            server_config={"name": "stale-name"},
            server_info={"name": "fresh-name", "version": "2.0.0"},
            metrics=metrics,
            query_results=results,
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=1.0,
            detected_tools={},
        )
        assert out["server"]["name"] == "fresh-name"

    def test_server_version_with_empty_server_info(self):
        """Empty server_info dict falls back to config."""
        metrics, results = self._stub_metrics()
        out = _build_result_json(
            run_id=str(uuid.uuid4()),
            server_config={"name": "srv", "version": "9.9.9"},
            server_info={},
            metrics=metrics,
            query_results=results,
            repos=[],
            replicate_metrics=[metrics],
            startup_ms=1.0,
            detected_tools={},
        )
        assert out["server"]["version"] == "9.9.9"
        assert out["server"]["name"] == "srv"


class TestAdapterPathScoping:
    def test_path_value_is_used_when_schema_has_path(self):
        adapter = RAGAdapter.__new__(RAGAdapter)
        adapter._query_params = {
            "query": "{query}", "top_k": "{top_k}", "path": "{path}",
        }
        values = adapter._query_values("hello", 10, "/tmp/repo")
        filled = adapter._fill_params(adapter._query_params, values)
        assert filled["query"] == "hello"
        assert filled["top_k"] == 10
        assert filled["path"] == "/tmp/repo"

    def test_path_omitted_when_schema_lacks_path(self):
        adapter = RAGAdapter.__new__(RAGAdapter)
        adapter._query_params = {"query": "{query}", "top_k": "{top_k}"}
        values = adapter._query_values("hello", 10, "/tmp/repo")
        filled = adapter._fill_params(adapter._query_params, values)
        assert "path" not in filled

    @pytest.mark.asyncio
    async def test_query_raw_passes_path_to_tool_call(self):
        client = MCPClient(command="true")
        captured: dict = {}

        async def fake_request(method, params):
            if method == "tools/call":
                captured["arguments"] = params["arguments"]
            return {"content": [{"type": "text", "text": "[]"}]}

        client._request = fake_request  # type: ignore[assignment]

        adapter = RAGAdapter(client)
        adapter._query_tool = "code_search"
        adapter._query_params = {
            "query": "{query}", "top_k": "{top_k}", "path": "{path}",
        }
        await adapter.query_raw("hello world", top_k=10, path="/tmp/flask")
        assert captured["arguments"]["query"] == "hello world"
        assert captured["arguments"]["top_k"] == 10
        assert captured["arguments"]["path"] == "/tmp/flask"


class TestAdapterDetectsPathParam:
    @pytest.mark.asyncio
    async def test_detect_tools_picks_up_path_in_query_schema(self):
        client = MCPClient(command="true")

        async def fake_request(method, params):
            if method == "initialize":
                return {"serverInfo": {"name": "stub"}}
            if method == "tools/list":
                return {"tools": [
                    {
                        "name": "rag_index",
                        "description": "ingest",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                    {
                        "name": "code_search",
                        "description": "search",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "top_k": {"type": "integer"},
                                "path": {"type": "string"},
                                "path_filter": {"type": "string"},
                            },
                        },
                    },
                ]}
            return {}

        client._request = fake_request  # type: ignore[assignment]

        async def fake_notify(method, params):
            return None

        client._notify = fake_notify  # type: ignore[assignment]

        adapter = RAGAdapter(client)
        await adapter.detect_tools()
        assert "path" in (adapter._query_params or {})
        assert adapter._query_params["path"] == "{path}"
        assert adapter._query_params["query"] == "{query}"
        assert adapter._query_params["top_k"] == "{top_k}"
