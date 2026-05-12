"""Tests for BM25 baseline preset (M6): BM25Searcher class and 'bm25' preset.

Covers VAL-BM25-001 through VAL-BM25-004.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

PRESETS_DIR = Path(__file__).parent.parent / "rag_bench" / "presets"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_PY_CODE = '''
def hello():
    """Say hello."""
    return "Hello, world!"

class Calculator:
    """Simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def subtract(self, a, b):
        """Subtract b from a."""
        return a - b

def factorial(n):
    """Compute factorial."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''


def _create_temp_repo(files: dict[str, str]) -> Path:
    """Create a temporary directory with sample code files."""
    tmp = tempfile.mkdtemp(prefix="bm25_test_")
    for rel_path, content in files.items():
        full_path = Path(tmp) / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    return Path(tmp)


# ---------------------------------------------------------------------------
# VAL-BM25-001: BM25 preset loads with InProcess transport, no API keys
# ---------------------------------------------------------------------------

class TestBM25PresetLoads:
    """Confirm BM25 preset loads correctly via InProcess transport without API keys."""

    def test_preset_exists(self):
        """The bm25.json preset file must exist."""
        preset_path = PRESETS_DIR / "bm25.json"
        assert preset_path.exists(), f"Expected preset at {preset_path}"

    def test_preset_is_valid_json_with_required_fields(self):
        """Preset must have transport=inprocess, module, class fields."""
        preset_path = PRESETS_DIR / "bm25.json"
        data = json.loads(preset_path.read_text())

        assert data["name"] == "bm25"
        assert data["transport"] == "inprocess"
        assert "module" in data
        assert "class" in data
        assert data["module"] == "rag_bench.bm25"
        assert data["class"] == "BM25Searcher"
        assert "tool_mapping" in data
        assert "ingest" in data["tool_mapping"]
        assert "query" in data["tool_mapping"]

    def test_no_api_key_fields_required(self):
        """Preset must not require any API key fields."""
        preset_path = PRESETS_DIR / "bm25.json"
        data = json.loads(preset_path.read_text())

        # No API key fields anywhere in the config
        serialized = json.dumps(data).lower()
        assert "api_key" not in serialized
        assert "token" not in serialized

    def test_preset_loads_via_cli_load_preset(self):
        """The preset must be discoverable via the CLI's preset loading."""
        from rag_bench.cli import _load_preset

        config = _load_preset("bm25")
        assert config is not None
        assert config["transport"] == "inprocess"

    def test_preset_creates_client_via_runner(self):
        """The preset must produce a valid InProcessClient via _create_client."""
        from rag_bench.runner import _create_client

        preset_path = PRESETS_DIR / "bm25.json"
        config = json.loads(preset_path.read_text())

        client = _create_client(config)
        assert client is not None
        from rag_bench.inprocess_client import InProcessClient
        assert isinstance(client, InProcessClient)

    def test_inprocess_client_tool_discovery(self):
        """InProcessClient wrapping BM25Searcher should expose tools."""
        from rag_bench.runner import _create_client

        preset_path = PRESETS_DIR / "bm25.json"
        config = json.loads(preset_path.read_text())

        client = _create_client(config)
        tools = asyncio.run(client.list_tools())

        tool_names = [t.name for t in tools]
        assert "rag_index" in tool_names
        assert "code_search" in tool_names

    def test_list_tools_returns_index_and_search(self):
        """list_tools() should expose rag_index and code_search."""
        from rag_bench.bm25 import BM25Searcher

        searcher = BM25Searcher()
        tools = searcher.list_tools()

        tool_names = {t["name"] for t in tools}
        assert "rag_index" in tool_names
        assert "code_search" in tool_names

    def test_bm25_searcher_importable(self):
        """BM25Searcher class must be importable without errors."""
        from rag_bench.bm25 import BM25Searcher
        searcher = BM25Searcher()
        assert searcher is not None

    def test_no_api_key_needed_for_index_or_search(self):
        """BM25Searcher must work without any API keys set."""
        # Clear all API-key-related env vars
        clean_env = {
            k: v for k, v in os.environ.items()
            if "API_KEY" not in k and "TOKEN" not in k and "SECRET" not in k
        }
        with patch.dict(os.environ, clean_env, clear=True):
            from rag_bench.bm25 import BM25Searcher
            searcher = BM25Searcher()
            # Should not raise
            tools = searcher.list_tools()
            assert len(tools) == 2


# ---------------------------------------------------------------------------
# VAL-BM25-002: BM25 search returns file_path + score for code queries
# ---------------------------------------------------------------------------

class TestBM25SearchResults:
    """Confirm BM25 search returns proper results with file_path and score."""

    def test_index_and_search_returns_results(self):
        """Index a small repo, then search — must return results with
        file_path and score."""
        from rag_bench.bm25 import BM25Searcher

        repo = _create_temp_repo({
            "main.py": SAMPLE_PY_CODE,
        })

        try:
            searcher = BM25Searcher()
            # Index
            index_result = searcher.call_tool("rag_index", path=str(repo))
            assert index_result.get("status") == "ok"
            assert index_result.get("files_indexed", 0) > 0
            assert index_result.get("chunks_indexed", 0) > 0

            # Search
            results = searcher.call_tool(
                "code_search", query="add two numbers", top_k=5,
            )
            assert isinstance(results, list)
            assert len(results) > 0

            for r in results:
                assert "file_path" in r, f"Result missing file_path: {r}"
                assert r["file_path"], f"Empty file_path in result: {r}"
                assert "score" in r, f"Result missing score: {r}"
                assert isinstance(r["score"], (int, float)), (
                    f"Score is not numeric: {r['score']}"
                )
        finally:
            import shutil
            shutil.rmtree(str(repo))

    def test_search_returns_relevant_file(self):
        """Search for 'add' should return a file containing Calculator."""
        from rag_bench.bm25 import BM25Searcher

        repo = _create_temp_repo({
            "math.py": SAMPLE_PY_CODE,
            "unrelated.py": "def foo(): pass\ndef bar(): pass\n",
        })

        try:
            searcher = BM25Searcher()
            searcher.call_tool("rag_index", path=str(repo))
            results = searcher.call_tool(
                "code_search", query="add subtract calculator", top_k=5,
            )
            # The file containing Calculator.add should be in results
            file_paths = [r["file_path"] for r in results]
            assert "math.py" in file_paths, (
                f"Expected 'math.py' in results, got: {file_paths}"
            )
        finally:
            import shutil
            shutil.rmtree(str(repo))

    def test_search_empty_index_returns_empty_list(self):
        """Searching with no index should return empty list."""
        from rag_bench.bm25 import BM25Searcher

        searcher = BM25Searcher()
        results = searcher.call_tool("code_search", query="anything", top_k=5)
        assert results == []

    def test_search_empty_query_returns_empty_list(self):
        """Searching with empty query returns empty list."""
        from rag_bench.bm25 import BM25Searcher

        repo = _create_temp_repo({"a.py": "def foo(): pass"})
        try:
            searcher = BM25Searcher()
            searcher.call_tool("rag_index", path=str(repo))
            results = searcher.call_tool("code_search", query="", top_k=5)
            assert results == []
        finally:
            import shutil
            shutil.rmtree(str(repo))

    def test_top_k_respected(self):
        """BM25 search respects the top_k parameter."""
        from rag_bench.bm25 import BM25Searcher

        # Create 10 distinct files
        files = {
            f"file_{i:02d}.py": f"def func_{i}():\n    return {i}\n"
            for i in range(10)
        }
        repo = _create_temp_repo(files)
        try:
            searcher = BM25Searcher()
            searcher.call_tool("rag_index", path=str(repo))
            results = searcher.call_tool(
                "code_search", query="func", top_k=3,
            )
            assert len(results) <= 3
        finally:
            import shutil
            shutil.rmtree(str(repo))

    def test_scores_are_numeric_and_descending(self):
        """BM25 scores should be non-negative and sorted descending."""
        from rag_bench.bm25 import BM25Searcher

        files = {
            f"mod_{i}.py": f"def search_{i}():\n    return {i * 100}\n"
            for i in range(5)
        }
        repo = _create_temp_repo(files)
        try:
            searcher = BM25Searcher()
            searcher.call_tool("rag_index", path=str(repo))
            results = searcher.call_tool(
                "code_search", query="search", top_k=5,
            )
            assert len(results) > 1
            scores = [r["score"] for r in results]
            # Scores should be descending (or equal)
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"Scores not descending: {scores}"
                )
        finally:
            import shutil
            shutil.rmtree(str(repo))

    def test_file_paths_are_relative_strings(self):
        """Every result must have a non-empty, relative file_path string."""
        from rag_bench.bm25 import BM25Searcher

        repo = _create_temp_repo({
            "src/auth.py": SAMPLE_PY_CODE,
            "src/utils.py": "def helper(): pass",
        })

        try:
            searcher = BM25Searcher()
            searcher.call_tool("rag_index", path=str(repo))
            results = searcher.call_tool(
                "code_search", query="authentication", top_k=5,
            )

            for r in results:
                assert isinstance(r["file_path"], str)
                assert len(r["file_path"]) > 0
                # Should be relative to repo root (no leading /)
                assert not r["file_path"].startswith("/")
        finally:
            import shutil
            shutil.rmtree(str(repo))

    def test_hit_at_5_computation(self):
        """Compute Hit@5 from BM25 results against ground truth."""
        from rag_bench.bm25 import BM25Searcher
        from rag_bench.metrics import compute_metrics, QueryResult, file_matches

        repo = _create_temp_repo({
            "core/models.py": '''
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
    def save(self):
        pass
''',
            "core/views.py": '''
def render_template(template_name, context):
    return f"Rendered {template_name} with {context}"
''',
        })

        try:
            searcher = BM25Searcher()
            searcher.call_tool("rag_index", path=str(repo))

            query_results = []
            test_queries = [
                ("save user to database", ["core/models.py"]),
                ("render html template", ["core/views.py"]),
            ]

            for query_text, expected in test_queries:
                raw = searcher.call_tool("code_search", query=query_text, top_k=5)
                returned_files = [r["file_path"] for r in raw]

                qr = QueryResult(
                    query_id=f"test_{query_text[:10]}",
                    query_text=query_text,
                    query_type="locate",
                    difficulty="easy",
                    expected_files=expected,
                    expected_symbols=[],
                    returned_files=returned_files,
                    returned_symbols=[],
                    latency_ms=50.0,
                    tool_calls=1,
                    repo="test",
                )
                qr.found_file = any(
                    any(file_matches(ret, exp) for ret in returned_files[:5])
                    for exp in expected
                )
                query_results.append(qr)

            metrics = compute_metrics(query_results)
            assert metrics.total_queries == 2
            assert 0.0 <= metrics.hit_at_5 <= 1.0
        finally:
            import shutil
            shutil.rmtree(str(repo))


# ---------------------------------------------------------------------------
# VAL-BM25-003: BM25 completes benchmark on all 3 repos
# ---------------------------------------------------------------------------

class TestBM25AllRepos:
    """Index all 3 benchmark repos without crashes.

    NOTE: These tests require cloned repos at ~/.cache/rag-bench/repos/.
    They are skipped if the repos are not available.
    """

    @pytest.fixture(autouse=True)
    def _check_repos_available(self):
        """Skip all tests in this class if repos aren't cloned."""
        repo_dir = Path.home() / ".cache" / "rag-bench" / "repos" / "flask"
        if not repo_dir.exists():
            pytest.skip("Benchmark repos not cloned. Run the benchmark first.")

    def test_index_flask(self):
        """Index Flask repo without crash."""
        from rag_bench.bm25 import BM25Searcher

        repo_path = str(Path.home() / ".cache" / "rag-bench" / "repos" / "flask")
        searcher = BM25Searcher()

        result = searcher.call_tool("rag_index", path=repo_path)
        assert result["status"] == "ok"
        assert result["files_indexed"] > 0
        assert result["chunks_indexed"] > 0

        # Search should work
        results = searcher.call_tool(
            "code_search", query="handle HTTP request", top_k=5,
        )
        assert len(results) > 0

    def test_index_fastapi(self):
        """Index FastAPI repo without crash."""
        from rag_bench.bm25 import BM25Searcher

        repo_path = str(Path.home() / ".cache" / "rag-bench" / "repos" / "fastapi")
        searcher = BM25Searcher()

        result = searcher.call_tool("rag_index", path=repo_path)
        assert result["status"] == "ok"
        assert result["files_indexed"] > 0
        assert result["chunks_indexed"] > 0

        results = searcher.call_tool(
            "code_search", query="path parameter dependency", top_k=5,
        )
        assert len(results) > 0

    def test_index_express(self):
        """Index Express repo without crash."""
        from rag_bench.bm25 import BM25Searcher

        repo_path = str(Path.home() / ".cache" / "rag-bench" / "repos" / "express")
        searcher = BM25Searcher()

        result = searcher.call_tool("rag_index", path=repo_path)
        assert result["status"] == "ok"
        assert result["files_indexed"] > 0
        assert result["chunks_indexed"] > 0

        results = searcher.call_tool(
            "code_search", query="route middleware", top_k=5,
        )
        assert len(results) > 0

    def test_all_three_repos_sequentially(self):
        """Index all three repos one after another — no crash."""
        from rag_bench.bm25 import BM25Searcher

        searcher = BM25Searcher()
        cache_root = Path.home() / ".cache" / "rag-bench" / "repos"

        for repo_name in ["flask", "fastapi", "express"]:
            repo_path = str(cache_root / repo_name)
            result = searcher.call_tool("rag_index", path=repo_path)
            assert result["status"] == "ok"
            assert result["files_indexed"] > 0

            # Verify search works after each index
            results = searcher.call_tool(
                "code_search", query="test query", top_k=3,
            )
            assert isinstance(results, list)


# ---------------------------------------------------------------------------
# VAL-BM25-004: BM25 Hit@5 > 0 (finds relevant files)
# ---------------------------------------------------------------------------

class TestBM25HitAt5:
    """Confirm BM25 finds at least some relevant files (Hit@5 > 0)."""

    @pytest.fixture(autouse=True)
    def _check_repos_available(self):
        """Skip all tests in this class if repos aren't cloned."""
        repo_dir = Path.home() / ".cache" / "rag-bench" / "repos" / "flask"
        if not repo_dir.exists():
            pytest.skip("Benchmark repos not cloned. Run the benchmark first.")

    def test_hit_at_5_positive_on_flask(self):
        """BM25 should have Hit@5 > 0 when run against Flask queries."""
        import asyncio
        from rag_bench.runner import _create_client

        preset_path = PRESETS_DIR / "bm25.json"
        config = json.loads(preset_path.read_text())

        client = _create_client(config)

        # Load Flask queries
        from rag_bench.datasets.loader import load_queries, clone_repo, load_repos, load_warmup_queries
        from rag_bench.adapter import RAGAdapter
        from rag_bench.metrics import compute_metrics, QueryResult, file_matches

        queries = load_queries("flask")
        repos = [r for r in load_repos() if r.name == "flask"]
        repo_dirs = {r.name: clone_repo(r) for r in repos}
        repo_dir = repo_dirs["flask"]

        try:
            # Index
            adapter = RAGAdapter(client, config)
            detected = asyncio.run(adapter.detect_tools())
            assert detected.get("ingest") is not None
            assert detected.get("query") is not None

            ingest_result = asyncio.run(
                adapter.client.call_tool(
                    detected["ingest"],
                    {"path": str(repo_dir)},
                )
            )
            ingest_data = json.loads(ingest_result.text)
            assert ingest_data.get("status") == "ok"
            assert ingest_data.get("chunks_indexed", 0) > 0

            # Run queries
            results: list[QueryResult] = []
            for q in queries:
                raw = asyncio.run(
                    adapter.client.call_tool(
                        detected["query"],
                        {"query": q.query, "top_k": 5},
                    )
                )
                search_results = json.loads(raw.text)
                returned_files = [r.get("file_path", "") for r in search_results]

                qr = QueryResult(
                    query_id=q.id,
                    query_text=q.query,
                    query_type=q.type,
                    difficulty=q.difficulty,
                    expected_files=q.expected_files,
                    expected_symbols=q.expected_symbols,
                    returned_files=returned_files,
                    returned_symbols=[],
                    latency_ms=raw.latency_ms,
                    tool_calls=1,
                    repo=q.repo,
                )
                qr.found_file = any(
                    any(file_matches(ret, exp) for ret in returned_files[:5])
                    for exp in q.expected_files
                )
                results.append(qr)

            metrics = compute_metrics(results)
            assert metrics.total_queries == len(queries)
            assert metrics.hit_at_5 > 0.0, (
                f"BM25 Hit@5 is {metrics.hit_at_5}, expected > 0"
            )
        finally:
            import shutil
            # Don't clean up cloned repo — it's cached
            pass


# ---------------------------------------------------------------------------
# Tree-sitter chunking unit tests
# ---------------------------------------------------------------------------

class TestBM25Chunking:
    """Test the tree-sitter-based code chunking logic."""

    def test_python_chunking_with_tree_sitter(self):
        """Python code should be chunked at function and class boundaries via tree-sitter."""
        from rag_bench.bm25 import BM25Searcher

        searcher = BM25Searcher()
        chunks = searcher._chunk_file(SAMPLE_PY_CODE, "test.py", ".py")

        assert len(chunks) >= 3  # hello(), Calculator, factorial()
        # At least one chunk should mention 'Calculator'
        class_chunks = [c for c in chunks if "Calculator" in c["content"]]
        assert len(class_chunks) >= 1

    def test_chunks_have_required_fields(self):
        """Each chunk must have content, file_path, and start_line."""
        from rag_bench.bm25 import BM25Searcher

        searcher = BM25Searcher()
        chunks = searcher._chunk_file(SAMPLE_PY_CODE, "test.py", ".py")

        for chunk in chunks:
            assert "content" in chunk
            assert "file_path" in chunk
            assert chunk["file_path"] == "test.py"
            assert "start_line" in chunk

    def test_empty_file_returns_no_chunks(self):
        """Empty or whitespace-only files produce no chunks."""
        from rag_bench.bm25 import BM25Searcher

        searcher = BM25Searcher()
        chunks = searcher._chunk_file("  \n\n  ", "empty.py", ".py")
        assert len(chunks) == 0

    def test_symbol_detection(self):
        """Tree-sitter should detect function/class names as symbols."""
        from rag_bench.bm25 import BM25Searcher

        searcher = BM25Searcher()
        chunks = searcher._chunk_file(SAMPLE_PY_CODE, "test.py", ".py")

        symbols = [c.get("symbol") for c in chunks if c.get("symbol")]
        # Should detect at least some named functions/classes
        assert "hello" in symbols or "Calculator" in symbols or "factorial" in symbols

    def test_javascript_chunking_with_tree_sitter(self):
        """JavaScript code should be chunked via tree-sitter."""
        from rag_bench.bm25 import BM25Searcher

        js_code = '''
function greet(name) {
    return "Hello, " + name;
}

class User {
    constructor(name) {
        this.name = name;
    }

    getName() {
        return this.name;
    }
}
'''
        searcher = BM25Searcher()
        chunks = searcher._chunk_file(js_code, "app.js", ".js")
        assert len(chunks) >= 2  # greet + User


# ---------------------------------------------------------------------------
# Preset integration via CLI
# ---------------------------------------------------------------------------

class TestBM25CLIIntegration:
    """Verify the BM25 preset works end-to-end via the CLI."""

    def test_validate_command_succeeds(self):
        """rag-bench validate --preset bm25 --config-only should succeed."""
        from click.testing import CliRunner
        from rag_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate", "--preset", "bm25", "--config-only",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"

    def test_run_command_discovers_preset(self):
        """rag-bench run --preset bm25 --repo flask should not fail with
        'preset not found'."""
        from click.testing import CliRunner
        from rag_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--preset", "bm25", "--repo", "nonexistent_repo_xyz",
        ])
        # Should fail because of the repo name, not because preset not found
        output = result.output.lower()
        assert "preset 'bm25' not found" not in output, (
            f"Preset 'bm25' was not found: {result.output}"
        )
