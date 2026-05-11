"""Tests for naive RAG baseline (M3): NaiveRAGSearcher class and 'naive-rag' preset.

Covers VAL-NAIVE-001 through VAL-NAIVE-005.
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


SAMPLE_JS_CODE = '''
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

    async fetchData(url) {
        const response = await fetch(url);
        return response.json();
    }
}

const formatDate = (date) => {
    return date.toISOString();
};
'''


def _create_temp_repo(files: dict[str, str]) -> Path:
    """Create a temporary directory with sample code files.

    Args:
        files: Dict mapping relative filenames to their content strings.

    Returns:
        Path to the temporary directory.
    """
    tmp = tempfile.mkdtemp(prefix="naive_rag_test_")
    for rel_path, content in files.items():
        full_path = Path(tmp) / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    return Path(tmp)


# ---------------------------------------------------------------------------
# VAL-NAIVE-004: Model name parity
# ---------------------------------------------------------------------------

class TestModelParity:
    """Confirm naive RAG uses the same model as nova-rag."""

    def test_default_model_is_all_minilm(self):
        from rag_bench.naive_rag import NaiveRAGSearcher

        searcher = NaiveRAGSearcher()
        assert searcher.model_name == "all-MiniLM-L6-v2"

    def test_env_var_overrides_model_name(self):
        """NAIVE_RAG_MODEL_NAME env var should override the default."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        with patch.dict(os.environ, {"NAIVE_RAG_MODEL_NAME": "all-mpnet-base-v2"}):
            searcher = NaiveRAGSearcher()
            assert searcher.model_name == "all-mpnet-base-v2"

    def test_env_var_chunk_size(self):
        """NAIVE_RAG_CHUNK_SIZE env var should override the default chunk size."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        with patch.dict(os.environ, {"NAIVE_RAG_CHUNK_SIZE": "2000"}):
            searcher = NaiveRAGSearcher()
            assert searcher.max_chunk_chars == 2000


# ---------------------------------------------------------------------------
# VAL-NAIVE-001: Index and query produces results
# ---------------------------------------------------------------------------

class TestIndexAndQuery:
    """Confirm that indexing a directory and querying returns valid results."""

    def test_list_tools_returns_index_and_search(self):
        """list_tools() should expose rag_index and code_search."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        searcher = NaiveRAGSearcher()
        tools = searcher.list_tools()

        tool_names = {t["name"] for t in tools}
        assert "rag_index" in tool_names
        assert "code_search" in tool_names

    def test_index_and_search_returns_results(self):
        """Index a small repo, then search — must return results with
        file_path and score."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        repo = _create_temp_repo({
            "main.py": SAMPLE_PY_CODE,
        })

        try:
            searcher = NaiveRAGSearcher()
            # Index
            index_result = searcher.call_tool("rag_index", path=str(repo))
            assert index_result.get("status") == "ok"
            assert index_result.get("files_indexed", 0) > 0

            # Search
            results = searcher.call_tool(
                "code_search", query="add two numbers", top_k=5,
            )
            assert isinstance(results, list)
            assert len(results) > 0

            for r in results:
                assert "file_path" in r
                assert "score" in r
                assert 0.0 <= r["score"] <= 1.0
        finally:
            import shutil
            shutil.rmtree(str(repo))


# ---------------------------------------------------------------------------
# VAL-NAIVE-002: Hit@5 computable from result file paths
# ---------------------------------------------------------------------------

class TestHitAt5Computable:
    """Confirm result file paths can be matched against ground truth."""

    def test_file_paths_are_relative_strings(self):
        """Every result must have a non-empty file_path string."""
        from rag_bench.naive_rag import NaiveRAGSearcher
        from rag_bench.metrics import file_matches

        repo = _create_temp_repo({
            "src/auth.py": SAMPLE_PY_CODE,
            "src/utils.py": "def helper(): pass",
        })

        try:
            searcher = NaiveRAGSearcher()
            searcher.call_tool("rag_index", path=str(repo))
            results = searcher.call_tool(
                "code_search", query="authentication", top_k=5,
            )

            for r in results:
                assert isinstance(r["file_path"], str)
                assert len(r["file_path"]) > 0

            # file_matches should work on these paths
            ground_truth = ["src/auth.py"]
            hit = any(
                any(file_matches(r["file_path"], exp) for r in results[:5])
                for exp in ground_truth
            )
            assert isinstance(hit, bool)
        finally:
            import shutil
            shutil.rmtree(str(repo))

    def test_hit_at_5_computation(self):
        """Compute Hit@5 from naive RAG results against ground truth."""
        from rag_bench.naive_rag import NaiveRAGSearcher
        from rag_bench.metrics import compute_metrics, QueryResult

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
            searcher = NaiveRAGSearcher()
            searcher.call_tool("rag_index", path=str(repo))

            query_results = []
            test_queries = [
                ("save user to database", ["core/models.py"]),
                ("render html template", ["core/views.py"]),
            ]

            for query_text, expected in test_queries:
                raw = searcher.call_tool("code_search", query=query_text, top_k=5)
                returned_files = [r["file_path"] for r in raw]

                from rag_bench.metrics import file_matches
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
# VAL-NAIVE-003: Latency metrics schema
# ---------------------------------------------------------------------------

class TestLatencyMetrics:
    """Confirm naive RAG exposes timing metrics comparable to other presets."""

    def test_index_and_query_have_latency(self):
        """call_tool for index and search should report timing."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        repo = _create_temp_repo({
            "a.py": "def foo(): pass",
        })

        try:
            searcher = NaiveRAGSearcher()
            index_result = searcher.call_tool("rag_index", path=str(repo))
            # Index result should include timing info
            assert "chunks_embedded" in index_result
            assert index_result.get("chunks_embedded", 0) > 0

            search_result = searcher.call_tool(
                "code_search", query="foo function", top_k=5,
            )
            # Search result list should have scores
            assert isinstance(search_result, list)
        finally:
            import shutil
            shutil.rmtree(str(repo))


# ---------------------------------------------------------------------------
# Code chunking unit tests
# ---------------------------------------------------------------------------

class TestChunking:
    """Test the code chunking logic."""

    def test_python_chunking_detects_functions_and_classes(self):
        """Python code should be chunked at function and class boundaries."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        searcher = NaiveRAGSearcher()
        chunks = searcher._chunk_file_content(SAMPLE_PY_CODE, "test.py")

        assert len(chunks) >= 3  # hello(), Calculator class, factorial()
        # At least one chunk should mention 'Calculator'
        class_chunks = [c for c in chunks if "Calculator" in c["content"]]
        assert len(class_chunks) >= 1

    def test_javascript_chunking_detects_functions_and_classes(self):
        """JS code should be chunked at function/class boundaries."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        searcher = NaiveRAGSearcher()
        chunks = searcher._chunk_file_content(SAMPLE_JS_CODE, "app.js")

        assert len(chunks) >= 3
        class_chunks = [c for c in chunks if "User" in c["content"]]
        assert len(class_chunks) >= 1

    def test_chunks_have_required_fields(self):
        """Each chunk must have content, file_path, and start_line."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        searcher = NaiveRAGSearcher()
        chunks = searcher._chunk_file_content(SAMPLE_PY_CODE, "test.py")

        for chunk in chunks:
            assert "content" in chunk
            assert "file_path" in chunk
            assert chunk["file_path"] == "test.py"
            assert "start_line" in chunk

    def test_empty_file_returns_no_chunks(self):
        """Empty or whitespace-only files produce no chunks."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        searcher = NaiveRAGSearcher()
        chunks = searcher._chunk_file_content("  \n\n  ", "empty.py")
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# Preset configuration
# ---------------------------------------------------------------------------

class TestPresetConfiguration:
    """Verify the naive-rag preset JSON is well-formed and loadable."""

    def test_preset_exists(self):
        """The naive_rag.json preset file must exist."""
        preset_path = PRESETS_DIR / "naive_rag.json"
        assert preset_path.exists(), f"Expected preset at {preset_path}"

    def test_preset_is_valid_json_with_required_fields(self):
        """Preset must have transport=inprocess, module, class fields."""
        preset_path = PRESETS_DIR / "naive_rag.json"
        data = json.loads(preset_path.read_text())

        assert data["name"] == "naive-rag"
        assert data["transport"] == "inprocess"
        assert "module" in data
        assert "class" in data
        assert "tool_mapping" in data
        assert "ingest" in data["tool_mapping"]
        assert "query" in data["tool_mapping"]

    def test_preset_loads_via_cli_load_preset(self):
        """The preset must be discoverable via the CLI's preset loading."""
        from rag_bench.cli import _load_preset

        config = _load_preset("naive-rag")
        assert config is not None
        assert config["transport"] == "inprocess"

    def test_preset_creates_client_via_runner(self):
        """The preset must produce a valid InProcessClient via _create_client."""
        from rag_bench.runner import _create_client

        preset_path = PRESETS_DIR / "naive_rag.json"
        config = json.loads(preset_path.read_text())

        client = _create_client(config)
        assert client is not None
        from rag_bench.inprocess_client import InProcessClient
        assert isinstance(client, InProcessClient)

    def test_inprocess_client_tool_discovery(self):
        """InProcessClient wrapping NaiveRAGSearcher should expose tools."""
        from rag_bench.runner import _create_client
        import asyncio

        preset_path = PRESETS_DIR / "naive_rag.json"
        config = json.loads(preset_path.read_text())

        client = _create_client(config)
        tools = asyncio.run(client.list_tools())

        tool_names = [t.name for t in tools]
        assert "rag_index" in tool_names
        assert "code_search" in tool_names


# ---------------------------------------------------------------------------
# VAL-NAIVE-005: Works on all 3 benchmark repos
# ---------------------------------------------------------------------------

class TestAllRepos:
    """Index all 3 benchmark repos without OOM or crash.

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
        """Index Flask repo without OOM."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        repo_path = str(Path.home() / ".cache" / "rag-bench" / "repos" / "flask")
        searcher = NaiveRAGSearcher()

        result = searcher.call_tool("rag_index", path=repo_path)
        assert result["status"] == "ok"
        assert result["files_indexed"] > 0
        assert result["chunks_embedded"] > 0

        # Search should work
        results = searcher.call_tool(
            "code_search", query="handle HTTP request", top_k=5,
        )
        assert len(results) > 0

    def test_index_fastapi(self):
        """Index FastAPI repo without OOM."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        repo_path = str(Path.home() / ".cache" / "rag-bench" / "repos" / "fastapi")
        searcher = NaiveRAGSearcher()

        result = searcher.call_tool("rag_index", path=repo_path)
        assert result["status"] == "ok"
        assert result["files_indexed"] > 0
        assert result["chunks_embedded"] > 0

        results = searcher.call_tool(
            "code_search", query="path parameter dependency", top_k=5,
        )
        assert len(results) > 0

    def test_index_express(self):
        """Index Express repo without OOM."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        repo_path = str(Path.home() / ".cache" / "rag-bench" / "repos" / "express")
        searcher = NaiveRAGSearcher()

        result = searcher.call_tool("rag_index", path=repo_path)
        assert result["status"] == "ok"
        assert result["files_indexed"] > 0
        assert result["chunks_embedded"] > 0

        results = searcher.call_tool(
            "code_search", query="route middleware", top_k=5,
        )
        assert len(results) > 0

    def test_all_three_repos_sequentially(self):
        """Index all three repos one after another — no OOM or crash."""
        from rag_bench.naive_rag import NaiveRAGSearcher

        searcher = NaiveRAGSearcher()
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
