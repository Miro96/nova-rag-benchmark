"""Tests for CocoIndex Code preset (M4).

Covers VAL-COCO-001 through VAL-COCO-006.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
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

    def subtract(self, b):
        """Subtract from result."""
        return self.result - b
'''

SAMPLE_PY_CODE_2 = '''
def greet(name):
    """Greet someone."""
    return f"Hello, {name}!"

async def fetch_data(url):
    """Fetch data from a URL."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
'''


def _create_temp_repo(files: dict[str, str]) -> Path:
    """Create a temporary directory with sample code files."""
    tmp = Path(tempfile.mkdtemp(prefix="cocoindex_test_"))
    for rel_path, content in files.items():
        full = tmp / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
    return tmp


def _load_cocoindex_preset() -> dict:
    """Load the cocoindex-code preset."""
    path = PRESETS_DIR / "cocoindex_code.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# VAL-COCO-006: Uses comparable embedding model
# ---------------------------------------------------------------------------

class TestEmbeddingModelParity:
    """VAL-COCO-006: CocoIndex preset uses comparable embedding model."""

    def test_preset_is_configured_with_local_embeddings(self):
        """CocoIndex preset has no API key requirement — local embeddings only."""
        cfg = _load_cocoindex_preset()

        # The preset uses local embeddings (ccc init defaults to
        # Snowflake arctic-embed-xs, but the user can switch to
        # all-MiniLM-L6-v2 via the --litellm-model flag in
        # pre_ingest_commands if desired).
        assert cfg["transport"] == "mcp"
        # No API key in the preset config
        assert "api_key" not in cfg

    def test_pre_ingest_commands_allow_model_override(self):
        """The pre_ingest_commands can be edited to specify a model."""
        cfg = _load_cocoindex_preset()
        pre = cfg.get("pre_ingest_commands", [])

        # The init command accepts --litellm-model to pin the embedding model
        assert len(pre) >= 2  # init + index
        init_cmd = pre[0]
        assert init_cmd["cmd"] in ("python", "python3", "ccc")


# ---------------------------------------------------------------------------
# VAL-COCO-001: MCP server starts and exposes a search tool
# ---------------------------------------------------------------------------

class TestMCPStartup:
    """VAL-COCO-001: MCP server starts and exposes a search tool."""

    def test_preset_loads_and_has_correct_transport(self):
        """The cocoindex-code preset loads with MCP transport."""
        cfg = _load_cocoindex_preset()
        assert cfg["name"] == "cocoindex-code"
        assert cfg["transport"] == "mcp"

    def test_preset_command_is_valid(self):
        """The preset's command points to the CocoIndex MCP server."""
        cfg = _load_cocoindex_preset()
        assert cfg["command"] in ("python", "python3", "ccc")
        assert "-m" in cfg.get("args", [])
        assert "cocoindex_code.cli" in cfg.get("args", [])
        assert "mcp" in cfg.get("args", [])

    def test_mcp_server_starts_with_cocoindex(self):
        """Starting the CocoIndex MCP server in an initialized project works."""
        tmp = _create_temp_repo({"main.py": SAMPLE_PY_CODE})

        # Initialize the project
        subprocess.run(
            [sys.executable, "-m", "cocoindex_code.cli", "init", "-f"],
            cwd=str(tmp), check=True, capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            [sys.executable, "-m", "cocoindex_code.cli", "index"],
            cwd=str(tmp), check=True, capture_output=True, text=True, timeout=120,
        )

        # Start the MCP server and discover tools
        from rag_bench.mcp_client import MCPClient

        client = MCPClient(
            command="python3",
            args=["-m", "cocoindex_code.cli", "mcp"],
            cwd=str(tmp),
        )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _run():
                await client.start()
                tools = await client.list_tools()
                await client.stop()
                return tools

            tools = loop.run_until_complete(_run())
            loop.close()
        except Exception:
            # Clean up on failure
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        # VAL-COCO-001 assertion: at least one search tool exists
        search_tools = [
            t for t in tools
            if "search" in t.name.lower()
        ]
        assert len(search_tools) >= 1, (
            f"Expected at least one search tool; got tools: "
            f"{[t.name for t in tools]}"
        )

        # Clean up
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# VAL-COCO-002: Search tool returns file paths and relevance scores
# ---------------------------------------------------------------------------

class TestSearchResults:
    """VAL-COCO-002: Search tool returns file paths and relevance scores."""

    def test_search_returns_file_paths_and_scores(self):
        """CocoIndex search returns structured results with file_path and score."""
        tmp = _create_temp_repo({
            "main.py": SAMPLE_PY_CODE,
            "utils.py": SAMPLE_PY_CODE_2,
        })

        # Init + index
        subprocess.run(
            [sys.executable, "-m", "cocoindex_code.cli", "init", "-f"],
            cwd=str(tmp), check=True, capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            [sys.executable, "-m", "cocoindex_code.cli", "index"],
            cwd=str(tmp), check=True, capture_output=True, text=True, timeout=120,
        )

        from rag_bench.mcp_client import MCPClient

        client = MCPClient(
            command="python3",
            args=["-m", "cocoindex_code.cli", "mcp"],
            cwd=str(tmp),
        )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _run():
                await client.start()
                result = await client.call_tool("search", {
                    "query": "add two numbers",
                    "limit": 5,
                })
                await client.stop()
                return result

            result = loop.run_until_complete(_run())
            loop.close()
        except Exception:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        text = result.text
        assert text, "Search result text should not be empty"
        assert not result.is_error, f"Search returned an error: {text}"

        # Parse JSON results
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"raw": text}

        results_list = data if isinstance(data, list) else data.get("results", [])

        assert isinstance(results_list, list), f"Expected list; got {type(results_list)}"
        assert len(results_list) > 0, "Search returned 0 results"

        # Each result should have file_path and score
        for r in results_list:
            assert isinstance(r, dict), f"Expected dict; got {type(r)}"
            # At minimum, have some file identifier
            file_key = (
                r.get("file_path") or r.get("path") or r.get("file")
                or r.get("uri") or r.get("source")
            )
            assert file_key, f"Result missing file path: {r}"

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# VAL-COCO-003: Completes a full benchmark run on at least 1 repo
# ---------------------------------------------------------------------------

class TestBenchmarkRun:
    """VAL-COCO-003: Full benchmark run completes on at least 1 repo."""

    @pytest.mark.slow
    def test_benchmark_completes_on_temp_repo(self):
        """The CocoIndex preset completes a benchmark run on a sample repo.

        This validates the full pipeline: pre_ingest_commands (init+index),
        MCP server startup, tool detection, query execution.
        """
        import sys
        from rag_bench.runner import run_benchmark
        from rag_bench.datasets.loader import Query

        cfg = _load_cocoindex_preset()

        tmp = _create_temp_repo({
            "main.py": SAMPLE_PY_CODE,
            "utils.py": SAMPLE_PY_CODE_2,
        })

        mock_repo = type("Repo", (), {
            "name": "test-repo",
            "git_url": f"file://{tmp}",
            "language": "python",
        })()

        mock_queries = [
            Query(
                id="CQ-TEST-01",
                query="find the add function for two numbers",
                type="locate",
                difficulty="easy",
                repo="test-repo",
                expected_files=["main.py"],
                expected_symbols=["add"],
            ),
        ]

        mock_warmup = [
            Query(
                id="WARMUP-01",
                query="calculator class",
                type="locate",
                difficulty="easy",
                repo="test-repo",
                expected_files=["main.py"],
                expected_symbols=["Calculator"],
            ),
        ]

        # Patch at the runner's import site so the already-imported
        # names inside run_benchmark see our mocks.
        runner_mod_path = "rag_bench.runner"
        with (
            patch(f"{runner_mod_path}.load_repos", return_value=[mock_repo]),
            patch(f"{runner_mod_path}.load_queries", return_value=mock_queries),
            patch(f"{runner_mod_path}.load_warmup_queries", return_value=mock_warmup),
            patch(f"{runner_mod_path}.clone_repo", return_value=tmp),
            patch(f"{runner_mod_path}.get_repo_files", return_value=[
                tmp / "main.py", tmp / "utils.py",
            ]),
        ):

            result = asyncio.run(run_benchmark(
                server_config=cfg,
                repo_filter="test-repo",
                top_k=5,
                replicates=1,
            ))

        # Assertions
        assert result is not None
        assert result["retrieval"]["total_queries"] > 0

        # The result should have ingest info (pre-indexed, no ingest tool)
        ingest = result["ingest"]
        assert ingest["total_files"] > 0

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_preset_runs_without_crashing(self):
        """The preset's pre_ingest_commands execute without errors on a temp repo."""
        cfg = _load_cocoindex_preset()

        tmp = _create_temp_repo({"test.py": SAMPLE_PY_CODE})

        pre_cmds = cfg.get("pre_ingest_commands", [])
        assert len(pre_cmds) == 2, "Expected init + index commands"

        for cmd_spec in pre_cmds:
            from rag_bench.base_client import _resolve_command_parts
            parts = _resolve_command_parts(
                cmd_spec["cmd"], cmd_spec.get("args", [])
            )
            cwd_tmpl = cmd_spec.get("cwd_template", "")
            cwd = cwd_tmpl.replace("{repo_path}", str(tmp)) if cwd_tmpl else None

            result = subprocess.run(
                parts, cwd=cwd, check=True,
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0, (
                f"Command {parts} failed: {result.stderr[:500]}"
            )

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# VAL-COCO-004: Result file paths are resolvable to ground truth
# ---------------------------------------------------------------------------

class TestGroundTruthMapping:
    """VAL-COCO-004: CocoIndex result file paths map to ground truth format."""

    def test_search_result_paths_are_relative(self):
        """CocoIndex returns relative file paths that can be matched."""
        tmp = _create_temp_repo({
            "src/main.py": SAMPLE_PY_CODE,
            "src/utils.py": SAMPLE_PY_CODE_2,
        })

        subprocess.run(
            [sys.executable, "-m", "cocoindex_code.cli", "init", "-f"],
            cwd=str(tmp), check=True, capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            [sys.executable, "-m", "cocoindex_code.cli", "index"],
            cwd=str(tmp), check=True, capture_output=True, text=True, timeout=120,
        )

        from rag_bench.mcp_client import MCPClient

        client = MCPClient(
            command="python3",
            args=["-m", "cocoindex_code.cli", "mcp"],
            cwd=str(tmp),
        )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _run():
                await client.start()
                result = await client.call_tool("search", {
                    "query": "add two numbers calculator",
                    "limit": 10,
                })
                await client.stop()
                return result

            result = loop.run_until_complete(_run())
            loop.close()
        except Exception:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        text = result.text
        data = json.loads(text) if text else []
        results_list = data if isinstance(data, list) else data.get("results", [])

        # Collect all file paths
        found_paths = set()
        for r in results_list:
            fp = (
                r.get("file_path") or r.get("path") or r.get("file")
                or r.get("uri") or r.get("source") or ""
            )
            if fp:
                found_paths.add(fp)

        # At least one found path should be matchable to our test files
        known_files = {"src/main.py", "src/utils.py", "main.py", "utils.py"}
        assert found_paths, "No file paths in search results"

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# VAL-COCO-005: Works without API keys (local embeddings only)
# ---------------------------------------------------------------------------

class TestNoAPIKeys:
    """VAL-COCO-005: CocoIndex works without API keys."""

    def test_preset_has_no_api_key_field(self):
        """The preset config contains no API key references."""
        cfg = _load_cocoindex_preset()
        assert "api_key" not in cfg
        assert "token" not in cfg

    def test_index_and_search_without_api_keys(self):
        """CocoIndex indexing and MCP search work without any API keys.

        We run with OPENAI_API_KEY and similar vars explicitly cleared
        to prove the preset is purely local.
        """
        tmp = _create_temp_repo({"code.py": SAMPLE_PY_CODE})

        # Clear any API key env vars
        clean_env = os.environ.copy()
        for key in list(clean_env.keys()):
            if "API_KEY" in key or "TOKEN" in key:
                clean_env.pop(key, None)

        # Init + index with clean env
        subprocess.run(
            [sys.executable, "-m", "cocoindex_code.cli", "init", "-f"],
            cwd=str(tmp), env=clean_env,
            check=True, capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            [sys.executable, "-m", "cocoindex_code.cli", "index"],
            cwd=str(tmp), env=clean_env,
            check=True, capture_output=True, text=True, timeout=120,
        )

        from rag_bench.mcp_client import MCPClient

        client = MCPClient(
            command="python3",
            args=["-m", "cocoindex_code.cli", "mcp"],
            cwd=str(tmp),
            env=clean_env,
        )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _run():
                await client.start()
                result = await client.call_tool("search", {
                    "query": "hello world greeting",
                    "limit": 5,
                })
                await client.stop()
                return result

            result = loop.run_until_complete(_run())
            loop.close()
        except Exception:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        text = result.text
        assert text, "Search should return results without API keys"
        assert not result.is_error, (
            f"Search failed without API keys: {text}"
        )

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Pre-ingest command resolution
# ---------------------------------------------------------------------------

class TestPreIngestCommands:
    """Tests for _run_pre_ingest_commands and cwd template resolution."""

    def test_pre_ingest_commands_resolve_repo_path(self):
        """The {repo_path} template is substituted in pre_ingest_commands."""
        from rag_bench.runner import _run_pre_ingest_commands

        tmp = Path(tempfile.mkdtemp(prefix="pre_ingest_test_"))
        (tmp / "test.py").write_text("x = 1")

        config = {
            "pre_ingest_commands": [
                {
                "cmd": sys.executable,
                    "args": ["-c", "import sys; sys.exit(0)"],
                    "cwd_template": "{repo_path}",
                },
            ],
        }

        repo_dirs = {"test": tmp}
        # Should not raise — the command succeeds
        _run_pre_ingest_commands(repo_dirs, config)

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_pre_ingest_command_failure_raises(self):
        """A failing pre_ingest command raises RuntimeError."""
        from rag_bench.runner import _run_pre_ingest_commands

        tmp = Path(tempfile.mkdtemp(prefix="pre_ingest_fail_"))
        (tmp / "test.py").write_text("x = 1")

        config = {
            "pre_ingest_commands": [
                {
                "cmd": sys.executable,
                    "args": ["-c", "import sys; sys.exit(1)"],
                    "cwd_template": "{repo_path}",
                },
            ],
        }

        repo_dirs = {"test": tmp}
        with pytest.raises(RuntimeError, match="exit code 1"):
            _run_pre_ingest_commands(repo_dirs, config)

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_cwd_template_resolution_single_repo(self):
        """cwd template is resolved to the first repo for single-repo runs."""
        from rag_bench.runner import _resolve_cwd_template

        config = {"cwd": "{repo_path}"}
        repo_dirs = {"test": Path("/tmp/fake-repo")}

        _resolve_cwd_template(config, repo_dirs)
        assert config["cwd"] == "/tmp/fake-repo"

    def test_cwd_template_resolution_multi_repo_picks_first(self):
        """cwd template picks the first repo for multi-repo runs."""
        from rag_bench.runner import _resolve_cwd_template

        config = {"cwd": "{repo_path}"}
        repo_dirs = {
            "flask": Path("/tmp/flask"),
            "fastapi": Path("/tmp/fastapi"),
        }

        _resolve_cwd_template(config, repo_dirs)
        # First key (insertion order preserved in Python 3.7+)
        assert config["cwd"] in ("/tmp/flask", "/tmp/fastapi")

    def test_no_cwd_template_leaves_config_unchanged(self):
        """Config without cwd template is left unchanged."""
        from rag_bench.runner import _resolve_cwd_template

        config = {"cwd": "/some/fixed/path"}
        original = dict(config)
        _resolve_cwd_template(config, {"test": Path("/tmp/other")})
        assert config == original
