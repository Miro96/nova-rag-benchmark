"""Grep/Glob baseline — keyword extraction + local grep (or DeepSeek agent).

Provides a search-only (no ingest) in-process preset that searches a code
repository using grep, glob, and file-reading tools.  Supports two back-ends:

1. **Local grep/glob** — fast keyword-based search (default).
2. **DeepSeek-powered agent** — LLM-driven function-calling search
   (activated when ``DEEPSEEK_API_KEY`` is set).

Use as a baseline to measure how much code-intelligence (nova-rag, naive RAG,
CocoIndex) adds beyond plain text search.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from rag_bench.baseline import DeepSeekBaselineAgent, run_baseline_search

logger = logging.getLogger(__name__)


class GrepGlobSearcher:
    """In-process grep/glob baseline search.

    Satisfies the ``InProcessClient`` contract: ``list_tools()`` +
    ``call_tool(name, **params)``.

    The searcher is **stateless** — it does not maintain an index or cache
    between queries.  Each ``code_search`` call searches directly against the
    repo directory on disk.

    Environment variables
    ---------------------
    DEEPSEEK_API_KEY : str, optional
        When set the DeepSeek-powered agent is used; otherwise local grep/glob.
    GREP_GLOB_MODEL : str
        DeepSeek model name (default: ``deepseek-v4-flash``).
    GREP_GLOB_MAX_ITERATIONS : int
        Max agent tool-calling rounds (default: 5).
    """

    def __init__(self) -> None:
        self._api_key = os.getenv("DEEPSEEK_API_KEY")
        self._model = os.getenv("GREP_GLOB_MODEL", "deepseek-v4-flash")
        self._max_iterations = int(os.getenv("GREP_GLOB_MAX_ITERATIONS", "5"))

    # -- Tool contract -------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the tool descriptor for code_search.

        The grep/glob baseline is search-only — it has no index/ingest tool.
        """
        return [
            {
                "name": "code_search",
                "description": (
                    "Search a code repository using grep, glob, and file-reading "
                    "tools.  Supports keyword extraction + shell grep (fast, local) "
                    "or LLM-powered function-calling search via DeepSeek API "
                    "(smarter but requires an API key)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language or keyword query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of results.",
                            "default": 5,
                        },
                        "path": {
                            "type": "string",
                            "description": "Root directory of the repo to search.",
                        },
                    },
                    "required": ["query", "path"],
                },
            },
        ]

    def call_tool(self, name: str, **params: Any) -> Any:
        if name == "code_search":
            return self._search(
                query=params.get("query", ""),
                top_k=int(params.get("top_k", 5)),
                path=params.get("path", "."),
            )
        raise ValueError(f"Unknown tool: {name}")

    # -- Search --------------------------------------------------------------

    def _search(
        self, query: str, top_k: int = 5, path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the repo directory using grep/glob or DeepSeek agent.

        Returns a list of result dicts with ``file_path`` and ``score`` keys.
        """
        repo_dir = Path(path).expanduser().resolve() if path else Path.cwd()

        if not repo_dir.is_dir():
            logger.warning("GrepGlob: not a directory: %s", repo_dir)
            return []

        t0 = time.perf_counter()

        if self._api_key:
            # DeepSeek-powered agent — run synchronously via asyncio
            import asyncio
            try:
                agent = DeepSeekBaselineAgent(
                    api_key=self._api_key,
                    model=self._model,
                    max_iterations=self._max_iterations,
                )
                result = asyncio.run(agent.search(query=query, repo_dir=repo_dir))
                found_files = result.get("found_files", [])[:top_k]
                found_symbols = result.get("found_symbols", [])
                latency_ms = result.get("total_time_ms", 0)
            except Exception as e:
                logger.warning(
                    "DeepSeek agent failed, falling back to local grep/glob: %s", e,
                )
                # Fall through to local
                return self._local_search(query, top_k, repo_dir)
        else:
            return self._local_search(query, top_k, repo_dir)

        # Build result list from agent response
        results: list[dict[str, Any]] = []
        for i, fp in enumerate(found_files):
            results.append({
                "file_path": fp,
                "score": 1.0 - (i * 0.01),  # rank-based pseudo-score
                "symbol": found_symbols[i] if i < len(found_symbols) else None,
            })
        return results

    def _local_search(
        self, query: str, top_k: int, repo_dir: Path,
    ) -> list[dict[str, Any]]:
        """Run the local grep/glob strategy (no LLM)."""
        br = run_baseline_search(
            query=query,
            query_type="locate",
            repo_dir=repo_dir,
            expected_symbols=[],
        )

        results: list[dict[str, Any]] = []
        for i, fp in enumerate(br.found_files[:top_k]):
            results.append({
                "file_path": fp,
                "score": 1.0 - (i * 0.01),  # rank-based pseudo-score
                "symbol": (
                    br.found_symbols[i] if i < len(br.found_symbols) else None
                ),
            })
        return results
