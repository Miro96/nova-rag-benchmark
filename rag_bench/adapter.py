"""Adapters for normalizing different RAG MCP server interfaces."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from rag_bench.mcp_client import CallResult, MCPClient, ToolInfo

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from a RAG server."""
    file_path: str
    content: str
    score: float | None = None
    symbol: str | None = None


class RAGAdapter:
    """Normalizes different RAG MCP servers into a common interface."""

    def __init__(self, client: MCPClient, config: dict[str, Any] | None = None):
        self.client = client
        self.config = config or {}
        self._ingest_tool: str | None = None
        self._ingest_params: dict[str, str] | None = None
        self._query_tool: str | None = None
        self._query_params: dict[str, str] | None = None
        self._clear_tool: str | None = None
        self._status_tool: str | None = None

    INGEST_PATTERNS = ("rag_index", "ingest", "index", "embed", "process", "add_document")
    QUERY_PATTERNS = ("code_search", "rag_search", "query", "search", "ask", "retrieve", "find")
    CLEAR_PATTERNS = ("clear", "delete_all", "remove_all", "reset")
    STATUS_PATTERNS = ("status", "info", "stats")

    INGEST_PATH_KEYS = ["file_path", "path", "directory", "dir", "folder"]
    QUERY_TEXT_KEYS = ["query", "text", "question", "q", "search"]
    QUERY_LIMIT_KEYS = ["top_k", "limit", "k", "n", "num_results", "max_results"]

    async def detect_tools(self) -> dict[str, str | None]:
        """Resolve which tools to call and which parameter names to use.

        Tool name resolution: if the preset's `tool_mapping` specifies a tool,
        prefer it (when it actually exists on the server); otherwise fall back
        to pattern-based auto-detection.

        Parameter resolution: always anchor on the live `tools/list` schema.
        Preset-supplied params are kept only when their key exists in the
        schema; missing canonical params (path / query / top_k) are filled in
        from the schema. This protects against stale presets that reference
        non-existent params (e.g. `limit` when the server only accepts `top_k`).
        """
        tools = await self.client.list_tools()
        tool_names = [t.name for t in tools]
        logger.info("Available tools: %s", tool_names)

        mapping = self.config.get("tool_mapping", {}) or {}

        self._ingest_tool = self._resolve_tool(
            tools, mapping.get("ingest", {}).get("tool"), self.INGEST_PATTERNS
        )
        self._query_tool = self._resolve_tool(
            tools, mapping.get("query", {}).get("tool"), self.QUERY_PATTERNS
        )
        self._clear_tool = self._resolve_tool(
            tools, mapping.get("clear", {}).get("tool"), self.CLEAR_PATTERNS
        )
        self._status_tool = self._resolve_tool(
            tools, mapping.get("status", {}).get("tool"), self.STATUS_PATTERNS
        )

        if self._ingest_tool:
            schema_params = self._detect_params(
                tools, self._ingest_tool, path_keys=self.INGEST_PATH_KEYS,
            )
            preset_params = mapping.get("ingest", {}).get("params") or {}
            self._ingest_params = self._reconcile_params(
                tools, self._ingest_tool, preset_params, schema_params,
            )

        if self._query_tool:
            schema_params = self._detect_params(
                tools, self._query_tool,
                query_keys=self.QUERY_TEXT_KEYS,
                limit_keys=self.QUERY_LIMIT_KEYS,
            )
            preset_params = mapping.get("query", {}).get("params") or {}
            self._query_params = self._reconcile_params(
                tools, self._query_tool, preset_params, schema_params,
            )

        detected = {
            "ingest": self._ingest_tool,
            "query": self._query_tool,
            "clear": self._clear_tool,
            "status": self._status_tool,
        }
        logger.info("Detected tool mapping: %s", detected)
        return detected

    def _resolve_tool(
        self,
        tools: list[ToolInfo],
        configured: str | None,
        patterns: tuple[str, ...],
    ) -> str | None:
        if configured:
            if any(t.name == configured for t in tools):
                return configured
            logger.warning(
                "Configured tool '%s' not present on server; falling back to auto-detect",
                configured,
            )
        return self._find_tool(tools, *patterns)

    def _reconcile_params(
        self,
        tools: list[ToolInfo],
        tool_name: str,
        preset_params: dict[str, Any],
        schema_params: dict[str, str],
    ) -> dict[str, Any]:
        """Drop preset params not in schema, then backfill missing canonical params from schema."""
        tool = next((t for t in tools if t.name == tool_name), None)
        schema_props = tool.input_schema.get("properties", {}) if tool else {}

        merged: dict[str, Any] = {}
        for key, value in preset_params.items():
            if key in schema_props:
                merged[key] = value
            else:
                logger.warning(
                    "Preset param '%s' not in inputSchema of '%s'; dropping",
                    key, tool_name,
                )

        for key, value in schema_params.items():
            if key not in merged:
                merged[key] = value

        return merged

    async def ingest_directory(self, path: str) -> CallResult:
        """Ingest a directory or file into the RAG index."""
        if not self._ingest_tool:
            raise RuntimeError("No ingest tool detected")

        params = self._fill_params(self._ingest_params or {}, {
            "file_path": path, "path": path, "directory": path,
            "dir": path, "folder": path,
        })
        return await self.client.call_tool(self._ingest_tool, params)

    async def ingest_file(self, path: str) -> CallResult:
        """Ingest a single file."""
        if not self._ingest_tool:
            raise RuntimeError("No ingest tool detected")

        params = self._fill_params(self._ingest_params or {}, {
            "file_path": path, "path": path, "file": path,
        })
        return await self.client.call_tool(self._ingest_tool, params)

    async def query(self, text: str, top_k: int = 10) -> list[SearchResult]:
        """Query the RAG index and return normalized results."""
        if not self._query_tool:
            raise RuntimeError("No query tool detected")

        params = self._fill_params(self._query_params or {}, {
            "query": text, "text": text, "question": text,
            "q": text, "search": text,
            "limit": top_k, "top_k": top_k, "k": top_k,
            "n": top_k, "num_results": top_k, "max_results": top_k,
        })
        result = await self.client.call_tool(self._query_tool, params)
        return self._parse_search_results(result)

    async def query_raw(self, text: str, top_k: int = 10) -> CallResult:
        """Query and return raw CallResult (for latency measurement)."""
        if not self._query_tool:
            raise RuntimeError("No query tool detected")

        params = self._fill_params(self._query_params or {}, {
            "query": text, "text": text, "question": text,
            "q": text, "search": text,
            "limit": top_k, "top_k": top_k, "k": top_k,
            "n": top_k, "num_results": top_k, "max_results": top_k,
        })
        return await self.client.call_tool(self._query_tool, params)

    async def clear(self) -> CallResult | None:
        """Clear the index if supported."""
        if not self._clear_tool:
            return None
        return await self.client.call_tool(self._clear_tool, {})

    def _find_tool(self, tools: list[ToolInfo], *patterns: str) -> str | None:
        """Find a tool by name patterns."""
        for tool in tools:
            name_lower = tool.name.lower()
            for pattern in patterns:
                if pattern in name_lower:
                    return tool.name
        return None

    def _detect_params(
        self,
        tools: list[ToolInfo],
        tool_name: str,
        **key_groups: list[str],
    ) -> dict[str, str]:
        """Detect parameter names from tool schema."""
        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            return {}

        props = tool.input_schema.get("properties", {})
        detected = {}

        for group_name, candidates in key_groups.items():
            for candidate in candidates:
                if candidate in props:
                    detected[candidate] = f"{{{candidate}}}"
                    break

        return detected

    def _fill_params(self, template: dict[str, str], values: dict[str, Any]) -> dict[str, Any]:
        """Fill parameter template with actual values."""
        filled = {}
        for key, tmpl in template.items():
            if isinstance(tmpl, str) and tmpl.startswith("{") and tmpl.endswith("}"):
                var_name = tmpl[1:-1]
                if var_name in values:
                    filled[key] = values[var_name]
                else:
                    filled[key] = tmpl
            else:
                filled[key] = tmpl
        return filled

    def _parse_search_results(self, result: CallResult) -> list[SearchResult]:
        """Parse raw MCP result into SearchResult list."""
        results = []
        text = result.text

        # Try JSON parse first
        try:
            import json
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    results.append(self._item_to_search_result(item))
                return results
            if isinstance(data, dict) and "results" in data:
                for item in data["results"]:
                    results.append(self._item_to_search_result(item))
                return results
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: parse text for file paths
        file_pattern = re.compile(r'(?:file|path|source)[\s:=]+["\']?([^\s"\']+\.\w+)', re.I)
        for match in file_pattern.finditer(text):
            results.append(SearchResult(file_path=match.group(1), content=text))

        # If nothing found, treat whole text as a single result
        if not results and text.strip():
            results.append(SearchResult(file_path="", content=text))

        return results

    def _item_to_search_result(self, item: dict) -> SearchResult:
        """Convert a dict item to SearchResult."""
        file_path = (
            item.get("file_path")
            or item.get("path")
            or item.get("source")
            or item.get("file")
            or item.get("filename")
            or ""
        )
        content = (
            item.get("content")
            or item.get("text")
            or item.get("chunk")
            or item.get("snippet")
            or ""
        )
        score = item.get("score") or item.get("relevance") or item.get("similarity")
        symbol = item.get("symbol") or item.get("function") or item.get("name")
        return SearchResult(
            file_path=str(file_path),
            content=str(content),
            score=float(score) if score is not None else None,
            symbol=str(symbol) if symbol else None,
        )
