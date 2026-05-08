"""Tests for RAGAdapter — preset loading, tool detection, schema-driven param discovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rag_bench.adapter import RAGAdapter
from rag_bench.mcp_client import CallResult, ToolInfo, _resolve_command_parts

PRESETS_DIR = Path(__file__).parent.parent / "rag_bench" / "presets"


class FakeMCPClient:
    """Test double for MCPClient that returns canned tools/list and records calls."""

    def __init__(self, tools: list[dict[str, Any]]):
        self._tools_data = tools
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[ToolInfo]:
        return [
            ToolInfo(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in self._tools_data
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallResult:
        self.calls.append((name, dict(arguments)))
        return CallResult(content=[{"type": "text", "text": "{}"}], latency_ms=1.0)


def _nova_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "code_search",
            "inputSchema": {
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "path_filter": {"type": "string"},
                    "language": {"type": "string"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "rag_index",
            "inputSchema": {
                "properties": {
                    "path": {"type": "string"},
                    "force": {"type": "boolean"},
                },
            },
        },
        {
            "name": "rag_status",
            "inputSchema": {"properties": {"path": {"type": "string"}}},
        },
    ]


class TestPresetFile:
    def test_preset_loads_nova_rag_config(self):
        preset = json.loads((PRESETS_DIR / "nova_rag.json").read_text())
        assert preset["name"] == "nova-rag"
        assert preset["transport"] == "stdio"

        ingest = preset["tool_mapping"]["ingest"]
        assert ingest["tool"] == "rag_index"
        assert "path" in ingest["params"]

        query = preset["tool_mapping"]["query"]
        assert query["tool"] == "code_search"
        params = query["params"]
        assert "top_k" in params
        assert "limit" not in params

    def test_preset_no_nonexistent_tools(self):
        """Preset must not reference tools that nova-rag doesn't expose."""
        preset = json.loads((PRESETS_DIR / "nova_rag.json").read_text())
        assert preset["tool_mapping"]["ingest"]["tool"] != "index_directory"


@pytest.mark.asyncio
class TestNovaRagAdapter:
    async def test_ingest_uses_rag_index_not_index_directory(self):
        client = FakeMCPClient(_nova_tools())
        preset = json.loads((PRESETS_DIR / "nova_rag.json").read_text())
        adapter = RAGAdapter(client, preset)

        detected = await adapter.detect_tools()
        assert detected["ingest"] == "rag_index"
        assert detected["query"] == "code_search"

        await adapter.ingest_directory("/tmp/repo")
        called_tool, called_args = client.calls[-1]
        assert called_tool == "rag_index"
        assert called_args == {"path": "/tmp/repo"}

    async def test_query_param_detected_from_schema(self):
        client = FakeMCPClient(_nova_tools())
        preset = json.loads((PRESETS_DIR / "nova_rag.json").read_text())
        adapter = RAGAdapter(client, preset)
        await adapter.detect_tools()

        await adapter.query("what does Flask do?", top_k=7)

        called_tool, called_args = client.calls[-1]
        assert called_tool == "code_search"
        assert called_args.get("query") == "what does Flask do?"
        assert called_args.get("top_k") == 7
        assert "limit" not in called_args

    async def test_auto_detect_when_no_mapping(self):
        """When no preset config supplied, adapter detects tools from name + schema."""
        client = FakeMCPClient(_nova_tools())
        adapter = RAGAdapter(client, {})

        detected = await adapter.detect_tools()
        assert detected["ingest"] == "rag_index"
        assert detected["query"] == "code_search"

        await adapter.query("hello", top_k=3)
        _, args = client.calls[-1]
        assert args.get("top_k") == 3
        assert "limit" not in args

    async def test_invalid_param_in_preset_dropped(self):
        """If preset specifies a param not in the tool's inputSchema, it must be dropped."""
        client = FakeMCPClient(_nova_tools())
        bad_preset = {
            "tool_mapping": {
                "query": {
                    "tool": "code_search",
                    "params": {
                        "query": "{query}",
                        "limit": "{top_k}",
                        "top_k": "{top_k}",
                    },
                },
            },
        }
        adapter = RAGAdapter(client, bad_preset)
        await adapter.detect_tools()

        await adapter.query("hi", top_k=4)
        _, args = client.calls[-1]
        assert "limit" not in args
        assert args.get("top_k") == 4


class TestCommandResolution:
    def test_python_substituted_with_current_interpreter(self):
        parts = _resolve_command_parts("python -m nova_rag", [])
        assert parts[0] == sys.executable
        assert parts[1:] == ["-m", "nova_rag"]

    def test_python3_substituted(self):
        parts = _resolve_command_parts("python3", ["-m", "nova_rag"])
        assert parts[0] == sys.executable
        assert parts[1:] == ["-m", "nova_rag"]

    def test_other_command_left_intact(self):
        parts = _resolve_command_parts("uvx", ["chroma-mcp-server"])
        assert parts == ["uvx", "chroma-mcp-server"]
