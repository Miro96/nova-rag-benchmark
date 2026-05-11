"""Tests for transport layer: BaseClient, MCPClient, CLIClient, InProcessClient,
preset schema with transport field, and RAGAdapter transport-agnosticism.

Covers VAL-ADAPTER-001 through VAL-ADAPTER-008.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from rag_bench.adapter import RAGAdapter
from rag_bench.base_client import (
    BaseClient,
    CallResult,
    ToolError,
    ToolInfo,
    _resolve_command_parts,
)
from rag_bench.cli import cli
from rag_bench.cli_client import CLIClient
from rag_bench.inprocess_client import InProcessClient
from rag_bench.mcp_client import MCPClient, create_client

PRESETS_DIR = Path(__file__).parent.parent / "rag_bench" / "presets"


# ---------------------------------------------------------------------------
# Fake clients for testing
# ---------------------------------------------------------------------------

class FakeMCPClient:
    """Test double for MCPClient that returns canned tools/list and records calls."""

    def __init__(self, tools: list[dict] | None = None):
        self._tools_data = tools or _nova_tools()
        self.calls: list[tuple[str, dict]] = []
        self.call_count = 0
        self.server_info: dict[str, str] = {}

    async def list_tools(self) -> list[ToolInfo]:
        return [
            ToolInfo(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in self._tools_data
        ]

    async def call_tool(self, name: str, arguments: dict) -> CallResult:
        self.calls.append((name, dict(arguments)))
        self.call_count += 1
        return CallResult(content=[{"type": "text", "text": "{}"}], latency_ms=1.0)


def _nova_tools() -> list[dict]:
    return [
        {
            "name": "code_search",
            "inputSchema": {
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "rag_index",
            "inputSchema": {
                "properties": {"path": {"type": "string"}},
            },
        },
    ]


# ---------------------------------------------------------------------------
# VAL-ADAPTER-001: Existing MCP presets work unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """nova-rag preset must continue to work with MCP transport."""

    def test_nova_rag_preset_loads(self):
        preset = json.loads((PRESETS_DIR / "nova_rag.json").read_text())
        assert preset["name"] == "nova-rag"
        # Transport may be "stdio" (legacy) or "mcp"
        assert preset.get("transport", "mcp") in ("stdio", "mcp", "sse")

    def test_legacy_preset_creates_mcp_client(self):
        """Presets with no transport field default to MCP."""
        from rag_bench.runner import _create_client
        config = {"name": "test", "command": "echo"}
        client = _create_client(config)
        assert isinstance(client, MCPClient)

    def test_stdio_transport_creates_mcp_client(self):
        """Legacy 'stdio' transport maps to MCPClient."""
        from rag_bench.runner import _create_client
        config = {"name": "test", "command": "echo", "transport": "stdio"}
        client = _create_client(config)
        assert isinstance(client, MCPClient)

    def test_sse_transport_creates_mcp_client(self):
        """Legacy 'sse' transport maps to MCPClient."""
        from rag_bench.runner import _create_client
        config = {"name": "test", "command": "echo", "transport": "sse"}
        client = _create_client(config)
        assert isinstance(client, MCPClient)

    def test_mcp_transport_creates_mcp_client(self):
        """Explicit 'mcp' transport creates MCPClient."""
        from rag_bench.runner import _create_client
        config = {"name": "test", "command": "echo", "transport": "mcp"}
        client = _create_client(config)
        assert isinstance(client, MCPClient)

    @pytest.mark.asyncio
    async def test_adapter_works_with_mcp_client(self):
        """RAGAdapter works with MCPClient (FakeMCPClient simulates it)."""
        client = FakeMCPClient()
        adapter = RAGAdapter(client, {})
        detected = await adapter.detect_tools()
        assert detected["ingest"] == "rag_index"
        assert detected["query"] == "code_search"


# ---------------------------------------------------------------------------
# VAL-ADAPTER-002: Preset schema accepts "transport": "cli"
# ---------------------------------------------------------------------------

class TestCLITransportSchema:
    """Preset with transport: cli must be accepted."""

    def test_cli_transport_accepted_in_preset(self):
        """Schema validation accepts transport: cli."""
        from rag_bench.runner import _create_client

        # Simulate a CLI tool that returns valid tools on list-tools
        config = {
            "name": "test-cli",
            "command": "echo",
            "args": [],
            "transport": "cli",
        }
        client = _create_client(config)
        assert isinstance(client, CLIClient)

    def test_cli_transport_requires_command(self):
        """CLI transport without command raises clear error."""
        from rag_bench.runner import _create_client
        with pytest.raises(RuntimeError, match="command"):
            _create_client({"name": "test", "transport": "cli"})


# ---------------------------------------------------------------------------
# VAL-ADAPTER-003: Preset schema accepts "transport": "inprocess"
# ---------------------------------------------------------------------------

class TestInProcessTransportSchema:
    """Preset with transport: inprocess must be accepted."""

    def test_inprocess_transport_accepted_in_preset(self):
        """Schema validation accepts transport: inprocess."""
        from rag_bench.runner import _create_client
        config = {
            "name": "test-inproc",
            "transport": "inprocess",
            "module": "json",
            "class": "JSONDecoder",
        }
        client = _create_client(config)
        assert isinstance(client, InProcessClient)

    def test_inprocess_transport_requires_module_and_class(self):
        """InProcess transport without module raises clear error."""
        from rag_bench.runner import _create_client
        with pytest.raises(RuntimeError, match="module"):
            _create_client({"name": "test", "transport": "inprocess"})

        with pytest.raises(RuntimeError, match="module"):
            _create_client({
                "name": "test", "transport": "inprocess",
                "module": "", "class": "Foo",
            })


# ---------------------------------------------------------------------------
# VAL-ADAPTER-004: CLIClient correctly parses structured output
# ---------------------------------------------------------------------------

class TestCLIClientParsing:
    """CLIClient edge cases: whitespace, stderr noise, non-zero exit, empty stdout."""

    @pytest.mark.asyncio
    async def test_parse_with_leading_whitespace(self):
        """JSON with leading whitespace parses correctly."""
        raw = '  \n{"tools": [{"name": "search", "description": "find", "inputSchema": {}}]}\n'
        result = CLIClient._parse_json_output(raw, "test")
        assert isinstance(result, dict)
        assert "tools" in result

    @pytest.mark.asyncio
    async def test_parse_with_stderr_noise(self):
        """Stdout JSON parses regardless of what stderr contains."""
        # CLIClient._parse_json_output only looks at stdout, so stderr is irrelevant
        stdout = '{"results": ["a.py", "b.py"]}\n'
        result = CLIClient._parse_json_output(stdout, "test")
        assert isinstance(result, dict)
        assert "results" in result

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_tool_error(self):
        """Non-zero exit code produces ToolError with exit code and message."""
        client = CLIClient(command="false", args=[])
        with pytest.raises(ToolError) as exc_info:
            await client.list_tools()
        assert exc_info.value.exit_code is not None
        assert exc_info.value.exit_code != 0

    @pytest.mark.asyncio
    async def test_empty_stdout_raises_tool_error(self):
        """Empty stdout with exit 0 produces clear error, not JSONDecodeError."""
        client = CLIClient(command="true", args=[])
        with pytest.raises(ToolError, match="empty stdout"):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_nonzero_exit_raises_tool_error(self):
        """call_tool non-zero exit produces ToolError."""
        client = CLIClient(command="false", args=[])
        with pytest.raises(ToolError) as exc_info:
            await client.call_tool("search", {"query": "test"})
        assert exc_info.value.exit_code is not None
        assert exc_info.value.exit_code != 0

    @pytest.mark.asyncio
    async def test_list_tools_accepts_wrapped_dict(self):
        """CLI returning {"tools": [...]} is accepted."""
        # We'll mock _run to return a canned response
        client = CLIClient(command="echo", args=[])
        client._run = AsyncMock(return_value=(
            0,
            '{"tools": [{"name": "search", "description": "find files", "inputSchema": {"type": "object"}}]}',
            "",
        ))
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "search"

    @pytest.mark.asyncio
    async def test_list_tools_accepts_flat_array(self):
        """CLI returning a raw array of tools is accepted."""
        client = CLIClient(command="echo", args=[])
        client._run = AsyncMock(return_value=(
            0,
            '[{"name": "search", "description": "find", "inputSchema": {}}]',
            "",
        ))
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "search"

    @pytest.mark.asyncio
    async def test_call_tool_with_result_content_key(self):
        """call_tool result with 'content' key is passed through."""
        client = CLIClient(command="echo", args=[])
        client._run = AsyncMock(return_value=(
            0,
            '{"content": [{"type": "text", "text": "hello"}], "isError": false}',
            "",
        ))
        result = await client.call_tool("search", {"q": "x"})
        assert result.text == "hello"

    @pytest.mark.asyncio
    async def test_call_tool_with_list_result(self):
        """call_tool result as a list is wrapped in content items."""
        client = CLIClient(command="echo", args=[])
        client._run = AsyncMock(return_value=(
            0,
            '[{"file_path": "a.py"}, {"file_path": "b.py"}]',
            "",
        ))
        result = await client.call_tool("search", {"q": "x"})
        assert len(result.content) == 2  # two items in the list

    @pytest.mark.asyncio
    async def test_call_count_increments(self):
        """call_count increments on each call_tool invocation."""
        client = CLIClient(command="echo", args=[])
        client._run = AsyncMock(return_value=(
            0,
            '{"results": []}',
            "",
        ))
        assert client.call_count == 0
        await client.call_tool("search", {"q": "a"})
        assert client.call_count == 1
        await client.call_tool("search", {"q": "b"})
        assert client.call_count == 2


# ---------------------------------------------------------------------------
# VAL-ADAPTER-005: InProcessClient correctly routes calls
# ---------------------------------------------------------------------------

class MockCallableTool:
    """A well-behaved in-process tool for testing."""

    def list_tools(self) -> list[dict]:
        return [
            {"name": "search", "description": "search code", "inputSchema": {}},
            {"name": "index", "description": "index repo", "inputSchema": {}},
        ]

    def call_tool(self, name: str, **params) -> dict:
        if name == "search":
            return {"results": [{"file_path": "a.py", "score": 0.9}]}
        if name == "index":
            return {"status": "ok", "files": params.get("path", "")}
        raise ValueError(f"Unknown tool: {name}")


class FailingCallableTool:
    """A tool that always raises."""

    def list_tools(self) -> list[dict]:
        raise RuntimeError("cannot list tools")

    def call_tool(self, name: str, **params) -> dict:
        raise ValueError("bad input")


class TestInProcessClient:
    """InProcessClient correctly routes calls and wraps exceptions."""

    @pytest.mark.asyncio
    async def test_list_tools_propagates_result(self):
        """list_tools returns the callable's result as ToolInfo list."""
        client = InProcessClient(
            module_path="tests.test_transport",
            class_name="MockCallableTool",
        )
        tools = await client.list_tools()
        assert len(tools) == 2
        assert {t.name for t in tools} == {"search", "index"}

    @pytest.mark.asyncio
    async def test_call_tool_propagates_result(self):
        """call_tool returns the callable's result wrapped in CallResult."""
        client = InProcessClient(
            module_path="tests.test_transport",
            class_name="MockCallableTool",
        )
        result = await client.call_tool("search", {"query": "test"})
        assert isinstance(result, CallResult)
        assert "a.py" in result.text

    @pytest.mark.asyncio
    async def test_exception_wrapped_in_tool_error(self):
        """callable exceptions are wrapped in ToolError, not raw."""
        client = InProcessClient(
            module_path="tests.test_transport",
            class_name="FailingCallableTool",
        )
        with pytest.raises(ToolError, match="cannot list tools"):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_exception_wrapped(self):
        """call_tool exception is wrapped in ToolError."""
        client = InProcessClient(
            module_path="tests.test_transport",
            class_name="FailingCallableTool",
        )
        with pytest.raises(ToolError, match="bad input"):
            await client.call_tool("search", {"query": "x"})

    @pytest.mark.asyncio
    async def test_import_error_wrapped(self):
        """ImportError is wrapped in ToolError."""
        client = InProcessClient(
            module_path="nonexistent.module.xyz",
            class_name="Foo",
        )
        with pytest.raises(ToolError, match="Cannot import"):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_attribute_error_wrapped(self):
        """AttributeError (class not found) is wrapped in ToolError."""
        client = InProcessClient(
            module_path="json",
            class_name="NonExistentClass",
        )
        with pytest.raises(ToolError, match="has no class"):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_call_count_increments(self):
        """call_count increments on each call_tool."""
        client = InProcessClient(
            module_path="tests.test_transport",
            class_name="MockCallableTool",
        )
        assert client.call_count == 0
        await client.call_tool("search", {"q": "a"})
        assert client.call_count == 1
        await client.call_tool("search", {"q": "b"})
        assert client.call_count == 2

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool(self):
        """Unknown tool name raises ToolError from the callable."""
        client = InProcessClient(
            module_path="tests.test_transport",
            class_name="MockCallableTool",
        )
        with pytest.raises(ToolError, match="Unknown tool"):
            await client.call_tool("nonexistent", {})


# ---------------------------------------------------------------------------
# VAL-ADAPTER-006: RAGAdapter works with any BaseClient
# ---------------------------------------------------------------------------

class TestRAGAdapterTransportAgnostic:
    """RAGAdapter must produce identical behavior with any BaseClient implementation."""

    @pytest.mark.asyncio
    async def test_adapter_with_fake_client(self):
        """Adapter with a non-MCP BaseClient detects tools correctly."""
        client = FakeMCPClient()
        adapter = RAGAdapter(client, {})
        detected = await adapter.detect_tools()
        assert detected["ingest"] == "rag_index"
        assert detected["query"] == "code_search"

    @pytest.mark.asyncio
    async def test_adapter_with_fake_client_calls_correct_tools(self):
        """Adapter calls the right tool with the right params on a generic client."""
        client = FakeMCPClient()
        adapter = RAGAdapter(client, {})
        await adapter.detect_tools()

        await adapter.query("find main", top_k=5)
        called_tool, called_args = client.calls[-1]
        assert called_tool == "code_search"
        assert called_args.get("query") == "find main"
        assert called_args.get("top_k") == 5


# ---------------------------------------------------------------------------
# VAL-ADAPTER-008: Invalid transport type produces clear error
# ---------------------------------------------------------------------------

class TestInvalidTransportError:
    """Invalid transport type must produce a clear error with supported transports."""

    def test_unknown_transport_in_create_client(self):
        """_create_client with unknown transport raises clear RuntimeError."""
        from rag_bench.runner import _create_client
        with pytest.raises(RuntimeError) as exc_info:
            _create_client({"name": "test", "transport": "grpc"})
        msg = str(exc_info.value).lower()
        assert "grpc" in msg or "unknown" in msg
        assert "mcp" in msg
        assert "cli" in msg
        assert "inprocess" in msg

    def test_unknown_transport_in_cli(self):
        """CLI raises clear error for unknown transport in preset."""
        runner = CliRunner()
        # Create a temp preset file with bad transport
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump({"name": "bad", "command": "echo", "transport": "grpc"}, f)
            f.flush()
            result = runner.invoke(cli, ["run", "--config", f.name])
            Path(f.name).unlink(missing_ok=True)

        assert result.exit_code != 0
        combined = (result.output or "") + (
            str(result.exception) if result.exception else ""
        )
        assert "transport" in combined.lower()


# ---------------------------------------------------------------------------
# BaseClient abstractness
# ---------------------------------------------------------------------------

class TestBaseClientAbstract:
    """BaseClient cannot be instantiated directly and enforces interface."""

    def test_cannot_instantiate_directly(self):
        """BaseClient is abstract — must subclass."""
        with pytest.raises(TypeError):
            BaseClient()  # type: ignore[abstract]

    def test_subclass_must_implement_abstract_methods(self):
        """Subclass missing abstract methods raises TypeError."""
        with pytest.raises(TypeError):

            class IncompleteClient(BaseClient):  # type: ignore[abstract]
                pass

            IncompleteClient()

    def test_valid_subclass_instantiates(self):
        """Subclass with all abstract methods instantiates fine."""

        class CompleteClient(BaseClient):
            async def list_tools(self) -> list[ToolInfo]:
                return []

            async def call_tool(self, name: str, arguments: dict) -> CallResult:
                self._calls = getattr(self, '_calls', 0) + 1
                return CallResult(content=[], latency_ms=0)

            @property
            def call_count(self) -> int:
                return getattr(self, '_calls', 0)

        client = CompleteClient()
        assert isinstance(client, BaseClient)


# ---------------------------------------------------------------------------
# ToolError
# ---------------------------------------------------------------------------

class TestToolError:
    """ToolError carries exit_code and message."""

    def test_tool_error_with_exit_code(self):
        err = ToolError("command failed", exit_code=1)
        assert str(err) == "command failed"
        assert err.exit_code == 1

    def test_tool_error_without_exit_code(self):
        err = ToolError("something went wrong")
        assert str(err) == "something went wrong"
        assert err.exit_code is None
