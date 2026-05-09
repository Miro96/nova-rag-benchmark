"""MCP client for connecting to RAG servers via stdio or SSE."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_command_parts(command: str, args: list[str]) -> list[str]:
    """Split a command string and substitute the current interpreter for `python`.

    Presets routinely declare `python -m foo` so the user doesn't have to know
    the absolute interpreter path, but `subprocess` resolves bare `python` via
    PATH which can pick up the system interpreter rather than the venv that
    has the target package installed.
    """
    parts = command.split() + list(args or [])
    if parts and parts[0] in ("python", "python3"):
        parts[0] = sys.executable
    return parts


@dataclass
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class CallResult:
    content: list[dict[str, Any]]
    latency_ms: float
    is_error: bool = False

    @property
    def text(self) -> str:
        parts = []
        for item in self.content:
            if item.get("type") == "text":
                parts.append(item["text"])
        return "\n".join(parts)


@dataclass
class MCPClient:
    """JSON-RPC client for MCP servers over stdio."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    _process: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)
    _request_id: int = field(default=0, init=False, repr=False)
    _tools: list[ToolInfo] = field(default_factory=list, init=False, repr=False)
    _read_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    call_count: int = field(default=0, init=False, repr=False)
    server_info: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    async def start(self) -> None:
        """Start the MCP server subprocess and initialize."""
        cmd_parts = _resolve_command_parts(self.command, self.args)
        self._process = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )
        await self._initialize()

    async def stop(self) -> None:
        """Stop the MCP server subprocess."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()

    async def _initialize(self) -> None:
        """Send initialize request per MCP protocol."""
        result = await self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "rag-bench", "version": "0.1.0"},
        })
        # Store server info from the handshake for later use in output metadata
        self.server_info = result.get("serverInfo", {}) or {}
        # Send initialized notification
        await self._notify("notifications/initialized", {})
        logger.info("MCP server initialized: %s", self.server_info)

    async def list_tools(self) -> list[ToolInfo]:
        """Discover available tools from the server."""
        result = await self._request("tools/list", {})
        self._tools = [
            ToolInfo(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in result.get("tools", [])
        ]
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallResult:
        """Call an MCP tool and measure latency.

        ``call_count`` tracks attempted invocations (including failures) so the
        runner can compute per-query MCP tool-call counts via before/after
        deltas, instead of hardcoding 1.0.
        """
        self.call_count += 1
        t0 = time.perf_counter()
        result = await self._request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        latency_ms = (time.perf_counter() - t0) * 1000

        content = result.get("content", [])
        is_error = result.get("isError", False)
        return CallResult(content=content, latency_ms=latency_ms, is_error=is_error)

    def find_tools_by_pattern(self, *patterns: str) -> list[ToolInfo]:
        """Find tools whose name contains any of the given patterns."""
        matches = []
        for tool in self._tools:
            name_lower = tool.name.lower()
            desc_lower = tool.description.lower()
            for pattern in patterns:
                if pattern in name_lower or pattern in desc_lower:
                    matches.append(tool)
                    break
        return matches

    async def _request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and read response."""
        self._request_id += 1
        msg = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        await self._send(msg)
        return await self._read_response(self._request_id)

    async def _notify(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._send(msg)

    async def _send(self, msg: dict) -> None:
        assert self._process and self._process.stdin
        data = json.dumps(msg)
        self._process.stdin.write(data.encode() + b"\n")
        await self._process.stdin.drain()

    async def _read_response(self, expected_id: int, timeout: float = 30.0) -> dict:
        assert self._process and self._process.stdout
        async with self._read_lock:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"No response for request {expected_id}")
                try:
                    line = await asyncio.wait_for(
                        self._process.stdout.readline(),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"No response for request {expected_id}")

                if not line:
                    stderr_data = b""
                    if self._process.stderr:
                        try:
                            stderr_data = await asyncio.wait_for(
                                self._process.stderr.read(4096), timeout=1
                            )
                        except asyncio.TimeoutError:
                            pass
                    raise ConnectionError(
                        f"MCP server closed. stderr: {stderr_data.decode(errors='replace')}"
                    )

                line = line.strip()
                if not line:
                    continue

                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON line from server: %s", line[:200])
                    continue

                # Skip notifications
                if "id" not in response:
                    continue

                if response["id"] == expected_id:
                    if "error" in response:
                        err = response["error"]
                        raise RuntimeError(
                            f"MCP error {err.get('code')}: {err.get('message')}"
                        )
                    return response.get("result", {})

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()


def create_client(server_config: dict) -> MCPClient:
    """Create an MCPClient from a server config dict."""
    command = server_config.get("command", "")
    args = server_config.get("args", [])
    env = server_config.get("env")
    return MCPClient(command=command, args=args, env=env)
