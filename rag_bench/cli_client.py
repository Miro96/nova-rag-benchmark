"""CLI client — wraps a subprocess and parses structured output."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from rag_bench.base_client import (
    BaseClient,
    CallResult,
    ToolError,
    ToolInfo,
    _resolve_command_parts,
)

logger = logging.getLogger(__name__)

_SUPPORTED_TRANSPORTS = ("mcp", "cli", "inprocess")


@dataclass
class CLIClient(BaseClient):
    """Client that launches a CLI subprocess and parses structured (JSON) output.

    The subprocess must support two invocation patterns:

    * ``list-tools`` — write a JSON array of tool descriptors to stdout.
    * ``call-tool <name> <json-params>`` — write a JSON result object to stdout.

    If the subprocess exits non-zero, a ``ToolError`` is raised.  Stderr noise
    is tolerated and does not interfere with stdout parsing.
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    _call_count: int = field(default=0, init=False, repr=False)
    _tools: list[ToolInfo] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------

    async def _run(
        self,
        subcommand: str,
        stdin_data: str | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, str, str]:
        """Run the CLI once and return (returncode, stdout, stderr)."""
        cmd_parts = _resolve_command_parts(self.command, self.args)
        full_cmd = cmd_parts + subcommand.split()

        proc = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdin=asyncio.subprocess.PIPE if stdin_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(
                    stdin_data.encode() if stdin_data else None,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise ToolError(
                f"CLI subprocess timed out after {timeout}s: {full_cmd}"
            )

        return (
            proc.returncode or 0,
            stdout_bytes.decode(errors="replace"),
            stderr_bytes.decode(errors="replace"),
        )

    @staticmethod
    def _parse_json_output(raw_stdout: str, label: str) -> Any:
        """Extract and parse JSON from potentially noisy stdout.

        Handles leading/trailing whitespace and tolerates non-JSON prefix/suffix
        by scanning for the first ``{`` or ``[`` character.
        """
        stripped = raw_stdout.strip()
        if not stripped:
            raise ToolError(f"CLI produced empty stdout for {label}")

        # Fast path: whole output is valid JSON
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Scan for JSON start
        for start_char in ("{", "["):
            idx = stripped.find(start_char)
            if idx == -1:
                continue

            # Try from that offset onward
            for end in range(len(stripped), idx, -1):
                candidate = stripped[idx:end]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        raise ToolError(
            f"CLI {label} stdout is not valid JSON: {stripped[:200]}"
        )

    # ------------------------------------------------------------------
    # CLI protocol: list-tools
    # ------------------------------------------------------------------

    _LIST_TOOLS_SUBCOMMAND = "list-tools"

    async def list_tools(self) -> list[ToolInfo]:
        returncode, stdout, stderr = await self._run(
            self._LIST_TOOLS_SUBCOMMAND,
        )

        if returncode != 0:
            raise ToolError(
                f"CLI list-tools failed (exit {returncode}): {stderr.strip()[:200]}",
                exit_code=returncode,
            )

        tools_data = self._parse_json_output(stdout, "list-tools")
        if not isinstance(tools_data, list):
            # Accept {"tools": [...]} wrapper
            if isinstance(tools_data, dict) and "tools" in tools_data:
                tools_data = tools_data["tools"]
            else:
                raise ToolError(
                    f"CLI list-tools returned non-list: {type(tools_data).__name__}"
                )

        self._tools = [
            ToolInfo(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", t.get("input_schema", {})),
            )
            for t in tools_data
        ]
        return self._tools

    # ------------------------------------------------------------------
    # CLI protocol: call-tool <name> <params_json>
    # ------------------------------------------------------------------

    _CALL_TOOL_SUBCOMMAND = "call-tool"

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallResult:
        self._call_count += 1
        params_json = json.dumps(arguments)
        subcommand = f"{self._CALL_TOOL_SUBCOMMAND} {name} {_shell_quote(params_json)}"

        t0 = time.perf_counter()
        returncode, stdout, stderr = await self._run(subcommand)
        latency_ms = (time.perf_counter() - t0) * 1000

        if returncode != 0:
            raise ToolError(
                f"CLI call-tool '{name}' failed (exit {returncode}): {stderr.strip()[:200]}",
                exit_code=returncode,
            )

        result_data = self._parse_json_output(stdout, f"call-tool {name}")

        # Normalise to CallResult
        if isinstance(result_data, list):
            content = [
                {"type": "text", "text": json.dumps(item)}
                for item in result_data
            ]
            return CallResult(content=content, latency_ms=latency_ms)

        if isinstance(result_data, dict):
            # If the result already has "content", use it directly
            if "content" in result_data:
                return CallResult(
                    content=result_data["content"],
                    latency_ms=latency_ms,
                    is_error=result_data.get("isError", False),
                )
            # Wrap the dict as a text result
            return CallResult(
                content=[{"type": "text", "text": json.dumps(result_data)}],
                latency_ms=latency_ms,
            )

        # Fallback: wrap as text
        return CallResult(
            content=[{"type": "text", "text": str(result_data)}],
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        return self._call_count


def _shell_quote(s: str) -> str:
    """Minimal shell quoting — wraps in single quotes, escapes inner quotes."""
    escaped = s.replace("'", "'\\''")
    return f"'{escaped}'"
