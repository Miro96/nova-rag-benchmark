"""Abstract base for RAG tool clients — MCP, CLI, or in-process.

Also hosts the shared data types (``ToolInfo``, ``CallResult``) and the
command-resolution helper so that every transport backend can import them
without circular dependencies.
"""

from __future__ import annotations

import abc
import sys
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

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


class ToolError(RuntimeError):
    """Unified error for tool-call failures across all transport backends."""

    def __init__(self, message: str, exit_code: int | None = None):
        super().__init__(message)
        self.exit_code = exit_code


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


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseClient(abc.ABC):
    """Abstract client for discovering and calling RAG server tools.

    All transport backends (MCP stdio, CLI subprocess, in-process Python)
    must implement this interface so that ``RAGAdapter`` can remain
    transport-agnostic.
    """

    # Optional metadata about the backend (name, version, etc.).
    # Subclasses can set this to a dict of their choosing.
    server_info: dict[str, str] = field(default_factory=dict)

    @abc.abstractmethod
    async def list_tools(self) -> list[ToolInfo]:
        """Return the tools available on the backend."""

    @abc.abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallResult:
        """Call a named tool with keyword arguments and return the result."""

    @property
    @abc.abstractmethod
    def call_count(self) -> int:
        """Number of ``call_tool`` invocations attempted (including failures)."""
