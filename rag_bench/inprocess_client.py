"""In-process client — routes calls to a Python callable."""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from rag_bench.base_client import BaseClient, CallResult, ToolError, ToolInfo

logger = logging.getLogger(__name__)


@dataclass
class InProcessClient(BaseClient):
    """Client that calls a Python object directly — no subprocess, no network.

    The module/class referenced by the preset must expose two methods:

    * ``list_tools() -> list[dict]`` — returns tool descriptors as dicts.
    * ``call_tool(name: str, **params) -> dict | list`` — executes a tool.

    Exceptions raised by the callable are wrapped in ``ToolError`` so the
    benchmark runner can handle them uniformly.
    """

    module_path: str  # dotted Python import path
    class_name: str   # class name within that module
    _instance: Any = field(default=None, init=False, repr=False)
    _call_count: int = field(default=0, init=False, repr=False)

    async def _get_instance(self) -> Any:
        if self._instance is None:
            try:
                mod = importlib.import_module(self.module_path)
                cls = getattr(mod, self.class_name)
                self._instance = cls()
            except ImportError as e:
                raise ToolError(
                    f"Cannot import module '{self.module_path}': {e}"
                ) from e
            except AttributeError as e:
                raise ToolError(
                    f"Module '{self.module_path}' has no class '{self.class_name}': {e}"
                ) from e
        return self._instance

    async def list_tools(self) -> list[ToolInfo]:
        instance = await self._get_instance()
        try:
            raw_tools = instance.list_tools()
        except Exception as e:
            raise ToolError(f"InProcess list_tools() failed: {e}") from e

        # Sync callables are OK since list_tools is typically cheap
        tools: list[ToolInfo] = []
        for t in raw_tools:
            tools.append(ToolInfo(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", t.get("input_schema", {})),
            ))
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallResult:
        self._call_count += 1
        instance = await self._get_instance()

        t0 = time.perf_counter()
        try:
            result = instance.call_tool(name, **arguments)
        except Exception as e:
            raise ToolError(
                f"InProcess call_tool('{name}') failed: {e}"
            ) from e
        latency_ms = (time.perf_counter() - t0) * 1000

        # Normalise result to CallResult
        if isinstance(result, list):
            import json
            content = [
                {"type": "text", "text": json.dumps(item)}
                for item in result
            ]
            return CallResult(content=content, latency_ms=latency_ms)

        if isinstance(result, dict):
            import json
            if "content" in result:
                return CallResult(
                    content=result["content"],
                    latency_ms=latency_ms,
                    is_error=result.get("isError", False),
                )
            return CallResult(
                content=[{"type": "text", "text": json.dumps(result)}],
                latency_ms=latency_ms,
            )

        return CallResult(
            content=[{"type": "text", "text": str(result)}],
            latency_ms=latency_ms,
        )

    @property
    def call_count(self) -> int:
        return self._call_count
