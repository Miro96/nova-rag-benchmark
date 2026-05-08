"""Baseline agent: uses grep/glob/read to find code (no RAG).

Provides two baseline strategies:

1. **Local grep/glob** — ``run_baseline_search``: keyword extraction + shell
   grep. Fast, no API dependency, but less intelligent search.

2. **DeepSeek-powered agent** — ``DeepSeekBaselineAgent``: LLM-driven search
   using function-calling (grep / glob / read_file). Slower but smarter at
   figuring out *which* patterns to search for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MAX_LISTING_FILES = 300  # cap file listings sent to the LLM to control cost


def _list_files(repo_dir: Path, max_files: int = _MAX_LISTING_FILES) -> list[str]:
    """Collect relative file paths from *repo_dir* (source files only).

    Returns at most *max_files* entries to keep LLM context manageable.
    """
    from rag_bench.datasets.loader import get_repo_files

    files = get_repo_files(repo_dir)
    rel = []
    for f in files[:max_files]:
        try:
            rel.append(str(f.relative_to(repo_dir)))
        except ValueError:
            rel.append(str(f))
    return rel


# ---------------------------------------------------------------------------
# 1. Local grep/glob baseline
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    """Result from baseline grep/glob search."""
    query_id: str
    found_files: list[str]
    found_symbols: list[str]
    tool_calls: int
    total_time_ms: float


def run_baseline_search(
    query: str,
    query_type: str,
    repo_dir: Path,
    expected_symbols: list[str],
) -> BaselineResult:
    """Simulate a grep/glob search strategy for a code query.

    This emulates what an LLM agent would do without RAG:
    extract keywords, grep for them, find files.
    """
    t0 = time.perf_counter()
    tool_calls = 0
    found_files: list[str] = []
    found_symbols: list[str] = []

    # Strategy 1: grep for expected symbol names (most direct approach)
    keywords = _extract_keywords(query, expected_symbols)

    for keyword in keywords[:5]:  # limit to avoid too many calls
        tool_calls += 1
        matches = _grep(repo_dir, keyword)
        for file_path in matches:
            rel = str(file_path.relative_to(repo_dir))
            if rel not in found_files:
                found_files.append(rel)

    # Strategy 2: if nothing found, try broader keyword search
    if not found_files:
        broad_keywords = _extract_broad_keywords(query)
        for keyword in broad_keywords[:3]:
            tool_calls += 1
            matches = _grep(repo_dir, keyword)
            for file_path in matches:
                rel = str(file_path.relative_to(repo_dir))
                if rel not in found_files:
                    found_files.append(rel)

    # Strategy 3: glob for file patterns
    if not found_files:
        patterns = _extract_file_patterns(query)
        for pattern in patterns[:3]:
            tool_calls += 1
            for f in repo_dir.rglob(pattern):
                if f.is_file():
                    rel = str(f.relative_to(repo_dir))
                    if rel not in found_files:
                        found_files.append(rel)

    # Check which symbols were found
    for sym in expected_symbols:
        for f in found_files[:10]:
            full_path = repo_dir / f
            if full_path.exists():
                tool_calls += 1
                content = full_path.read_text(errors="replace")
                if sym in content:
                    found_symbols.append(sym)
                    break

    total_ms = (time.perf_counter() - t0) * 1000

    return BaselineResult(
        query_id="",
        found_files=found_files[:20],
        found_symbols=found_symbols,
        tool_calls=tool_calls,
        total_time_ms=total_ms,
    )


def _extract_keywords(query: str, expected_symbols: list[str]) -> list[str]:
    """Extract search keywords from query and expected symbols."""
    keywords = list(expected_symbols)

    # Also extract likely identifiers from the query
    words = query.lower().split()
    code_words = [
        w for w in words
        if w not in {
            "where", "is", "the", "how", "does", "what", "are", "a", "an",
            "in", "of", "for", "to", "and", "or", "from", "by", "with",
            "would", "break", "if", "i", "change", "modify", "that",
            "calls", "uses", "defined", "handled", "implemented", "work",
        }
    ]
    keywords.extend(code_words)
    return keywords


def _extract_broad_keywords(query: str) -> list[str]:
    """Extract broader keywords when specific search fails."""
    patterns = {
        "route": ["route", "router", "url", "path"],
        "auth": ["auth", "login", "session", "token"],
        "middleware": ["middleware", "use(", "before_request"],
        "template": ["template", "render", "view"],
        "error": ["error", "exception", "handler"],
        "config": ["config", "settings", "env"],
        "test": ["test", "client", "mock"],
        "static": ["static", "file", "send"],
        "json": ["json", "serialize", "encode"],
        "database": ["database", "db", "query", "model"],
        "dependency": ["dependency", "inject", "depends"],
        "validation": ["valid", "schema", "pydantic"],
        "websocket": ["websocket", "ws", "socket"],
        "security": ["security", "oauth", "bearer"],
        "openapi": ["openapi", "swagger", "schema"],
    }

    query_lower = query.lower()
    keywords = []
    for trigger, kws in patterns.items():
        if trigger in query_lower:
            keywords.extend(kws)

    return keywords or query.lower().split()[:3]


def _extract_file_patterns(query: str) -> list[str]:
    """Extract likely file glob patterns from query."""
    patterns = []
    query_lower = query.lower()

    if "test" in query_lower:
        patterns.extend(["*test*.py", "*test*.js", "*spec*.js"])
    if "config" in query_lower:
        patterns.extend(["*config*", "*settings*"])
    if "route" in query_lower or "router" in query_lower:
        patterns.extend(["*rout*"])
    if "middleware" in query_lower:
        patterns.extend(["*middleware*"])

    return patterns or ["*.py", "*.js"]


def _grep(repo_dir: Path, pattern: str, max_results: int = 20) -> list[Path]:
    """Run grep and return matching file paths."""
    try:
        result = subprocess.run(
            ["grep", "-rl", "--include=*.py", "--include=*.js", "--include=*.ts",
             "-m", "1", pattern, str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        files = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                p = Path(line.strip())
                if p.exists():
                    files.append(p)
                if len(files) >= max_results:
                    break
        return files
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


# ---------------------------------------------------------------------------
# 2. DeepSeek-powered agent (function-calling over chat completions)
# ---------------------------------------------------------------------------

# Tool definitions sent to DeepSeek as function-calling tools.
_BASELINE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search for a regex pattern in text files inside the repo "
                "directory. Returns matching file paths (relative to repo "
                "root). Use this to find files that contain a specific "
                "function name, class, string literal, or code pattern."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex or literal string to search for.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": (
                "Find file paths matching a glob pattern (e.g. "
                "'**/handlers/*.py'). Returns a list of matching relative "
                "paths. Use this to find files by naming convention."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match file paths.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full contents of a file identified by its "
                "repo-relative path. Use this after grep/glob to inspect "
                "candidate files and determine which one actually contains "
                "the code the user asked about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative file path to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
]


_SYSTEM_PROMPT = """You are a code search agent. You search a code repository using
grep, glob, and file-reading tools — you do NOT have RAG or semantic search.

Your goal: find the file(s) that best answer the user's question.

Workflow:
1. Analyze the query — what function, class, or concept is the user asking about?
2. Pick the best keyword(s) or file patterns and call grep or glob.
3. If grep returns many results, use read_file on the most promising candidates.
4. Once you've identified the best matching files, output ONLY their
   repo-relative paths, one per line, prefixed with "FILES:".

Example final answer:
FILES:
src/auth/login.py
src/auth/session.py

If you found specific symbols (function/class names), also list them with
"SYMBOLS:":
SYMBOLS:
login_user
create_session

Be efficient — prefer grep with specific patterns over glob for code searches.
Do NOT read every file. Do NOT explain your reasoning in the final message.
Just list FILES: and optionally SYMBOLS:.

Available files in the repo (you can grep, glob, or read any of these):
{file_listing}"""


class DeepSeekBaselineAgent:
    """LLM-powered agent that searches a repo using grep/glob/read_file tools.

    Communicates with the DeepSeek API via OpenAI-compatible chat completions
    with function calling. The agent iterates: it sends the current
    conversation to DeepSeek, executes any requested tool calls, and feeds
    the results back — repeating until the model emits a final text response
    (no more tool calls) or *max_iterations* is reached.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com/v1",
        max_iterations: int = 5,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_iterations = max_iterations
        self.timeout = timeout
        self._http_client: httpx.AsyncClient | None = None

    async def search(
        self,
        query: str,
        repo_dir: Path,
    ) -> dict[str, Any]:
        """Run the agent for a single query.

        Returns a dict with keys:
          * ``found_files`` — list of repo-relative file paths.
          * ``found_symbols`` — list of symbol names extracted from response.
          * ``tool_calls`` — number of tool invocations.
          * ``total_time_ms`` — wall-clock duration in milliseconds.
        """
        t0 = time.perf_counter()
        tool_calls = 0
        found_files: list[str] = []
        found_symbols: list[str] = []

        # Build file listing
        file_paths = _list_files(repo_dir)
        listing = "\n".join(file_paths)
        if len(file_paths) >= _MAX_LISTING_FILES:
            listing += f"\n... ({_MAX_LISTING_FILES} files shown; repo contains more)"

        system_content = _SYSTEM_PROMPT.format(file_listing=listing)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)

        for iteration in range(self.max_iterations):
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": _BASELINE_TOOLS,
                "temperature": 0.1,
                "max_tokens": 1024,
            }

            response = await self._http_client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                content=json.dumps(payload),
            )

            if response.status_code != 200:
                body = ""
                try:
                    body = response.text[:500]
                except Exception:
                    pass
                raise RuntimeError(
                    f"DeepSeek API error {response.status_code}: {body}"
                )

            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]

            # If the model produced text content (no tool calls), parse it
            # and finish.
            if message.get("content") and not message.get("tool_calls"):
                text = message["content"]
                found_files, found_symbols = _parse_agent_response(text, repo_dir, file_paths)
                break

            # Handle tool calls
            tool_calls_list = message.get("tool_calls") or []
            if not tool_calls_list:
                # No tool calls and no content — treat as done
                break

            # Record the assistant message with tool calls
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in tool_calls_list
                ],
            }
            # DeepSeek thinking mode: reasoning_content must be passed back
            if message.get("reasoning_content"):
                assistant_msg["reasoning_content"] = message["reasoning_content"]
            messages.append(assistant_msg)

            for tc in tool_calls_list:
                fn = tc["function"]
                tool_name = fn["name"]
                try:
                    tool_args = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}

                tool_calls += 1
                result_str = _execute_tool(tool_name, tool_args, repo_dir)

                # Append tool result message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        total_ms = (time.perf_counter() - t0) * 1000

        return {
            "found_files": found_files[:20],
            "found_symbols": found_symbols,
            "tool_calls": tool_calls,
            "total_time_ms": total_ms,
        }


def _execute_tool(
    name: str,
    args: dict[str, Any],
    repo_dir: Path,
) -> str:
    """Execute a single tool call and return the result as a string."""
    if name == "grep":
        pattern = args.get("pattern", "")
        if not pattern:
            return "Error: 'pattern' argument is required for grep."
        try:
            result = subprocess.run(
                [
                    "grep", "-rln", "--include=*.py", "--include=*.js",
                    "--include=*.ts", "--include=*.go", "--include=*.rs",
                    "--include=*.java", "--include=*.rb",
                    "-m", "1", "-e", pattern, str(repo_dir),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            # Convert absolute paths to repo-relative
            rel_lines = []
            for line in lines[:20]:
                try:
                    rel_lines.append(str(Path(line).relative_to(repo_dir)))
                except ValueError:
                    rel_lines.append(line)
            if not rel_lines:
                return f"No matches found for pattern: {pattern}"
            return "Matching files:\n" + "\n".join(rel_lines)
        except subprocess.TimeoutExpired:
            return f"Grep timed out for pattern: {pattern}"
        except FileNotFoundError:
            return "Error: grep command not available."

    elif name == "glob":
        pattern = args.get("pattern", "*")
        try:
            matches = sorted(repo_dir.glob(pattern))
            files = []
            for m in matches:
                if m.is_file():
                    try:
                        files.append(str(m.relative_to(repo_dir)))
                    except ValueError:
                        files.append(str(m))
            if not files:
                return f"No files matched glob pattern: {pattern}"
            return "Matching files:\n" + "\n".join(files[:30])
        except Exception as e:
            return f"Glob error: {e}"

    elif name == "read_file":
        path_str = args.get("path", "")
        if not path_str:
            return "Error: 'path' argument is required for read_file."

        # Security: prevent path traversal
        full_path = (repo_dir / path_str).resolve()
        try:
            full_path.relative_to(repo_dir.resolve())
        except ValueError:
            return f"Error: path '{path_str}' is outside the repo directory."

        if not full_path.is_file():
            return f"Error: file '{path_str}' does not exist or is not a regular file."

        try:
            content = full_path.read_text(errors="replace")
            # Truncate to reasonable size
            if len(content) > 8000:
                content = content[:8000] + "\n... [truncated]"
            return f"Content of {path_str}:\n```\n{content}\n```"
        except Exception as e:
            return f"Error reading {path_str}: {e}"

    else:
        return f"Unknown tool: {name}"


def _parse_agent_response(
    text: str,
    repo_dir: Path,
    file_paths: list[str],
) -> tuple[list[str], list[str]]:
    """Parse the agent's final text response for FILES: and SYMBOLS: sections."""
    found_files: list[str] = []
    found_symbols: list[str] = []

    lines = text.split("\n")
    section: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("FILES"):
            section = "files"
            # Check for inline path on same line
            rest = stripped[len("FILES"):].lstrip(": ").strip()
            if rest:
                found_files.append(rest)
            continue
        if stripped.upper().startswith("SYMBOLS"):
            section = "symbols"
            rest = stripped[len("SYMBOLS"):].lstrip(": ").strip()
            if rest:
                found_symbols.append(rest)
            continue

        if section == "files" and stripped and not stripped.startswith("#"):
            # Clean up any markdown formatting
            cleaned = stripped.lstrip("-* ").strip("` ")
            if cleaned:
                found_files.append(cleaned)
        elif section == "symbols" and stripped and not stripped.startswith("#"):
            cleaned = stripped.lstrip("-* ").strip("` ")
            if cleaned:
                found_symbols.append(cleaned)

    # Validate files exist in the repo
    validated_files = []
    for f in found_files:
        candidate = repo_dir / f
        if candidate.is_file():
            validated_files.append(f)
        else:
            # Try fuzzy matching against file listing
            for known in file_paths:
                if known.endswith(f) or f.endswith(known.split("/")[-1]):
                    validated_files.append(known)
                    break

    return validated_files, found_symbols
