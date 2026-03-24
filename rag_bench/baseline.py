"""Baseline agent: uses grep/glob/read to find code (no RAG)."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


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
