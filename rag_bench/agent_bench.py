"""End-to-end agent benchmark: Claude Code with vs without a code-RAG MCP.

The retrieval benchmark (rag_bench.runner) measures the search engine in
isolation. This module measures what actually matters to users: does adding
a code-intelligence MCP server make a *real coding agent* answer codebase
questions more accurately, in fewer turns, with fewer tokens?

Methodology mirrors Cursor's Context Bench (cursor.com/blog/semsearch):
the same agent, the same questions, two tool configurations:

- ``baseline``  — Claude Code with built-in lexical tools only (Grep, Glob, Read)
- ``nova-rag``  — the same, plus the nova-rag MCP server

Grading is programmatic and transparent: an answer is correct when it cites
at least one expected file AND (if the query has expected symbols) at least
one expected symbol. An optional LLM-judge mode is available for rubric
grading. All raw answers are kept in the output JSON for auditing.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from rag_bench.datasets.loader import Query, clone_repo, load_queries, load_repos

logger = logging.getLogger(__name__)

#: Tools available to the baseline agent. Read-only lexical navigation —
#: what Claude Code uses out of the box for codebase questions.
BASELINE_TOOLS = ["Grep", "Glob", "Read"]

#: Prompt template. Deliberately neutral: it does not mention nova-rag or
#: hint at any tool, so neither condition gets prompt-level help.
PROMPT_TEMPLATE = (
    "You are answering a question about the codebase in the current "
    "directory. Question: {query}\n\n"
    "Answer concisely. You MUST cite the relevant file path(s) and the "
    "relevant function/class name(s) in your answer."
)

JUDGE_TEMPLATE = (
    "You are grading an answer to a codebase question.\n"
    "Question: {query}\n"
    "Ground truth files: {files}\n"
    "Ground truth symbols: {symbols}\n"
    "Answer to grade:\n---\n{answer}\n---\n"
    "Reply with exactly one word: CORRECT if the answer identifies the "
    "right file(s) and symbol(s), otherwise WRONG."
)


@dataclass
class AgentCondition:
    """One arm of the A/B experiment."""

    name: str
    allowed_tools: list[str]
    mcp_config: Path | None = None  # extra MCP servers for this arm


@dataclass
class AgentQueryResult:
    query_id: str
    repo: str
    query_type: str
    difficulty: str
    condition: str
    answer: str = ""
    file_hit: bool = False
    symbol_hit: bool = False
    correct: bool = False
    judge_correct: bool | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    num_turns: int = 0
    duration_ms: int = 0
    tool_calls: dict = field(default_factory=dict)
    mcp_calls: int = 0
    error: str | None = None


# ── Claude Code invocation ──


def _build_claude_cmd(
    prompt: str,
    condition: AgentCondition,
    model: str | None,
    max_turns: int,
) -> list[str]:
    """Assemble the headless Claude Code command for one query."""
    cmd = [
        "claude",
        "-p", prompt,
        # stream-json exposes every tool_use event — without it, MCP
        # adoption is unmeasurable (we proved this the hard way: 0%
        # answer-text fingerprints over 91 queries). --verbose is
        # required by the CLI for stream-json in print mode.
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", str(max_turns),
        # Hermetic runs: ignore the user's configured MCP servers and
        # settings/CLAUDE.md so only what the benchmark passes applies.
        # (NOT --bare: bare mode also skips OAuth keychain reads and
        # breaks authentication in headless runs — verified empirically.)
        "--strict-mcp-config",
        "--setting-sources", "",
    ]
    if model:
        cmd += ["--model", model]
    if condition.mcp_config is not None:
        cmd += ["--mcp-config", str(condition.mcp_config)]
    cmd += ["--allowedTools", ",".join(condition.allowed_tools)]
    # dontAsk: auto-approve exactly the allowlisted (read-only) tools,
    # deny everything else without prompting — reproducible and safe.
    cmd += ["--permission-mode", "dontAsk"]
    return cmd


def parse_stream_json(stdout: str) -> dict:
    """Parse stream-json output into the final result dict + tool counts.

    Returns the terminal ``result`` event augmented with ``_tool_calls``
    (tool name → invocation count). Falls back to plain-JSON parsing for
    older CLI output.
    """
    result: dict | None = None
    tool_calls: dict[str, int] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "assistant":
            content = (event.get("message") or {}).get("content") or []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    tool_calls[name] = tool_calls.get(name, 0) + 1
        elif event.get("type") == "result":
            result = event

    if result is None:
        # Old CLI / plain json fallback: the whole stdout is one object
        result = json.loads(stdout)
    result["_tool_calls"] = tool_calls
    return result


def _run_claude(
    prompt: str,
    condition: AgentCondition,
    cwd: Path,
    model: str | None,
    max_turns: int,
    timeout: int,
) -> dict:
    """Run one headless Claude Code query, return the parsed JSON result."""
    cmd = _build_claude_cmd(prompt, condition, model, max_turns)
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(
            f"claude exited {proc.returncode}: {proc.stderr.strip()[:500]}"
        )
    raw = parse_stream_json(proc.stdout)
    if raw.get("is_error"):
        # Auth failures etc. come back success-shaped with is_error=true —
        # grading that text would silently poison the results.
        raise RuntimeError(f"claude error result: {str(raw.get('result'))[:300]}")
    return raw


# ── Grading ──


def _normalize_path(p: str) -> str:
    return p.replace("\\", "/").strip("/").lower()


#: Basenames too generic to count as a citation on their own — they
#: appear in nearly every repo and answer.
_GENERIC_BASENAMES = {
    "index.js", "index.ts", "index.py", "utils.js", "utils.py", "app.py",
    "app.js", "main.py", "main.js", "base.py", "__init__.py", "mod.rs",
    "lib.rs", "setup.py", "config.py", "types.ts",
}


def grade_files(answer: str, expected_files: list[str]) -> bool:
    """True if the answer cites at least one expected file.

    Primary rule: the last two path components ("flask/app.py") appear
    contiguously — repo-relative and absolute citations both count, a
    bare generic basename does not.

    Secondary rule (split citations): answers often name the directory
    and the file apart ("in `Payments.Api.Services/`: `BillingService.cs`
    orchestrates…"). A distinctive basename plus its parent directory
    appearing anywhere in the answer counts too; generic basenames
    (index.js, utils.py…) stay excluded from this relaxation.
    """
    if not expected_files:
        return True
    haystack = _normalize_path(answer)
    for f in expected_files:
        parts = _normalize_path(f).split("/")
        needle = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        if needle in haystack:
            return True
        if (len(parts) >= 2
                and parts[-1] not in _GENERIC_BASENAMES
                and parts[-1] in haystack
                and parts[-2] in haystack):
            return True
    return False


def grade_symbols(answer: str, expected_symbols: list[str]) -> bool:
    """True if the answer mentions at least one expected symbol (word-bounded)."""
    if not expected_symbols:
        return True
    for sym in expected_symbols:
        if re.search(rf"\b{re.escape(sym)}\b", answer):
            return True
    return False


def grade(answer: str, query: Query) -> tuple[bool, bool, bool]:
    """Returns (file_hit, symbol_hit, correct)."""
    file_hit = grade_files(answer, query.expected_files)
    symbol_hit = grade_symbols(answer, query.expected_symbols)
    return file_hit, symbol_hit, file_hit and symbol_hit


# ── Aggregation ──


def aggregate(results: list[AgentQueryResult]) -> dict:
    """Aggregate per-condition metrics from individual query results."""
    out: dict = {}
    conditions = sorted({r.condition for r in results})
    for cond in conditions:
        rows = [r for r in results if r.condition == cond and r.error is None]
        errors = [r for r in results if r.condition == cond and r.error]
        n = len(rows)
        if n == 0:
            out[cond] = {"queries": 0, "errors": len(errors)}
            continue
        toks = [r.input_tokens + r.output_tokens for r in rows]
        out[cond] = {
            "queries": n,
            "errors": len(errors),
            "accuracy": round(sum(r.correct for r in rows) / n, 4),
            "file_accuracy": round(sum(r.file_hit for r in rows) / n, 4),
            "symbol_accuracy": round(sum(r.symbol_hit for r in rows) / n, 4),
            "judge_accuracy": _judge_accuracy(rows),
            "tokens_mean": round(statistics.mean(toks), 1),
            "tokens_median": statistics.median(toks),
            "cost_usd_total": round(sum(r.cost_usd for r in rows), 4),
            "turns_mean": round(statistics.mean(r.num_turns for r in rows), 2),
            "duration_ms_median": statistics.median(r.duration_ms for r in rows),
            # Share of queries where at least one MCP tool was invoked —
            # the adoption metric; nothing downstream matters if the
            # agent never picks the tool.
            "mcp_adoption": round(sum(r.mcp_calls > 0 for r in rows) / n, 4),
            "tool_calls_total": dict(sum(
                (Counter(r.tool_calls) for r in rows), Counter()
            )),
            "by_type": _by_type(rows),
        }
    if len(conditions) == 2:
        out["delta"] = _deltas(out[conditions[0]], out[conditions[1]],
                               conditions[0], conditions[1])
    return out


def _judge_accuracy(rows: list[AgentQueryResult]) -> float | None:
    judged = [r for r in rows if r.judge_correct is not None]
    if not judged:
        return None
    return round(sum(r.judge_correct for r in judged) / len(judged), 4)


def _by_type(rows: list[AgentQueryResult]) -> dict:
    types: dict[str, dict] = {}
    for t in sorted({r.query_type for r in rows}):
        sub = [r for r in rows if r.query_type == t]
        types[t] = {
            "queries": len(sub),
            "accuracy": round(sum(r.correct for r in sub) / len(sub), 4),
        }
    return types


def _deltas(a: dict, b: dict, name_a: str, name_b: str) -> dict:
    """B-minus-A deltas (positive = second condition better)."""
    d: dict = {"baseline": name_a, "treatment": name_b}
    if a.get("queries") and b.get("queries"):
        d["accuracy_pp"] = round((b["accuracy"] - a["accuracy"]) * 100, 1)
        d["file_accuracy_pp"] = round((b["file_accuracy"] - a["file_accuracy"]) * 100, 1)
        d["symbol_accuracy_pp"] = round((b["symbol_accuracy"] - a["symbol_accuracy"]) * 100, 1)
        if a["tokens_mean"]:
            d["tokens_change_pct"] = round(
                (b["tokens_mean"] - a["tokens_mean"]) / a["tokens_mean"] * 100, 1
            )
        d["turns_change"] = round(b["turns_mean"] - a["turns_mean"], 2)
    return d


# ── Orchestration ──


def _make_nova_rag_mcp_config(work_dir: Path, server_command: list[str]) -> Path:
    """Write a single-server MCP config pointing nova-rag at a temp data dir."""
    data_dir = work_dir / "nova-rag-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "mcpServers": {
            "nova-rag": {
                "type": "stdio",
                "command": server_command[0],
                "args": server_command[1:],
                "env": {"NOVA_RAG_DATA_DIR": str(data_dir)},
                # Without this, Claude Code defers MCP tool schemas behind
                # ToolSearch and the model never sees the tool descriptions
                # at decision time — measured at 9-18% adoption. This is
                # also the documented install recommendation for users.
                "alwaysLoad": True,
            }
        }
    }
    path = work_dir / "mcp-nova-rag.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def run_agent_benchmark(
    repo_filter: str | None = None,
    model: str | None = None,
    max_turns: int = 15,
    limit: int | None = None,
    timeout: int = 300,
    judge: bool = False,
    nova_rag_command: list[str] | None = None,
    conditions: list[str] | None = None,
    runner: Callable[..., dict] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Run the A/B agent benchmark. Returns the full result document.

    ``runner`` is injectable for tests (same signature as ``_run_claude``).
    """
    runner = runner or _run_claude
    nova_rag_command = nova_rag_command or ["nova-rag"]
    wanted = conditions or ["baseline", "nova-rag"]
    progress = on_progress or (lambda msg: logger.info(msg))

    work_dir = Path(tempfile.mkdtemp(prefix="rag-bench-agent-"))
    conds: list[AgentCondition] = []
    if "baseline" in wanted:
        conds.append(AgentCondition("baseline", list(BASELINE_TOOLS)))
    if "nova-rag" in wanted:
        mcp_config = _make_nova_rag_mcp_config(work_dir, nova_rag_command)
        conds.append(AgentCondition(
            "nova-rag",
            list(BASELINE_TOOLS) + ["mcp__nova-rag"],
            mcp_config=mcp_config,
        ))

    repos = [r for r in load_repos() if repo_filter in (None, r.name)]
    results: list[AgentQueryResult] = []

    for repo in repos:
        repo_dir = clone_repo(repo)
        queries = load_queries(repo.name)
        if limit:
            queries = queries[:limit]

        # Warmup per condition: triggers nova-rag auto-indexing so index
        # build time is excluded from per-query metrics (an index, like
        # Cursor's, exists before the user asks questions). The baseline
        # warmup keeps both arms symmetric.
        for cond in conds:
            progress(f"[{repo.name}/{cond.name}] warmup (indexing if needed)...")
            try:
                runner("What does this project do? One sentence.",
                       cond, repo_dir, model, max_turns, timeout * 2)
            except Exception as exc:  # noqa: BLE001
                progress(f"[{repo.name}/{cond.name}] warmup failed: {exc}")

        for i, q in enumerate(queries, 1):
            # Interleave conditions per query so drift (cache state, API
            # weather) is balanced across arms.
            for cond in conds:
                r = AgentQueryResult(
                    query_id=q.id, repo=repo.name, query_type=q.type,
                    difficulty=q.difficulty, condition=cond.name,
                )
                t0 = time.time()
                try:
                    raw = runner(
                        PROMPT_TEMPLATE.format(query=q.query),
                        cond, repo_dir, model, max_turns, timeout,
                    )
                    r.answer = raw.get("result") or ""
                    usage = raw.get("usage") or {}
                    r.input_tokens = int(usage.get("input_tokens", 0))
                    r.output_tokens = int(usage.get("output_tokens", 0))
                    # Cost moved under "cost" in newer CLI versions;
                    # older ones report it top-level.
                    cost = raw.get("cost") or {}
                    r.cost_usd = float(
                        cost.get("total_cost_usd", raw.get("total_cost_usd", 0.0))
                    )
                    r.num_turns = int(raw.get("num_turns", 0))
                    if "duration_ms" in raw:
                        r.duration_ms = int(raw["duration_ms"])
                    elif "duration_seconds" in raw:
                        r.duration_ms = int(float(raw["duration_seconds"]) * 1000)
                    else:
                        r.duration_ms = int((time.time() - t0) * 1000)
                    r.tool_calls = raw.get("_tool_calls", {})
                    r.mcp_calls = sum(
                        n for name, n in r.tool_calls.items()
                        if name.startswith("mcp__")
                    )
                    r.file_hit, r.symbol_hit, r.correct = grade(r.answer, q)
                    if judge and r.answer:
                        r.judge_correct = _judge_one(q, r.answer, model, runner_cwd=repo_dir)
                except Exception as exc:  # noqa: BLE001
                    r.error = str(exc)[:500]
                results.append(r)
            done = sum(1 for x in results if x.repo == repo.name) // len(conds)
            progress(f"[{repo.name}] {done}/{len(queries)} queries done")

    doc = {
        "benchmark": "agent-ab",
        "run_id": str(uuid.uuid4()),
        "model": model or "(claude code default)",
        "max_turns": max_turns,
        "conditions": [c.name for c in conds],
        "repos": [r.name for r in repos],
        "summary": aggregate(results),
        "queries": [asdict(r) for r in results],
    }
    shutil.rmtree(work_dir, ignore_errors=True)
    return doc


def _judge_one(q: Query, answer: str, model: str | None, runner_cwd: Path) -> bool | None:
    """LLM-judge grading via a tool-less headless call."""
    prompt = JUDGE_TEMPLATE.format(
        query=q.query,
        files=", ".join(q.expected_files),
        symbols=", ".join(q.expected_symbols) or "(none)",
        answer=answer[:4000],
    )
    try:
        cmd = ["claude", "-p", prompt, "--output-format", "json",
               "--max-turns", "1", "--strict-mcp-config",
               "--setting-sources", "", "--allowedTools", ""]
        if model:
            cmd += ["--model", model]
        proc = subprocess.run(cmd, cwd=str(runner_cwd), capture_output=True,
                              text=True, timeout=120)
        verdict = (json.loads(proc.stdout).get("result") or "").strip().upper()
        return verdict.startswith("CORRECT")
    except Exception:  # noqa: BLE001
        return None


# ── Reporting ──


def render_markdown(doc: dict) -> str:
    """Render the A/B summary as a publishable markdown table."""
    s = doc["summary"]
    conds = [c for c in doc["conditions"] if c in s]
    lines = [
        f"### Agent benchmark: Claude Code {' vs '.join(conds)}",
        "",
        f"Model: `{doc['model']}` · max {doc['max_turns']} turns · "
        f"repos: {', '.join(doc['repos'])} · grading: file+symbol citation "
        "vs ground truth",
        "",
        "| Metric | " + " | ".join(conds) + " |",
        "|---|" + "---|" * len(conds),
    ]

    def row(label: str, key: str, fmt: str = "{}") -> str:
        vals = []
        for c in conds:
            v = s[c].get(key)
            vals.append(fmt.format(v) if v is not None else "—")
        return f"| {label} | " + " | ".join(vals) + " |"

    lines.append(row("Accuracy (file+symbol)", "accuracy", "{:.1%}"))
    lines.append(row("File accuracy", "file_accuracy", "{:.1%}"))
    lines.append(row("Symbol accuracy", "symbol_accuracy", "{:.1%}"))
    lines.append(row("Tokens / query (mean)", "tokens_mean", "{:,.0f}"))
    lines.append(row("Agent turns (mean)", "turns_mean", "{}"))
    lines.append(row("Latency p50 (ms)", "duration_ms_median", "{:,.0f}"))
    lines.append(row("MCP adoption", "mcp_adoption", "{:.0%}"))
    lines.append(row("Total cost (USD)", "cost_usd_total", "${}"))

    delta = s.get("delta")
    if delta and "accuracy_pp" in delta:
        lines += [
            "",
            f"**Δ {delta['treatment']} vs {delta['baseline']}:** "
            f"accuracy {delta['accuracy_pp']:+.1f} pp · "
            f"tokens {delta.get('tokens_change_pct', 0):+.1f}% · "
            f"turns {delta.get('turns_change', 0):+.2f}",
        ]
    return "\n".join(lines)
