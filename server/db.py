"""SQLite database for leaderboard storage."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).parent / "leaderboard.db"


class DuplicateRunError(Exception):
    """Raised when a run_id already exists in the database."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"Run '{run_id}' already exists")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    server_name TEXT NOT NULL,
    git_url TEXT DEFAULT '',
    git_user TEXT DEFAULT '',
    server_version TEXT DEFAULT '',
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- ingest metrics
    ingest_total_files INTEGER DEFAULT 0,
    ingest_total_sec REAL DEFAULT 0,
    ingest_files_per_sec REAL DEFAULT 0,
    index_size_mb REAL DEFAULT 0,
    ram_peak_mb REAL DEFAULT 0,

    -- retrieval quality
    hit_at_1 REAL DEFAULT 0,
    hit_at_3 REAL DEFAULT 0,
    hit_at_5 REAL DEFAULT 0,
    hit_at_10 REAL DEFAULT 0,
    symbol_hit_at_5 REAL DEFAULT 0,
    mrr REAL DEFAULT 0,

    -- latency
    query_latency_p50_ms REAL DEFAULT 0,
    query_latency_p95_ms REAL DEFAULT 0,
    query_latency_p99_ms REAL DEFAULT 0,
    query_latency_mean_ms REAL DEFAULT 0,

    -- efficiency
    avg_tool_calls REAL DEFAULT 0,

    -- composite
    composite_score REAL DEFAULT 0,
    chunk_hit_at_5 REAL DEFAULT 0,

    -- token usage
    avg_response_tokens REAL DEFAULT 0,
    p95_response_tokens REAL DEFAULT 0,
    total_response_tokens REAL DEFAULT 0,
    avg_llm_tokens REAL DEFAULT 0,

    -- meta
    total_queries INTEGER DEFAULT 0,
    total_hits INTEGER DEFAULT 0,
    bench_version TEXT DEFAULT '',
    dataset_version TEXT DEFAULT '',
    environment TEXT DEFAULT '{}',  -- JSON
    by_difficulty TEXT DEFAULT '{}', -- JSON
    by_type TEXT DEFAULT '{}',       -- JSON
    repos TEXT DEFAULT '[]',         -- JSON
    queries TEXT DEFAULT '[]'        -- JSON
);
"""


async def init_db() -> None:
    """Initialize the database schema.

    Idempotent: after creating the table, introspects via PRAGMA table_info(runs)
    and runs ALTER TABLE ADD COLUMN for any missing token columns so legacy DBs
    are upgraded in-place without data loss.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()

        # Introspect existing columns to detect missing token columns
        cursor = await db.execute("PRAGMA table_info(runs)")
        rows = await cursor.fetchall()
        existing_columns = {row[1] for row in rows}

        token_columns = [
            ("avg_response_tokens", "REAL DEFAULT 0"),
            ("p95_response_tokens", "REAL DEFAULT 0"),
            ("total_response_tokens", "REAL DEFAULT 0"),
            ("avg_llm_tokens", "REAL DEFAULT 0"),
        ]

        for col_name, col_def in token_columns:
            if col_name not in existing_columns:
                await db.execute(
                    f"ALTER TABLE runs ADD COLUMN {col_name} {col_def}"
                )

        chunk_columns = [
            ("chunk_hit_at_5", "REAL DEFAULT 0"),
            ("queries", "TEXT DEFAULT '[]'"),
        ]

        for col_name, col_def in chunk_columns:
            if col_name not in existing_columns:
                await db.execute(
                    f"ALTER TABLE runs ADD COLUMN {col_name} {col_def}"
                )

        await db.commit()


async def insert_run(data: dict) -> str:
    """Insert a benchmark run into the database.

    Raises DuplicateRunError if the run_id already exists.
    """
    run_id = data.get("run_id", "")
    server = data.get("server", {})
    ingest = data.get("ingest", {})
    retrieval = data.get("retrieval", {})
    latency = retrieval.get("latency", {})
    efficiency = data.get("efficiency", {})
    tokens = retrieval.get("tokens") or {}

    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(
                """
                INSERT INTO runs (
                    id, server_name, git_url, git_user, server_version,
                    ingest_total_files, ingest_total_sec, ingest_files_per_sec,
                    index_size_mb, ram_peak_mb,
                    hit_at_1, hit_at_3, hit_at_5, hit_at_10,
                    symbol_hit_at_5, mrr,
                    query_latency_p50_ms, query_latency_p95_ms,
                    query_latency_p99_ms, query_latency_mean_ms,
                    avg_tool_calls, composite_score, chunk_hit_at_5,
                    avg_response_tokens, p95_response_tokens,
                    total_response_tokens, avg_llm_tokens,
                    total_queries, total_hits,
                    bench_version, dataset_version,
                    environment, by_difficulty, by_type, repos, queries
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?, ?, ?
                )
                """,
                (
                    run_id,
                    server.get("name", ""),
                    server.get("git_url", ""),
                    server.get("git_user", ""),
                    server.get("version", ""),
                    ingest.get("total_files", 0),
                    ingest.get("total_sec", 0),
                    ingest.get("files_per_sec", 0),
                    ingest.get("index_size_mb", 0),
                    ingest.get("ram_peak_mb", 0),
                    retrieval.get("hit_at_1", 0),
                    retrieval.get("hit_at_3", 0),
                    retrieval.get("hit_at_5", 0),
                    retrieval.get("hit_at_10", 0),
                    retrieval.get("symbol_hit_at_5", 0),
                    retrieval.get("mrr", 0),
                    latency.get("p50_ms", 0),
                    latency.get("p95_ms", 0),
                    latency.get("p99_ms", 0),
                    latency.get("mean_ms", 0),
                    efficiency.get("avg_tool_calls", 0),
                    data.get("composite_score", 0),
                    retrieval.get("chunk_hit_at_5", 0),
                    tokens.get("avg", 0),
                    tokens.get("p95", 0),
                    tokens.get("total", 0),
                    efficiency.get("avg_total_llm_tokens", 0),
                    retrieval.get("total_queries", 0),
                    retrieval.get("total_hits", 0),
                    data.get("bench_version", ""),
                    data.get("dataset_version", ""),
                    json.dumps(data.get("environment", {})),
                    json.dumps(data.get("by_difficulty", {})),
                    json.dumps(data.get("by_type", {})),
                    json.dumps(data.get("repos", [])),
                    json.dumps(data.get("query_details", [])),
                ),
            )
            await db.commit()
        except Exception as e:
            if "UNIQUE constraint failed: runs.id" in str(e):
                raise DuplicateRunError(run_id) from e
            raise
    return run_id


async def get_leaderboard(
    sort_by: str = "composite_score",
    order: str = "desc",
    limit: int = 50,
) -> list[dict]:
    """Get leaderboard data."""
    valid_sorts = {
        "composite_score",
        "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
        "symbol_hit_at_5", "mrr",
        "query_latency_p50_ms", "query_latency_p95_ms",
        "query_latency_p99_ms", "query_latency_mean_ms",
        "ingest_total_files", "ingest_total_sec", "ingest_files_per_sec",
        "index_size_mb", "ram_peak_mb",
        "avg_tool_calls", "total_queries", "total_hits",
        "submitted_at",
        "avg_response_tokens", "p95_response_tokens",
        "total_response_tokens", "avg_llm_tokens",
        "chunk_hit_at_5",
    }
    if sort_by not in valid_sorts:
        sort_by = "composite_score"

    order = "DESC" if order.lower() == "desc" else "ASC"

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            f"""
            SELECT * FROM runs
            ORDER BY {sort_by} {order}
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_run(run_id: str) -> dict | None:
    """Get a specific run by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
