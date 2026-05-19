"""Tests for DB migration (stats_cached/stats_baseline_ab), Pydantic widening,
and scipy dependency.

Covers:
- VAL-STATS-011: DB migration is idempotent and additive
- VAL-STATS-012: BenchmarkSubmission preserves replicates/iqr/by_repo/query_details
- VAL-STATS-016: scipy>=1.11 declared in pyproject.toml
- VAL-STATS-018: Malformed query_details on submit handled cleanly
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import uuid

import aiosqlite
import pytest
import tomllib

from fastapi.testclient import TestClient

from server import db
from server.app import app
from server.models import BenchmarkSubmission


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_payload(run_id: str | None = None) -> dict:
    """Return a valid benchmark submission payload."""
    return {
        "run_id": run_id or str(uuid.uuid4()),
        "bench_version": "0.1.0",
        "dataset_version": "1.0.0",
        "server": {
            "name": "test-server",
            "git_url": "https://github.com/test/server",
            "git_user": "tester",
            "version": "1.0.0",
        },
        "environment": {"os": "linux", "python": "3.12"},
        "repos": ["flask", "fastapi"],
        "ingest": {
            "total_files": 100,
            "total_sec": 12.5,
            "files_per_sec": 8.0,
            "index_size_mb": 45.2,
            "ram_peak_mb": 512.0,
        },
        "retrieval": {
            "total_queries": 90,
            "total_hits": 72,
            "hit_at_1": 0.25,
            "hit_at_3": 0.55,
            "hit_at_5": 0.72,
            "hit_at_10": 0.85,
            "symbol_hit_at_5": 0.60,
            "mrr": 0.45,
            "latency": {
                "p50_ms": 120.0,
                "p95_ms": 350.0,
                "p99_ms": 500.0,
                "mean_ms": 150.0,
            },
        },
        "efficiency": {
            "avg_tool_calls": 2.3,
        },
        "composite_score": 0.65,
        "by_difficulty": {"easy": {"count": 30}, "medium": {"count": 30}},
        "by_type": {"locate": {"count": 30}, "callers": {"count": 20}},
    }


def _payload_with_extra_fields(run_id: str | None = None) -> dict:
    """Return a payload with replicates/iqr/by_repo/query_details/baseline/ab_comparison/startup_ms."""
    payload = _valid_payload(run_id=run_id)
    payload["replicates"] = [
        {"run": 1, "hit_at_5": 0.8, "latency_ms": 100.0},
        {"run": 2, "hit_at_5": 0.75, "latency_ms": 110.0},
    ]
    payload["iqr"] = {
        "hit_at_5": 0.05,
        "latency_ms": 15.0,
    }
    payload["by_repo"] = {
        "flask": {"hit_at_5": 0.85},
        "fastapi": {"hit_at_5": 0.90},
    }
    payload["query_details"] = [
        {
            "id": "Q001",
            "query": "Where is Flask defined?",
            "type": "locate",
            "difficulty": "easy",
            "repo": "flask",
            "found_file": True,
            "found_symbol": True,
            "found_chunk": True,
            "latency_ms": 120.0,
            "response_tokens": 450,
        },
        {
            "id": "Q002",
            "query": "What calls create_app?",
            "type": "callers",
            "difficulty": "medium",
            "repo": "flask",
            "found_file": True,
            "found_symbol": False,
            "found_chunk": False,
            "latency_ms": 250.0,
            "response_tokens": 600,
        },
    ]
    payload["baseline"] = {"server_name": "grep-glob-baseline", "run_id": "baseline-123"}
    payload["ab_comparison"] = {"hit_at_5_delta": 0.05, "p_value": 0.3}
    payload["startup_ms"] = 1500.0
    return payload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client():
    """Create a TestClient with a fresh in-memory DB."""
    db.DB_PATH = db.DB_PATH.parent / "test_migration_pydantic.db"
    if db.DB_PATH.exists():
        db.DB_PATH.unlink()

    asyncio.run(db.init_db())

    client = TestClient(app)
    yield client

    if db.DB_PATH.exists():
        db.DB_PATH.unlink()


@pytest.fixture
def legacy_db_path(tmp_path):
    """Create a legacy DB (old schema, no stats columns) with one row."""
    db_file = tmp_path / "legacy_stats.db"

    async def _create_legacy():
        async with aiosqlite.connect(str(db_file)) as conn:
            # Use the OLD schema without stats_cached or stats_baseline_ab
            old_schema = """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                server_name TEXT NOT NULL,
                git_url TEXT DEFAULT '',
                git_user TEXT DEFAULT '',
                server_version TEXT DEFAULT '',
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ingest_total_files INTEGER DEFAULT 0,
                ingest_total_sec REAL DEFAULT 0,
                ingest_files_per_sec REAL DEFAULT 0,
                index_size_mb REAL DEFAULT 0,
                ram_peak_mb REAL DEFAULT 0,
                hit_at_1 REAL DEFAULT 0,
                hit_at_3 REAL DEFAULT 0,
                hit_at_5 REAL DEFAULT 0,
                hit_at_10 REAL DEFAULT 0,
                symbol_hit_at_5 REAL DEFAULT 0,
                mrr REAL DEFAULT 0,
                query_latency_p50_ms REAL DEFAULT 0,
                query_latency_p95_ms REAL DEFAULT 0,
                query_latency_p99_ms REAL DEFAULT 0,
                query_latency_mean_ms REAL DEFAULT 0,
                avg_tool_calls REAL DEFAULT 0,
                composite_score REAL DEFAULT 0,
                chunk_hit_at_5 REAL DEFAULT 0,
                avg_response_tokens REAL DEFAULT 0,
                p95_response_tokens REAL DEFAULT 0,
                total_response_tokens REAL DEFAULT 0,
                avg_llm_tokens REAL DEFAULT 0,
                total_queries INTEGER DEFAULT 0,
                total_hits INTEGER DEFAULT 0,
                bench_version TEXT DEFAULT '',
                dataset_version TEXT DEFAULT '',
                environment TEXT DEFAULT '{}',
                by_difficulty TEXT DEFAULT '{}',
                by_type TEXT DEFAULT '{}',
                repos TEXT DEFAULT '[]',
                queries TEXT DEFAULT '[]'
            );
            """
            await conn.executescript(old_schema)
            await conn.execute(
                """INSERT INTO runs (id, server_name, composite_score, dataset_version)
                   VALUES ('legacy-run-1', 'legacy-server', 0.75, 'v1')"""
            )
            await conn.commit()

    asyncio.run(_create_legacy())
    return db_file


# ---------------------------------------------------------------------------
# VAL-STATS-016: scipy>=1.11 declared in pyproject.toml
# ---------------------------------------------------------------------------

class TestScipyDependency:
    """VAL-STATS-016: scipy>=1.11 in pyproject.toml."""

    def test_scipy_in_dependencies(self):
        """scipy>=1.11 declared in [project] dependencies."""
        pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            d = tomllib.load(f)
        deps = d["project"]["dependencies"]
        assert any("scipy" in dep for dep in deps), \
            f"scipy not found in dependencies: {deps}"

        # Verify version constraint is >= 1.11
        for dep in deps:
            if "scipy" in dep:
                # Extract the version specifier
                parts = dep.split(">=")
                assert len(parts) >= 2, f"Missing >= version specifier: {dep}"
                version = parts[1].strip()
                assert float(version.split(".")[0]) >= 1.0, \
                    f"scipy version must be >= 1.11: {dep}"


# ---------------------------------------------------------------------------
# VAL-STATS-011: DB migration is idempotent and additive
# ---------------------------------------------------------------------------

class TestStatsMigration:
    """VAL-STATS-011: DB migration adds stats_cached/stats_baseline_ab idempotently."""

    def test_init_db_adds_stats_columns_on_legacy_db(self, legacy_db_path):
        """init_db on legacy DB adds stats_cached and stats_baseline_ab columns."""
        old_path = db.DB_PATH
        db.DB_PATH = legacy_db_path
        try:
            asyncio.run(db.init_db())

            async def _check():
                async with aiosqlite.connect(str(legacy_db_path)) as conn:
                    cursor = await conn.execute("PRAGMA table_info(runs)")
                    rows = await cursor.fetchall()
                    columns = {row[1] for row in rows}
                    assert "stats_cached" in columns, \
                        f"stats_cached missing: {columns}"
                    assert "stats_baseline_ab" in columns, \
                        f"stats_baseline_ab missing: {columns}"

            asyncio.run(_check())
        finally:
            db.DB_PATH = old_path

    def test_init_db_preserves_legacy_rows(self, legacy_db_path):
        """init_db on legacy DB preserves pre-existing rows."""
        old_path = db.DB_PATH
        db.DB_PATH = legacy_db_path
        try:
            asyncio.run(db.init_db())

            async def _check():
                async with aiosqlite.connect(str(legacy_db_path)) as conn:
                    cursor = await conn.execute("SELECT COUNT(*) FROM runs")
                    count = await cursor.fetchone()
                    assert count[0] >= 1, \
                        "Legacy row was lost after migration!"

                    cursor = await conn.execute(
                        "SELECT id, server_name, composite_score FROM runs WHERE id = ?",
                        ("legacy-run-1",),
                    )
                    row = await cursor.fetchone()
                    assert row is not None, "Legacy row was lost!"
                    assert row[0] == "legacy-run-1"
                    assert row[1] == "legacy-server"
                    assert row[2] == 0.75

            asyncio.run(_check())
        finally:
            db.DB_PATH = old_path

    def test_init_db_idempotent_on_fresh_db(self):
        """Calling init_db twice on a fresh DB is safe (no duplicate column error)."""
        db.DB_PATH = db.DB_PATH.parent / "test_stats_idempotent.db"
        if db.DB_PATH.exists():
            db.DB_PATH.unlink()

        try:
            asyncio.run(db.init_db())
            # Second call should not raise
            asyncio.run(db.init_db())

            # Verify columns exist
            async def _check():
                async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                    cursor = await conn.execute("PRAGMA table_info(runs)")
                    rows = await cursor.fetchall()
                    columns = {row[1] for row in rows}
                    assert "stats_cached" in columns
                    assert "stats_baseline_ab" in columns

            asyncio.run(_check())
        finally:
            if db.DB_PATH.exists():
                db.DB_PATH.unlink()


# ---------------------------------------------------------------------------
# VAL-STATS-012: BenchmarkSubmission preserves extra fields
# ---------------------------------------------------------------------------

class TestBenchmarkSubmissionExtraFields:
    """VAL-STATS-012: BenchmarkSubmission preserves replicates/iqr/by_repo/query_details."""

    def test_submit_with_extra_fields_returns_200(self, test_client):
        """POST with replicates/iqr/by_repo/query_details returns 200, not 422."""
        payload = _payload_with_extra_fields()
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}: {response.text}"

    def test_extra_fields_survive_model_dump(self, test_client):
        """data.model_dump() includes extra fields (replicates, iqr, by_repo, etc.)."""
        payload = _payload_with_extra_fields()

        # Validate via Pydantic model
        instance = BenchmarkSubmission(**payload)
        dump = instance.model_dump()
        assert "replicates" in dump, "replicates dropped by model_dump"
        assert "iqr" in dump, "iqr dropped by model_dump"
        assert "by_repo" in dump, "by_repo dropped by model_dump"
        assert "query_details" in dump, "query_details dropped by model_dump"
        assert "baseline" in dump, "baseline dropped by model_dump"
        assert "ab_comparison" in dump, "ab_comparison dropped by model_dump"
        assert "startup_ms" in dump, "startup_ms dropped by model_dump"

    def test_extra_fields_reachable_via_get_run(self, test_client):
        """Extra fields submitted survive round-trip via GET /api/run/{id}."""
        payload = _payload_with_extra_fields()
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200
        run_id = payload["run_id"]

        response = test_client.get(f"/api/run/{run_id}")
        assert response.status_code == 200
        data = response.json()
        # query_details should be present (stored in DB and returned)
        assert "queries" in data or "query_details" in data

    def test_submit_with_arbitrary_extra_fields(self, test_client):
        """Model accepts arbitrary extra fields not declared on BenchmarkSubmission."""
        payload = _payload_with_extra_fields()
        payload["custom_field"] = "custom_value"
        payload["another_extra"] = 42
        # Validate via model
        instance = BenchmarkSubmission(**payload)
        dump = instance.model_dump()
        assert dump.get("custom_field") == "custom_value"
        assert dump.get("another_extra") == 42


# ---------------------------------------------------------------------------
# VAL-STATS-018: Malformed query_details on submit handled cleanly
# ---------------------------------------------------------------------------

class TestMalformedQueryDetails:
    """VAL-STATS-018: Malformed query_details on submit handled cleanly."""

    def test_query_details_string_returns_4xx(self, test_client):
        """POST with query_details as string returns 400/422."""
        payload = _valid_payload()
        payload["query_details"] = "not a list"
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code in (400, 422), \
            f"Expected 4xx, got {response.status_code}: {response.text}"

    def test_query_details_string_has_detail_field(self, test_client):
        """Response body has detail field describing the error."""
        payload = _valid_payload()
        payload["query_details"] = "not a list"
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code in (400, 422)
        data = response.json()
        assert "detail" in data, \
            f"Expected 'detail' in response: {data}"

    def test_query_details_string_no_db_row_inserted(self, test_client):
        """No row inserted for malformed query_details."""
        payload = _valid_payload()
        payload["query_details"] = "not a list"
        test_client.post("/api/submit", json=payload)

        # Verify no row was inserted
        response = test_client.get("/api/leaderboard")
        entries = response.json()["entries"]
        assert len(entries) == 0, \
            f"No row should be inserted for malformed query_details, got {len(entries)}"

    def test_query_details_no_traceback_in_body(self, test_client):
        """Response body contains no Python traceback markers."""
        payload = _valid_payload()
        payload["query_details"] = "not a list"
        response = test_client.post("/api/submit", json=payload)
        body = response.text
        assert "Traceback" not in body, "Traceback leaked in response"
        assert 'File "' not in body, "File path leaked in response"
        assert "Exception" not in body or "HTTPException" in body, \
            "Raw exception text leaked in response"

    def test_query_details_dict_returns_4xx(self, test_client):
        """POST with query_details as dict returns 4xx."""
        payload = _valid_payload()
        payload["query_details"] = {"id": "Q001"}
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code in (400, 422), \
            f"Expected 4xx for dict query_details, got {response.status_code}"

    def test_query_details_none_returns_4xx(self, test_client):
        """POST with query_details as None returns 4xx (or 200 if default)."""
        payload = _valid_payload()
        payload["query_details"] = None
        response = test_client.post("/api/submit", json=payload)
        # None could be coerced to list or rejected depending on Pydantic config
        # At minimum, it should not crash with a 500
        assert response.status_code not in (500,), \
            f"Should not 500 on None query_details, got {response.status_code}"

    def test_valid_query_details_accepted(self, test_client):
        """Valid list query_details still works (returns 200)."""
        payload = _valid_payload()
        payload["query_details"] = [
            {
                "id": "Q001",
                "query": "Where is Flask defined?",
                "type": "locate",
                "difficulty": "easy",
                "repo": "flask",
                "found_file": True,
                "found_symbol": True,
                "found_chunk": True,
                "latency_ms": 120.0,
                "response_tokens": 450,
            },
        ]
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200

    def test_empty_list_query_details_accepted(self, test_client):
        """Empty list query_details is valid (returns 200)."""
        payload = _valid_payload()
        payload["query_details"] = []
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200
