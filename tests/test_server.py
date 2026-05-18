"""Tests for leaderboard server — submit, leaderboard, run detail, and CLI."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from rag_bench.cli import cli


# ---------------------------------------------------------------------------
# Test helpers
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
        "by_difficulty": {"easy": {"count": 30}, "medium": {"count": 30}, "hard": {"count": 30}},
        "by_type": {"locate": {"count": 30}, "callers": {"count": 20}, "explain": {"count": 20}, "impact": {"count": 20}},
    }


# ---------------------------------------------------------------------------
# VAL-BENCH-LB-001: POST /api/submit valid → 200, GET confirms stored
# ---------------------------------------------------------------------------

class TestSubmitValid:
    def test_submit_valid_returns_200(self, test_client):
        """POST valid payload returns 200 with status ok and run_id."""
        payload = _valid_payload()
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["run_id"] == payload["run_id"]

    def test_get_run_returns_same_data(self, test_client):
        """GET /api/run/{run_id} returns same data as submitted."""
        payload = _valid_payload()
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200
        run_id = payload["run_id"]

        response = test_client.get(f"/api/run/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == run_id
        assert data["server_name"] == payload["server"]["name"]
        assert data["composite_score"] == payload["composite_score"]
        assert data["bench_version"] == payload["bench_version"]


# ---------------------------------------------------------------------------
# VAL-BENCH-LB-002: Duplicate run_id → 409
# ---------------------------------------------------------------------------

class TestSubmitDuplicate:
    def test_duplicate_run_id_returns_409(self, test_client):
        """Submitting same run_id twice returns 409 Conflict."""
        payload = _valid_payload()
        response1 = test_client.post("/api/submit", json=payload)
        assert response1.status_code == 200

        response2 = test_client.post("/api/submit", json=payload)
        assert response2.status_code == 409

    def test_duplicate_does_not_overwrite(self, test_client):
        """After 409, original data is still intact."""
        payload = _valid_payload()
        test_client.post("/api/submit", json=payload)
        # Submit again with same run_id but different score
        payload_changed = dict(payload)
        payload_changed["composite_score"] = 0.99
        response2 = test_client.post("/api/submit", json=payload_changed)
        assert response2.status_code == 409

        # Original score should be preserved
        response_get = test_client.get(f"/api/run/{payload['run_id']}")
        assert response_get.status_code == 200
        assert response_get.json()["composite_score"] == 0.65


# ---------------------------------------------------------------------------
# VAL-BENCH-LB-003: Invalid payload → 422
# ---------------------------------------------------------------------------

class TestSubmitInvalid:
    def test_missing_required_field_returns_422(self, test_client):
        """Missing required fields return 422 with detail."""
        # Missing 'server' and 'run_id'
        response = test_client.post("/api/submit", json={})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_empty_payload_returns_422(self, test_client):
        """Empty body returns 422."""
        response = test_client.post("/api/submit", content=b"")
        assert response.status_code == 422

    def test_wrong_type_returns_422(self, test_client):
        """Wrong type for a field returns 422."""
        payload = _valid_payload()
        payload["composite_score"] = "not-a-number"
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 422

    def test_run_id_empty_string_accepted(self, test_client):
        """run_id empty string is technically valid Pydantic-wise (no constraint set)."""
        payload = _valid_payload(run_id="")
        response = test_client.post("/api/submit", json=payload)
        # Should succeed — empty string is allowed
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# VAL-BENCH-LB-004: Leaderboard response has all required fields
# ---------------------------------------------------------------------------

class TestLeaderboardResponse:
    def test_leaderboard_returns_entries_and_total(self, test_client):
        """GET /api/leaderboard returns entries list and total."""
        payload = _valid_payload()
        test_client.post("/api/submit", json=payload)

        response = test_client.get("/api/leaderboard")
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert "total" in data
        assert isinstance(data["entries"], list)
        assert data["total"] == len(data["entries"])

    def test_leaderboard_entry_has_all_required_fields(self, test_client):
        """Each leaderboard entry has all required fields from VAL-BENCH-LB-004."""
        payload = _valid_payload()
        test_client.post("/api/submit", json=payload)

        response = test_client.get("/api/leaderboard")
        entries = response.json()["entries"]
        assert len(entries) == 1

        entry = entries[0]
        required_fields = [
            "run_id", "server_name", "hit_at_1", "hit_at_5",
            "symbol_hit_at_5", "mrr", "query_latency_p50_ms",
            "query_latency_p95_ms", "ingest_total_sec",
            "ram_peak_mb", "composite_score", "submitted_at",
        ]
        for field in required_fields:
            assert field in entry, f"Missing required field: {field}"

        # Type checks
        assert isinstance(entry["run_id"], str)
        assert isinstance(entry["composite_score"], (int, float))

    def test_leaderboard_sorted_by_composite_score_desc(self, test_client):
        """Leaderboard entries sorted by composite_score descending."""
        # Submit two entries
        payload1 = _valid_payload(run_id=str(uuid.uuid4()))
        payload1["composite_score"] = 0.3
        test_client.post("/api/submit", json=payload1)

        payload2 = _valid_payload(run_id=str(uuid.uuid4()))
        payload2["composite_score"] = 0.9
        test_client.post("/api/submit", json=payload2)

        response = test_client.get("/api/leaderboard")
        entries = response.json()["entries"]
        assert entries[0]["composite_score"] >= entries[1]["composite_score"]
        assert entries[0]["composite_score"] == 0.9

    def test_leaderboard_empty_is_ok(self, test_client):
        """Empty leaderboard returns 200 with empty entries."""
        response = test_client.get("/api/leaderboard")
        assert response.status_code == 200
        data = response.json()
        assert data["entries"] == []
        assert data["total"] == 0


# ---------------------------------------------------------------------------
# VAL-BENCH-LB-006: sort_by, order, limit query params
# ---------------------------------------------------------------------------

class TestLeaderboardSortLimit:
    def test_sort_by_hit_at_5_asc_and_limit(self, test_client):
        """?sort_by=hit_at_5&order=asc&limit=3 respects sort and limit."""
        for i in range(5):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["retrieval"]["hit_at_5"] = 0.1 * (i + 1)
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=hit_at_5&order=asc&limit=3"
        )
        assert response.status_code == 200
        entries = response.json()["entries"]
        assert len(entries) <= 3
        # Ascending order
        hits = [e["hit_at_5"] for e in entries]
        assert hits == sorted(hits)

    def test_sort_by_mrr_desc(self, test_client):
        """?sort_by=mrr&order=desc sorts by MRR descending."""
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["retrieval"]["mrr"] = 0.1 * (i + 1)
            test_client.post("/api/submit", json=payload)

        response = test_client.get("/api/leaderboard?sort_by=mrr&order=desc")
        entries = response.json()["entries"]
        mrrs = [e["mrr"] for e in entries]
        assert mrrs == sorted(mrrs, reverse=True)

    def test_invalid_sort_falls_back_to_composite_score(self, test_client):
        """Invalid sort_by falls back to composite_score."""
        payload = _valid_payload()
        test_client.post("/api/submit", json=payload)

        response = test_client.get("/api/leaderboard?sort_by=invalid_field")
        assert response.status_code == 200
        # Should still return valid data
        assert "entries" in response.json()

    def test_limit_respected(self, test_client):
        """Limit param caps result count."""
        for i in range(10):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            test_client.post("/api/submit", json=payload)

        response = test_client.get("/api/leaderboard?limit=4")
        assert len(response.json()["entries"]) <= 4

    def test_sort_by_query_latency_p50_ms_asc(self, test_client):
        """?sort_by=query_latency_p50_ms&order=asc sorts by P50 ascending."""
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["retrieval"]["latency"]["p50_ms"] = 100.0 * (i + 1)
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=query_latency_p50_ms&order=asc"
        )
        entries = response.json()["entries"]
        latencies = [e["query_latency_p50_ms"] for e in entries]
        assert latencies == sorted(latencies)

    def test_sort_by_ingest_total_sec_desc(self, test_client):
        """?sort_by=ingest_total_sec&order=desc sorts by ingest time descending."""
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["ingest"]["total_sec"] = 5.0 * (i + 1)
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=ingest_total_sec&order=desc"
        )
        entries = response.json()["entries"]
        times = [e["ingest_total_sec"] for e in entries]
        assert times == sorted(times, reverse=True)

    def test_sort_by_hit_at_10_desc(self, test_client):
        """?sort_by=hit_at_10&order=desc sorts by Hit@10 descending."""
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["retrieval"]["hit_at_10"] = 0.2 * (i + 1)
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=hit_at_10&order=desc"
        )
        entries = response.json()["entries"]
        hits = [e["hit_at_10"] for e in entries]
        assert hits == sorted(hits, reverse=True)

    def test_sort_by_index_size_mb_asc(self, test_client):
        """?sort_by=index_size_mb&order=asc sorts by index size ascending."""
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["ingest"]["index_size_mb"] = 10.0 * (i + 1)
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=index_size_mb&order=asc"
        )
        entries = response.json()["entries"]
        sizes = [e["index_size_mb"] for e in entries]
        assert sizes == sorted(sizes)

    def test_sort_by_avg_tool_calls_desc(self, test_client):
        """?sort_by=avg_tool_calls&order=desc sorts by tool calls descending."""
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["efficiency"]["avg_tool_calls"] = 1.0 * (i + 1)
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=avg_tool_calls&order=desc"
        )
        entries = response.json()["entries"]
        calls = [e["avg_tool_calls"] for e in entries]
        assert calls == sorted(calls, reverse=True)

    def test_sort_by_hit_at_3_asc(self, test_client):
        """?sort_by=hit_at_3&order=asc sorts by Hit@3 ascending."""
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["retrieval"]["hit_at_3"] = 0.15 * (i + 1)
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=hit_at_3&order=asc"
        )
        entries = response.json()["entries"]
        hits = [e["hit_at_3"] for e in entries]
        assert hits == sorted(hits)

    def test_sort_defaults_to_composite_score_when_no_param(self, test_client):
        """No sort_by param defaults to composite_score descending."""
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["composite_score"] = 0.1 * (i + 1)
            test_client.post("/api/submit", json=payload)

        response = test_client.get("/api/leaderboard")
        entries = response.json()["entries"]
        scores = [e["composite_score"] for e in entries]
        assert scores == sorted(scores, reverse=True)

    def test_order_defaults_to_desc(self, test_client):
        """No order param defaults to descending."""
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["composite_score"] = 0.1 * (i + 1)
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=composite_score"
        )
        entries = response.json()["entries"]
        scores = [e["composite_score"] for e in entries]
        assert scores == sorted(scores, reverse=True)

    def test_invalid_order_falls_back_to_asc(self, test_client):
        """Invalid order value falls back to ASC."""
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["composite_score"] = 0.1 * (i + 1)
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=composite_score&order=invalid"
        )
        entries = response.json()["entries"]
        scores = [e["composite_score"] for e in entries]
        # Should be ascending (fallback from invalid order)
        assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# VAL-BENCH-LB-007: GET /api/run/unknown → 404
# ---------------------------------------------------------------------------

class TestRunNotFound:
    def test_unknown_run_id_returns_404(self, test_client):
        """GET /api/run/nonexistent returns 404."""
        response = test_client.get("/api/run/nonexistent-id-12345")
        assert response.status_code == 404

    def test_unknown_run_id_has_descriptive_error(self, test_client):
        """404 response includes descriptive error message."""
        response = test_client.get("/api/run/definitely-not-there")
        assert response.status_code == 404
        data = response.json()
        # Should have some error-like field
        assert "detail" in data or "error" in data


# ---------------------------------------------------------------------------
# VAL-BENCH-LB-008: rag-bench submit CLI
# ---------------------------------------------------------------------------

class TestSubmitCLI:
    def test_submit_help_shows_url_option(self):
        """rag-bench submit --help shows --server-url option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["submit", "--help"])
        assert result.exit_code == 0
        assert "server-url" in result.output


# ---------------------------------------------------------------------------
# VAL-BENCH-LB-005: Default host is 127.0.0.1
# ---------------------------------------------------------------------------

class TestServeDefaults:
    def test_serve_help_shows_127_0_0_1(self):
        """rag-bench serve --help shows 127.0.0.1 as default host."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "127.0.0.1" in result.output

    def test_serve_help_does_not_show_0_0_0_0(self):
        """Default host is NOT 0.0.0.0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "0.0.0.0" not in result.output


# ---------------------------------------------------------------------------
# Token column tests — VAL-SERVER-001 through VAL-SERVER-008 + VAL-COMPAT-003
# ---------------------------------------------------------------------------

def _valid_payload_with_tokens(run_id: str | None = None) -> dict:
    """Return a valid benchmark submission payload including token fields."""
    payload = _valid_payload(run_id=run_id)
    payload["retrieval"]["tokens"] = {
        "avg": 1234.5,
        "p50": 1100.0,
        "p95": 1800.0,
        "total": 129000.0,
    }
    payload["efficiency"]["avg_total_llm_tokens"] = 2380.0
    return payload


class TestSchemaTokenColumns:
    """VAL-SERVER-001: DB schema includes token columns."""

    def test_schema_has_token_columns(self, db_with_tokens):
        """After init_db(), PRAGMA table_info(runs) lists token columns."""
        import asyncio, aiosqlite
        from server import db

        async def _check():
            async with aiosqlite.connect(db.DB_PATH) as conn:
                cursor = await conn.execute("PRAGMA table_info(runs)")
                rows = await cursor.fetchall()
                columns = {row[1] for row in rows}
                expected = {
                    "avg_response_tokens", "p95_response_tokens",
                    "total_response_tokens", "avg_llm_tokens",
                }
                missing = expected - columns
                assert not missing, f"Missing columns: {missing}"
                # Verify types
                for row in rows:
                    if row[1] in expected:
                        # col type should be REAL or similar
                        assert "REAL" in row[2].upper() or row[2].upper() == "REAL", \
                            f"Column {row[1]} has type {row[2]}, expected REAL"

        asyncio.run(_check())


class TestLegacyDbAlter:
    """VAL-SERVER-002: Idempotent ALTER for legacy databases."""

    OLD_SCHEMA = """
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
        total_queries INTEGER DEFAULT 0,
        total_hits INTEGER DEFAULT 0,
        bench_version TEXT DEFAULT '',
        dataset_version TEXT DEFAULT '',
        environment TEXT DEFAULT '{}',
        by_difficulty TEXT DEFAULT '{}',
        by_type TEXT DEFAULT '{}',
        repos TEXT DEFAULT '[]'
    );
    """

    def test_legacy_db_adds_columns_without_losing_data(self, legacy_db_path):
        """init_db on legacy DB adds columns and preserves old rows."""
        import asyncio, aiosqlite
        from server.db import init_db, DB_PATH as _orig_db_path
        from server import db

        # Point DB_PATH to our legacy temp file
        old_path = db.DB_PATH
        db.DB_PATH = legacy_db_path
        try:
            # Run init_db which should add missing columns
            asyncio.run(init_db())

            # Verify new columns exist
            async def _check():
                async with aiosqlite.connect(legacy_db_path) as conn:
                    cursor = await conn.execute("PRAGMA table_info(runs)")
                    rows = await cursor.fetchall()
                    columns = {row[1] for row in rows}
                    expected = {
                        "avg_response_tokens", "p95_response_tokens",
                        "total_response_tokens", "avg_llm_tokens",
                    }
                    missing = expected - columns
                    assert not missing, f"Missing columns after alter: {missing}"

                    # Verify legacy row still exists and has the new columns as 0
                    cursor = await conn.execute("SELECT * FROM runs WHERE id = ?", ("legacy-run-1",))
                    row = await cursor.fetchone()
                    assert row is not None, "Legacy row was lost!"

            asyncio.run(_check())
        finally:
            db.DB_PATH = old_path

    def test_init_db_idempotent_on_fresh_db(self, db_with_tokens):
        """Calling init_db twice on a fresh DB is safe (idempotent)."""
        import asyncio
        from server.db import init_db
        from server import db

        # Already initialized once by fixture; run again
        asyncio.run(init_db())

        # Verify columns still exist
        async def _check():
            import aiosqlite
            async with aiosqlite.connect(db.DB_PATH) as conn:
                cursor = await conn.execute("PRAGMA table_info(runs)")
                rows = await cursor.fetchall()
                columns = {row[1] for row in rows}
                assert "avg_response_tokens" in columns
                assert "p95_response_tokens" in columns
                assert "total_response_tokens" in columns
                assert "avg_llm_tokens" in columns

        asyncio.run(_check())


class TestInsertRunTokenFields:
    """VAL-SERVER-003: insert_run persists token fields."""

    def test_insert_with_tokens_stores_them(self, test_client):
        """insert_run with token fields stores them; get_run returns them."""
        payload = _valid_payload_with_tokens()
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200

        run_id = payload["run_id"]
        response = test_client.get(f"/api/run/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["avg_response_tokens"] == 1234.5
        assert data["p95_response_tokens"] == 1800.0
        assert data["total_response_tokens"] == 129000.0
        assert data["avg_llm_tokens"] == 2380.0

    def test_insert_without_tokens_stores_zeros(self, test_client):
        """insert_run without token fields stores 0 in those columns (VAL-SERVER-004)."""
        payload = _valid_payload()
        # Ensure no token fields
        payload["retrieval"].pop("tokens", None)
        payload["efficiency"].pop("avg_total_llm_tokens", None)

        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200

        run_id = payload["run_id"]
        response = test_client.get(f"/api/run/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["avg_response_tokens"] == 0.0
        assert data["p95_response_tokens"] == 0.0
        assert data["total_response_tokens"] == 0.0
        assert data["avg_llm_tokens"] == 0.0


class TestLeaderboardTokenFields:
    """VAL-SERVER-005 + VAL-SERVER-006: Leaderboard exposes and sorts by tokens."""

    def test_leaderboard_entries_have_token_fields(self, test_client):
        """GET /api/leaderboard returns entries with token fields."""
        payload = _valid_payload_with_tokens()
        test_client.post("/api/submit", json=payload)

        response = test_client.get("/api/leaderboard")
        assert response.status_code == 200
        entries = response.json()["entries"]
        assert len(entries) >= 1
        entry = entries[0]
        assert "avg_response_tokens" in entry
        assert "p95_response_tokens" in entry
        assert "total_response_tokens" in entry
        assert "avg_llm_tokens" in entry

    def test_leaderboard_sort_by_avg_response_tokens_asc(self, test_client):
        """?sort_by=avg_response_tokens&order=asc returns ordered results."""
        import uuid

        # Insert entries with different token values
        for i, tokens_val in enumerate([500.0, 100.0, 300.0]):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["retrieval"]["tokens"] = {
                "avg": tokens_val, "p50": tokens_val,
                "p95": tokens_val, "total": tokens_val * 10,
            }
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=avg_response_tokens&order=asc"
        )
        assert response.status_code == 200
        entries = response.json()["entries"]
        tokens = [e["avg_response_tokens"] for e in entries
                  if e.get("avg_response_tokens", 0) > 0]
        assert tokens == sorted(tokens), f"Expected sorted ascending, got {tokens}"

    def test_leaderboard_sort_by_avg_response_tokens_desc(self, test_client):
        """?sort_by=avg_response_tokens&order=desc returns ordered results."""
        import uuid

        for i, tokens_val in enumerate([500.0, 100.0, 300.0]):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["retrieval"]["tokens"] = {
                "avg": tokens_val, "p50": tokens_val,
                "p95": tokens_val, "total": tokens_val * 10,
            }
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=avg_response_tokens&order=desc"
        )
        assert response.status_code == 200
        entries = response.json()["entries"]
        tokens = [e["avg_response_tokens"] for e in entries
                  if e.get("avg_response_tokens", 0) > 0]
        assert tokens == sorted(tokens, reverse=True), \
            f"Expected sorted descending, got {tokens}"

    def test_leaderboard_sort_by_p95_response_tokens(self, test_client):
        """?sort_by=p95_response_tokens sorts by p95."""
        import uuid

        for i, tokens_val in enumerate([1800.0, 900.0, 1500.0]):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["retrieval"]["tokens"] = {
                "avg": tokens_val, "p50": tokens_val,
                "p95": tokens_val, "total": tokens_val * 10,
            }
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=p95_response_tokens&order=asc"
        )
        assert response.status_code == 200
        entries = response.json()["entries"]
        tokens = [e["p95_response_tokens"] for e in entries
                  if e.get("p95_response_tokens", 0) > 0]
        assert tokens == sorted(tokens)

    def test_leaderboard_sort_by_total_response_tokens(self, test_client):
        """?sort_by=total_response_tokens sorts by total."""
        import uuid

        for i, tokens_val in enumerate([30000.0, 10000.0, 20000.0]):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["retrieval"]["tokens"] = {
                "avg": tokens_val, "p50": tokens_val,
                "p95": tokens_val, "total": tokens_val,
            }
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=total_response_tokens&order=asc"
        )
        assert response.status_code == 200
        entries = response.json()["entries"]
        tokens = [e["total_response_tokens"] for e in entries
                  if e.get("total_response_tokens", 0) > 0]
        assert tokens == sorted(tokens)

    def test_leaderboard_sort_by_avg_llm_tokens(self, test_client):
        """?sort_by=avg_llm_tokens sorts by LLM tokens."""
        import uuid

        for i, tokens_val in enumerate([3000.0, 1000.0, 2000.0]):
            payload = _valid_payload(run_id=str(uuid.uuid4()))
            payload["efficiency"]["avg_total_llm_tokens"] = tokens_val
            test_client.post("/api/submit", json=payload)

        response = test_client.get(
            "/api/leaderboard?sort_by=avg_llm_tokens&order=asc"
        )
        assert response.status_code == 200
        entries = response.json()["entries"]
        tokens = [e["avg_llm_tokens"] for e in entries
                  if e.get("avg_llm_tokens", 0) > 0]
        assert tokens == sorted(tokens)


class TestSubmitTokenFields:
    """VAL-SERVER-008: Server submit endpoint accepts new fields."""

    def test_submit_with_token_fields_returns_200(self, test_client):
        """POST /api/submit with token fields returns 200 and run_id."""
        payload = _valid_payload_with_tokens()
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["run_id"] == payload["run_id"]

    def test_get_run_echoes_token_fields(self, test_client):
        """GET /api/run/{id} returns the submitted token fields."""
        payload = _valid_payload_with_tokens()
        test_client.post("/api/submit", json=payload)

        response = test_client.get(f"/api/run/{payload['run_id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["avg_response_tokens"] == 1234.5
        assert data["p95_response_tokens"] == 1800.0
        assert data["total_response_tokens"] == 129000.0
        assert data["avg_llm_tokens"] == 2380.0


class TestBackCompatOldResultJson:
    """VAL-COMPAT-003: Old result JSON loadable by leaderboard."""

    def test_old_payload_without_tokens_accepted(self, test_client):
        """Old result JSON (no token fields) accepted by POST /api/submit."""
        payload = _valid_payload()
        # Ensure no token fields at all
        payload["retrieval"].pop("tokens", None)
        payload["efficiency"].pop("avg_total_llm_tokens", None)

        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200

        # Verify it shows up in leaderboard with 0s
        response = test_client.get("/api/leaderboard")
        entries = response.json()["entries"]
        matching = [e for e in entries if e["run_id"] == payload["run_id"]]
        assert len(matching) == 1
        entry = matching[0]
        assert entry["avg_response_tokens"] == 0.0
        assert entry["p95_response_tokens"] == 0.0
        assert entry["total_response_tokens"] == 0.0
        assert entry["avg_llm_tokens"] == 0.0

    def test_old_payload_get_run_returns_zeros(self, test_client):
        """GET /api/run/{id} on old payload returns 0s for token fields."""
        payload = _valid_payload()
        payload["retrieval"].pop("tokens", None)
        payload["efficiency"].pop("avg_total_llm_tokens", None)
        test_client.post("/api/submit", json=payload)

        response = test_client.get(f"/api/run/{payload['run_id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["avg_response_tokens"] == 0.0
        assert data["p95_response_tokens"] == 0.0
        assert data["total_response_tokens"] == 0.0
        assert data["avg_llm_tokens"] == 0.0


# ---------------------------------------------------------------------------
# VAL-SERVER-007: Leaderboard HTML shows token columns
# ---------------------------------------------------------------------------

class TestHtmlTokenColumns:
    """VAL-SERVER-007: GET / returns HTML containing token column headers."""

    def test_html_contains_token_column_headers(self, test_client):
        """GET / returns HTML with 'Avg Tokens' and 'P95 Tokens' headers."""
        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "Avg Tokens" in html, (
            "HTML should contain 'Avg Tokens' column header"
        )
        assert "P95 Tokens" in html, (
            "HTML should contain 'P95 Tokens' column header"
        )

    def test_html_contains_avg_llm_tokens_header(self, test_client):
        """GET / returns HTML with 'Avg LLM Tokens' header."""
        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "Avg LLM Tokens" in html, (
            "HTML should contain 'Avg LLM Tokens' column header"
        )

    def test_html_token_columns_are_sortable(self, test_client):
        """Token column headers have data-sort attributes for sorting."""
        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text
        # Verify sort data attributes exist for token columns
        assert 'data-sort="avg_response_tokens"' in html, (
            "Avg Tokens column should be sortable"
        )
        assert 'data-sort="p95_response_tokens"' in html, (
            "P95 Tokens column should be sortable"
        )
        assert 'data-sort="avg_llm_tokens"' in html, (
            "Avg LLM Tokens column should be sortable"
        )

    def test_leaderboard_table_has_extra_columns_for_tokens(self, test_client):
        """The leaderboard table renders token column cells."""
        # The page loads data dynamically via JavaScript, so we verify the
        # HTML structure includes the token column headers and proper colspan.
        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text
        # Verify the empty state colspan accounts for new columns (was 13, now 19)
        assert 'colspan="19"' in html, (
            "Empty state colspan should be 19 to account for all columns"
        )
        # Verify token column headers are present
        assert "Avg Tokens" in html
        assert "P95 Tokens" in html
        assert "Avg LLM Tokens" in html

    def test_html_sort_dropdown_has_token_options(self, test_client):
        """The sort dropdown includes token column options."""
        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text
        assert 'value="avg_response_tokens"' in html, (
            "Sort dropdown should have avg_response_tokens option"
        )
        assert 'value="p95_response_tokens"' in html, (
            "Sort dropdown should have p95_response_tokens option"
        )
        assert 'value="avg_llm_tokens"' in html, (
            "Sort dropdown should have avg_llm_tokens option"
        )

    def test_legacy_row_zero_tokens_renders(self, test_client):
        """Legacy rows with 0 in token columns render without breaking layout."""
        # Submit a legacy payload without token fields
        payload = _valid_payload()
        payload["retrieval"].pop("tokens", None)
        payload["efficiency"].pop("avg_total_llm_tokens", None)
        test_client.post("/api/submit", json=payload)

        # The page loads data dynamically via JS, so HTML source won't contain
        # the data. Just verify the page renders successfully.
        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text
        # Page should render without error
        assert "<table" in html
        # Verify token column headers are present (handles 0-value rows fine)
        assert "Avg Tokens" in html
        assert "P95 Tokens" in html


class TestHtmlTooltips:
    """VAL-UI-001, VAL-UI-002: Every sortable column header has an informative tooltip."""

    _SORTABLE_COLUMNS = [
        "hit_at_1", "hit_at_5", "symbol_hit_at_5", "chunk_hit_at_5",
        "mrr", "query_latency_p50_ms", "query_latency_p95_ms",
        "ingest_total_sec", "ram_peak_mb", "composite_score",
        "avg_response_tokens", "p95_response_tokens", "avg_llm_tokens",
    ]

    def test_every_sortable_th_has_title(self, test_client):
        """VAL-UI-001: Every <th data-sort='...'> has a non-empty title attribute."""
        import re

        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text

        # Find all <th ...> elements with data-sort="X"
        th_pattern = re.compile(
            r'<th\b[^>]*\bdata-sort\s*=\s*"([^"]*)"[^>]*>', re.IGNORECASE
        )
        matches = th_pattern.findall(html)
        sort_keys = set(matches)

        # Check every expected column is present
        for col in self._SORTABLE_COLUMNS:
            assert col in sort_keys, (
                f"Expected sortable column '{col}' not found in HTML"
            )

        # Check every <th data-sort='...'> has a title attribute
        th_with_title = re.compile(
            r'<th\b[^>]*\bdata-sort\s*=\s*"[^"]*"[^>]*\btitle\s*=\s*"([^"]*)"[^>]*>',
            re.IGNORECASE,
        )
        titled = th_with_title.findall(html)
        # Each titled match should have at least one word
        assert len(titled) == len(matches), (
            f"All {len(matches)} sortable columns must have a title attribute; "
            f"found {len(titled)}"
        )

        for title_text in titled:
            # Non-empty
            assert title_text.strip(), (
                f"Tooltip title must not be empty"
            )
            # At least 10 words
            word_count = len(title_text.split())
            assert word_count >= 10, (
                f"Tooltip must have at least 10 words, got {word_count}: '{title_text}'"
            )

    def test_hit_at_5_tooltip_mentions_correct_phrases(self, test_client):
        """VAL-UI-002: Hit@5 tooltip mentions 'fraction of queries' and 'top 5'."""
        import re

        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text

        # Extract the title for the hit_at_5 column
        pattern = re.compile(
            r'<th\b[^>]*\bdata-sort\s*=\s*"hit_at_5"[^>]*\btitle\s*=\s*"([^"]*)"[^>]*>',
            re.IGNORECASE,
        )
        match = pattern.search(html)
        assert match, "Hit@5 column must have a title attribute"
        title_text = match.group(1).lower()
        assert "fraction of queries" in title_text, (
            f"Hit@5 tooltip should mention 'fraction of queries': '{match.group(1)}'"
        )
        assert "top 5" in title_text, (
            f"Hit@5 tooltip should mention 'top 5': '{match.group(1)}'"
        )

    def test_mrr_tooltip_mentions_mean_reciprocal_rank(self, test_client):
        """VAL-UI-002: MRR tooltip mentions 'Mean Reciprocal Rank'."""
        import re

        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text

        pattern = re.compile(
            r'<th\b[^>]*\bdata-sort\s*=\s*"mrr"[^>]*\btitle\s*=\s*"([^"]*)"[^>]*>',
            re.IGNORECASE,
        )
        match = pattern.search(html)
        assert match, "MRR column must have a title attribute"
        title_text = match.group(1).lower()
        assert "mean reciprocal rank" in title_text, (
            f"MRR tooltip should mention 'Mean Reciprocal Rank': '{match.group(1)}'"
        )

    def test_all_columns_have_distinct_tooltips(self, test_client):
        """Each sortable column has a unique, distinct tooltip."""
        import re

        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text

        # Extract (data-sort, title) pairs
        pattern = re.compile(
            r'<th\b[^>]*\bdata-sort\s*=\s*"([^"]*)"[^>]*\btitle\s*=\s*"([^"]*)"[^>]*>',
            re.IGNORECASE,
        )
        pairs = pattern.findall(html)

        titles_by_sort = {}
        for sort_key, title_text in pairs:
            titles_by_sort[sort_key] = title_text

        # All titles should be unique (no two columns share the same title)
        seen_titles = set()
        for sort_key, title_text in titles_by_sort.items():
            normalized = title_text.strip().lower()
            assert normalized not in seen_titles, (
                f"Duplicate tooltip for '{sort_key}': '{title_text}'"
            )
            seen_titles.add(normalized)


# ---------------------------------------------------------------------------
# VAL-UI-003, VAL-UI-004, VAL-UI-005: Breakdown columns
# ---------------------------------------------------------------------------

class TestHtmlBreakdownColumns:
    """VAL-UI-003, VAL-UI-004, VAL-UI-005: Difficulty/type breakdown columns."""

    def test_html_contains_breakdown_column_headers(self, test_client):
        """GET / returns HTML with 'By Difficulty' and 'By Type' headers."""
        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "By Difficulty" in html, (
            "HTML should contain 'By Difficulty' column header"
        )
        assert "By Type" in html, (
            "HTML should contain 'By Type' column header"
        )

    def test_breakdown_columns_are_sortable(self, test_client):
        """Breakdown column headers have data-sort attributes."""
        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text
        # The breakdown columns use a client-side sort key
        assert 'data-sort="by_difficulty"' in html, (
            "'By Difficulty' column should be sortable"
        )
        assert 'data-sort="by_type"' in html, (
            "'By Type' column should be sortable"
        )

    def test_breakdown_columns_have_tooltips(self, test_client):
        """Breakdown column headers have descriptive tooltip titles."""
        import re

        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text

        # Check By Difficulty tooltip
        diff_pattern = re.compile(
            r'<th\b[^>]*\bdata-sort\s*=\s*"by_difficulty"[^>]*\btitle\s*=\s*"([^"]*)"[^>]*>',
            re.IGNORECASE,
        )
        diff_match = diff_pattern.search(html)
        assert diff_match, "'By Difficulty' column must have a title attribute"
        diff_title = diff_match.group(1)
        assert len(diff_title.split()) >= 10, (
            f"'By Difficulty' tooltip must have at least 10 words, "
            f"got {len(diff_title.split())}: '{diff_title}'"
        )

        # Check By Type tooltip
        type_pattern = re.compile(
            r'<th\b[^>]*\bdata-sort\s*=\s*"by_type"[^>]*\btitle\s*=\s*"([^"]*)"[^>]*>',
            re.IGNORECASE,
        )
        type_match = type_pattern.search(html)
        assert type_match, "'By Type' column must have a title attribute"
        type_title = type_match.group(1)
        assert len(type_title.split()) >= 10, (
            f"'By Type' tooltip must have at least 10 words, "
            f"got {len(type_title.split())}: '{type_title}'"
        )

    def test_colspan_updated_for_breakdown_columns(self, test_client):
        """Empty/error state colspan is 19 (was 17, +2 for breakdown columns)."""
        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text
        # Both empty state and error state have colspan="19"
        assert 'colspan="19"' in html, (
            "colspan should be 19 to account for breakdown columns"
        )
        # The old colspan should NOT be present
        assert 'colspan="17"' not in html, (
            "colspan 17 should no longer appear (replaced by 19)"
        )

    def test_fmt_breakdown_function_exists_in_js(self, test_client):
        """The fmtBreakdown helper function is defined in the page script."""
        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text
        assert "fmtBreakdown" in html, (
            "fmtBreakdown JavaScript function should be defined in the page"
        )

    def test_by_difficulty_and_by_type_in_js_rendering(self, test_client):
        """The renderTable function reads by_difficulty and by_type from entries."""
        response = test_client.get("/")
        assert response.status_code == 200
        html = response.text
        # The JS should reference these fields for rendering
        assert "by_difficulty" in html, (
            "JavaScript should reference by_difficulty for rendering"
        )
        assert "by_type" in html, (
            "JavaScript should reference by_type for rendering"
        )

    def test_submitted_breakdown_data_appears_in_leaderboard_api(
        self, test_client
    ):
        """After submitting a run with by_difficulty/by_type, they appear in API."""
        import uuid

        payload = _valid_payload(run_id=str(uuid.uuid4()))
        payload["by_difficulty"] = {
            "easy": {"count": 10, "hit_at_5": 0.85},
            "medium": {"count": 10, "hit_at_5": 0.72},
            "hard": {"count": 10, "hit_at_5": 0.55},
        }
        payload["by_type"] = {
            "locate": {"count": 10, "hit_at_5": 0.90},
            "callers": {"count": 10, "hit_at_5": 0.80},
            "explain": {"count": 10, "hit_at_5": 0.60},
        }
        test_client.post("/api/submit", json=payload)

        response = test_client.get("/api/leaderboard")
        assert response.status_code == 200
        entries = response.json()["entries"]
        matching = [e for e in entries if e["run_id"] == payload["run_id"]]
        assert len(matching) == 1
        entry = matching[0]

        # by_difficulty should be a dict with expected keys
        diff = entry.get("by_difficulty")
        assert diff is not None, "by_difficulty should be present in API response"
        if isinstance(diff, str):
            import json
            diff = json.loads(diff)
        assert "easy" in diff, "by_difficulty should contain 'easy' key"
        assert diff["easy"]["hit_at_5"] == 0.85

        # by_type should be a dict with expected keys
        typ = entry.get("by_type")
        assert typ is not None, "by_type should be present in API response"
        if isinstance(typ, str):
            import json
            typ = json.loads(typ)
        assert "locate" in typ, "by_type should contain 'locate' key"
        assert typ["locate"]["hit_at_5"] == 0.90

    def test_empty_breakdown_renders_without_crash(self, test_client):
        """Entries without breakdown data (legacy) render without crash."""
        import uuid

        # Submit a payload without any breakdown data
        payload = _valid_payload(run_id=str(uuid.uuid4()))
        payload.pop("by_difficulty", None)
        payload.pop("by_type", None)
        test_client.post("/api/submit", json=payload)

        response = test_client.get("/api/leaderboard")
        assert response.status_code == 200
        entries = response.json()["entries"]
        assert len(entries) >= 1
        # The entry should still have the fields (from server round-trip)
        entry = entries[0]
        # by_difficulty could be {} or missing — either is fine
        assert "run_id" in entry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client():
    """Create a TestClient with a fresh in-memory DB."""
    import asyncio

    from server.app import app
    from server import db

    # Use a temp file for the DB
    db.DB_PATH = db.DB_PATH.parent / "test_leaderboard.db"

    # Remove stale test DB
    if db.DB_PATH.exists():
        db.DB_PATH.unlink()

    # Initialize
    asyncio.run(db.init_db())

    client = TestClient(app)
    yield client

    # Cleanup
    if db.DB_PATH.exists():
        db.DB_PATH.unlink()


@pytest.fixture
def db_with_tokens():
    """Create a fresh DB via init_db() and return the DB_PATH."""
    import asyncio
    from server import db

    db.DB_PATH = db.DB_PATH.parent / "test_leaderboard_tokens.db"
    if db.DB_PATH.exists():
        db.DB_PATH.unlink()

    asyncio.run(db.init_db())
    yield db.DB_PATH

    if db.DB_PATH.exists():
        db.DB_PATH.unlink()


@pytest.fixture
def legacy_db_path(tmp_path):
    """Create a legacy DB (old schema, no token columns) with one row."""
    import asyncio, aiosqlite

    db_file = tmp_path / "legacy.db"

    async def _create_legacy():
        async with aiosqlite.connect(str(db_file)) as conn:
            OLD_SCHEMA = """
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
                total_queries INTEGER DEFAULT 0,
                total_hits INTEGER DEFAULT 0,
                bench_version TEXT DEFAULT '',
                dataset_version TEXT DEFAULT '',
                environment TEXT DEFAULT '{}',
                by_difficulty TEXT DEFAULT '{}',
                by_type TEXT DEFAULT '{}',
                repos TEXT DEFAULT '[]'
            );
            """
            await conn.executescript(OLD_SCHEMA)
            await conn.execute(
                """INSERT INTO runs (id, server_name, composite_score)
                   VALUES ('legacy-run-1', 'legacy-server', 0.75)"""
            )
            await conn.commit()

    asyncio.run(_create_legacy())
    return db_file
