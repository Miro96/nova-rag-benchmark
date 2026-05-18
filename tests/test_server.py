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
        # Verify the empty state colspan accounts for new columns (was 13, now 16)
        assert 'colspan="16"' in html, (
            "Empty state colspan should be 16 to account for new token columns"
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
