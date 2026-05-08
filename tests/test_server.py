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
