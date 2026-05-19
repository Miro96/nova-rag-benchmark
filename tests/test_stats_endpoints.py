"""Tests for stats endpoints: GET /api/run/{id}/stats and POST /api/run/{id}/recompute_stats.

Covers:
- VAL-STATS-001: GET /api/run/{id}/stats returns full JSON schema
- VAL-STATS-002: GET /api/run/unknown/stats returns 404
- VAL-STATS-004: POST /api/run/{id}/recompute_stats backfills legacy rows
- VAL-STATS-005: POST /api/run/unknown/recompute_stats returns 404
- VAL-STATS-014: Cached-path GET latency check (structural, timing via curl)
- VAL-CROSS-007: GET /api/run/{id}/queries preserves legacy shape
"""
from __future__ import annotations

import asyncio
import json
import uuid

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from server import db
from server.app import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_payload(run_id: str | None = None, server_name: str = "test-server") -> dict:
    """Return a valid benchmark submission payload with query_details."""
    rid = run_id or str(uuid.uuid4())
    return {
        "run_id": rid,
        "bench_version": "0.1.0",
        "dataset_version": "v1",
        "server": {
            "name": server_name,
            "git_url": "https://github.com/test/server",
            "git_user": "tester",
            "version": "1.0.0",
        },
        "environment": {"os": "linux"},
        "repos": ["test-repo"],
        "ingest": {
            "total_files": 100, "total_sec": 10.0,
            "files_per_sec": 10.0, "index_size_mb": 50.0, "ram_peak_mb": 500.0,
        },
        "retrieval": {
            "total_queries": 5,
            "total_hits": 3,
            "hit_at_1": 0.2, "hit_at_3": 0.4, "hit_at_5": 0.6, "hit_at_10": 0.8,
            "symbol_hit_at_5": 0.3,
            "chunk_hit_at_5": 0.4,
            "mrr": 0.5,
            "latency": {"p50_ms": 100, "p95_ms": 200, "p99_ms": 300, "mean_ms": 120},
            "tokens": {"avg": 500, "p50": 450, "p95": 900, "total": 2500},
        },
        "efficiency": {"avg_tool_calls": 2.0},
        "composite_score": 0.5,
        "by_difficulty": {"easy": {"count": 2}, "medium": {"count": 2}, "hard": {"count": 1}},
        "by_type": {"locate": {"count": 2}, "callers": {"count": 2}, "explain": {"count": 1}},
        "by_repo": {"test-repo": {"hit_at_5": 0.6}},
        "query_details": [
            {
                "id": f"Q{i:03d}",
                "type": ["locate", "callers", "explain", "locate", "callers"][i],
                "difficulty": ["easy", "easy", "medium", "medium", "hard"][i],
                "repo": "test-repo",
                "found_file": bool(i < 3),
                "found_symbol": bool(i < 2),
                "found_chunk": bool(i < 2),
                "latency_ms": 100.0 + i * 50,
                "response_tokens": 200.0 + i * 100,
                "tool_calls": 1 + i % 3,
                "returned_files": ["f1.py"],
                "returned_symbols": ["Foo"],
                "expected_files": ["f1.py"],
                "expected_symbols": ["Foo"],
            }
            for i in range(5)
        ],
        "replicates": [
            {"hit_at_5": 0.6, "latency_ms": 100.0},
            {"hit_at_5": 0.7, "latency_ms": 110.0},
        ],
        "iqr": {"hit_at_5": 0.1, "latency_ms": 15.0},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client():
    """Create a TestClient with a fresh DB."""
    db.DB_PATH = db.DB_PATH.parent / "test_stats_endpoints.db"
    if db.DB_PATH.exists():
        db.DB_PATH.unlink()

    asyncio.run(db.init_db())

    client = TestClient(app)
    yield client

    if db.DB_PATH.exists():
        db.DB_PATH.unlink()


# ---------------------------------------------------------------------------
# VAL-STATS-001: GET /api/run/{id}/stats returns full JSON schema
# ---------------------------------------------------------------------------

class TestGetStatsEndpoint:
    """VAL-STATS-001: GET /api/run/{id}/stats returns 200 with full schema."""

    def test_get_stats_returns_200_with_required_keys(self, test_client):
        """GET /api/run/{id}/stats returns 200 with run_id, metrics, by_difficulty,
        by_type, by_repo, correlations, iqr, cv, baseline_ab."""
        # Submit a run first (stats are populated at submit time)
        payload = _valid_payload()
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        # Now get stats
        r = test_client.get(f"/api/run/{run_id}/stats")
        assert r.status_code == 200
        data = r.json()

        required_keys = [
            "run_id", "metrics", "by_difficulty", "by_type",
            "by_repo", "correlations", "iqr", "cv", "baseline_ab",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

        # run_id matches
        assert data["run_id"] == run_id

    def test_get_stats_metrics_structure(self, test_client):
        """metrics.hit_at_5 has point, ci_low, ci_high, n."""
        payload = _valid_payload()
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        r = test_client.get(f"/api/run/{run_id}/stats")
        assert r.status_code == 200
        data = r.json()

        m = data["metrics"]["hit_at_5"]
        for field in ("point", "ci_low", "ci_high", "n"):
            assert field in m, f"hit_at_5 missing {field}"
            assert isinstance(m[field], (int, float)), f"hit_at_5.{field} not numeric"

        # n should equal len(query_details)
        assert m["n"] == 5
        # CI brackets point
        assert m["ci_low"] <= m["point"] <= m["ci_high"]

    def test_get_stats_chunk_hit_at_5_present(self, test_client):
        """metrics.chunk_hit_at_5 is present with full schema."""
        payload = _valid_payload()
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        r = test_client.get(f"/api/run/{run_id}/stats")
        assert r.status_code == 200
        cm = r.json()["metrics"]["chunk_hit_at_5"]
        for field in ("point", "ci_low", "ci_high", "n"):
            assert field in cm
        assert cm["n"] == 5
        assert cm["ci_low"] <= cm["point"] <= cm["ci_high"]

    def test_get_stats_correlations_structure(self, test_client):
        """correlations has matrix and labels with the six core metrics."""
        payload = _valid_payload()
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        r = test_client.get(f"/api/run/{run_id}/stats")
        assert r.status_code == 200
        corr = r.json()["correlations"]
        assert "labels" in corr
        assert "matrix" in corr
        assert len(corr["labels"]) == 6
        # Verify diagonal entries are 1.0
        matrix = corr["matrix"]
        for i in range(len(matrix)):
            assert abs(matrix[i][i] - 1.0) < 1e-9

    def test_get_stats_by_difficulty_present(self, test_client):
        """by_difficulty has bucket entries with metric stats."""
        payload = _valid_payload()
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        r = test_client.get(f"/api/run/{run_id}/stats")
        assert r.status_code == 200
        buckets = r.json()["by_difficulty"]
        assert len(buckets) >= 1
        for diff_name, diff_stats in buckets.items():
            assert "hit_at_5" in diff_stats, f"{diff_name} missing hit_at_5"
            assert "point" in diff_stats["hit_at_5"]

    def test_get_stats_by_type_present(self, test_client):
        """by_type has bucket entries."""
        payload = _valid_payload()
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        r = test_client.get(f"/api/run/{run_id}/stats")
        assert r.status_code == 200
        buckets = r.json()["by_type"]
        assert len(buckets) >= 1

    def test_get_stats_by_repo_present(self, test_client):
        """by_repo has bucket entries."""
        payload = _valid_payload()
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        r = test_client.get(f"/api/run/{run_id}/stats")
        assert r.status_code == 200
        buckets = r.json()["by_repo"]
        assert len(buckets) >= 1

    def test_get_stats_iqr_and_cv_present(self, test_client):
        """iqr and cv dicts are present."""
        payload = _valid_payload()
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        r = test_client.get(f"/api/run/{run_id}/stats")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["iqr"], dict)
        assert isinstance(data["cv"], dict)

    def test_get_stats_baseline_ab_is_null_when_no_baseline(self, test_client):
        """baseline_ab is null when no grep-glob baseline exists."""
        payload = _valid_payload()
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        r = test_client.get(f"/api/run/{run_id}/stats")
        assert r.status_code == 200
        data = r.json()
        # baseline_ab should be null or empty dict
        assert data["baseline_ab"] is None or data["baseline_ab"] == {}


# ---------------------------------------------------------------------------
# VAL-STATS-002: GET /api/run/unknown/stats returns 404
# ---------------------------------------------------------------------------

class TestGetStatsNotFound:
    """VAL-STATS-002: GET /api/run/unknown/stats returns 404."""

    def test_unknown_run_returns_404(self, test_client):
        """GET /api/run/does-not-exist/stats returns 404."""
        r = test_client.get("/api/run/does-not-exist-xyz/stats")
        assert r.status_code == 404

    def test_unknown_run_has_detail_field(self, test_client):
        """404 response has JSON detail field."""
        r = test_client.get("/api/run/definitely-not-there/stats")
        assert r.status_code == 404
        data = r.json()
        assert "detail" in data
        assert len(data["detail"]) > 0

    def test_unknown_run_no_traceback(self, test_client):
        """404 response contains no Python traceback."""
        r = test_client.get("/api/run/bad-run-id/stats")
        body = r.text
        assert "Traceback" not in body
        assert 'File "' not in body


# ---------------------------------------------------------------------------
# VAL-STATS-004: POST /api/run/{id}/recompute_stats backfills legacy rows
# ---------------------------------------------------------------------------

class TestRecomputeStats:
    """VAL-STATS-004: POST /api/run/{id}/recompute_stats backfills."""

    def test_recompute_stats_returns_200(self, test_client):
        """POST /api/run/{id}/recompute_stats returns 200 with status ok."""
        payload = _valid_payload()
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        # Clear the cached stats to simulate legacy row
        async def _clear():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                await conn.execute(
                    "UPDATE runs SET stats_cached = '{}' WHERE id = ?",
                    (run_id,),
                )
                await conn.commit()

        asyncio.run(_clear())

        # Now recompute
        r = test_client.post(f"/api/run/{run_id}/recompute_stats")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["run_id"] == run_id

    def test_recompute_stats_populates_cleared_cache(self, test_client):
        """After clearing stats_cached and recomputing, stats are populated."""
        payload = _valid_payload()
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        # Clear stats_cached
        async def _clear():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                await conn.execute(
                    "UPDATE runs SET stats_cached = '{}' WHERE id = ?",
                    (run_id,),
                )
                await conn.commit()

        asyncio.run(_clear())

        # Recompute
        r = test_client.post(f"/api/run/{run_id}/recompute_stats")
        assert r.status_code == 200

        # Verify stats are now populated
        r = test_client.get(f"/api/run/{run_id}/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["metrics"]["hit_at_5"]["n"] == 5
        assert "point" in data["metrics"]["hit_at_5"]

    def test_recompute_on_legacy_row_with_no_query_details(self, test_client):
        """Recompute on a run without query_details returns empty stats gracefully."""
        payload = _valid_payload()
        # Remove query_details
        payload.pop("query_details", None)
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        # Clear stats_cached
        async def _clear():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                await conn.execute(
                    "UPDATE runs SET stats_cached = '{}' WHERE id = ?",
                    (run_id,),
                )
                await conn.commit()

        asyncio.run(_clear())

        # Recompute should not crash
        r = test_client.post(f"/api/run/{run_id}/recompute_stats")
        assert r.status_code == 200

        # Stats should be returned (even if mostly empty)
        r = test_client.get(f"/api/run/{run_id}/stats")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# VAL-STATS-005: POST /api/run/unknown/recompute_stats returns 404
# ---------------------------------------------------------------------------

class TestRecomputeStatsNotFound:
    """VAL-STATS-005: POST /api/run/unknown/recompute_stats returns 404."""

    def test_unknown_run_returns_404(self, test_client):
        """POST /api/run/no-such-run/recompute_stats returns 404."""
        r = test_client.post("/api/run/no-such-run-123/recompute_stats")
        assert r.status_code == 404

    def test_unknown_run_has_detail_field(self, test_client):
        """404 response has JSON detail field."""
        r = test_client.post("/api/run/definitely-not-there/recompute_stats")
        assert r.status_code == 404
        data = r.json()
        assert "detail" in data
        assert len(data["detail"]) > 0

    def test_unknown_run_no_db_insert(self, test_client):
        """No row inserted as side effect of 404."""
        r = test_client.post("/api/run/no-such-run-insert/recompute_stats")
        assert r.status_code == 404

        # Verify row count unchanged
        async def _check():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM runs WHERE id = ?",
                    ("no-such-run-insert",),
                )
                count = await cursor.fetchone()
                assert count[0] == 0

        asyncio.run(_check())

    def test_unknown_run_no_traceback(self, test_client):
        """404 response contains no Python traceback."""
        r = test_client.post("/api/run/bad-run/recompute_stats")
        body = r.text
        assert "Traceback" not in body
        assert 'File "' not in body


# ---------------------------------------------------------------------------
# VAL-CROSS-007: GET /api/run/{id}/queries preserves legacy shape
# ---------------------------------------------------------------------------

class TestQueriesPreserveLegacyShape:
    """VAL-CROSS-007: GET /api/run/{id}/queries returns preserved shape."""

    def test_queries_endpoint_shape_preserved(self, test_client):
        """GET /api/run/{id}/queries returns the expected JSON array."""
        payload = _valid_payload()
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200
        run_id = payload["run_id"]

        r = test_client.get(f"/api/run/{run_id}/queries")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 5

        # Check that each query has the expected fields
        expected_item_keys = [
            "id", "type", "difficulty", "repo",
            "found_file", "found_symbol", "found_chunk",
            "latency_ms", "response_tokens", "tool_calls",
        ]
        for item in data:
            for key in expected_item_keys:
                assert key in item, f"Query item missing key: {key}"

    def test_queries_endpoint_404_unknown(self, test_client):
        """GET /api/run/unknown/queries returns 404."""
        r = test_client.get("/api/run/unknown-run-queries/queries")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Baseline A/B detection via stats endpoint
# ---------------------------------------------------------------------------

class TestStatsBaselineAB:
    """Verify baseline_ab via GET /api/run/{id}/stats."""

    def test_baseline_ab_populated_via_stats_endpoint(self, test_client):
        """When a grep-glob baseline exists, baseline_ab is non-null in stats response."""
        # Submit grep-glob baseline
        baseline_payload = _valid_payload(
            run_id="baseline-grep-stats", server_name="grep-glob-svr"
        )
        baseline_payload["dataset_version"] = "v1"
        baseline_payload["query_details"] = [
            {
                "id": f"shared_{i:03d}",
                "type": "locate",
                "difficulty": "easy",
                "repo": "test-repo",
                "found_file": bool(i % 2 == 0),
                "found_symbol": False,
                "found_chunk": False,
                "latency_ms": 100.0,
                "response_tokens": 200.0,
                "tool_calls": 1,
                "returned_files": ["f.py"],
                "returned_symbols": [],
                "expected_files": ["f.py"],
                "expected_symbols": [],
            }
            for i in range(4)
        ]
        r = test_client.post("/api/submit", json=baseline_payload)
        assert r.status_code == 200

        # Submit non-grep-glob with same dataset_version
        new_payload = _valid_payload(
            run_id="new-run-stats", server_name="custom-server"
        )
        new_payload["dataset_version"] = "v1"
        new_payload["query_details"] = [
            {
                "id": f"shared_{i:03d}",
                "type": "locate",
                "difficulty": "easy",
                "repo": "test-repo",
                "found_file": bool(i < 2),
                "found_symbol": False,
                "found_chunk": False,
                "latency_ms": 120.0,
                "response_tokens": 250.0,
                "tool_calls": 1,
                "returned_files": ["f.py"],
                "returned_symbols": [],
                "expected_files": ["f.py"],
                "expected_symbols": [],
            }
            for i in range(4)
        ]
        r = test_client.post("/api/submit", json=new_payload)
        assert r.status_code == 200

        # Stats should show baseline_ab populated
        r = test_client.get("/api/run/new-run-stats/stats")
        assert r.status_code == 200
        data = r.json()
        ab = data["baseline_ab"]
        assert ab is not None
        assert isinstance(ab, dict)
        assert ab.get("baseline_run_id") == "baseline-grep-stats"
        assert ab.get("dataset_version") == "v1"
        assert "p_values" in ab
        assert "cohens_d" in ab
        assert "cliffs_delta" in ab
