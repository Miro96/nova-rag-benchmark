"""Tests for GET /api/server/{name}/history endpoint.

Covers:
- VAL-TREND-001: Returns 200 with runs sorted ascending by submitted_at
- VAL-TREND-002: Returns 200 with empty array for unknown server name (NOT 404)
- VAL-TREND-003: Each entry contains required metric fields with correct types
- VAL-TREND-004: Non-first entries include `regressions` map with boolean flags
"""
from __future__ import annotations

import asyncio
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
        "efficiency": {"avg_tool_calls": 2.0, "avg_total_llm_tokens": 500},
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
    db.DB_PATH = db.DB_PATH.parent / "test_server_history.db"
    if db.DB_PATH.exists():
        db.DB_PATH.unlink()

    asyncio.run(db.init_db())

    client = TestClient(app)
    yield client

    if db.DB_PATH.exists():
        db.DB_PATH.unlink()


# ---------------------------------------------------------------------------
# VAL-TREND-001: Returns 200 with runs sorted ascending by submitted_at
# ---------------------------------------------------------------------------

class TestHistorySorted:
    """VAL-TREND-001: GET /api/server/{name}/history returns 200 with sorted entries."""

    def test_history_returns_200_with_entries(self, test_client):
        """Submit 3 runs for same server and verify history returns them in order."""
        server = "my-rag-server"
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
            payload["composite_score"] = 0.5 + i * 0.1
            r = test_client.post("/api/submit", json=payload)
            assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 3

        # Verify ascending by submitted_at
        timestamps = [entry["submitted_at"] for entry in data]
        assert timestamps == sorted(timestamps), (
            f"Expected submitted_at in ascending order, got {timestamps}"
        )

    def test_history_single_run_server(self, test_client):
        """Server with exactly 1 run returns 200 with one entry."""
        payload = _valid_payload(run_id=str(uuid.uuid4()), server_name="single-server")
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200

        r = test_client.get("/api/server/single-server/history")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 1


# ---------------------------------------------------------------------------
# VAL-TREND-002: Returns 200 with empty array for unknown server (NOT 404)
# ---------------------------------------------------------------------------

class TestHistoryUnknownServer:
    """VAL-TREND-002: GET /api/server/unknown/history returns 200 with empty array."""

    def test_unknown_server_returns_200_empty(self, test_client):
        """Unknown server name returns 200 (NOT 404) with empty array."""
        r = test_client.get("/api/server/__no_such_server__/history")
        assert r.status_code == 200, (
            f"Expected 200 for unknown server, got {r.status_code}"
        )
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 0, f"Expected empty array, got {len(data)} entries"

    def test_unknown_server_not_404(self, test_client):
        """Explicitly assert that unknown server does NOT return 404."""
        r = test_client.get("/api/server/completely-unknown-server/history")
        assert r.status_code != 404, (
            "Unknown server name should NOT return 404"
        )


# ---------------------------------------------------------------------------
# VAL-TREND-003: Each entry contains required metric fields with correct types
# ---------------------------------------------------------------------------

class TestHistoryEntryFields:
    """VAL-TREND-003: Each history entry contains all required keys."""

    REQUIRED_KEYS = [
        "run_id", "submitted_at", "composite_score",
        "hit_at_5", "symbol_hit_at_5", "mrr",
        "latency_p50_ms", "avg_response_tokens", "dataset_version",
    ]

    def test_entries_have_all_required_keys(self, test_client):
        """Every entry in history response contains all 9 required keys."""
        payload = _valid_payload(run_id=str(uuid.uuid4()), server_name="field-test")
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200

        r = test_client.get("/api/server/field-test/history")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1

        entry = data[0]
        missing = [k for k in self.REQUIRED_KEYS if k not in entry]
        assert not missing, f"Missing required keys: {missing}"

    def test_entry_types_are_correct(self, test_client):
        """Each field has the expected type."""
        payload = _valid_payload(run_id=str(uuid.uuid4()), server_name="type-test")
        r = test_client.post("/api/submit", json=payload)
        assert r.status_code == 200

        r = test_client.get("/api/server/type-test/history")
        assert r.status_code == 200
        entry = r.json()[0]

        assert isinstance(entry["run_id"], str)
        assert isinstance(entry["submitted_at"], str)
        assert isinstance(entry["composite_score"], (int, float))
        assert isinstance(entry["hit_at_5"], (int, float))
        assert isinstance(entry["symbol_hit_at_5"], (int, float))
        assert isinstance(entry["mrr"], (int, float))
        assert isinstance(entry["latency_p50_ms"], (int, float))
        assert isinstance(entry["avg_response_tokens"], (int, float))
        assert isinstance(entry["dataset_version"], str)

    def test_multiple_runs_all_have_keys(self, test_client):
        """When multiple runs exist, all entries have required keys."""
        server = "multi-field-test"
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
            r = test_client.post("/api/submit", json=payload)
            assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 3

        for i, entry in enumerate(data):
            missing = [k for k in self.REQUIRED_KEYS if k not in entry]
            assert not missing, f"Entry {i} missing keys: {missing}"


# ---------------------------------------------------------------------------
# VAL-TREND-004: Non-first entries include `regressions` map with boolean flags
# ---------------------------------------------------------------------------

class TestHistoryRegressions:
    """VAL-TREND-004: Regression flags computed via >5% rule (direction-aware)."""

    # Higher-is-better metrics: (prev - curr) / prev > 0.05
    # Lower-is-better metrics: (curr - prev) / prev > 0.05
    HIGHER_BETTER = ["composite_score", "hit_at_5", "symbol_hit_at_5", "mrr"]
    LOWER_BETTER = ["latency_p50_ms", "avg_response_tokens"]
    ALL_METRICS = HIGHER_BETTER + LOWER_BETTER

    def test_regressions_present_on_non_first_entries(self, test_client):
        """Every entry except the first has a `regressions` object."""
        server = "regression-test-server"
        for i in range(3):
            payload = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
            r = test_client.post("/api/submit", json=payload)
            assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 3

        # First entry: regressions may be absent or all false
        entry0 = data[0]
        if "regressions" in entry0:
            reg0 = entry0["regressions"]
            assert isinstance(reg0, dict)
            # All should be false for first entry
            for metric in self.ALL_METRICS:
                if metric in reg0:
                    assert reg0[metric] is False, (
                        f"First entry {metric} regression should be false"
                    )

        # Non-first entries must have regressions
        for i in range(1, len(data)):
            entry = data[i]
            assert "regressions" in entry, f"Entry {i} missing 'regressions'"
            reg = entry["regressions"]
            assert isinstance(reg, dict), f"Entry {i} regressions should be a dict"

            # All 6 metrics should be in regressions
            for metric in self.ALL_METRICS:
                assert metric in reg, (
                    f"Entry {i} regressions missing metric '{metric}'"
                )
                assert isinstance(reg[metric], bool), (
                    f"Entry {i} regressions.{metric} should be boolean, "
                    f"got {type(reg[metric]).__name__}"
                )

    def test_regression_higher_is_better_detected(self, test_client):
        """When composite_score drops >5%, regression flag is true."""
        server = "higher-better-test"
        # Run 1: composite_score = 100.0
        p1 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p1["composite_score"] = 100.0
        r = test_client.post("/api/submit", json=p1)
        assert r.status_code == 200

        # Run 2: composite_score = 90.0 (drop of 10%, which is >5%)
        p2 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p2["composite_score"] = 90.0
        r = test_client.post("/api/submit", json=p2)
        assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2

        entry = data[1]
        assert entry["regressions"]["composite_score"] is True, (
            f"Expected composite_score regression=True (100→90, drop 10%), "
            f"got {entry['regressions']['composite_score']}"
        )

    def test_regression_higher_is_better_not_triggered_on_small_change(self, test_client):
        """When composite_score drops only 3%, regression flag is false."""
        server = "small-drop-test"
        # Run 1: composite_score = 100.0
        p1 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p1["composite_score"] = 100.0
        r = test_client.post("/api/submit", json=p1)
        assert r.status_code == 200

        # Run 2: composite_score = 97.0 (drop of 3%, which is <=5%)
        p2 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p2["composite_score"] = 97.0
        r = test_client.post("/api/submit", json=p2)
        assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()

        entry = data[1]
        assert entry["regressions"]["composite_score"] is False, (
            f"Expected composite_score regression=False (100→97, drop 3%), "
            f"got {entry['regressions']['composite_score']}"
        )

    def test_regression_higher_is_better_improvement_not_regression(self, test_client):
        """When composite_score improves, regression flag is false."""
        server = "improvement-test"
        # Run 1: composite_score = 80.0
        p1 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p1["composite_score"] = 80.0
        r = test_client.post("/api/submit", json=p1)
        assert r.status_code == 200

        # Run 2: composite_score = 100.0 (improvement)
        p2 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p2["composite_score"] = 100.0
        r = test_client.post("/api/submit", json=p2)
        assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()

        entry = data[1]
        assert entry["regressions"]["composite_score"] is False, (
            f"Expected composite_score regression=False (80→100, improvement), "
            f"got {entry['regressions']['composite_score']}"
        )

    def test_regression_lower_is_better_detected(self, test_client):
        """When latency increases >5%, regression flag is true."""
        server = "lower-better-test"
        # Run 1: latency_p50_ms = 100.0
        p1 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p1["retrieval"]["latency"]["p50_ms"] = 100.0
        r = test_client.post("/api/submit", json=p1)
        assert r.status_code == 200

        # Run 2: latency_p50_ms = 110.0 (increase of 10%, which is >5%)
        p2 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p2["retrieval"]["latency"]["p50_ms"] = 110.0
        r = test_client.post("/api/submit", json=p2)
        assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()

        entry = data[1]
        assert entry["regressions"]["latency_p50_ms"] is True, (
            f"Expected latency_p50_ms regression=True (100→110, increase 10%), "
            f"got {entry['regressions']['latency_p50_ms']}"
        )

    def test_regression_lower_is_better_not_triggered_on_small_increase(self, test_client):
        """When latency increases only 3%, regression flag is false."""
        server = "small-increase-test"
        # Run 1: latency_p50_ms = 100.0
        p1 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p1["retrieval"]["latency"]["p50_ms"] = 100.0
        r = test_client.post("/api/submit", json=p1)
        assert r.status_code == 200

        # Run 2: latency_p50_ms = 103.0 (increase of 3%, which is <=5%)
        p2 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p2["retrieval"]["latency"]["p50_ms"] = 103.0
        r = test_client.post("/api/submit", json=p2)
        assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()

        entry = data[1]
        assert entry["regressions"]["latency_p50_ms"] is False, (
            f"Expected latency_p50_ms regression=False (100→103, increase 3%), "
            f"got {entry['regressions']['latency_p50_ms']}"
        )

    def test_regression_lower_is_better_improvement_not_regression(self, test_client):
        """When latency decreases (improves), regression flag is false."""
        server = "latency-improve-test"
        # Run 1: latency_p50_ms = 100.0
        p1 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p1["retrieval"]["latency"]["p50_ms"] = 100.0
        r = test_client.post("/api/submit", json=p1)
        assert r.status_code == 200

        # Run 2: latency_p50_ms = 80.0 (improvement, decrease of 20%)
        p2 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p2["retrieval"]["latency"]["p50_ms"] = 80.0
        r = test_client.post("/api/submit", json=p2)
        assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()

        entry = data[1]
        assert entry["regressions"]["latency_p50_ms"] is False, (
            f"Expected latency_p50_ms regression=False (100→80, improvement), "
            f"got {entry['regressions']['latency_p50_ms']}"
        )

    def test_regression_at_exactly_5_percent_boundary(self, test_client):
        """At exactly 5% boundary, regression is NOT triggered (>5%, not >=5%)."""
        server = "boundary-test"
        # Run 1: composite_score = 100.0
        p1 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p1["composite_score"] = 100.0
        r = test_client.post("/api/submit", json=p1)
        assert r.status_code == 200

        # Run 2: composite_score = 95.0 (exactly 5% drop)
        p2 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p2["composite_score"] = 95.0
        r = test_client.post("/api/submit", json=p2)
        assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()

        entry = data[1]
        # (100-95)/100 = 0.05, which is NOT >0.05, so no regression
        assert entry["regressions"]["composite_score"] is False, (
            f"Expected composite_score regression=False at exact 5% boundary, "
            f"got {entry['regressions']['composite_score']}"
        )

    def test_regression_zero_prev_handled(self, test_client):
        """When previous value is 0, division is safe (no ZeroDivisionError)."""
        server = "zero-prev-test"
        # Run 1: composite_score = 0.0
        p1 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p1["composite_score"] = 0.0
        r = test_client.post("/api/submit", json=p1)
        assert r.status_code == 200

        # Run 2: composite_score = 0.0 (no change)
        p2 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p2["composite_score"] = 0.0
        r = test_client.post("/api/submit", json=p2)
        assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2

        # Should not crash and regression should be False when prev=0
        entry = data[1]
        assert "regressions" in entry
        assert entry["regressions"]["composite_score"] is False

    def test_regression_multiple_metrics(self, test_client):
        """Verify all 6 metrics have regression flags."""
        server = "all-metrics-test"
        # Run 1
        p1 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p1["composite_score"] = 100.0
        p1["retrieval"]["hit_at_5"] = 0.8
        p1["retrieval"]["symbol_hit_at_5"] = 0.7
        p1["retrieval"]["mrr"] = 0.6
        p1["retrieval"]["latency"]["p50_ms"] = 100.0
        p1["retrieval"]["tokens"]["avg"] = 500.0
        r = test_client.post("/api/submit", json=p1)
        assert r.status_code == 200

        # Run 2: all higher-is-better drop by 10%, lower-is-better increase by 10%
        p2 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        p2["composite_score"] = 90.0  # 10% drop → regression
        p2["retrieval"]["hit_at_5"] = 0.72  # 10% drop → regression
        p2["retrieval"]["symbol_hit_at_5"] = 0.63  # 10% drop → regression
        p2["retrieval"]["mrr"] = 0.54  # 10% drop → regression
        p2["retrieval"]["latency"]["p50_ms"] = 110.0  # 10% increase → regression
        p2["retrieval"]["tokens"]["avg"] = 550.0  # 10% increase → regression
        r = test_client.post("/api/submit", json=p2)
        assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()

        entry = data[1]
        reg = entry["regressions"]

        # All higher-is-better: regression=True
        for m in self.HIGHER_BETTER:
            assert reg[m] is True, (
                f"Expected {m} regression=True (10% drop), got {reg[m]}"
            )

        # All lower-is-better: regression=True
        for m in self.LOWER_BETTER:
            assert reg[m] is True, (
                f"Expected {m} regression=True (10% increase), got {reg[m]}"
            )

    def test_single_run_no_regressions(self, test_client):
        """Single-run server: entry has no regressions or all false."""
        server = "single-run-reg-test"
        p1 = _valid_payload(run_id=str(uuid.uuid4()), server_name=server)
        r = test_client.post("/api/submit", json=p1)
        assert r.status_code == 200

        r = test_client.get(f"/api/server/{server}/history")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1

        entry = data[0]
        if "regressions" in entry:
            for v in entry["regressions"].values():
                assert v is False, "Single run should not have regression=True"
