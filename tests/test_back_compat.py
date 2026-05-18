"""Back-compatibility verification tests.

These tests ensure the mission did not regress existing functionality:
  1. No preset files were modified (VAL-COMPAT-002).
  2. The number of surviving test cases has not decreased below the pre-mission
     baseline — i.e. no pre-existing tests were deleted (VAL-COMPAT-001).
  3. Old result JSON without chunk/query fields is accepted by POST /api/submit
     and rendered by the leaderboard (VAL-COMPAT-003).

The pre-mission baseline was 325 total tests with 66 tests in the three
known-broken files (test_bm25.py, test_naive_rag.py, test_cocoindex.py),
leaving 259 tests in the surviving files.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRE_MISSION_SURVIVING_TEST_COUNT = 259

# ANSI escape sequence pattern for stripping colour codes from pytest output.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (colour codes) from a string."""
    return _ANSI_RE.sub("", text)


def _parse_collected_count(output: str) -> int:
    """Parse the 'N tests collected' number from pytest --collect-only output."""
    plain = _strip_ansi(output)
    for line in plain.splitlines():
        stripped = line.strip()
        if "tests collected" in stripped:
            try:
                return int(stripped.split()[0])
            except (ValueError, IndexError):
                pass
    raise RuntimeError(
        f"Could not parse collected test count from pytest output:\n{plain}"
    )


def _make_legacy_payload() -> dict:
    """Build a minimal legacy payload missing chunk/query fields.

    This simulates a pre-chunk-era submission: no ``chunk_hit_at_5`` in
    retrieval, ``query_details`` with old-style keys (no ``found_chunk``,
    ``returned_contents``, ``response_tokens``).
    """
    return {
        "run_id": str(uuid.uuid4()),
        "bench_version": "0.1.0",
        "dataset_version": "1.0.0",
        "server": {
            "name": "legacy-test-server",
            "git_url": "https://github.com/legacy/server",
            "git_user": "legacy",
            "version": "1.0.0",
        },
        "environment": {"os": "linux", "python": "3.12"},
        "repos": ["flask"],
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
        "query_details": [
            {
                "id": "Q001",
                "type": "locate",
                "difficulty": "easy",
                "found_file": True,
                "found_symbol": True,
                "latency_ms": 120.0,
                "returned_files": ["src/flask/app.py"],
            },
            {
                "id": "Q002",
                "type": "callers",
                "difficulty": "medium",
                "found_file": False,
                "found_symbol": False,
                "latency_ms": 250.0,
                "returned_files": [],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_presets_unchanged():
    """VAL-COMPAT-002: git diff shows zero changes in rag_bench/presets/."""
    result = subprocess.run(
        ["git", "diff", "--quiet", "rag_bench/presets/"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"git diff --quiet rag_bench/presets/ exited {result.returncode} — "
        f"preset files have been modified. stderr: {result.stderr!r}"
    )


def test_surviving_test_count_not_decreased():
    """VAL-COMPAT-001: collection count in surviving test files >= baseline.

    Runs ``pytest --collect-only`` on all test files *except* the three
    known-broken ones (test_bm25, test_naive_rag, test_cocoindex) and asserts
    that the collected count is ≥ the pre-mission count of 259.  This proves
    that no pre-existing tests were deleted during the mission.
    """
    venv_pytest = str(REPO_ROOT / ".venv" / "bin" / "pytest")
    ignore_args = [
        "--ignore=tests/test_bm25.py",
        "--ignore=tests/test_cocoindex.py",
        "--ignore=tests/test_naive_rag.py",
    ]
    result = subprocess.run(
        [venv_pytest, "tests/", "--collect-only", "-q"] + ignore_args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    # pytest may put the summary on stdout or stderr; try both.
    combined = result.stdout + "\n" + result.stderr
    count = _parse_collected_count(combined)

    assert count >= PRE_MISSION_SURVIVING_TEST_COUNT, (
        f"Surviving test count {count} is below the pre-mission baseline "
        f"of {PRE_MISSION_SURVIVING_TEST_COUNT}. "
        f"Pre-existing tests may have been deleted."
    )


# ---------------------------------------------------------------------------
# VAL-COMPAT-003: Legacy JSON accepted by leaderboard
# ---------------------------------------------------------------------------

def test_legacy_payload_without_chunk_fields_accepted():
    """VAL-COMPAT-003: Payload missing chunk_hit_at_5 and with old-style
    query_details is accepted and rendered by the leaderboard.

    Uses the FastAPI TestClient with a fresh temp DB to verify:
    1. POST /api/submit returns 200
    2. GET /api/leaderboard includes the run with chunk_hit_at_5 = 0
    3. GET /api/run/{id} returns chunk_hit_at_5 = 0
    4. GET /api/run/{id}/queries returns the stored queries (accepted)
    """
    from fastapi.testclient import TestClient
    from server.app import app
    from server import db

    # Use a temp file for the DB to avoid interfering with real data
    orig_path = db.DB_PATH
    temp_path = orig_path.parent / "test_back_compat_legacy.db"
    try:
        db.DB_PATH = temp_path
        if temp_path.exists():
            temp_path.unlink()
        asyncio.run(db.init_db())

        client = TestClient(app)
        payload = _make_legacy_payload()

        # 1. Submit — must be accepted (200)
        response = client.post("/api/submit", json=payload)
        assert response.status_code == 200, (
            f"Legacy payload rejected with {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["status"] == "ok"
        assert data["run_id"] == payload["run_id"]

        # 2. Leaderboard — run appears with chunk_hit_at_5 = 0
        response = client.get("/api/leaderboard")
        assert response.status_code == 200
        lb = response.json()
        entries = lb["entries"]
        matching = [e for e in entries if e["run_id"] == payload["run_id"]]
        assert len(matching) == 1, (
            f"Legacy run not found in leaderboard. Entries: {[e.get('run_id') for e in entries]}"
        )
        entry = matching[0]
        assert entry["chunk_hit_at_5"] == 0.0, (
            f"Expected chunk_hit_at_5=0.0 for legacy payload, got {entry['chunk_hit_at_5']}"
        )

        # 3. GET /api/run/{id} — returns chunk_hit_at_5 = 0
        response = client.get(f"/api/run/{payload['run_id']}")
        assert response.status_code == 200
        run = response.json()
        assert run["chunk_hit_at_5"] == 0.0

        # 4. GET /api/run/{id}/queries — returns legacy queries (not 404/500)
        response = client.get(f"/api/run/{payload['run_id']}/queries")
        assert response.status_code == 200, (
            f"Legacy queries endpoint returned {response.status_code}: {response.text}"
        )
        queries = response.json()
        assert isinstance(queries, list)
        assert len(queries) == len(payload["query_details"]), (
            f"Expected {len(payload['query_details'])} queries, got {len(queries)}"
        )
        # Old-style queries lack found_chunk — endpoint should still return them
        for q in queries:
            assert "id" in q
            assert "type" in q

    finally:
        # Restore original path and clean up
        db.DB_PATH = orig_path
        if temp_path.exists():
            temp_path.unlink()


def test_legacy_payload_without_query_details_accepted():
    """VAL-COMPAT-003 extended: Payload with no query_details at all is accepted
    and the queries endpoint returns an empty array (200, not 404/500).
    """
    from fastapi.testclient import TestClient
    from server.app import app
    from server import db

    orig_path = db.DB_PATH
    temp_path = orig_path.parent / "test_back_compat_nodetails.db"
    try:
        db.DB_PATH = temp_path
        if temp_path.exists():
            temp_path.unlink()
        asyncio.run(db.init_db())

        client = TestClient(app)
        payload = _make_legacy_payload()
        del payload["query_details"]  # entirely missing

        # Submit — must be accepted
        response = client.post("/api/submit", json=payload)
        assert response.status_code == 200, (
            f"Legacy payload (no query_details) rejected: {response.status_code}"
        )

        # Queries endpoint — returns [] (200, not 404/500)
        response = client.get(f"/api/run/{payload['run_id']}/queries")
        assert response.status_code == 200, (
            f"Expected 200 for legacy run without query_details, got {response.status_code}"
        )
        queries = response.json()
        assert queries == [], (
            f"Expected empty array for legacy run without query_details, got {queries}"
        )

    finally:
        db.DB_PATH = orig_path
        if temp_path.exists():
            temp_path.unlink()
