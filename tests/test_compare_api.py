"""Tests for GET /api/compare endpoint — multi-run comparison with paired A/B stats.

Covers VAL-COMPARE-001 through VAL-COMPARE-005 and VAL-COMPARE-015.
"""
from __future__ import annotations

import asyncio
import json
import uuid

import aiosqlite
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.stats import chi2, wilcoxon

from rag_bench.stats import cliffs_delta, cohens_d
from server import db
from server.app import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_payload(
    run_id: str | None = None,
    server_name: str = "test-server",
    dataset_version: str = "v1",
    hit_at_5: float = 0.6,
    mrr: float = 0.5,
    query_latency_p50_ms: float = 100.0,
    query_details: list[dict] | None = None,
) -> dict:
    """Return a valid benchmark submission payload."""
    rid = run_id or str(uuid.uuid4())
    return {
        "run_id": rid,
        "bench_version": "0.1.0",
        "dataset_version": dataset_version,
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
            "total_queries": len(query_details) if query_details else 5,
            "total_hits": int(hit_at_5 * (len(query_details) if query_details else 5)),
            "hit_at_1": 0.2, "hit_at_3": 0.4, "hit_at_5": hit_at_5, "hit_at_10": 0.8,
            "symbol_hit_at_5": 0.3,
            "chunk_hit_at_5": 0.4,
            "mrr": mrr,
            "latency": {
                "p50_ms": query_latency_p50_ms, "p95_ms": 200,
                "p99_ms": 300, "mean_ms": 120,
            },
            "tokens": {"avg": 500, "p95": 900, "total": 2500},
        },
        "efficiency": {"avg_tool_calls": 2.0},
        "composite_score": 0.5,
        "by_difficulty": {"easy": {"count": 2}, "medium": {"count": 2}, "hard": {"count": 1}},
        "by_type": {"locate": {"count": 2}, "callers": {"count": 2}, "explain": {"count": 1}},
        "by_repo": {"test-repo": {"hit_at_5": hit_at_5}},
        "query_details": query_details or [
            {
                "id": f"Q{i:03d}",
                "type": ["locate", "callers", "explain", "locate", "callers"][i],
                "difficulty": ["easy", "easy", "medium", "medium", "hard"][i],
                "repo": "test-repo",
                "found_file": bool(i < int(hit_at_5 * 5)),
                "found_symbol": bool(i < 2),
                "found_chunk": bool(i < 2),
                "latency_ms": 50.0 + i * 30,
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
            {"hit_at_5": hit_at_5, "mrr": mrr},
            {"hit_at_5": hit_at_5 + 0.05, "mrr": mrr + 0.05},
        ],
        "iqr": {"hit_at_5": 0.1, "latency_ms": 15.0},
    }


def _make_paired_query_details(
    run_id: str,
    n_queries: int = 10,
    found_file_rule: str = "even",
    latency_base: float = 100.0,
    mrr_style: str = "first",
) -> list[dict]:
    """Build query_details with controlled outcomes for paired testing.

    Parameters
    ----------
    run_id : run identifier (used as prefix for query ids like 'runA-Q000')
    n_queries : number of queries
    found_file_rule : how to set found_file:
        'all' → all True
        'none' → all False
        'even' → True for even i
        'first_half' → True for i < n//2
    latency_base : base latency value (each query = base + i*10)
    mrr_style :
        'first' → returned_files[0] matches expected_files → MRR=1.0
        'none' → no match → MRR=0.0
    """
    qds = []
    for i in range(n_queries):
        if found_file_rule == "all":
            ff = True
        elif found_file_rule == "none":
            ff = False
        elif found_file_rule == "even":
            ff = (i % 2 == 0)
        elif found_file_rule == "first_half":
            ff = (i < n_queries // 2)
        else:
            ff = False

        if mrr_style == "first":
            returned_files = ["match.py", "other.py"]
            expected_files = ["match.py"]
        elif mrr_style == "none":
            returned_files = ["other.py"]
            expected_files = ["match.py"]
        else:
            returned_files = []
            expected_files = []

        qds.append({
            "id": f"{run_id}-Q{i:03d}",
            "type": "locate",
            "difficulty": "medium",
            "repo": "test-repo",
            "found_file": ff,
            "found_symbol": False,
            "found_chunk": ff,
            "latency_ms": latency_base + i * 10.0,
            "response_tokens": 200.0,
            "tool_calls": 1,
            "returned_files": returned_files,
            "returned_symbols": [],
            "expected_files": expected_files,
            "expected_symbols": [],
        })
    return qds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client():
    """Create a TestClient with a fresh DB."""
    db.DB_PATH = db.DB_PATH.parent / "test_compare_api.db"
    if db.DB_PATH.exists():
        db.DB_PATH.unlink()

    asyncio.run(db.init_db())

    client = TestClient(app)
    yield client

    if db.DB_PATH.exists():
        db.DB_PATH.unlink()


# ---------------------------------------------------------------------------
# VAL-COMPARE-001: Compare API returns 200 with required top-level schema
# ---------------------------------------------------------------------------

class TestCompareAPISchema:
    """VAL-COMPARE-001: GET /api/compare?run_ids=a,b returns 200 with
    runs, pairwise_ab, per_metric_table."""

    def test_compare_returns_200_with_two_runs(self, test_client):
        """GET /api/compare?run_ids=a,b returns HTTP 200."""
        # Submit two runs
        p1 = _valid_payload("run-a", "server-a")
        p2 = _valid_payload("run-b", "server-b")
        for p in [p1, p2]:
            r = test_client.post("/api/submit", json=p)
            assert r.status_code == 200

        r = test_client.get("/api/compare?run_ids=run-a,run-b")
        assert r.status_code == 200

    def test_compare_returns_required_top_level_keys(self, test_client):
        """Response has runs, pairwise_ab, per_metric_table."""
        p1 = _valid_payload("run-a", "server-a")
        p2 = _valid_payload("run-b", "server-b")
        for p in [p1, p2]:
            test_client.post("/api/submit", json=p)

        r = test_client.get("/api/compare?run_ids=run-a,run-b")
        data = r.json()
        assert "runs" in data, "Missing 'runs' key"
        assert "pairwise_ab" in data, "Missing 'pairwise_ab' key"
        assert "per_metric_table" in data, "Missing 'per_metric_table' key"

    def test_runs_length_matches_request(self, test_client):
        """len(runs) == number of run_ids in request."""
        p1 = _valid_payload("run-a", "server-a")
        p2 = _valid_payload("run-b", "server-b")
        for p in [p1, p2]:
            test_client.post("/api/submit", json=p)

        r = test_client.get("/api/compare?run_ids=run-a,run-b")
        data = r.json()
        assert len(data["runs"]) == 2

    def test_runs_preserves_request_order(self, test_client):
        """runs array preserves the order from run_ids query param."""
        p1 = _valid_payload("run-a", "server-a")
        p2 = _valid_payload("run-b", "server-b")
        p3 = _valid_payload("run-c", "server-c")
        for p in [p1, p2, p3]:
            test_client.post("/api/submit", json=p)

        r = test_client.get("/api/compare?run_ids=run-c,run-a,run-b")
        data = r.json()
        assert [run["run_id"] for run in data["runs"]] == ["run-c", "run-a", "run-b"]

    def test_runs_entry_has_required_fields(self, test_client):
        """Each run entry has run_id, server_name, and key metrics."""
        p1 = _valid_payload("run-a", "server-a")
        p2 = _valid_payload("run-b", "server-b")
        for p in [p1, p2]:
            test_client.post("/api/submit", json=p)

        r = test_client.get("/api/compare?run_ids=run-a,run-b")
        data = r.json()
        for run_entry in data["runs"]:
            assert "run_id" in run_entry
            assert "server_name" in run_entry
            assert "hit_at_5" in run_entry
            assert "mrr" in run_entry
            assert "query_latency_p50_ms" in run_entry

    def test_pairwise_ab_covers_all_unordered_pairs(self, test_client):
        """For N runs, pairwise_ab covers C(N,2) pairs."""
        p1 = _valid_payload("run-a", "server-a")
        p2 = _valid_payload("run-b", "server-b")
        p3 = _valid_payload("run-c", "server-c")
        for p in [p1, p2, p3]:
            test_client.post("/api/submit", json=p)

        r = test_client.get("/api/compare?run_ids=run-a,run-b,run-c")
        data = r.json()
        pairs = data["pairwise_ab"]
        assert len(pairs) == 3  # C(3,2) = 3

    def test_per_metric_table_structure(self, test_client):
        """per_metric_table has metrics list and runs rows."""
        p1 = _valid_payload("run-a", "server-a")
        p2 = _valid_payload("run-b", "server-b")
        for p in [p1, p2]:
            test_client.post("/api/submit", json=p)

        r = test_client.get("/api/compare?run_ids=run-a,run-b")
        data = r.json()
        pmt = data["per_metric_table"]
        assert "metrics" in pmt
        assert "runs" in pmt
        assert "hit_at_5" in pmt["metrics"]
        assert "mrr" in pmt["metrics"]
        assert "query_latency_p50_ms" in pmt["metrics"]
        assert len(pmt["runs"]) == 2

    def test_three_runs_pairwise_ab(self, test_client):
        """For 3 runs: pairwise_ab covers all 3 pairs."""
        for i, rid in enumerate(["run-a", "run-b", "run-c"]):
            test_client.post("/api/submit", json=_valid_payload(rid, f"server-{rid}"))

        r = test_client.get("/api/compare?run_ids=run-a,run-b,run-c")
        data = r.json()
        pairs = data["pairwise_ab"]
        # Verify all 3 pairs exist
        pair_keys = list(pairs.keys())
        assert len(pair_keys) == 3

        # Verify all requested IDs appear across the pairs
        all_ids_in_pairs = set()
        for pk in pair_keys:
            a, b = pk.split(",")
            all_ids_in_pairs.add(a)
            all_ids_in_pairs.add(b)
        assert all_ids_in_pairs == {"run-a", "run-b", "run-c"}


# ---------------------------------------------------------------------------
# VAL-COMPARE-002: Compare API returns 404 with missing run id
# ---------------------------------------------------------------------------

class TestCompareAPINotFound:
    """VAL-COMPARE-002: GET /api/compare?run_ids=missing,b returns 404."""

    def test_unknown_run_returns_404(self, test_client):
        """GET /api/compare?run_ids=missing,b returns 404."""
        p1 = _valid_payload("run-b", "server-b")
        test_client.post("/api/submit", json=p1)

        r = test_client.get("/api/compare?run_ids=does-not-exist-xyz,run-b")
        assert r.status_code == 404

    def test_404_detail_names_missing_run(self, test_client):
        """404 response detail field names the missing run."""
        p1 = _valid_payload("run-b", "server-b")
        test_client.post("/api/submit", json=p1)

        r = test_client.get("/api/compare?run_ids=does-not-exist-xyz,run-b")
        assert r.status_code == 404
        data = r.json()
        assert "detail" in data
        assert "does-not-exist-xyz" in data["detail"]

    def test_404_no_traceback(self, test_client):
        """404 response contains no Python traceback."""
        p1 = _valid_payload("run-b", "server-b")
        test_client.post("/api/submit", json=p1)

        r = test_client.get("/api/compare?run_ids=bad-run-id,run-b")
        body = r.text
        assert "Traceback" not in body
        assert 'File "' not in body

    def test_multiple_missing_names_first(self, test_client):
        """When the first run_id is missing, 404 names it."""
        p1 = _valid_payload("run-b", "server-b")
        test_client.post("/api/submit", json=p1)

        r = test_client.get("/api/compare?run_ids=missing-first,run-b")
        assert r.status_code == 404
        assert "missing-first" in r.json()["detail"]


# ---------------------------------------------------------------------------
# VAL-COMPARE-003: Compare API returns 400/422 when run_ids absent
# ---------------------------------------------------------------------------

class TestCompareAPIMissingParam:
    """VAL-COMPARE-003: GET /api/compare without run_ids returns 400 or 422."""

    def test_no_run_ids_returns_4xx(self, test_client):
        """GET /api/compare returns 4xx when run_ids is absent."""
        r = test_client.get("/api/compare")
        assert r.status_code in (400, 422)

    def test_no_run_ids_has_error_message(self, test_client):
        """Error response has descriptive detail."""
        r = test_client.get("/api/compare")
        assert r.status_code in (400, 422)
        data = r.json()
        assert "detail" in data

    def test_empty_run_ids_returns_4xx(self, test_client):
        """GET /api/compare?run_ids= returns 4xx."""
        r = test_client.get("/api/compare?run_ids=")
        assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# VAL-COMPARE-004: Compare API rejects single-run requests
# ---------------------------------------------------------------------------

class TestCompareAPISingleRun:
    """VAL-COMPARE-004: GET /api/compare?run_ids=single returns 400."""

    def test_single_run_returns_400(self, test_client):
        """GET /api/compare?run_ids=a returns 400 with >=2 message."""
        p1 = _valid_payload("run-a", "server-a")
        test_client.post("/api/submit", json=p1)

        r = test_client.get("/api/compare?run_ids=run-a")
        assert r.status_code == 400

    def test_single_run_message_mentions_two(self, test_client):
        """Error message says at least 2 run_ids required."""
        p1 = _valid_payload("run-a", "server-a")
        test_client.post("/api/submit", json=p1)

        r = test_client.get("/api/compare?run_ids=run-a")
        assert r.status_code == 400
        detail = r.json()["detail"]
        # Should mention 2 or "at least" or similar
        assert "2" in detail.lower() or "two" in detail.lower() or "least" in detail.lower()

    def test_duplicate_run_ids_returns_400(self, test_client):
        """GET /api/compare?run_ids=a,a returns 400 (duplicate ids)."""
        p1 = _valid_payload("run-a", "server-a")
        test_client.post("/api/submit", json=p1)

        r = test_client.get("/api/compare?run_ids=run-a,run-a")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# VAL-COMPARE-005: Pairwise A/B has all required statistical fields
# ---------------------------------------------------------------------------

class TestPairwiseABFields:
    """VAL-COMPARE-005: Each pair × each metric has delta, delta_ci_lo,
    delta_ci_hi, p_value, cohens_d, cliffs_delta."""

    REQUIRED_METRICS = ["hit_at_5", "mrr", "query_latency_p50_ms"]
    REQUIRED_FIELDS = [
        "delta", "delta_ci_lo", "delta_ci_hi",
        "p_value", "cohens_d", "cliffs_delta",
    ]

    def test_all_required_metrics_and_fields_present(self, test_client):
        """Every pair has every required metric with all 6 fields."""
        p1 = _valid_payload("run-a", "server-a")
        p2 = _valid_payload("run-b", "server-b")
        for p in [p1, p2]:
            test_client.post("/api/submit", json=p)

        r = test_client.get("/api/compare?run_ids=run-a,run-b")
        data = r.json()
        pairs = data["pairwise_ab"]

        for pair_key, pair_data in pairs.items():
            for metric in self.REQUIRED_METRICS:
                assert metric in pair_data, (
                    f"Metric '{metric}' missing from pair '{pair_key}'"
                )
                mdata = pair_data[metric]
                for field in self.REQUIRED_FIELDS:
                    assert field in mdata, (
                        f"Field '{field}' missing from {pair_key}/{metric}"
                    )

    def test_ci_brackets_delta(self, test_client):
        """For finite values, delta_ci_lo <= delta <= delta_ci_hi."""
        p1 = _valid_payload("run-a", "server-a")
        p2 = _valid_payload("run-b", "server-b")
        for p in [p1, p2]:
            test_client.post("/api/submit", json=p)

        r = test_client.get("/api/compare?run_ids=run-a,run-b")
        data = r.json()

        for pair_key, pair_data in data["pairwise_ab"].items():
            for metric in self.REQUIRED_METRICS:
                mdata = pair_data[metric]
                d, lo, hi = mdata["delta"], mdata["delta_ci_lo"], mdata["delta_ci_hi"]
                if all(v is not None and not (isinstance(v, float) and (v != v))
                       for v in (d, lo, hi)):
                    assert lo <= d <= hi, (
                        f"{pair_key}/{metric}: ci_lo({lo}) <= delta({d}) <= ci_hi({hi})"
                    )

    def test_multiple_pairs_each_have_fields(self, test_client):
        """For 3 runs, all 3 pairs each have all fields."""
        for i, rid in enumerate(["run-a", "run-b", "run-c"]):
            test_client.post(
                "/api/submit",
                json=_valid_payload(rid, f"server-{rid}"),
            )

        r = test_client.get("/api/compare?run_ids=run-a,run-b,run-c")
        data = r.json()
        pairs = data["pairwise_ab"]

        assert len(pairs) == 3
        for pair_key, pair_data in pairs.items():
            for metric in self.REQUIRED_METRICS:
                assert metric in pair_data
                for field in self.REQUIRED_FIELDS:
                    assert field in pair_data[metric]


# ---------------------------------------------------------------------------
# VAL-COMPARE-015: P-value matches scipy reference
# ---------------------------------------------------------------------------

class TestComparePValueReference:
    """VAL-COMPARE-015: API p-values match independent scipy computations."""

    def test_mcnemar_pvalue_for_hit_at_5(self, test_client):
        """p_value for hit_at_5 matches scipy McNemar reference."""
        # Run A: hits on queries 0-7 (8 out of 10)
        # Run B: hits on queries 0-5 (6 out of 10)
        qds_a = _make_paired_query_details("runA", n_queries=10,
                                           found_file_rule="first_half",
                                           latency_base=100.0, mrr_style="first")
        # Run A: first 5 queries found (0-4)
        # Wait, first_half means i < n//2 = i < 5, so queries 0-4 found
        # Let me use explicit control
        qds_a = []
        qds_b = []
        for i in range(10):
            # A: hits on i < 8 (queries 0-7)
            # B: hits on i < 6 (queries 0-5)
            found_a = (i < 8)
            found_b = (i < 6)

            def _make_q(found):
                return {
                    "id": f"shared-Q{i:03d}",
                    "type": "locate",
                    "difficulty": "medium",
                    "repo": "test-repo",
                    "found_file": found,
                    "found_symbol": False,
                    "found_chunk": found,
                    "latency_ms": 100.0 + i * 10,
                    "response_tokens": 200.0,
                    "tool_calls": 1,
                    "returned_files": ["match.py"] if found else ["other.py"],
                    "returned_symbols": [],
                    "expected_files": ["match.py"],
                    "expected_symbols": [],
                }
            qds_a.append(_make_q(found_a))
            qds_b.append(_make_q(found_b))

        # Build contingency table:
        #   b = (A hit, B miss): found_a=True, found_b=False → i in [6,7]
        b = sum(1 for i in range(10) if (i < 8) and not (i < 6))  # 2 queries (6,7)
        #   c = (A miss, B hit): found_a=False, found_b=True → i in [8,9]? no, B only hits 0-5
        c = sum(1 for i in range(10) if not (i < 8) and (i < 6))  # 0

        # For b=2, c=0, McNemar's stat = (|2-0|-1)^2 / 2 = 1/2 = 0.5
        # p = 1 - chi2.cdf(0.5, 1)
        from scipy.stats import chi2 as _chi2
        if b + c > 0:
            expected_p = float(1.0 - _chi2.cdf((abs(b - c) - 1.0) ** 2 / (b + c), 1))
        else:
            expected_p = 1.0

        p_a = {
            **_valid_payload("run-compare-a", "server-a"),
            "query_details": qds_a,
            "retrieval": {
                **_valid_payload()["retrieval"],
                "total_queries": 10,
                "total_hits": 8,
                "hit_at_5": 0.8,
            },
        }
        p_b = {
            **_valid_payload("run-compare-b", "server-b"),
            "query_details": qds_b,
            "retrieval": {
                **_valid_payload()["retrieval"],
                "total_queries": 10,
                "total_hits": 6,
                "hit_at_5": 0.6,
            },
        }

        for p in [p_a, p_b]:
            r = test_client.post("/api/submit", json=p)
            assert r.status_code == 200

        r = test_client.get("/api/compare?run_ids=run-compare-a,run-compare-b")
        assert r.status_code == 200
        data = r.json()

        pair_key = "run-compare-a,run-compare-b"
        api_p = data["pairwise_ab"][pair_key]["hit_at_5"]["p_value"]
        assert api_p is not None
        assert abs(api_p - expected_p) < 1e-6, (
            f"McNemar p-value: API={api_p}, expected={expected_p}"
        )

    def test_wilcoxon_pvalue_for_latency(self, test_client):
        """p_value for query_latency_p50_ms matches scipy Wilcoxon reference."""
        # Build paired latency arrays with a known difference
        np.random.seed(42)
        base_lat = np.random.normal(100, 10, 10).tolist()
        new_lat = [v + np.random.normal(5, 3) for v in base_lat]  # slightly higher

        qds_a = []
        qds_b = []
        for i in range(10):
            def _make_q(lat):
                return {
                    "id": f"shared-Q{i:03d}",
                    "type": "locate",
                    "difficulty": "medium",
                    "repo": "test-repo",
                    "found_file": True,
                    "found_symbol": False,
                    "found_chunk": True,
                    "latency_ms": lat,
                    "response_tokens": 200.0,
                    "tool_calls": 1,
                    "returned_files": ["match.py"],
                    "returned_symbols": [],
                    "expected_files": ["match.py"],
                    "expected_symbols": [],
                }
            qds_a.append(_make_q(base_lat[i]))
            qds_b.append(_make_q(new_lat[i]))

        # Independent reference
        expected_p = float(wilcoxon(base_lat, new_lat, zero_method="wilcox",
                                    alternative="two-sided").pvalue)

        p_a = {
            **_valid_payload("run-lat-a", "server-a"),
            "query_details": qds_a,
        }
        p_b = {
            **_valid_payload("run-lat-b", "server-b"),
            "query_details": qds_b,
        }

        for p in [p_a, p_b]:
            r = test_client.post("/api/submit", json=p)
            assert r.status_code == 200

        r = test_client.get("/api/compare?run_ids=run-lat-a,run-lat-b")
        assert r.status_code == 200
        data = r.json()

        pair_key = "run-lat-a,run-lat-b"
        api_p = data["pairwise_ab"][pair_key]["query_latency_p50_ms"]["p_value"]
        assert api_p is not None
        assert abs(api_p - expected_p) < 1e-6, (
            f"Wilcoxon p-value: API={api_p}, expected={expected_p}"
        )

    def test_wilcoxon_pvalue_for_mrr(self, test_client):
        """p_value for mrr matches scipy Wilcoxon reference."""
        # Build paired MRR with known values
        # Run A: MRR=1.0 for first 5, 0.0 for last 5
        # Run B: MRR=0.5 for all (returned_files[1] matches)
        qds_a = []
        qds_b = []
        for i in range(10):
            qds_a.append({
                "id": f"MRR-Q{i:03d}",
                "type": "locate",
                "difficulty": "medium",
                "repo": "test-repo",
                "found_file": True,
                "found_symbol": False,
                "found_chunk": True,
                "latency_ms": 100.0,
                "response_tokens": 200.0,
                "tool_calls": 1,
                "returned_files": (["match.py"] if i < 5 else ["other.py", "match.py"]),
                "returned_symbols": [],
                "expected_files": ["match.py"],
                "expected_symbols": [],
            })
            qds_b.append({
                "id": f"MRR-Q{i:03d}",
                "type": "locate",
                "difficulty": "medium",
                "repo": "test-repo",
                "found_file": True,
                "found_symbol": False,
                "found_chunk": True,
                "latency_ms": 100.0,
                "response_tokens": 200.0,
                "tool_calls": 1,
                "returned_files": ["other.py", "match.py"],
                "returned_symbols": [],
                "expected_files": ["match.py"],
                "expected_symbols": [],
            })

        # MRR values: A = [1.0]*5 + [0.5]*5, B = [0.5]*10
        mrr_a = [1.0] * 5 + [0.5] * 5
        mrr_b = [0.5] * 10
        expected_p = float(wilcoxon(mrr_a, mrr_b, zero_method="wilcox",
                                    alternative="two-sided").pvalue)

        p_a = {
            **_valid_payload("run-mrr-a", "server-a"),
            "query_details": qds_a,
        }
        p_b = {
            **_valid_payload("run-mrr-b", "server-b"),
            "query_details": qds_b,
        }

        for p in [p_a, p_b]:
            r = test_client.post("/api/submit", json=p)
            assert r.status_code == 200

        r = test_client.get("/api/compare?run_ids=run-mrr-a,run-mrr-b")
        assert r.status_code == 200
        data = r.json()

        pair_key = "run-mrr-a,run-mrr-b"
        api_p = data["pairwise_ab"][pair_key]["mrr"]["p_value"]
        assert api_p is not None
        assert abs(api_p - expected_p) < 1e-6, (
            f"Wilcoxon MRR p-value: API={api_p}, expected={expected_p}"
        )

    def test_delta_ci_coverage(self, test_client):
        """Delta CI is reported and brackets the delta for known pairs."""
        p1 = _valid_payload("run-ci-a", "server-a")
        p2 = _valid_payload("run-ci-b", "server-b")
        for p in [p1, p2]:
            test_client.post("/api/submit", json=p)

        r = test_client.get("/api/compare?run_ids=run-ci-a,run-ci-b")
        assert r.status_code == 200
        data = r.json()

        pair_key = "run-ci-a,run-ci-b"
        for metric in ["hit_at_5", "mrr", "query_latency_p50_ms"]:
            mdata = data["pairwise_ab"][pair_key][metric]
            d, lo, hi = mdata["delta"], mdata["delta_ci_lo"], mdata["delta_ci_hi"]
            if all(v is not None and not (isinstance(v, float) and (v != v))
                   for v in (d, lo, hi)):
                assert lo <= d <= hi, f"{metric}: CI [{lo}, {hi}] does not bracket delta {d}"
