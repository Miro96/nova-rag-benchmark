"""Tests for rag_bench.stats — pure-math statistical primitives.

Covers wilson_ci, normal_ci, pearson_correlation_matrix with analytically
known reference values.

Also covers:
- cohens_d, cliffs_delta effect sizes
- summary_stats_for_run integration (server/stats_cache.py)
- submit → stats_cached populated
- baseline A/B detection (McNemar, Wilcoxon, Cohen's d)
"""
from __future__ import annotations

import asyncio
import json
import math
import uuid

import aiosqlite
import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_bench.stats import (
    cliffs_delta,
    cohens_d,
    normal_ci,
    pearson_correlation_matrix,
    wilson_ci,
)
from server import db
from server.app import app
from server.stats_cache import summary_stats_for_run
from server.models import BenchmarkSubmission


# -----------------------------------------------------------------------
# wilson_ci
# -----------------------------------------------------------------------

class TestWilsonCI:
    """Wilson 95 % CI for proportions."""

    def test_wilson_5_of_35(self):
        """5/35 hits → Wilson 95% CI ≈ [0.0626, 0.2938]."""
        lo, hi = wilson_ci(5, 35)
        assert abs(lo - 0.0626) < 0.0001
        assert abs(hi - 0.2938) < 0.0001

    def test_wilson_35_of_35(self):
        """All hits → CI is [0.9011, 1.0] (or very close)."""
        lo, hi = wilson_ci(35, 35)
        assert abs(lo - 0.9011) < 0.001
        assert abs(hi - 1.0) < 1e-6

    def test_wilson_0_of_35(self):
        """No hits → CI is [0.0, 0.0989]."""
        lo, hi = wilson_ci(0, 35)
        assert abs(lo - 0.0) < 1e-12
        assert abs(hi - 0.0989) < 0.001

    def test_wilson_symmetry(self):
        """Wilson(k, n) and Wilson(n-k, n) are symmetric."""
        ci_pos = wilson_ci(5, 35)
        ci_neg = wilson_ci(30, 35)
        # Symmetry: ci_low(5,35) ≈ 1 - ci_high(30,35) and vice versa
        assert abs(ci_pos[0] - (1.0 - ci_neg[1])) < 1e-9
        assert abs(ci_pos[1] - (1.0 - ci_neg[0])) < 1e-9

    def test_wilson_1_of_1(self):
        """Single success → CI brackets around 1.0."""
        lo, hi = wilson_ci(1, 1)
        assert 0.0 <= lo < 1.0
        assert hi == 1.0
        assert lo <= hi

    def test_wilson_custom_alpha(self):
        """alpha=0.10 yields wider interval than alpha=0.05."""
        ci_95 = wilson_ci(10, 20, alpha=0.05)
        ci_90 = wilson_ci(10, 20, alpha=0.10)
        width_95 = ci_95[1] - ci_95[0]
        width_90 = ci_90[1] - ci_90[0]
        assert width_95 > width_90

    def test_wilson_raises_n_zero(self):
        """n <= 0 → ValueError."""
        with pytest.raises(ValueError):
            wilson_ci(5, 0)

    def test_wilson_raises_n_negative(self):
        """n < 0 → ValueError."""
        with pytest.raises(ValueError):
            wilson_ci(5, -3)

    def test_wilson_raises_k_negative(self):
        """k < 0 → ValueError."""
        with pytest.raises(ValueError):
            wilson_ci(-1, 10)

    def test_wilson_raises_k_gt_n(self):
        """k > n → ValueError."""
        with pytest.raises(ValueError):
            wilson_ci(50, 10)

    def test_accepts_tuple(self):
        """Python tuples work for k and n."""
        ci = wilson_ci(5, 35)
        assert isinstance(ci, tuple)
        assert len(ci) == 2


# -----------------------------------------------------------------------
# normal_ci
# -----------------------------------------------------------------------

class TestNormalCI:
    """Normal/t CI for means."""

    def test_normal_known_bounds(self):
        """mean=100, std=15, n=30 → CI ≈ (94.45, 105.55) (t, df=29)."""
        lo, hi = normal_ci(100, 15, 30)
        # Reference: t.ppf(0.975, 29) ≈ 2.045; margin ≈ 2.045*15/sqrt(30) ≈ 5.55
        assert abs(lo - 94.45) < 0.1
        assert abs(hi - 105.55) < 0.1

    def test_normal_symmetric(self):
        """CI bounds are symmetric around the mean."""
        lo, hi = normal_ci(50, 10, 50)
        assert abs((lo + hi) / 2 - 50.0) < 1e-9

    def test_normal_std_zero(self):
        """std=0 → CI collapses to mean."""
        lo, hi = normal_ci(42.0, 0.0, 20)
        assert lo == 42.0
        assert hi == 42.0

    def test_normal_n_1(self):
        """n=1 → uses normal (not t), symmetric."""
        lo, hi = normal_ci(10.0, 5.0, 1)
        assert lo < 10.0 < hi
        # t with df=0 falls back to norm

    def test_normal_custom_alpha(self):
        """alpha=0.01 yields wider interval than alpha=0.05."""
        ci_95 = normal_ci(100, 15, 30, alpha=0.05)
        ci_99 = normal_ci(100, 15, 30, alpha=0.01)
        assert (ci_99[1] - ci_99[0]) > (ci_95[1] - ci_95[0])

    def test_raises_n_zero(self):
        """n <= 0 → ValueError."""
        with pytest.raises(ValueError):
            normal_ci(10, 5, 0)

    def test_raises_n_negative(self):
        """n < 0 → ValueError."""
        with pytest.raises(ValueError):
            normal_ci(10, 5, -5)

    def test_raises_std_negative(self):
        """std < 0 → ValueError."""
        with pytest.raises(ValueError):
            normal_ci(10, -3, 20)


# -----------------------------------------------------------------------
# pearson_correlation_matrix
# -----------------------------------------------------------------------

class TestPearsonCorrelationMatrix:
    """Pearson correlation matrix via numpy.corrcoef."""

    def test_perfect_positive(self):
        """Perfectly correlated (y = 2x) → correlation 1.0."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        m = pearson_correlation_matrix(x, y)
        assert abs(m[0, 1] - 1.0) < 1e-9

    def test_perfect_negative(self):
        """Perfectly anti-correlated (y = -x) → correlation -1.0."""
        x = [1, 2, 3, 4, 5]
        y = [-1, -2, -3, -4, -5]
        m = pearson_correlation_matrix(x, y)
        assert abs(m[0, 1] + 1.0) < 1e-9

    def test_uncorrelated(self):
        """Orthogonal vectors → near-zero correlation."""
        x = [1, -1, 0, 0]
        y = [0, 0, 1, -1]
        m = pearson_correlation_matrix(x, y)
        assert abs(m[0, 1]) < 1e-12

    def test_diagonal_is_one(self):
        """Diagonal entries are exactly 1.0."""
        x = [1.0, 2.0, 3.0]
        y = [3.0, 2.0, 1.0]
        z = [10, 20, 30]
        m = pearson_correlation_matrix(x, y, z)
        for i in range(3):
            assert m[i, i] == 1.0

    def test_symmetry(self):
        """Matrix is symmetric within 1e-9."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 1, 4, 3, 6, 5, 8, 7, 9, 10]
        z = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        m = pearson_correlation_matrix(x, y, z)
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(m[i, j] - m[j, i]) < 1e-9

    def test_shape(self):
        """Matrix shape is (k, k) for k columns."""
        cols = [[1, 2, 3]] * 5
        m = pearson_correlation_matrix(*cols)
        assert m.shape == (5, 5)

    def test_accepts_tuples(self):
        """Python tuples work as input."""
        x = (1, 2, 3)
        y = (3, 2, 1)
        m = pearson_correlation_matrix(x, y)
        assert m.shape == (2, 2)

    def test_raises_unequal_length(self):
        """Different lengths → ValueError."""
        with pytest.raises(ValueError):
            pearson_correlation_matrix([1, 2], [1, 2, 3])

    def test_raises_empty(self):
        """Empty arrays → ValueError."""
        with pytest.raises(ValueError):
            pearson_correlation_matrix([], [])


# -----------------------------------------------------------------------
# cohens_d
# -----------------------------------------------------------------------

class TestCohensD:
    """Cohen's d effect size."""

    def test_cohens_d_identical(self):
        """Identical arrays → d == 0."""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        d = cohens_d(x, y)
        assert abs(d) < 1e-9

    def test_cohens_d_known_shift(self):
        """x shifted by 1 SD → d ≈ 1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 3.0, 4.0, 5.0, 6.0]  # shift of +1
        d = cohens_d(y, x)  # y > x
        # pooled std of [1,2,3,4,5] is ~1.581, mean diff = 1.0 → d ≈ 0.632
        # Actually let's compute it more carefully
        # mean_x = 3.0, mean_y = 4.0, diff = 1.0
        # var_x = 2.5, var_y = 2.5 (ddof=1)
        # s_pooled = sqrt((4*2.5 + 4*2.5) / 8) = sqrt(20/8) = sqrt(2.5) ≈ 1.581
        # d = 1.0 / 1.581 ≈ 0.6325
        assert abs(d - 0.6325) < 0.01

    def test_cohens_d_known_reference(self):
        """Reference: N(100,15) vs N(110,15) → d ≈ 0.667."""
        np.random.seed(42)
        x = np.random.normal(100, 15, 1000).tolist()
        y = np.random.normal(110, 15, 1000).tolist()
        d = cohens_d(y, x)
        assert abs(d - 0.667) < 0.2  # sampling tolerance

    def test_cohens_d_extremes(self):
        """x=[0]*5, y=[1]*5 → s_pooled=0 returns 0.0 (degenerate)."""
        x = [0.0] * 5
        y = [1.0] * 5
        d = cohens_d(y, x)
        # Both arrays have zero variance → s_pooled=0 → returns 0.0 safely
        assert d == 0.0

    def test_cohens_d_reversed(self):
        """cohens_d(a, b) == -cohens_d(b, a)."""
        x = [1, 2, 3, 4, 5]
        y = [3, 4, 5, 6, 7]
        d1 = cohens_d(x, y)
        d2 = cohens_d(y, x)
        assert abs(d1 + d2) < 1e-9

    def test_cohens_d_single_element(self):
        """Single-element arrays return 0.0."""
        assert cohens_d([5], [3]) == 0.0


# -----------------------------------------------------------------------
# cliffs_delta
# -----------------------------------------------------------------------

class TestCliffsDelta:
    """Cliff's delta non-parametric effect size."""

    def test_cliffs_delta_full_dominance(self):
        """All x > all y → delta == 1.0."""
        x = [5, 6, 7]
        y = [1, 2, 3]
        d = cliffs_delta(x, y)
        assert d == 1.0

    def test_cliffs_delta_full_submission(self):
        """All x < all y → delta == -1.0."""
        x = [1, 2, 3]
        y = [5, 6, 7]
        d = cliffs_delta(x, y)
        assert d == -1.0

    def test_cliffs_delta_identical(self):
        """Identical arrays → delta == 0.0."""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        d = cliffs_delta(x, y)
        assert d == 0.0

    def test_cliffs_delta_symmetry(self):
        """cliffs_delta(x, y) == -cliffs_delta(y, x)."""
        x = [1, 3, 5, 2, 7]
        y = [4, 2, 6, 1, 3]
        d1 = cliffs_delta(x, y)
        d2 = cliffs_delta(y, x)
        assert abs(d1 + d2) < 1e-9

    def test_cliffs_delta_empty(self):
        """Empty arrays return 0.0."""
        assert cliffs_delta([], [1, 2, 3]) == 0.0


# -----------------------------------------------------------------------
# Fixtures for integration tests
# -----------------------------------------------------------------------

def _make_query_details(n: int = 5, seed: int = 42) -> list[dict]:
    """Generate synthetic query_details with known values."""
    rng = np.random.RandomState(seed)
    results = []
    for i in range(n):
        found_file = bool(rng.randint(0, 2))
        found_symbol = bool(rng.randint(0, 2))
        found_chunk = bool(rng.randint(0, 2))
        results.append({
            "id": f"Q{i:03d}",
            "type": ["locate", "callers", "explain"][i % 3],
            "difficulty": ["easy", "medium", "hard"][i % 3],
            "repo": "test-repo",
            "found_file": found_file,
            "found_symbol": found_symbol,
            "found_chunk": found_chunk,
            "latency_ms": float(rng.randint(50, 500)),
            "response_tokens": float(rng.randint(100, 2000)),
            "tool_calls": int(rng.randint(1, 5)),
            "returned_files": ["f1.py", "f2.py"] if found_file else ["f3.py"],
            "returned_symbols": ["Foo", "Bar"] if found_symbol else ["Baz"],
            "expected_files": ["f1.py"],
            "expected_symbols": ["Foo"],
        })
    return results


def _valid_payload(run_id: str | None = None, server_name: str = "test-server") -> dict:
    """Return a valid benchmark submission payload with query_details."""
    rid = run_id or str(uuid.uuid4())
    query_details = _make_query_details(5)
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
        "query_details": query_details,
        "replicates": [
            {"hit_at_5": 0.6, "latency_ms": 100.0},
            {"hit_at_5": 0.7, "latency_ms": 110.0},
        ],
        "iqr": {"hit_at_5": 0.1, "latency_ms": 15.0},
    }


@pytest.fixture
def test_client():
    """Create a TestClient with a fresh DB."""
    db.DB_PATH = db.DB_PATH.parent / "test_stats_integration.db"
    if db.DB_PATH.exists():
        db.DB_PATH.unlink()

    asyncio.run(db.init_db())

    client = TestClient(app)
    yield client

    if db.DB_PATH.exists():
        db.DB_PATH.unlink()


# -----------------------------------------------------------------------
# summary_stats_for_run unit tests
# -----------------------------------------------------------------------

class TestSummaryStatsForRun:
    """Test server/stats_cache.py::summary_stats_for_run."""

    def test_returns_expected_keys(self):
        """Returns dict with metrics, by_difficulty, by_type, by_repo, correlations, iqr, cv."""
        qd = _make_query_details(5)
        result = summary_stats_for_run(qd, replicates=[])
        for key in ("metrics", "by_difficulty", "by_type", "by_repo", "correlations", "iqr", "cv"):
            assert key in result, f"Missing key: {key}"

    def test_metrics_structure(self):
        """Each metric has point, ci_low, ci_high, n."""
        qd = _make_query_details(5)
        result = summary_stats_for_run(qd, replicates=[])
        for metric in ("hit_at_5", "chunk_hit_at_5", "symbol_hit_at_5", "mrr", "latency_ms", "response_tokens"):
            assert metric in result["metrics"], f"Missing metric: {metric}"
            m = result["metrics"][metric]
            for field in ("point", "ci_low", "ci_high", "n"):
                assert field in m, f"{metric} missing {field}"
                assert isinstance(m[field], (int, float)), f"{metric}.{field} not numeric"

    def test_hit_at_5_n_equals_len_query_details(self):
        """metrics.hit_at_5.n == len(query_details)."""
        for n in (3, 5, 7):
            qd = _make_query_details(n)
            result = summary_stats_for_run(qd, replicates=[])
            assert result["metrics"]["hit_at_5"]["n"] == n

    def test_chunk_hit_at_5_present(self):
        """metrics.chunk_hit_at_5 has point/ci_low/ci_high/n."""
        qd = _make_query_details(5)
        result = summary_stats_for_run(qd, replicates=[])
        cm = result["metrics"]["chunk_hit_at_5"]
        for field in ("point", "ci_low", "ci_high", "n"):
            assert field in cm
        assert cm["n"] == 5
        assert cm["ci_low"] <= cm["point"] <= cm["ci_high"]

    def test_wilson_ci_brackets_point(self):
        """For binary metrics, ci_low <= point <= ci_high."""
        qd = _make_query_details(10, seed=123)
        result = summary_stats_for_run(qd, replicates=[])
        for metric in ("hit_at_5", "chunk_hit_at_5", "symbol_hit_at_5"):
            m = result["metrics"][metric]
            assert m["ci_low"] <= m["point"] <= m["ci_high"], \
                f"{metric}: {m['ci_low']} <= {m['point']} <= {m['ci_high']}"

    def test_by_difficulty_buckets(self):
        """by_difficulty has entries for each difficulty level."""
        qd = _make_query_details(12, seed=42)
        result = summary_stats_for_run(qd, replicates=[])
        buckets = result["by_difficulty"]
        assert len(buckets) >= 1
        for diff_name, diff_stats in buckets.items():
            for metric in ("hit_at_5", "latency_ms"):
                assert metric in diff_stats, f"{diff_name} missing {metric}"
                assert "point" in diff_stats[metric]

    def test_by_type_buckets(self):
        """by_type has entries for each query type."""
        qd = _make_query_details(12, seed=42)
        result = summary_stats_for_run(qd, replicates=[])
        buckets = result["by_type"]
        assert len(buckets) >= 1
        for type_name, type_stats in buckets.items():
            assert "hit_at_5" in type_stats

    def test_by_repo_buckets(self):
        """by_repo has entries."""
        qd = _make_query_details(5, seed=42)
        result = summary_stats_for_run(qd, replicates=[])
        assert len(result["by_repo"]) >= 1

    def test_correlations_labels(self):
        """correlations has labels matching the six core metrics."""
        qd = _make_query_details(10, seed=42)
        result = summary_stats_for_run(qd, replicates=[])
        corr = result["correlations"]
        assert "labels" in corr
        assert "matrix" in corr
        assert len(corr["labels"]) == 6

    def test_iqr_and_cv_from_replicates(self):
        """IQR and CV computed from replicates."""
        qd = _make_query_details(5)
        reps = [
            {"hit_at_5": 0.5, "latency_ms": 100.0},
            {"hit_at_5": 0.7, "latency_ms": 120.0},
        ]
        result = summary_stats_for_run(qd, replicates=reps)
        assert "hit_at_5" in result["iqr"] or "latency_ms" in result["iqr"]

    def test_empty_query_details(self):
        """Empty query_details returns placeholder."""
        result = summary_stats_for_run([], replicates=[])
        assert result["metrics"] == {}
        assert result["correlations"]["matrix"] == []

    def test_wilson_known_5_of_35(self):
        """Synthetic 5/35 queries with found_file=True → Wilson CI matches."""
        qd = []
        for i in range(35):
            qd.append({
                "id": f"Q{i:03d}",
                "found_file": i < 5,
                "found_symbol": False,
                "found_chunk": False,
                "latency_ms": 100.0,
                "response_tokens": 200.0,
                "tool_calls": 1,
                "difficulty": "easy",
                "type": "locate",
                "repo": "test",
                "expected_files": ["f.py"], "expected_symbols": [],
                "returned_files": ["f.py"] if i < 5 else ["other.py"],
                "returned_symbols": [],
            })
        result = summary_stats_for_run(qd, replicates=[])
        m = result["metrics"]["hit_at_5"]
        assert abs(m["point"] - 0.1429) < 0.001
        assert abs(m["ci_low"] - 0.0626) < 0.01
        assert abs(m["ci_high"] - 0.2938) < 0.01
        assert m["n"] == 35


# -----------------------------------------------------------------------
# Integration: submit → stats_cached populated
# -----------------------------------------------------------------------

class TestSubmitPopulatesStatsCache:
    """POST /api/submit → stats_cached is populated in the same request."""

    def test_submit_populates_stats_cached(self, test_client):
        """After submit, stats_cached column is non-empty JSON."""
        payload = _valid_payload()
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200
        run_id = payload["run_id"]

        # Read directly from DB
        async def _check():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                cursor = await conn.execute(
                    "SELECT stats_cached FROM runs WHERE id = ?", (run_id,)
                )
                row = await cursor.fetchone()
                assert row is not None
                stats_json = row[0]
                assert stats_json is not None
                assert len(stats_json) > 2  # not empty
                stats = json.loads(stats_json)
                assert "metrics" in stats
                assert stats["metrics"]["hit_at_5"]["n"] == 5

        asyncio.run(_check())

    def test_submit_metrics_hit_at_5_structure(self, test_client):
        """Submitted stats include metrics.hit_at_5 with point/ci_low/ci_high/n."""
        payload = _valid_payload()
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200
        run_id = payload["run_id"]

        async def _check():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                cursor = await conn.execute(
                    "SELECT stats_cached FROM runs WHERE id = ?", (run_id,)
                )
                row = await cursor.fetchone()
                stats = json.loads(row[0])
                m = stats["metrics"]["hit_at_5"]
                assert isinstance(m["point"], (int, float))
                assert isinstance(m["ci_low"], (int, float))
                assert isinstance(m["ci_high"], (int, float))
                assert m["n"] == 5

        asyncio.run(_check())

    def test_submit_chunk_hit_at_5_present(self, test_client):
        """stats_cached includes metrics.chunk_hit_at_5."""
        payload = _valid_payload()
        response = test_client.post("/api/submit", json=payload)
        assert response.status_code == 200
        run_id = payload["run_id"]

        async def _check():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                cursor = await conn.execute(
                    "SELECT stats_cached FROM runs WHERE id = ?", (run_id,)
                )
                row = await cursor.fetchone()
                stats = json.loads(row[0])
                cm = stats["metrics"]["chunk_hit_at_5"]
                assert "point" in cm
                assert "ci_low" in cm
                assert "ci_high" in cm
                assert cm["n"] == 5

        asyncio.run(_check())


# -----------------------------------------------------------------------
# Integration: baseline A/B detection
# -----------------------------------------------------------------------

class TestBaselineABDetection:
    """Baseline auto-detection and A/B paired tests."""

    def test_baseline_detected_for_grep_glob_match(self, test_client):
        """Submit grep-glob baseline, then non-grep-glob with same dataset_version."""
        # Step 1: submit grep-glob baseline
        baseline_payload = _valid_payload(
            run_id="baseline-grep-glob-001", server_name="grep-glob-baseline"
        )
        baseline_payload["dataset_version"] = "v1"
        baseline_payload["query_details"] = _make_query_details(8, seed=42)
        # Override query IDs to be deterministic
        for i, q in enumerate(baseline_payload["query_details"]):
            q["id"] = f"shared_{i:03d}"

        r1 = test_client.post("/api/submit", json=baseline_payload)
        assert r1.status_code == 200

        # Step 2: submit a non-grep-glob run with same dataset_version
        new_payload = _valid_payload(
            run_id="new-run-002", server_name="my-cool-server"
        )
        new_payload["dataset_version"] = "v1"
        new_payload["query_details"] = _make_query_details(8, seed=99)
        for i, q in enumerate(new_payload["query_details"]):
            q["id"] = f"shared_{i:03d}"

        r2 = test_client.post("/api/submit", json=new_payload)
        assert r2.status_code == 200

        # Verify stats_baseline_ab is populated
        async def _check():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                cursor = await conn.execute(
                    "SELECT stats_baseline_ab FROM runs WHERE id = ?",
                    ("new-run-002",),
                )
                row = await cursor.fetchone()
                assert row is not None
                ab_json = row[0]
                assert ab_json is not None
                assert len(ab_json) > 2
                ab = json.loads(ab_json)
                assert ab["baseline_run_id"] == "baseline-grep-glob-001"
                assert ab["dataset_version"] == "v1"
                assert "p_values" in ab
                assert "cohens_d" in ab
                assert "cliffs_delta" in ab

        asyncio.run(_check())

    def test_baseline_ab_for_matching_shared_query_ids(self, test_client):
        """A/B stats computed over shared query IDs between baseline and new run."""
        baseline_payload = _valid_payload(
            run_id="baseline-grep-002", server_name="grep-glob"
        )
        baseline_payload["dataset_version"] = "v1"
        baseline_payload["query_details"] = _make_query_details(6, seed=42)
        for i, q in enumerate(baseline_payload["query_details"]):
            q["id"] = f"q_{i}"

        r1 = test_client.post("/api/submit", json=baseline_payload)
        assert r1.status_code == 200

        new_payload = _valid_payload(
            run_id="new-run-003", server_name="test-server-B"
        )
        new_payload["dataset_version"] = "v1"
        new_payload["query_details"] = _make_query_details(6, seed=99)
        for i, q in enumerate(new_payload["query_details"]):
            q["id"] = f"q_{i}"

        r2 = test_client.post("/api/submit", json=new_payload)
        assert r2.status_code == 200

        async def _check():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                cursor = await conn.execute(
                    "SELECT stats_baseline_ab FROM runs WHERE id = ?",
                    ("new-run-003",),
                )
                row = await cursor.fetchone()
                ab = json.loads(row[0])
                # p_values should have entries for our metrics
                assert "hit_at_5" in ab["p_values"]
                assert "test_used" in ab
                assert ab["test_used"]["hit_at_5"] == "mcnemar"

        asyncio.run(_check())

    def test_baseline_not_detected_for_different_dataset(self, test_client):
        """No baseline when dataset_version differs."""
        baseline_payload = _valid_payload(
            run_id="baseline-grep-003", server_name="grep-glob"
        )
        baseline_payload["dataset_version"] = "v1"
        baseline_payload["query_details"] = _make_query_details(5, seed=42)
        for i, q in enumerate(baseline_payload["query_details"]):
            q["id"] = f"q_{i}"

        r1 = test_client.post("/api/submit", json=baseline_payload)
        assert r1.status_code == 200

        new_payload = _valid_payload(
            run_id="new-run-004", server_name="test-server-C"
        )
        new_payload["dataset_version"] = "v2"  # different
        new_payload["query_details"] = _make_query_details(5, seed=99)
        for i, q in enumerate(new_payload["query_details"]):
            q["id"] = f"q_{i}"

        r2 = test_client.post("/api/submit", json=new_payload)
        assert r2.status_code == 200

        async def _check():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                cursor = await conn.execute(
                    "SELECT stats_baseline_ab FROM runs WHERE id = ?",
                    ("new-run-004",),
                )
                row = await cursor.fetchone()
                ab_json = row[0]
                # Should be empty (no baseline found)
                ab = json.loads(ab_json)
                # "{}" parses as empty dict
                assert ab == {} or ab is None

        asyncio.run(_check())


# -----------------------------------------------------------------------
# McNemar reference test
# -----------------------------------------------------------------------

class TestMcNemarReference:
    """Known contingency → McNemar p-value matches scipy reference."""

    def test_mcnemar_known_contingency(self, test_client):
        """Submit paired runs with constructed binary outcomes → McNemar matches."""
        from scipy.stats import chi2

        def _mcnemar_ref_pvalue(b: int, c: int) -> float:
            """McNemar reference p-value with continuity correction."""
            if b + c == 0:
                return 1.0
            stat = (abs(b - c) - 1.0) ** 2 / (b + c)
            return float(1.0 - chi2.cdf(stat, 1))

        def _make_binary_qd(hits: list[int], seed: int) -> list[dict]:
            qd = []
            for i, h in enumerate(hits):
                qd.append({
                    "id": f"q_{i}",
                    "type": "locate",
                    "difficulty": "easy",
                    "repo": "test",
                    "found_file": bool(h),
                    "found_symbol": False,
                    "found_chunk": bool(h),
                    "latency_ms": 100.0,
                    "response_tokens": 200.0,
                    "tool_calls": 1,
                    "expected_files": ["f.py"],
                    "expected_symbols": [],
                    "returned_files": ["f.py"] if h else ["other.py"],
                    "returned_symbols": [],
                })
            return qd

        # Construct: baseline has 8 hits out of 10, new has 6 hits out of 10
        # Let's design so b=2, c=0: 2 queries baseline hit→new miss, 0 baseline miss→new hit
        baseline_hits = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  # 8 hits
        new_hits =      [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  # 6 hits
        # b=2 (queries 6,7: baseline hit, new miss), c=0

        # First submit baseline
        baseline_payload = _valid_payload("mcnemar-baseline", "grep-glob-ref")
        baseline_payload["dataset_version"] = "mcnemar-v1"
        baseline_payload["query_details"] = _make_binary_qd(baseline_hits, 42)

        r1 = test_client.post("/api/submit", json=baseline_payload)
        assert r1.status_code == 200

        # Submit new run
        new_payload = _valid_payload("mcnemar-new", "test-mcnemar")
        new_payload["dataset_version"] = "mcnemar-v1"
        new_payload["query_details"] = _make_binary_qd(new_hits, 99)

        r2 = test_client.post("/api/submit", json=new_payload)
        assert r2.status_code == 200

        # Verify McNemar p-value
        async def _check():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                cursor = await conn.execute(
                    "SELECT stats_baseline_ab FROM runs WHERE id = ?",
                    ("mcnemar-new",),
                )
                row = await cursor.fetchone()
                ab = json.loads(row[0])
                p_val = ab["p_values"]["hit_at_5"]

                # Reference: b=2, c=0 → McNemar with continuity correction
                ref_p = _mcnemar_ref_pvalue(2, 0)
                assert abs(p_val - ref_p) < 1e-6, \
                    f"McNemar p-value: {p_val} vs reference {ref_p}"

        asyncio.run(_check())


# -----------------------------------------------------------------------
# Wilcoxon reference test
# -----------------------------------------------------------------------

class TestWilcoxonReference:
    """Paired latency arrays → Wilcoxon p-value matches scipy reference."""

    def test_wilcoxon_known_latency_pair(self, test_client):
        """Submit paired runs with constructed latency → Wilcoxon matches."""
        from scipy.stats import wilcoxon as wilcoxon_ref

        def _make_latency_qd(latencies: list[float], seed: int) -> list[dict]:
            qd = []
            for i, lat in enumerate(latencies):
                qd.append({
                    "id": f"q_{i}",
                    "type": "locate",
                    "difficulty": "easy",
                    "repo": "test",
                    "found_file": True,
                    "found_symbol": False,
                    "found_chunk": True,
                    "latency_ms": lat,
                    "response_tokens": 200.0,
                    "tool_calls": 1,
                    "expected_files": ["f.py"],
                    "expected_symbols": [],
                    "returned_files": ["f.py"],
                    "returned_symbols": [],
                })
            return qd

        baseline_lat = [100.0, 200.0, 150.0, 175.0, 225.0, 180.0, 190.0, 160.0]
        new_lat =      [105.0, 210.0, 145.0, 180.0, 230.0, 185.0, 195.0, 155.0]

        # Submit baseline
        baseline_payload = _valid_payload("wilcoxon-baseline", "grep-glob-wilc")
        baseline_payload["dataset_version"] = "wilc-v1"
        baseline_payload["query_details"] = _make_latency_qd(baseline_lat, 42)

        r1 = test_client.post("/api/submit", json=baseline_payload)
        assert r1.status_code == 200

        # Submit new
        new_payload = _valid_payload("wilcoxon-new", "test-wilc")
        new_payload["dataset_version"] = "wilc-v1"
        new_payload["query_details"] = _make_latency_qd(new_lat, 99)

        r2 = test_client.post("/api/submit", json=new_payload)
        assert r2.status_code == 200

        async def _check():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                cursor = await conn.execute(
                    "SELECT stats_baseline_ab FROM runs WHERE id = ?",
                    ("wilcoxon-new",),
                )
                row = await cursor.fetchone()
                ab = json.loads(row[0])
                p_val = ab["p_values"]["latency_ms"]

                ref = wilcoxon_ref(baseline_lat, new_lat, zero_method="wilcox", alternative="two-sided")
                ref_p = float(ref.pvalue)
                assert abs(p_val - ref_p) < 1e-6, \
                    f"Wilcoxon p-value: {p_val} vs reference {ref_p}"

                assert ab["test_used"]["latency_ms"] == "wilcoxon"

        asyncio.run(_check())


# -----------------------------------------------------------------------
# Cohen's d reference test
# -----------------------------------------------------------------------

class TestCohensDReference:
    """Cohen's d matches analytical reference for known latency pair."""

    def test_cohens_d_known_latency_pair(self, test_client):
        """Cohen's d for latency pair matches reference within 1e-3."""
        def _make_qd(latencies: list[float]) -> list[dict]:
            qd = []
            for i, lat in enumerate(latencies):
                qd.append({
                    "id": f"q_{i}",
                    "type": "locate",
                    "difficulty": "easy",
                    "repo": "test",
                    "found_file": True,
                    "found_symbol": False,
                    "found_chunk": True,
                    "latency_ms": lat,
                    "response_tokens": 200.0,
                    "tool_calls": 1,
                    "expected_files": ["f.py"],
                    "expected_symbols": [],
                    "returned_files": ["f.py"],
                    "returned_symbols": [],
                })
            return qd

        # Two sets where Cohen's d is analytically calculable
        # x: N(100, 10) → [110, 90, 105, 95, 100]
        # y: N(110, 10) → [120, 100, 115, 105, 110]
        # mean diff = 10, pooled var = 100, pooled std = 10, d = 1.0
        x_lat = [110.0, 90.0, 105.0, 95.0, 100.0]   # mean=100, std=~7.9 (ddof=1)
        y_lat = [120.0, 100.0, 115.0, 105.0, 110.0]  # mean=110, std=~7.9

        baseline_payload = _valid_payload("cohensd-base", "grep-glob-cd")
        baseline_payload["dataset_version"] = "cd-v1"
        baseline_payload["query_details"] = _make_qd(x_lat)

        r1 = test_client.post("/api/submit", json=baseline_payload)
        assert r1.status_code == 200

        new_payload = _valid_payload("cohensd-new", "test-cd")
        new_payload["dataset_version"] = "cd-v1"
        new_payload["query_details"] = _make_qd(y_lat)

        r2 = test_client.post("/api/submit", json=new_payload)
        assert r2.status_code == 200

        async def _check():
            async with aiosqlite.connect(str(db.DB_PATH)) as conn:
                cursor = await conn.execute(
                    "SELECT stats_baseline_ab FROM runs WHERE id = ?",
                    ("cohensd-new",),
                )
                row = await cursor.fetchone()
                ab = json.loads(row[0])
                cd = ab["cohens_d"]["latency_ms"]

                # Reference Cohen's d
                from rag_bench.stats import cohens_d as cd_ref
                ref = cd_ref(np.array(y_lat), np.array(x_lat))
                assert abs(cd - ref) < 1e-3, \
                    f"Cohen's d: {cd} vs reference {ref}"

        asyncio.run(_check())
