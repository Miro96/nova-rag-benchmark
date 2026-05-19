"""Statistics cache and A/B baseline detection for POST /api/submit.

Wraps rag_bench.stats primitives and scipy.stats paired tests to compute
and persist stats_cached / stats_baseline_ab JSON blobs for every run.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from scipy.stats import chi2, wilcoxon

from rag_bench.stats import (
    cliffs_delta,
    cohens_d,
    normal_ci,
    pearson_correlation_matrix,
    wilson_ci,
)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# Continuous metrics (use normal CI, Wilcoxon, Cohen's d, Cliff's delta)
_CONTINUOUS_METRICS = ["mrr", "latency_ms", "response_tokens"]

# Binary / proportion metrics (use Wilson CI, McNemar)
_BINARY_METRICS = ["hit_at_5", "chunk_hit_at_5", "symbol_hit_at_5"]

# Metrics used for the Pearson correlation matrix (in this order)
_CORR_METRIC_KEYS = [
    "latency_ms",
    "response_tokens",
    "tool_calls",
    "hit_at_5",
    "chunk_hit_at_5",
    "symbol_hit_at_5",
]

# Bucket keys
_BUCKET_KEYS = ["by_difficulty", "by_type", "by_repo"]


# ---------------------------------------------------------------------------
# JSON sanitization helper
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN, Inf, -Inf with None so the dict is JSON-safe."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _json_dumps_safe(obj: Any) -> str:
    """json.dumps with NaN/Inf → null sanitation."""
    return json.dumps(_sanitize_for_json(obj))


# ---------------------------------------------------------------------------
# McNemar helper (scipy 1.17 does not expose contingency_tables.mcnemar)
# ---------------------------------------------------------------------------

def _mcnemar_pvalue(b: int, c: int, correction: bool = True) -> float:
    """McNemar's test p-value for a 2×2 paired contingency table.

    Parameters
    ----------
    b : count of (baseline hit, new miss) discordant pairs.
    c : count of (baseline miss, new hit) discordant pairs.
    correction : if True, applies Yates' continuity correction.

    Returns
    -------
    float : two-sided p-value from the chi-square(1) distribution.
    """
    if b + c == 0:
        return 1.0
    if correction:
        stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    else:
        stat = (b - c) ** 2 / (b + c)
    # stat is chi-square with 1 df
    return float(1.0 - chi2.cdf(stat, 1))


# ---------------------------------------------------------------------------
# Per-query helpers
# ---------------------------------------------------------------------------

def _per_query_mrr(qd: dict) -> float:
    """Compute per-query MRR from expected/returned file and symbol lists.

    Returns 1 / rank of first match (1-based), or 0.0 if no match.
    """
    expected_files = set(qd.get("expected_files") or [])
    expected_symbols = set(qd.get("expected_symbols") or [])
    returned_files = qd.get("returned_files") or []
    returned_symbols = qd.get("returned_symbols") or []

    # Check files first
    for rank, f in enumerate(returned_files, start=1):
        if f in expected_files:
            return 1.0 / rank

    # Then check symbols
    for rank, s in enumerate(returned_symbols, start=1):
        if s in expected_symbols:
            return 1.0 / rank

    return 0.0


def _extract_per_query_values(query_details: list[dict]) -> dict[str, list]:
    """Extract per-query metric arrays from query_details.

    Returns a dict mapping metric name → list of values.
    """
    hit_at_5_vals: list[int] = []
    chunk_hit_vals: list[int] = []
    symbol_hit_vals: list[int] = []
    mrr_vals: list[float] = []
    latency_vals: list[float] = []
    token_vals: list[float] = []
    tool_call_vals: list[int] = []

    for q in query_details:
        hit_at_5_vals.append(1 if q.get("found_file") else 0)
        # chunk_hit: use found_chunk if present, otherwise fall back to found_file
        chunk_hit_vals.append(
            1 if q.get("found_chunk", q.get("found_file")) else 0
        )
        symbol_hit_vals.append(1 if q.get("found_symbol") else 0)
        mrr_vals.append(_per_query_mrr(q))
        latency_vals.append(float(q.get("latency_ms", 0) or 0))
        token_vals.append(float(q.get("response_tokens", 0) or 0))
        tool_call_vals.append(int(q.get("tool_calls", 0) or 0))

    return {
        "hit_at_5": hit_at_5_vals,
        "chunk_hit_at_5": chunk_hit_vals,
        "symbol_hit_at_5": symbol_hit_vals,
        "mrr": mrr_vals,
        "latency_ms": latency_vals,
        "response_tokens": token_vals,
        "tool_calls": tool_call_vals,
    }


# ---------------------------------------------------------------------------
# Metric-level summary
# ---------------------------------------------------------------------------

def _compute_metric_stats(
    name: str, values: list, n: int
) -> dict[str, Any]:
    """Compute {point, ci_low, ci_high, n} for a single metric."""
    if n == 0:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    is_binary = name in _BINARY_METRICS

    if is_binary:
        k = sum(values)
        point = k / n
        lo, hi = wilson_ci(k, n)
    else:
        arr = np.array(values, dtype=np.float64)
        point = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        lo, hi = normal_ci(point, std_val, n)

    return {"point": round(point, 8), "ci_low": round(lo, 8), "ci_high": round(hi, 8), "n": n}


def _compute_bucket_stats(
    query_details: list[dict], bucket_key: str
) -> dict[str, dict[str, Any]]:
    """Compute per-bucket stats.

    bucket_key is one of 'difficulty', 'type', 'repo'.
    """
    bucket_values: dict[str, dict[str, list]] = {}

    for q in query_details:
        bucket = q.get(bucket_key)
        if bucket is None:
            continue
        if bucket not in bucket_values:
            bucket_values[bucket] = {}
        # Extract per-query values for this query
        per_query = _extract_per_query_values([q])
        for metric, val in per_query.items():
            bucket_values[bucket].setdefault(metric, []).append(val[0])

    result: dict[str, dict[str, Any]] = {}
    for bucket, metric_vals in bucket_values.items():
        n_bucket = max(len(v) for v in metric_vals.values()) if metric_vals else 0
        result[bucket] = {}
        for metric in _BINARY_METRICS + _CONTINUOUS_METRICS:
            vals = metric_vals.get(metric, [])
            result[bucket][metric] = _compute_metric_stats(metric, vals, len(vals))

    return result


def _compute_correlations(query_details: list[dict]) -> dict[str, Any]:
    """Compute the Pearson correlation matrix over the six core metrics."""
    if len(query_details) < 3:
        # Not enough data for meaningful correlations
        return {
            "matrix": [],
            "labels": _CORR_METRIC_KEYS,
        }

    per_query = _extract_per_query_values(query_details)
    metric_arrays = [per_query[k] for k in _CORR_METRIC_KEYS]
    corr_matrix = pearson_correlation_matrix(*metric_arrays)

    return {
        "matrix": [[float(v) for v in row] for row in corr_matrix],
        "labels": _CORR_METRIC_KEYS,
    }


def _compute_iqr_cv(replicates: list[dict]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute IQR and CV from replicates.

    Returns (iqr_dict, cv_dict).
    """
    if not replicates:
        return {}, {}

    # Collect metric values across replicates
    replicate_metrics: dict[str, list[float]] = {}
    for rep in replicates:
        for key, val in rep.items():
            if isinstance(val, (int, float)) and key not in ("index", "run"):
                replicate_metrics.setdefault(key, []).append(float(val))

    iqr_dict: dict[str, float] = {}
    cv_dict: dict[str, float] = {}

    for metric, vals in replicate_metrics.items():
        if len(vals) < 2:
            continue
        arr = np.array(vals, dtype=np.float64)
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr_dict[metric] = round(q3 - q1, 8)

        mean_val = float(np.mean(arr))
        if mean_val != 0:
            std_val = float(np.std(arr, ddof=1))
            cv_dict[metric] = round(std_val / mean_val, 8)

    return iqr_dict, cv_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summary_stats_for_run(
    query_details: list[dict],
    replicates: list[dict] | None = None,
    by_difficulty: dict | None = None,
    by_type: dict | None = None,
    by_repo: dict | None = None,
) -> dict[str, Any]:
    """Compute summary statistics for a single run.

    Parameters
    ----------
    query_details : list of per-query detail dicts.
    replicates : list of replicate run summaries (optional).
    by_difficulty, by_type, by_repo : pre-grouped bucket dicts (optional,
        used to pick up bucket names even if query_details has little data).

    Returns
    -------
    dict with keys: metrics, by_difficulty, by_type, by_repo,
    correlations, iqr, cv.
    """
    n = len(query_details)
    if n == 0:
        # Return empty placeholder
        return {
            "metrics": {},
            "by_difficulty": {},
            "by_type": {},
            "by_repo": {},
            "correlations": {"matrix": [], "labels": _CORR_METRIC_KEYS},
            "iqr": {},
            "cv": {},
        }

    per_query = _extract_per_query_values(query_details)

    # --- Top-level metrics ---
    metrics: dict[str, Any] = {}
    for metric in _BINARY_METRICS + _CONTINUOUS_METRICS:
        vals = per_query.get(metric, [])
        metrics[metric] = _compute_metric_stats(metric, vals, n)

    # --- Buckets ---
    by_difficulty_stats = _compute_bucket_stats(query_details, "difficulty")
    by_type_stats = _compute_bucket_stats(query_details, "type")
    by_repo_stats = _compute_bucket_stats(query_details, "repo")

    # --- Correlations ---
    correlations = _compute_correlations(query_details)

    # --- IQR / CV from replicates ---
    iqr_dict, cv_dict = _compute_iqr_cv(replicates or [])

    return {
        "metrics": metrics,
        "by_difficulty": by_difficulty_stats,
        "by_type": by_type_stats,
        "by_repo": by_repo_stats,
        "correlations": correlations,
        "iqr": iqr_dict,
        "cv": cv_dict,
    }


async def detect_baseline_and_compute_ab(
    run: dict, db_conn
) -> dict[str, Any] | None:
    """Detect a grep-glob baseline and compute A/B paired statistics.

    Baseline rule:
    - server_name contains *both* "grep" and "glob" (case-insensitive)
    - same dataset_version as the submitted run
    - picks the first matching run (by submitted_at)

    If no baseline is found, returns None.

    Parameters
    ----------
    run : the submitted run data dict (as returned by model_dump).
    db_conn : an aiosqlite connection.

    Returns
    -------
    dict with baseline_run_id, dataset_version, p_values, cohens_d,
    cliffs_delta, delta, test_used — or None.
    """
    server_name = (run.get("server") or {}).get("name", "")
    dataset_version = run.get("dataset_version", "")

    # Rule: baseline must be grep-glob
    name_lower = server_name.lower()
    if "grep" in name_lower and "glob" in name_lower:
        # The submitted run is itself grep-glob — could serve as a
        # baseline for other runs but cannot have a baseline itself.
        return None

    # Find a grep-glob run with matching dataset_version
    cursor = await db_conn.execute(
        """
        SELECT id, queries
        FROM runs
        WHERE LOWER(server_name) LIKE '%grep%glob%'
          AND dataset_version = ?
        ORDER BY submitted_at ASC
        LIMIT 1
        """,
        (dataset_version,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None

    baseline_run_id = row[0]
    baseline_queries_raw = row[1] or "[]"
    try:
        baseline_queries = json.loads(baseline_queries_raw)
    except (json.JSONDecodeError, TypeError):
        baseline_queries = []

    # Get the new run's queries
    new_queries = run.get("query_details") or []

    if not baseline_queries or not new_queries:
        return None

    # Build lookup by query id for alignment
    baseline_by_id: dict[str, dict] = {
        q["id"]: q for q in baseline_queries if "id" in q
    }
    new_by_id: dict[str, dict] = {
        q["id"]: q for q in new_queries if "id" in q
    }

    # Intersection of query ids (paired)
    common_ids = sorted(set(baseline_by_id) & set(new_by_id))
    if not common_ids:
        return None

    # --- Paired arrays ---
    def _paired_values(extract_fn, common_ids, baseline_by_id, new_by_id):
        """Return (baseline_vals, new_vals) aligned by common_ids."""
        b_vals, n_vals = [], []
        for qid in common_ids:
            bq = baseline_by_id[qid]
            nq = new_by_id[qid]
            b_vals.append(extract_fn(bq))
            n_vals.append(extract_fn(nq))
        return b_vals, n_vals

    # Binary metrics via McNemar
    def _binary_outcome(qd: dict, key: str) -> int:
        if key == "hit_at_5":
            return 1 if qd.get("found_file") else 0
        elif key == "chunk_hit_at_5":
            return 1 if qd.get("found_chunk", qd.get("found_file")) else 0
        elif key == "symbol_hit_at_5":
            return 1 if qd.get("found_symbol") else 0
        return 0

    # Continuous metrics via paired tests
    def _continuous_value(qd: dict, key: str) -> float:
        if key == "mrr":
            return _per_query_mrr(qd)
        elif key == "latency_ms":
            return float(qd.get("latency_ms", 0) or 0)
        elif key == "response_tokens":
            return float(qd.get("response_tokens", 0) or 0)
        return 0.0

    p_values: dict[str, float | None] = {}
    cohens_d_dict: dict[str, float | None] = {}
    cliffs_delta_dict: dict[str, float | None] = {}
    delta_dict: dict[str, float | None] = {}
    test_used: dict[str, str | None] = {}

    # Binary metrics: McNemar
    for metric in _BINARY_METRICS:
        b_vals, n_vals = _paired_values(
            lambda q, m=metric: _binary_outcome(q, m),
            common_ids, baseline_by_id, new_by_id,
        )
        # Build contingency table:
        #   b=1,n=1  |  b=1,n=0
        #   b=0,n=1  |  b=0,n=0
        b = 0  # baseline hit, new miss
        c = 0  # baseline miss, new hit
        for bv, nv in zip(b_vals, n_vals):
            if bv == 1 and nv == 0:
                b += 1
            elif bv == 0 and nv == 1:
                c += 1

        if b + c == 0:
            p_values[metric] = 1.0
            delta_dict[metric] = 0.0
            cohens_d_dict[metric] = None
            cliffs_delta_dict[metric] = None
        else:
            # McNemar's test with continuity correction
            p_values[metric] = _mcnemar_pvalue(b, c, correction=True)
            # Delta = difference in proportions
            new_rate = sum(n_vals) / len(n_vals) if n_vals else 0.0
            base_rate = sum(b_vals) / len(b_vals) if b_vals else 0.0
            delta_dict[metric] = round(new_rate - base_rate, 8)

        test_used[metric] = "mcnemar"

    # Continuous metrics: Wilcoxon + Cohen's d + Cliff's delta
    for metric in _CONTINUOUS_METRICS:
        b_vals, n_vals = _paired_values(
            lambda q, m=metric: _continuous_value(q, m),
            common_ids, baseline_by_id, new_by_id,
        )

        b_arr = np.array(b_vals, dtype=np.float64)
        n_arr = np.array(n_vals, dtype=np.float64)

        # Check for zero variance
        if np.allclose(b_arr, n_arr):
            p_values[metric] = 1.0
            delta_dict[metric] = 0.0
            cohens_d_dict[metric] = 0.0
            cliffs_delta_dict[metric] = 0.0
        elif np.array_equal(b_arr, n_arr):
            p_values[metric] = 1.0
            delta_dict[metric] = 0.0
            cohens_d_dict[metric] = 0.0
            cliffs_delta_dict[metric] = 0.0
        else:
            try:
                w_result = wilcoxon(b_arr, n_arr, zero_method="wilcox", alternative="two-sided")
                p_values[metric] = float(w_result.pvalue)
            except Exception:
                p_values[metric] = None

            new_mean = float(np.mean(n_arr))
            base_mean = float(np.mean(b_arr))
            delta_dict[metric] = round(new_mean - base_mean, 8)

            cohens_d_dict[metric] = round(cohens_d(n_arr, b_arr), 8)
            cliffs_delta_dict[metric] = round(cliffs_delta(n_arr, b_arr), 8)

        test_used[metric] = "wilcoxon"

    return {
        "baseline_run_id": baseline_run_id,
        "dataset_version": dataset_version,
        "p_values": p_values,
        "cohens_d": cohens_d_dict,
        "cliffs_delta": cliffs_delta_dict,
        "delta": delta_dict,
        "test_used": test_used,
    }


# ---------------------------------------------------------------------------
# Multi-run pairwise comparison (for GET /api/compare)
# ---------------------------------------------------------------------------

# Metrics required by the compare API
_COMPARE_BINARY_METRICS = ["hit_at_5"]
_COMPARE_CONTINUOUS_METRICS = ["mrr", "query_latency_p50_ms"]


def _per_query_binary(qd: dict, metric: str) -> int:
    """Extract binary outcome for a compare metric."""
    if metric == "hit_at_5":
        return 1 if qd.get("found_file") else 0
    return 0


def _per_query_continuous(qd: dict, metric: str) -> float:
    """Extract continuous value for a compare metric."""
    if metric == "mrr":
        return _per_query_mrr(qd)
    elif metric == "query_latency_p50_ms":
        # Use per-query latency for the paired test
        return float(qd.get("latency_ms", 0) or 0)
    return 0.0


def _paired_mean_diff_ci(
    a_vals: list[float], b_vals: list[float], alpha: float = 0.05
) -> tuple[float, float, float]:
    """Compute mean difference and its 95% CI using paired t-interval.

    Returns (mean_diff, ci_low, ci_high).
    """
    n = len(a_vals)
    if n < 2:
        return 0.0, 0.0, 0.0

    diffs = np.array(b_vals, dtype=np.float64) - np.array(a_vals, dtype=np.float64)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se = std_diff / math.sqrt(n)
    df = n - 1
    try:
        from scipy.stats import t as _t
        t_crit = float(_t.ppf(1 - alpha / 2, df=df))
    except Exception:
        from scipy.stats import norm as _norm
        t_crit = float(_norm.ppf(1 - alpha / 2))
    margin = t_crit * se
    return round(mean_diff, 8), round(mean_diff - margin, 8), round(mean_diff + margin, 8)


def _binary_proportion_diff_ci(
    a_vals: list[int], b_vals: list[int], alpha: float = 0.05
) -> tuple[float, float, float]:
    """Compute difference in proportions and its CI (normal approximation).

    Returns (prop_diff, ci_low, ci_high).
    """
    n = len(a_vals)
    if n == 0:
        return 0.0, 0.0, 0.0

    p_a = sum(a_vals) / n
    p_b = sum(b_vals) / n
    delta = p_b - p_a

    if n == 1 or (p_a == 0 and p_b == 0) or (p_a == 1 and p_b == 1):
        return round(delta, 8), round(delta, 8), round(delta, 8)

    try:
        from scipy.stats import norm as _norm
        z = float(_norm.ppf(1 - alpha / 2))
    except Exception:
        z = 1.96

    se = math.sqrt(p_a * (1 - p_a) / n + p_b * (1 - p_b) / n)
    margin = z * se
    return round(delta, 8), round(delta - margin, 8), round(delta + margin, 8)


def compute_pairwise_stats(runs_data: list[dict]) -> dict[str, dict]:
    """Compute pairwise A/B statistics for all unordered pairs of runs.

    Parameters
    ----------
    runs_data : list of dicts, each containing at least:
        run_id, query_details (list of per-query dicts)

    Returns
    -------
    dict keyed by "runA,runB" (the two run_ids joined by comma, preserving
    the order from runs_data), each value being a per-metric dict with
    delta, delta_ci_lo, delta_ci_hi, p_value, cohens_d, cliffs_delta.
    """
    if len(runs_data) < 2:
        return {}

    # Build lookup by run_id for quick access
    runs_by_id: dict[str, dict] = {
        rd["run_id"]: rd for rd in runs_data
    }

    result: dict[str, dict] = {}

    all_metrics = _COMPARE_BINARY_METRICS + _COMPARE_CONTINUOUS_METRICS

    for i in range(len(runs_data)):
        for j in range(i + 1, len(runs_data)):
            run_a = runs_data[i]
            run_b = runs_data[j]
            qds_a = run_a.get("query_details") or []
            qds_b = run_b.get("query_details") or []

            # Build lookup by query id
            a_by_id = {q["id"]: q for q in qds_a if "id" in q}
            b_by_id = {q["id"]: q for q in qds_b if "id" in q}

            common_ids = sorted(set(a_by_id) & set(b_by_id))

            pair_key = f"{run_a['run_id']},{run_b['run_id']}"
            pair_result: dict[str, dict] = {}

            for metric in all_metrics:
                if not common_ids:
                    pair_result[metric] = {
                        "delta": None,
                        "delta_ci_lo": None,
                        "delta_ci_hi": None,
                        "p_value": None,
                        "cohens_d": None,
                        "cliffs_delta": None,
                    }
                    continue

                if metric in _COMPARE_BINARY_METRICS:
                    # --- Binary metric: McNemar ---
                    a_vals = [_per_query_binary(a_by_id[qid], metric) for qid in common_ids]
                    b_vals = [_per_query_binary(b_by_id[qid], metric) for qid in common_ids]

                    # Build contingency table
                    b_disc = 0  # a hit, b miss
                    c_disc = 0  # a miss, b hit
                    for av, bv in zip(a_vals, b_vals):
                        if av == 1 and bv == 0:
                            b_disc += 1
                        elif av == 0 and bv == 1:
                            c_disc += 1

                    if b_disc + c_disc == 0:
                        p_val = 1.0
                    else:
                        p_val = _mcnemar_pvalue(b_disc, c_disc, correction=True)

                    delta, ci_lo, ci_hi = _binary_proportion_diff_ci(a_vals, b_vals)

                    # Cohen's d for binary data
                    a_arr = np.array(a_vals, dtype=np.float64)
                    b_arr = np.array(b_vals, dtype=np.float64)
                    cd = cohens_d(b_arr, a_arr)
                    cld = cliffs_delta(b_arr, a_arr)

                else:
                    # --- Continuous metric: Wilcoxon ---
                    a_vals_f = [_per_query_continuous(a_by_id[qid], metric) for qid in common_ids]
                    b_vals_f = [_per_query_continuous(b_by_id[qid], metric) for qid in common_ids]

                    a_arr = np.array(a_vals_f, dtype=np.float64)
                    b_arr = np.array(b_vals_f, dtype=np.float64)

                    if np.allclose(a_arr, b_arr) or np.array_equal(a_arr, b_arr):
                        p_val = 1.0
                        cd = 0.0
                        cld = 0.0
                    else:
                        try:
                            w_result = wilcoxon(a_arr, b_arr, zero_method="wilcox",
                                                alternative="two-sided")
                            p_val = float(w_result.pvalue)
                        except Exception:
                            p_val = None
                        cd = cohens_d(b_arr, a_arr)
                        cld = cliffs_delta(b_arr, a_arr)

                    delta, ci_lo, ci_hi = _paired_mean_diff_ci(a_vals_f, b_vals_f)

                pair_result[metric] = {
                    "delta": delta,
                    "delta_ci_lo": ci_lo,
                    "delta_ci_hi": ci_hi,
                    "p_value": round(p_val, 8) if p_val is not None else None,
                    "cohens_d": round(cd, 8) if cd is not None else None,
                    "cliffs_delta": round(cld, 8) if cld is not None else None,
                }

            result[pair_key] = pair_result

    return result
