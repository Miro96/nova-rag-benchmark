"""FastAPI leaderboard server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from server.db import (
    DuplicateRunError,
    get_full_run_data,
    get_leaderboard,
    get_queries,
    get_run,
    get_runs_by_ids,
    get_stats,
    init_db,
    insert_run,
    update_stats,
)
from server.models import BenchmarkSubmission


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="rag-bench Leaderboard",
    version="0.1.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent / "static"

# Mount static files directory at /static for shared CSS/JS and future assets.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.post("/api/submit")
async def submit_run(data: BenchmarkSubmission):
    """Submit benchmark results."""
    try:
        run_id = await insert_run(data.model_dump())
    except DuplicateRunError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "ok", "run_id": run_id}


def _map_row(row: dict) -> dict:
    """Map DB row fields to API response fields."""
    # Rename 'id' to 'run_id' for the API response
    if "id" in row:
        row["run_id"] = row.pop("id")
    # Parse JSON fields if stored as strings
    for field in ("environment", "by_difficulty", "by_type", "repos"):
        if isinstance(row.get(field), str):
            try:
                import json
                row[field] = json.loads(row[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return row


@app.get("/api/leaderboard")
async def leaderboard(
    sort_by: str = Query("composite_score"),
    order: str = Query("desc"),
    limit: int = Query(50, ge=1, le=200),
):
    """Get leaderboard data."""
    rows = await get_leaderboard(sort_by=sort_by, order=order, limit=limit)
    entries = []
    for rank, row in enumerate(rows, 1):
        entry = _map_row(dict(row))
        entry["rank"] = rank
        entries.append(entry)
    return {"entries": entries, "total": len(entries)}


@app.get("/api/run/{run_id}")
async def run_detail(run_id: str):
    """Get details for a specific run."""
    run = await get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return _map_row(dict(run))


@app.get("/api/run/{run_id}/queries")
async def run_queries(run_id: str):
    """Get per-query details for a specific run.

    Returns a JSON array of query objects. Returns 404 if the run
    does not exist. Returns an empty array (200) if the run exists
    but has no query_details.
    """
    queries = await get_queries(run_id)
    if queries is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return queries


@app.get("/api/run/{run_id}/stats")
async def run_stats(run_id: str):
    """Get cached statistics for a specific run.

    Reads stats_cached and stats_baseline_ab from the database and merges
    them into a single JSON response. Returns 404 if the run does not exist.
    """
    stats_tuple = await get_stats(run_id)
    if stats_tuple is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    stats_cached, stats_baseline_ab = stats_tuple

    # Merge: add run_id and baseline_ab to the cached stats dict
    # Treat empty baseline_ab dict as null
    baseline_ab = stats_baseline_ab if stats_baseline_ab else None

    return {
        "run_id": run_id,
        **stats_cached,
        "baseline_ab": baseline_ab,
    }


@app.post("/api/run/{run_id}/recompute_stats")
async def recompute_stats(run_id: str):
    """Recompute and cache statistics for a specific run.

    Reads the stored query_details and other data for the run, recomputes
    stats_cached and stats_baseline_ab, and writes them back. Returns 404
    if the run does not exist.
    """
    import json as _json

    run_data = await get_full_run_data(run_id)
    if run_data is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    from server.stats_cache import (
        _json_dumps_safe,
        detect_baseline_and_compute_ab,
        summary_stats_for_run,
    )

    query_details = run_data.get("query_details") or []
    replicates = run_data.get("replicates") or []
    by_difficulty = run_data.get("by_difficulty") or {}
    by_type = run_data.get("by_type") or {}
    by_repo = run_data.get("by_repo") or {}

    stats_cached_json = _json_dumps_safe(
        summary_stats_for_run(
            query_details=query_details,
            replicates=replicates,
            by_difficulty=by_difficulty,
            by_type=by_type,
            by_repo=by_repo,
        )
    )

    # Baseline A/B detection
    import aiosqlite
    from server.db import DB_PATH

    async with aiosqlite.connect(DB_PATH) as db_conn:
        baseline_ab = await detect_baseline_and_compute_ab(run_data, db_conn)

    stats_baseline_ab_json = _json_dumps_safe(baseline_ab) if baseline_ab else "{}"

    await update_stats(run_id, stats_cached_json, stats_baseline_ab_json)

    return {"status": "ok", "run_id": run_id}


@app.get("/api/compare")
async def compare_runs(run_ids: str = Query(None)):
    """Compare multiple runs with pairwise A/B statistics.

    Query parameters:
    - run_ids: comma-separated list of run IDs (minimum 2 required).

    Returns 200 with keys: runs, pairwise_ab, per_metric_table.
    Returns 400 if run_ids is missing, empty, single, or has duplicates.
    Returns 404 if any run_id is not found.
    """
    # --- Validation ---
    if not run_ids:
        raise HTTPException(
            status_code=400,
            detail="Missing required query parameter 'run_ids'. "
                   "Provide at least 2 comma-separated run IDs.",
        )

    ids = [rid.strip() for rid in run_ids.split(",") if rid.strip()]

    if not ids:
        raise HTTPException(
            status_code=400,
            detail="Missing required query parameter 'run_ids'. "
                   "Provide at least 2 comma-separated run IDs.",
        )

    if len(ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 run_ids are required for comparison, "
                   f"got {len(ids)}.",
        )

    if len(set(ids)) != len(ids):
        raise HTTPException(
            status_code=400,
            detail="Duplicate run_ids are not allowed in comparison.",
        )

    # --- Fetch runs ---
    runs = await get_runs_by_ids(ids)
    found_ids = {r["run_id"] for r in runs}

    # Check for missing IDs (check in request order to report the first missing)
    for rid in ids:
        if rid not in found_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Run '{rid}' not found.",
            )

    # Reorder runs to match request order
    run_order: dict[str, int] = {rid: idx for idx, rid in enumerate(ids)}
    runs.sort(key=lambda r: run_order.get(r["run_id"], 999))

    # --- Compute pairwise A/B stats ---
    from server.stats_cache import compute_pairwise_stats

    pairwise_ab = compute_pairwise_stats(runs)

    # --- Build per_metric_table ---
    COMPARE_METRICS = ["hit_at_5", "mrr", "query_latency_p50_ms"]
    per_metric_table = {
        "metrics": COMPARE_METRICS,
        "runs": [
            {
                "run_id": r["run_id"],
                "server_name": r["server_name"],
                "hit_at_5": r["hit_at_5"],
                "mrr": r["mrr"],
                "query_latency_p50_ms": r["query_latency_p50_ms"],
            }
            for r in runs
        ],
    }

    # --- Build runs array for response ---
    runs_response = [
        {
            "run_id": r["run_id"],
            "server_name": r["server_name"],
            "hit_at_5": r["hit_at_5"],
            "mrr": r["mrr"],
            "query_latency_p50_ms": r["query_latency_p50_ms"],
        }
        for r in runs
    ]

    return {
        "runs": runs_response,
        "pairwise_ab": pairwise_ab,
        "per_metric_table": per_metric_table,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the leaderboard HTML page."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>rag-bench Leaderboard</h1><p>Static files not found.</p>"


@app.get("/detail.html", response_class=HTMLResponse)
async def detail_page():
    """Serve the query detail page."""
    html_path = STATIC_DIR / "detail.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Query Detail</h1><p>Static files not found.</p>"


@app.get("/run/{run_id}/report", response_class=HTMLResponse)
async def run_report(run_id: str):
    """Serve the single-run report page with Plotly visualizations."""
    html_path = STATIC_DIR / "report.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Run Report</h1><p>Static files not found.</p>"


@app.get("/compare", response_class=HTMLResponse)
async def compare_page():
    """Serve the multi-run comparison page with Plotly visualizations."""
    html_path = STATIC_DIR / "compare.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Multi-Run Comparison</h1><p>Static files not found.</p>"
