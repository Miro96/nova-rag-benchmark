"""FastAPI leaderboard server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from server.db import DuplicateRunError, get_leaderboard, get_run, init_db, insert_run
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


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the leaderboard HTML page."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>rag-bench Leaderboard</h1><p>Static files not found.</p>"
