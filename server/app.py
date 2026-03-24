"""FastAPI leaderboard server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from server.db import get_leaderboard, get_run, init_db, insert_run
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
    run_id = await insert_run(data.model_dump())
    return {"status": "ok", "run_id": run_id}


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
        entries.append({
            "rank": rank,
            **row,
        })
    return {"entries": entries, "total": len(entries)}


@app.get("/api/run/{run_id}")
async def run_detail(run_id: str):
    """Get details for a specific run."""
    run = await get_run(run_id)
    if not run:
        return {"error": "Run not found"}, 404
    return run


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the leaderboard HTML page."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>rag-bench Leaderboard</h1><p>Static files not found.</p>"
