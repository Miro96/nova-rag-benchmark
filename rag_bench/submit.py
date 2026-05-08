"""Submit benchmark results to leaderboard server."""

from __future__ import annotations

import httpx


async def submit_results(server_url: str, data: dict) -> dict:
    """Submit benchmark results to the leaderboard server.

    Returns the server response dict, which includes 'run_id'.
    """
    url = f"{server_url.rstrip('/')}/api/submit"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=data)
        response.raise_for_status()
        return response.json()
