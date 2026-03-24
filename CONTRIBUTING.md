# Contributing to rag-bench

## Adding queries

Each query in `rag_bench/datasets/queries/<repo>.jsonl` has this format:

```json
{
  "id": "flask_031",
  "type": "locate",
  "query": "where is session handling?",
  "expected_files": ["src/flask/sessions.py"],
  "expected_symbols": ["SecureCookieSessionInterface"],
  "difficulty": "medium"
}
```

**Fields:**
- `id` — unique, format: `{repo}_{number}`
- `type` — one of: `locate`, `callers`, `explain`, `impact`
- `query` — natural language question a developer would ask
- `expected_files` — list of file paths (relative to repo root) that contain the answer
- `expected_symbols` — list of function/class names that should be found
- `difficulty` — `easy` (one obvious file), `medium` (requires understanding), `hard` (cross-cutting)

**Guidelines:**
- Questions should be realistic — things a developer would actually ask
- Ground truth files must exist at the pinned version (check `repos.json` for the tag)
- Prefer specific symbols over generic ones

## Adding presets

Create `rag_bench/presets/my_server.json`:

```json
{
  "name": "my-server",
  "git_url": "https://github.com/you/my-server",
  "command": "python -m my_server",
  "args": [],
  "transport": "stdio",
  "tool_mapping": {
    "ingest": {
      "tool": "ingest_tool_name",
      "params": { "path": "{path}" }
    },
    "query": {
      "tool": "search_tool_name",
      "params": { "query": "{query}", "limit": "{top_k}" }
    }
  }
}
```

## Adding test repositories

Add to `rag_bench/datasets/repos.json`:

```json
{
  "name": "myrepo",
  "git_url": "https://github.com/org/repo.git",
  "ref": "v1.0.0",
  "language": "python",
  "size": "medium"
}
```

Then create `rag_bench/datasets/queries/myrepo.jsonl` with 30 queries.

**Important:** Always pin to a specific tag/commit for reproducibility.

## Running tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Code style

- Python 3.11+
- Type hints everywhere
- No unnecessary abstractions
