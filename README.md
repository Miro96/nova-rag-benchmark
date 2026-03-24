# rag-bench

**Benchmark for Code RAG MCP Servers** — measure how well RAG helps AI find the right code.

No existing benchmark covers the intersection of **code search + MCP protocol + A/B comparison**. rag-bench fills that gap.

## What it measures

rag-bench indexes real open-source repositories (Flask, FastAPI, Express) and runs 90 code search queries with known correct answers. It measures:

| Metric | What it tells you |
|--------|-------------------|
| **Hit@1 / Hit@5** | Did the RAG find the correct file in top results? |
| **Symbol Hit@5** | Did it find the correct function/class? |
| **MRR** | How high is the first correct result ranked? |
| **Latency p50/p95** | How fast are queries? |
| **Ingest Speed** | How fast does it index a codebase? |
| **RAM / Index Size** | Resource consumption |
| **Composite Score** | Weighted combination of all metrics |

### Query types

- `locate` — "Where is X defined?" (most common developer question)
- `callers` — "What calls X?"
- `explain` — "How does X work?" (needs multiple related files)
- `impact` — "What breaks if I change X?"

### A/B mode

Compare RAG vs grep/glob baseline to measure the actual improvement RAG provides:
- How many fewer tool calls?
- How much faster to find the right code?
- Does RAG actually help or is grep enough?

## Quick start

```bash
# Clone
git clone https://github.com/Miro96/nova-rag-benchmark.git
cd nova-rag-benchmark

# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run benchmark on your RAG MCP server
rag-bench run --command "python -m your_server" --transport stdio

# Or use a preset
rag-bench run --preset mcp-local-rag

# Run tests
pytest tests/ -v
```

## Usage

### Benchmark a single server

```bash
# With a preset
rag-bench run --preset nova-rag

# With a custom command
rag-bench run --command "npx -y mcp-local-rag" --transport stdio

# With a config file
rag-bench run --config my-server.json

# Only test on one repo
rag-bench run --preset nova-rag --repo flask

# A/B comparison vs grep baseline
rag-bench run --preset nova-rag --ab-baseline
```

### Compare multiple servers

```bash
rag-bench compare --presets nova-rag,mcp-local-rag,chroma-mcp
```

### Leaderboard

```bash
# Start the web leaderboard
rag-bench serve --port 8080

# Submit results
rag-bench submit results/run_*.json \
  --git-url https://github.com/you/your-rag \
  --git-user yourusername \
  --server-url http://localhost:8080
```

Open `http://localhost:8080` to see the leaderboard with sorting, filtering, and radar chart comparison.

## Server config format

Create a JSON config for your RAG MCP server:

```json
{
  "name": "my-rag-server",
  "git_url": "https://github.com/you/my-rag",
  "command": "python -m my_rag.server",
  "transport": "stdio",
  "tool_mapping": {
    "ingest": {
      "tool": "index_directory",
      "params": { "path": "{path}" }
    },
    "query": {
      "tool": "search",
      "params": { "query": "{query}", "limit": "{top_k}" }
    }
  }
}
```

If `tool_mapping` is omitted, rag-bench will auto-detect tools by analyzing tool names and schemas from `tools/list`.

## Built-in presets

| Preset | Server | Auto-detect |
|--------|--------|-------------|
| `nova-rag` | [nova-rag](https://github.com/user/nova-rag) | No (preset config) |
| `mcp-local-rag` | [mcp-local-rag](https://github.com/shinpr/mcp-local-rag) | No (preset config) |
| `chroma-mcp` | [chroma-mcp](https://github.com/chroma-core/chroma-mcp) | No (preset config) |

## Dataset

3 real open-source repositories, 90 queries with ground truth:

| Repository | Language | Size | Queries |
|------------|----------|------|---------|
| [Flask](https://github.com/pallets/flask) | Python | ~15K LOC | 30 |
| [FastAPI](https://github.com/fastapi/fastapi) | Python | ~40K LOC | 30 |
| [Express](https://github.com/expressjs/express) | JavaScript | ~15K LOC | 30 |

Each query has:
- Expected files (ground truth)
- Expected symbols (function/class names)
- Difficulty level (easy / medium / hard)
- Query type (locate / callers / explain / impact)

## Metrics

### Composite Score formula

```
Score = 0.30 * Hit@5
      + 0.15 * SymbolHit@5
      + 0.15 * MRR
      + 0.15 * ToolCallEfficiency
      + 0.15 * LatencyScore
      + 0.10 * ResourceScore
```

## Architecture

```
rag_bench/
├── cli.py          # CLI entry point (click)
├── mcp_client.py   # MCP JSON-RPC client (stdio)
├── adapter.py      # Normalizes different RAG server interfaces
├── runner.py       # Orchestrates: ingest → warmup → benchmark → metrics
├── metrics.py      # Hit@K, MRR, latency percentiles, composite score
├── baseline.py     # Grep/Glob baseline for A/B comparison
├── report.py       # Rich terminal tables
├── submit.py       # HTTP submit to leaderboard
├── presets/        # JSON configs for known servers
└── datasets/       # Repos + 90 queries with ground truth

server/
├── app.py          # FastAPI leaderboard server
├── db.py           # SQLite storage
├── models.py       # Pydantic models
└── static/         # Leaderboard web UI
```

## Contributing

1. **Add queries** — More queries improve benchmark reliability. Add to `rag_bench/datasets/queries/`
2. **Add presets** — Config files for new RAG MCP servers in `rag_bench/presets/`
3. **Add repos** — New test repositories in `rag_bench/datasets/repos.json`
4. **Submit results** — Run the benchmark and submit to the public leaderboard

## Why this exists

Existing benchmarks don't cover the RAG + MCP + code search intersection:

| | CodeRAG-Bench | GrepRAG | MCP-Bench | **rag-bench** |
|---|---|---|---|---|
| RAG for code | Yes | Yes | No | **Yes** |
| MCP protocol | No | No | Yes | **Yes** |
| A/B: RAG vs no RAG | Partial | Yes | No | **Yes** |
| Custom repos | No | No | No | **Yes** |
| Leaderboard | No | No | HuggingFace | **Self-hosted** |

## License

MIT
