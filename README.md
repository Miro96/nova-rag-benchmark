# rag-bench

**Benchmark for Code RAG MCP Servers** — measure how well RAG helps AI find the right code.

No existing benchmark covers the intersection of **code search + MCP protocol + A/B comparison**. rag-bench fills that gap.

## What it measures

rag-bench indexes real open-source repositories (Flask, FastAPI, Express) and runs 105 code search queries with known correct answers. It measures:

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
- `multi_hop` — "Trace X through multiple files to Y" (transitive reasoning)
- `cross_package` — "How does module A interact with module B?"
- `architecture` — "How is the project structured and layered?"
- `dead_code` — "Is function X actually called anywhere?"
- `conditional_path` — "What code paths exist when condition C is true?"
- `test_traceability` — "Which tests cover function X?"

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

# Run tests (325 tests)
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

### Transport modes

rag-bench supports three client transport modes via the `BaseClient` interface:

| Mode | Flag | Description |
|------|------|-------------|
| **MCP stdio** | `--transport stdio` | JSON-RPC over subprocess stdin/stdout (default) |
| **CLI subprocess** | `--transport cli` | Subprocess with CLI args, stdout capture |
| **In-process** | `--transport inprocess` | Direct Python import (presets only) |

```bash
# MCP stdio (default)
rag-bench run --command "python -m my_server" --transport stdio

# CLI subprocess
rag-bench run --command "my-rag search" --transport cli

# In-process (requires Python module)
rag-bench run --preset naive-rag --transport inprocess
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

### Pre-ingest commands

You can declare shell commands to run before indexing via the `pre_ingest` field. Use `{repo_path}` as a template variable for the repository directory and `{cwd}` for the working directory at benchmark time:

```json
{
  "name": "my-rag-server",
  "pre_ingest": [
    "pip install -r {repo_path}/requirements.txt",
    "cd {repo_path} && npm install"
  ]
}
```

### Index directories

Preset configs can declare which directories to index (defaults to the repo root):

```json
{
  "name": "my-rag-server",
  "index_directories": ["src/", "lib/"]
}
```

By default all non-ignored files are indexed; `index_directories` restricts indexing to specific subdirectories.

## Built-in presets

| Preset | Server | Description |
|--------|--------|-------------|
| `nova-rag` | [nova-rag](https://github.com/user/nova-rag) | Hybrid semantic + graph + keyword RAG |
| `naive-rag` | Embedding-only baseline | Naive embedding similarity (no code graph) |
| `cocoindex-code` | [CocoIndex](https://github.com/cocoindex/cocoindex) v0.2.33 | CocoIndex Code retrieval preset |
| `grep-glob` | Keyword baseline | Pure grep/glob keyword search (no embeddings) |
| `mcp-local-rag` | [mcp-local-rag](https://github.com/shinpr/mcp-local-rag) | MCP local RAG server |
| `chroma-mcp` | [chroma-mcp](https://github.com/chroma-core/chroma-mcp) | Chroma vector DB MCP server |

### Comparison results (105 queries, 3 replicates)

4-way comparison across all built-in presets on Flask + FastAPI + Express:

```bash
rag-bench compare --presets nova-rag,naive-rag,cocoindex-code,grep-glob --replicates 3
```

| Metric | nova-rag | naive-rag | cocoindex-code | grep-glob |
|--------|----------|-----------|----------------|-----------|
| **Hit@1** | 20.0% | 1.9% | 4.8% | 20.0% |
| **Hit@5** | 45.7% | 6.7% | 16.2% | 55.2% |
| **Hit@10** | 53.3% | 7.6% | 23.8% | 74.3% |
| **Symbol Hit@5** | 33.3% | 13.3% | 0.0% | 0.0% |
| **MRR** | 0.315 | 0.037 | 0.099 | 0.340 |
| **Latency p50** | 14.6 ms | 3.5 ms | 447.1 ms | 22.3 ms |
| **Latency p95** | 24.4 ms | 4.0 ms | 573.5 ms | 90.6 ms |
| **Ingest Speed** | 2,675 f/s | 206 f/s | — | — |
| **Composite Score** | **0.550** | 0.364 | 0.327 | 0.523 |

**Key takeaways:**
- **nova-rag** is the only preset that resolves symbols (33.3% Symbol Hit@5) — grep/glob and embedding-only approaches can't identify function/class names.
- **grep-glob** wins on raw file recall (55.2% Hit@5) but returns no symbol-level results.
- **naive-rag** (embedding-only) performs worst on file recall (6.7% Hit@5), showing that semantic similarity alone is insufficient for code search.
- **cocoindex-code** has high latency (~450 ms p50) vs sub-25ms for other approaches.
- **nova-rag** indexes 13× faster than naive-rag (2,675 vs 206 files/sec).

Reproducibility: all presets had CV < 0.05 across 3 replicates for composite score.

## Dataset

3 real open-source repositories, 105 queries with ground truth:

| Repository | Language | Size | Queries |
|------------|----------|------|---------|
| [Flask](https://github.com/pallets/flask) | Python | ~15K LOC | 35 |
| [FastAPI](https://github.com/fastapi/fastapi) | Python | ~40K LOC | 35 |
| [Express](https://github.com/expressjs/express) | JavaScript | ~15K LOC | 35 |

Each query has:
- Expected files (ground truth)
- Expected symbols (function/class names)
- Difficulty level (easy / medium / hard)
- Query type (locate / callers / explain / impact / multi_hop / cross_package / architecture / dead_code / conditional_path / test_traceability)

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
├── cli.py              # CLI entry point (click)
├── transport/          # Client transport layer
│   ├── base.py         # BaseClient interface
│   ├── mcp_client.py   # MCP JSON-RPC client (stdio)
│   ├── cli_client.py   # Subprocess CLI client
│   └── in_process.py   # In-process (Python import) client
├── adapter.py          # Normalizes different RAG server interfaces
├── runner.py           # Orchestrates: ingest → warmup → benchmark → metrics
├── metrics.py          # Hit@K, MRR, latency percentiles, composite score
├── baseline.py         # Grep/Glob baseline for A/B comparison
├── report.py           # Rich terminal tables
├── submit.py           # HTTP submit to leaderboard
├── presets/            # JSON configs for known servers
└── datasets/           # Repos + 105 queries with ground truth

server/
├── app.py              # FastAPI leaderboard server
├── db.py               # SQLite storage
├── models.py           # Pydantic models
└── static/             # Leaderboard web UI
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
