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

## How It Works

### Architecture pipeline

```
Presets (JSON configs)
    │
    ▼
Runner ──► ingest ──► warmup ──► benchmark ──► results
    │
    ▼
Report generator (Rich tables / JSON / leaderboard submit)
```

The benchmark follows a strict three-phase pipeline per run:

1. **Ingest** — The RAG server indexes each repository in the dataset. Wall-clock time is measured as "ingest speed" (files/sec). Servers that support pre-existing indexes can skip this phase after the first run.

2. **Warmup** — Each query type is run once (results discarded) to ensure caches, JIT compilers, and connection pools are in steady state before measurement begins.

3. **Benchmark** — All 105 queries are executed against every repository in randomized order. Each query is timed individually (p50/p95/mean latency). Results are compared against known ground truth to compute Hit@K, MRR, MAP, and Symbol Hit@5.

4. **Results** — Metrics are aggregated across 3 independent replicates. Median values are reported with interquartile range (IQR). Statistical significance (A/B deltas) is computed for multi-server comparisons.

### Presets

Each preset is a JSON configuration file that declares a server's transport, tool mapping, and optional pre-ingest steps. The benchmark ships with 5 presets spanning the full spectrum of code retrieval approaches:

| Preset | Approach | Transport | Description |
|--------|----------|-----------|-------------|
| `nova-rag` | Hybrid (semantic + graph + keyword) | MCP stdio | Full code graph with callers, callees, impact analysis, and dead code detection |
| `grep-glob` | Lexical (keyword) | InProcess | Pure ripgrep + glob pattern matching — no embeddings, no indexing |
| `bm25` | Keyword (statistical) | CLI | BM25 term-frequency scoring via CLI subprocess |
| `cocoindex-code` | Vector (embedding) | MCP stdio | CocoIndex code retrieval pipeline with chunked embeddings |
| `naive-rag` | Vector (embedding) | InProcess | Minimal embedding-only similarity search — no code graph, no chunking heuristics |

### Query types

The benchmark includes 105 queries across 10 distinct categories, each testing a different real-world code search need:

| # | Type | Description | Example question |
|---|------|-------------|------------------|
| 1 | `locate` | Find where a symbol is defined | "Where is `create_app` defined?" |
| 2 | `explain` | Understand how a feature works (multi-file) | "How does request dispatching work?" |
| 3 | `callers` | Find all call sites of a function | "Who calls `full_dispatch_request`?" |
| 4 | `callees` | Find what a function depends on | "What does `wsgi_app` call?" |
| 5 | `architecture` | Understand project structure and layering | "How is the routing layer organized?" |
| 6 | `impact` | Blast radius of changing a function | "What breaks if I change `url_for`?" |
| 7 | `dead_code` | Detect unused functions and methods | "Is `_get_request` called anywhere?" |
| 8 | `multi_hop` | Trace transitive relationships across files | "Trace request from WSGI to response" |
| 9 | `conditional_path` | Find code paths under specific conditions | "What code paths handle 404 errors?" |
| 10 | `test_traceability` | Map tests to the functions they cover | "Which tests cover `make_response`?" |

### Metrics

Every run computes a comprehensive set of retrieval quality, performance, and efficiency metrics:

| Metric | Category | What it measures | Why it matters |
|--------|----------|------------------|----------------|
| **Hit@1 / Hit@5 / Hit@10** | Retrieval | Fraction of queries where at least one correct file appears in top-K results | Measures raw recall — did the engine surface the right file at all? |
| **Symbol Hit@5** | Retrieval | Fraction of queries where the correct function/class name appears in top-5 results | Unique to rag-bench. Tests whether the engine understands code structure, not just text similarity. Only graph-aware engines can score here. |
| **MRR** (Mean Reciprocal Rank) | Retrieval | Average of 1/rank of the first correct result | Rewards engines that rank the correct answer higher rather than burying it |
| **MAP** (Mean Average Precision) | Retrieval | Average precision across all recall levels | Rewards engines that cluster all relevant results at the top |
| **Latency p50 / p95 / mean** | Performance | Query response time percentiles in milliseconds | Real-world responsiveness. p95 captures tail latency that users notice |
| **Ingest Speed** (files/sec) | Efficiency | Number of source files indexed per second | Matters for large codebases and CI/CD integration |
| **RAM / Index Size** | Efficiency | Peak memory and on-disk index footprint | Resource budget for constrained environments |
| **Composite Score** | Aggregate | Weighted combination (see formula below) | Single number for leaderboard ranking |
| **A/B Delta** | Comparative | Statistical difference between two servers across all metrics | Answers "is server A actually better than server B?" with confidence intervals |

### Reproducibility

Every result in the comparison table is the **median of 3 independent replicates** with seeded randomness (seed fixed per replicate for deterministic shuffling). This ensures:

- **Within-replicate determinism**: Same seed → same query order → identical results for a given server version
- **Across-replicate variance**: Each replicate uses a different random seed, capturing natural variability in server startup, OS scheduling, and I/O
- **IQR reporting**: Interquartile range shows the spread — small IQR means the metric is stable; large IQR flags noise
- **Statistical rigor**: A/B comparison uses paired testing across replicates to compute delta confidence intervals

All presets in the comparison table had coefficient of variation (CV) < 0.05 for composite score across 3 replicates.

### Code layout

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

## Comparison with Other Benchmarks

Existing benchmarks evaluate either embedding models, document retrieval, or coding agents — none measures end-to-end code intelligence retrieval pipelines. Here's how rag-bench differs:

| Benchmark | Domain | What it measures | Graph queries? | Symbol matching? | Leaderboard? | Why rag-bench is different |
|-----------|--------|------------------|----------------|-------------------|--------------|---------------------------|
| [**MTEB**](https://huggingface.co/spaces/mteb/leaderboard) | Text embeddings | Embedding model quality on 100+ datasets (clustering, classification, STS, retrieval) | No | No | Yes (HuggingFace) | Tests only the embedding layer, not the full retrieval pipeline. No code-specific tasks — models are ranked by how well they embed StackOverflow questions, not how well they find the right function in a codebase. No graph queries, no symbol matching. |
| [**CoIR**](https://github.com/CoIR-team/coir) (ACL 2025) | Code retrieval | Text-to-code and code-to-code retrieval quality using 10 code-specific datasets | No | No | No (dataset only) | Closest to rag-bench in domain, but limited to embedding-based retrieval: "given a natural language description, find the code" or "given code, find similar code." No graph queries (callers, callees, impact), no dead code detection, no symbol-level accuracy. No public leaderboard — results must be reproduced from paper. |
| [**CRAG**](https://www.cragbenchmark.org/) (Comprehensive RAG) | Document QA + Web | Factual question answering over documents, web search, and knowledge graphs | No | No | Yes | General-purpose document RAG, not code. Queries are Wikipedia-style facts ("When did X happen?"), not code search. No code graph navigation, no multilanguage codebase testing. |
| [**SWE-bench**](https://www.swebench.com/) | Coding agents | End-to-end bug fixing: given a GitHub issue, produce a correct patch | No | No | Yes | Measures agent coding ability (can the LLM write a correct fix?), not retrieval quality. A SWE-bench agent may use RAG internally, but the benchmark doesn't isolate or measure search quality — it measures patch correctness. Complements rag-bench: SWE-bench tests the agent, rag-bench tests the retrieval engine the agent uses. |
| **rag-bench** (this repo) | **Code intelligence retrieval** | End-to-end search pipeline: hybrid, lexical, vector, and graph-based retrieval compared side-by-side on real codebases | **Yes** (callers, callees, impact, multi-hop, architecture) | **Yes** (Symbol Hit@5) | **Yes** (self-hosted) | The only benchmark that: (1) tests graph queries essential for real code navigation, (2) measures symbol-level accuracy (did you find the right function, not just the right file?), (3) detects dead code, and (4) compares multiple retrieval strategies head-to-head — not just embedding models but also grep/glob and hybrid approaches. |

### Why these differences matter

- **Embedding-only benchmarks (MTEB, CoIR)** tell you which model to use, but not whether your retrieval pipeline works. A great embedding model with poor chunking, no graph, and no hybrid fallback will still fail on real code queries.
- **Document QA benchmarks (CRAG)** test a fundamentally different problem: finding facts in prose documents vs finding the right function in a codebase with imports, inheritance, and control flow.
- **Agent benchmarks (SWE-bench)** conflate retrieval quality with coding ability. A perfect retriever paired with a weak LLM scores poorly; a weak retriever with a strong LLM that re-reads the entire repo can still pass.

rag-bench isolates retrieval quality from the LLM so you can measure and improve the search engine independently.

## License

MIT
