### Agent benchmark: Claude Code baseline vs nova-rag

Model: `claude-sonnet-4-6` · max 15 turns · repos: novastorm-api · grading: file+symbol citation vs ground truth

| Metric | baseline | nova-rag |
|---|---|---|
| Accuracy (file+symbol) | 100.0% | 100.0% |
| File accuracy | 100.0% | 100.0% |
| Symbol accuracy | 100.0% | 100.0% |
| Tokens / query (mean) | 668 | 675 |
| Agent turns (mean) | 2.64 | 3.93 |
| Latency p50 (ms) | 68,962 | 19,394 |
| MCP adoption | 0% | 100% |
| Total cost (USD) | $4.823 | $3.4989 |

**Δ nova-rag vs baseline:** accuracy +0.0 pp · tokens +0.9% · turns +1.29
