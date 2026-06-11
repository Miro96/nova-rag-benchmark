### Agent benchmark: Claude Code baseline vs nova-rag

Model: `claude-sonnet-4-6` · max 15 turns · repos: express · grading: file+symbol citation vs ground truth

| Metric | baseline | nova-rag |
|---|---|---|
| Accuracy (file+symbol) | 97.1% | 97.1% |
| File accuracy | 97.1% | 97.1% |
| Symbol accuracy | 100.0% | 100.0% |
| Tokens / query (mean) | 612 | 676 |
| Agent turns (mean) | 2.94 | 3.17 |
| Latency p50 (ms) | 40,033 | 43,810 |
| MCP adoption | 0% | 9% |
| Total cost (USD) | $3.9551 | $4.2213 |

**Δ nova-rag vs baseline:** accuracy +0.0 pp · tokens +10.4% · turns +0.23
