### Agent benchmark: Claude Code baseline vs nova-rag

Model: `claude-sonnet-4-6` · max 15 turns · repos: express · grading: file+symbol citation vs ground truth

| Metric | baseline | nova-rag |
|---|---|---|
| Accuracy (file+symbol) | 94.1% | 82.9% |
| File accuracy | 97.1% | 85.7% |
| Symbol accuracy | 97.1% | 97.1% |
| Tokens / query (mean) | 583 | 937 |
| Agent turns (mean) | 2.71 | 6.69 |
| Latency p50 (ms) | 40,819 | 28,150 |
| MCP adoption | 0% | 97% |
| Total cost (USD) | $3.7815 | $5.4463 |

**Δ nova-rag vs baseline:** accuracy -11.3 pp · tokens +60.8% · turns +3.98
