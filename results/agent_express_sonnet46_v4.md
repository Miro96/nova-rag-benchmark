### Agent benchmark: Claude Code baseline vs nova-rag

Model: `claude-sonnet-4-6` · max 15 turns · repos: express · grading: file+symbol citation vs ground truth

| Metric | baseline | nova-rag |
|---|---|---|
| Accuracy (file+symbol) | 100.0% | 85.7% |
| File accuracy | 100.0% | 91.4% |
| Symbol accuracy | 100.0% | 94.3% |
| Tokens / query (mean) | 586 | 899 |
| Agent turns (mean) | 2.65 | 6.49 |
| Latency p50 (ms) | 30,634 | 24,154 |
| MCP adoption | 0% | 97% |
| Total cost (USD) | $3.7193 | $5.283 |

**Δ nova-rag vs baseline:** accuracy -14.3 pp · tokens +53.2% · turns +3.84
