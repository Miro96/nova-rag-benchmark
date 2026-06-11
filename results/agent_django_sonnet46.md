### Agent benchmark: Claude Code baseline vs nova-rag

Model: `claude-sonnet-4-6` · max 15 turns · repos: django · grading: file+symbol citation vs ground truth

| Metric | baseline | nova-rag |
|---|---|---|
| Accuracy (file+symbol) | 100.0% | 100.0% |
| File accuracy | 100.0% | 100.0% |
| Symbol accuracy | 100.0% | 100.0% |
| Tokens / query (mean) | 518 | 514 |
| Agent turns (mean) | 2.1 | 2.13 |
| Latency p50 (ms) | 19,786 | 19,016 |
| Total cost (USD) | $3.2921 | $3.212 |

**Δ nova-rag vs baseline:** accuracy +0.0 pp · tokens -0.8% · turns +0.03
