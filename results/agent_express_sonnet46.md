### Agent benchmark: Claude Code baseline vs nova-rag

Model: `claude-sonnet-4-6` · max 15 turns · repos: express · grading: file+symbol citation vs ground truth

| Metric | baseline | nova-rag |
|---|---|---|
| Accuracy (file+symbol) | 97.1% | 97.0% |
| File accuracy | 97.1% | 97.0% |
| Symbol accuracy | 100.0% | 100.0% |
| Tokens / query (mean) | 640 | 593 |
| Agent turns (mean) | 3 | 2.7 |
| Latency p50 (ms) | 37,389 | 23,957 |
| Total cost (USD) | $3.8745 | $3.5939 |

**Δ nova-rag vs baseline:** accuracy -0.2 pp · tokens -7.3% · turns -0.30
