### Agent benchmark: Claude Code baseline vs nova-rag

Model: `claude-sonnet-4-6` · max 15 turns · repos: novastorm-api · grading: file+symbol citation vs ground truth

| Metric | baseline | nova-rag |
|---|---|---|
| Accuracy (file+symbol) | 100.0% | 92.9% |
| File accuracy | 100.0% | 92.9% |
| Symbol accuracy | 100.0% | 96.4% |
| Tokens / query (mean) | 686 | 571 |
| Agent turns (mean) | 2.43 | 3.36 |
| Latency p50 (ms) | 60,678 | 11,829 |
| MCP adoption | 0% | 100% |
| Total cost (USD) | $4.8732 | $3.3714 |

**Δ nova-rag vs baseline:** accuracy -7.1 pp · tokens -16.8% · turns +0.93
