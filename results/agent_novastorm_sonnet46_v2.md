### Agent benchmark: Claude Code baseline vs nova-rag

Model: `claude-sonnet-4-6` · max 15 turns · repos: novastorm-api · grading: file+symbol citation vs ground truth

| Metric | baseline | nova-rag |
|---|---|---|
| Accuracy (file+symbol) | 100.0% | 96.4% |
| File accuracy | 100.0% | 96.4% |
| Symbol accuracy | 100.0% | 100.0% |
| Tokens / query (mean) | 681 | 748 |
| Agent turns (mean) | 2.43 | 2.79 |
| Latency p50 (ms) | 66,629 | 65,232 |
| MCP adoption | 0% | 18% |
| Total cost (USD) | $4.6169 | $4.6854 |

**Δ nova-rag vs baseline:** accuracy -3.6 pp · tokens +9.9% · turns +0.36
