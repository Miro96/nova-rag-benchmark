"""Re-grade a saved agent-benchmark result against current ground truth.

Grading is programmatic (file+symbol citation), so dataset fixes can be
applied retroactively to stored answers without re-running any agents:

    python scripts/regrade.py results/agent_express_sonnet46_v3.json

Rewrites the file's per-query grades and summary in place (a .bak copy
is kept) and prints the before/after accuracy per condition.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_bench.agent_bench import AgentQueryResult, aggregate, grade  # noqa: E402
from rag_bench.datasets.loader import load_queries  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    doc = json.loads(path.read_text())

    queries = {q.id: q for repo in doc["repos"] for q in load_queries(repo)}

    before = {c: doc["summary"][c].get("accuracy") for c in doc["conditions"]
              if c in doc["summary"]}

    rows = []
    for q in doc["queries"]:
        gt = queries.get(q["query_id"])
        if gt is not None and not q.get("error"):
            q["file_hit"], q["symbol_hit"], q["correct"] = grade(q["answer"], gt)
        known = {f.name for f in AgentQueryResult.__dataclass_fields__.values()}
        rows.append(AgentQueryResult(**{k: v for k, v in q.items() if k in known}))

    doc["summary"] = aggregate(rows)
    doc["regraded"] = True

    shutil.copy(path, path.with_suffix(".json.bak"))
    path.write_text(json.dumps(doc, indent=2))

    for cond in doc["conditions"]:
        s = doc["summary"].get(cond, {})
        if s.get("queries"):
            print(f"{cond}: accuracy {before.get(cond)} -> {s['accuracy']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
