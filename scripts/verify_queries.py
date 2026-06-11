"""Verify a query set's ground truth against the actual repo checkout.

Generic version of verify_django_queries.py — works for any repo in the
benchmark cache (including private overlay repos) and any language:

    python scripts/verify_queries.py <repo-name>

For every query:
- each expected_file must exist in ~/.cache/rag-bench/repos/<repo-name>;
- at least one expected_symbol must be defined (def/class/interface/
  record/struct/function) or at least present in an expected_file;
- expected_content is auto-filled for locate/callers entries with the
  definition line (extended to >= 20 chars), and stripped from other
  types per dataset convention.

Exit code 1 with a report if anything fails — nothing hand-waved.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CACHE = Path.home() / ".cache" / "rag-bench" / "repos"
QUERIES_DIR = Path(__file__).parent.parent / "rag_bench" / "datasets" / "queries"

# Definition patterns across the benchmark's languages
_DEF_TEMPLATES = [
    # Python
    r"\s*(?:class|def|async def)\s+{sym}\b",
    # C# / Java / TS-ish type declarations
    r".*\b(?:class|interface|record|struct|enum)\s+{sym}\b",
    # C# / Java methods & properties: visibility ... Name( or Name {{
    r"\s*(?:public|private|protected|internal|static|override|virtual|async|sealed|partial|\s)+[\w<>\[\],?\s]+\s+{sym}\s*[(<{{]",
    # JS/TS functions
    r"\s*(?:export\s+)?(?:async\s+)?function\s+{sym}\b",
]


def find_symbol_line(lines: list[str], symbol: str) -> str | None:
    def _extend(idx: int) -> str:
        chunk = lines[idx].rstrip()
        j = idx + 1
        while len(chunk) < 20 and j < len(lines):
            chunk += "\n" + lines[j].rstrip()
            j += 1
        return chunk

    for template in _DEF_TEMPLATES:
        pat = re.compile(template.format(sym=re.escape(symbol)))
        for i, line in enumerate(lines):
            if pat.match(line):
                return _extend(i)
    word = re.compile(rf"\b{re.escape(symbol)}\b")
    for i, line in enumerate(lines):
        if word.search(line):
            return _extend(i)
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    repo_name = sys.argv[1]
    repo = CACHE / repo_name
    queries_path = QUERIES_DIR / f"{repo_name}.jsonl"
    if not repo.exists():
        print(f"repo checkout not found: {repo}")
        return 1
    if not queries_path.exists():
        print(f"query set not found: {queries_path}")
        return 1

    failures: list[str] = []
    rows = []
    for raw in queries_path.read_text().splitlines():
        if not raw.strip():
            continue
        q = json.loads(raw)
        content_line = None
        for f in q["expected_files"]:
            path = repo / f
            if not path.exists():
                failures.append(f"{q['id']}: missing file {f}")
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for sym in q["expected_symbols"]:
                line = find_symbol_line(lines, sym)
                if line and content_line is None:
                    content_line = line
        if content_line is None:
            failures.append(
                f"{q['id']}: NO expected symbol found in any expected file "
                f"(files={q['expected_files']}, symbols={q['expected_symbols']})"
            )
        elif q["type"] in ("locate", "callers"):
            q["expected_content"] = [content_line]
        else:
            q.pop("expected_content", None)
        rows.append(q)

    if failures:
        print("VERIFICATION FAILED:")
        for f in failures:
            print(" -", f)
        return 1

    queries_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"OK: {len(rows)} queries verified against {repo}; expected_content filled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
