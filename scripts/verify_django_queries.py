"""Verify django.jsonl ground truth against the actual Django 5.2 checkout.

For every query:
- each expected_file must exist in the repo;
- at least one expected_symbol must be *defined* (def/class) or present
  in at least one expected_file;
- expected_content is auto-filled with the definition line of the first
  matched symbol (keeps the dataset honest and reproducible).

Exit code 1 with a report if anything fails — nothing hand-waved.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path.home() / ".cache" / "rag-bench" / "repos" / "django"
QUERIES = Path(__file__).parent.parent / "rag_bench" / "datasets" / "queries" / "django.jsonl"


def find_symbol_line(text: str, symbol: str) -> str | None:
    """Prefer a def/class definition line; fall back to any word-bounded use.

    Entries must be exact contiguous substrings of the file (the retrieval
    bench matches them against chunk text) and at least 20 chars long
    (enforced by tests/test_loader.py) — short multi-line signatures like
    ``def save(`` get extended with the following source lines.
    """
    lines = text.splitlines()

    def _extend(idx: int) -> str:
        chunk = lines[idx].rstrip()
        j = idx + 1
        while len(chunk) < 20 and j < len(lines):
            chunk += "\n" + lines[j].rstrip()
            j += 1
        return chunk

    for i, line in enumerate(lines):
        if re.match(rf"\s*(class|def|async def)\s+{re.escape(symbol)}\b", line):
            return _extend(i)
    for i, line in enumerate(lines):
        if re.search(rf"\b{re.escape(symbol)}\b", line):
            return _extend(i)
    return None


def main() -> int:
    failures: list[str] = []
    rows = []
    for raw in QUERIES.read_text().splitlines():
        if not raw.strip():
            continue
        q = json.loads(raw)
        content_line = None
        for f in q["expected_files"]:
            path = REPO / f
            if not path.exists():
                failures.append(f"{q['id']}: missing file {f}")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for sym in q["expected_symbols"]:
                line = find_symbol_line(text, sym)
                if line and content_line is None:
                    content_line = line
        if content_line is None:
            failures.append(
                f"{q['id']}: NO expected symbol found in any expected file "
                f"(files={q['expected_files']}, symbols={q['expected_symbols']})"
            )
        elif q["type"] in ("locate", "callers"):
            # Dataset convention (enforced by tests/test_loader.py):
            # only locate/callers entries carry expected_content.
            q["expected_content"] = [content_line]
        else:
            q.pop("expected_content", None)
        rows.append(q)

    if failures:
        print("VERIFICATION FAILED:")
        for f in failures:
            print(" -", f)
        return 1

    QUERIES.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"OK: {len(rows)} queries verified against {REPO}; expected_content filled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
