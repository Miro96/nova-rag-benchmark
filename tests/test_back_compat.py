"""Back-compatibility verification tests.

These tests ensure the mission did not regress existing functionality:
  1. No preset files were modified (VAL-COMPAT-002).
  2. The number of surviving test cases has not decreased below the pre-mission
     baseline — i.e. no pre-existing tests were deleted (VAL-COMPAT-001).

The pre-mission baseline was 325 total tests with 66 tests in the three
known-broken files (test_bm25.py, test_naive_rag.py, test_cocoindex.py),
leaving 259 tests in the surviving files.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRE_MISSION_SURVIVING_TEST_COUNT = 259

# ANSI escape sequence pattern for stripping colour codes from pytest output.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (colour codes) from a string."""
    return _ANSI_RE.sub("", text)


def _parse_collected_count(output: str) -> int:
    """Parse the 'N tests collected' number from pytest --collect-only output."""
    plain = _strip_ansi(output)
    for line in plain.splitlines():
        stripped = line.strip()
        if "tests collected" in stripped:
            # Line looks like "429 tests collected in 0.17s"
            try:
                return int(stripped.split()[0])
            except (ValueError, IndexError):
                pass
    raise RuntimeError(
        f"Could not parse collected test count from pytest output:\n{plain}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_presets_unchanged():
    """VAL-COMPAT-002: git diff shows zero changes in rag_bench/presets/."""
    result = subprocess.run(
        ["git", "diff", "--quiet", "rag_bench/presets/"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"git diff --quiet rag_bench/presets/ exited {result.returncode} — "
        f"preset files have been modified. stderr: {result.stderr!r}"
    )


def test_surviving_test_count_not_decreased():
    """VAL-COMPAT-001: collection count in surviving test files >= baseline.

    Runs ``pytest --collect-only`` on all test files *except* the three
    known-broken ones (test_bm25, test_naive_rag, test_cocoindex) and asserts
    that the collected count is ≥ the pre-mission count of 259.  This proves
    that no pre-existing tests were deleted during the mission.
    """
    venv_pytest = str(REPO_ROOT / ".venv" / "bin" / "pytest")
    ignore_args = [
        "--ignore=tests/test_bm25.py",
        "--ignore=tests/test_cocoindex.py",
        "--ignore=tests/test_naive_rag.py",
    ]
    result = subprocess.run(
        [venv_pytest, "tests/", "--collect-only", "-q"] + ignore_args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    # pytest may put the summary on stdout or stderr; try both.
    combined = result.stdout + "\n" + result.stderr
    count = _parse_collected_count(combined)

    assert count >= PRE_MISSION_SURVIVING_TEST_COUNT, (
        f"Surviving test count {count} is below the pre-mission baseline "
        f"of {PRE_MISSION_SURVIVING_TEST_COUNT}. "
        f"Pre-existing tests may have been deleted."
    )
