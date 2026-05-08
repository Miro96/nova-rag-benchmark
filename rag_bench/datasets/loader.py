"""Dataset loader: clones repos and loads queries."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).parent
REPOS_JSON = DATASETS_DIR / "repos.json"
QUERIES_DIR = DATASETS_DIR / "queries"
WARMUP_JSONL = DATASETS_DIR / "warmup.jsonl"
CACHE_DIR = Path.home() / ".cache" / "rag-bench" / "repos"


@dataclass
class Query:
    id: str
    type: str
    query: str
    expected_files: list[str]
    expected_symbols: list[str]
    difficulty: str
    repo: str


@dataclass
class WarmupQuery:
    id: str
    query: str


@dataclass
class RepoInfo:
    name: str
    git_url: str
    ref: str  # tag or commit for reproducibility
    language: str
    size: str  # small, medium, large


def load_repos() -> list[RepoInfo]:
    """Load repository definitions."""
    data = json.loads(REPOS_JSON.read_text())
    return [RepoInfo(**r) for r in data]


def clone_repo(repo: RepoInfo) -> Path:
    """Clone a repo to cache dir, return local path."""
    repo_dir = CACHE_DIR / repo.name
    if repo_dir.exists():
        logger.info("Repo %s already cached at %s", repo.name, repo_dir)
        return repo_dir

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s (%s)...", repo.name, repo.git_url)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", repo.ref,
         repo.git_url, str(repo_dir)],
        check=True,
        capture_output=True,
    )
    logger.info("Cloned %s to %s", repo.name, repo_dir)
    return repo_dir


def load_queries(repo_filter: str | None = None) -> list[Query]:
    """Load all queries, optionally filtered by repo name."""
    queries = []
    for qfile in sorted(QUERIES_DIR.glob("*.jsonl")):
        repo_name = qfile.stem
        if repo_filter and repo_name != repo_filter:
            continue
        for line in qfile.read_text().strip().split("\n"):
            if not line.strip():
                continue
            data = json.loads(line)
            queries.append(Query(
                id=data["id"],
                type=data["type"],
                query=data["query"],
                expected_files=data["expected_files"],
                expected_symbols=data.get("expected_symbols", []),
                difficulty=data["difficulty"],
                repo=repo_name,
            ))
    logger.info("Loaded %d queries%s", len(queries),
                f" (repo={repo_filter})" if repo_filter else "")
    return queries


def load_warmup_queries() -> list[WarmupQuery]:
    """Load the dedicated warmup query set.

    Warmup queries live in ``datasets/warmup.jsonl`` and are intentionally
    disjoint from the scored benchmark set so they can warm caches/JITs
    without polluting Hit@K or latency measurements with the same items.
    """
    if not WARMUP_JSONL.exists():
        return []
    items: list[WarmupQuery] = []
    text = WARMUP_JSONL.read_text().strip()
    if not text:
        return items
    for line in text.split("\n"):
        if not line.strip():
            continue
        data = json.loads(line)
        items.append(WarmupQuery(id=data["id"], query=data["query"]))
    return items


def get_repo_files(repo_dir: Path, extensions: set[str] | None = None) -> list[Path]:
    """Get all source files in a repo directory."""
    if extensions is None:
        extensions = {".py", ".js", ".ts", ".go", ".rs", ".java", ".rb",
                      ".md", ".txt", ".yaml", ".yml", ".toml", ".json"}

    files = []
    for f in repo_dir.rglob("*"):
        if f.is_file() and f.suffix in extensions:
            # Skip hidden dirs, node_modules, __pycache__, .git
            parts = f.relative_to(repo_dir).parts
            if any(p.startswith(".") or p in ("node_modules", "__pycache__", "venv")
                   for p in parts):
                continue
            files.append(f)
    return sorted(files)
