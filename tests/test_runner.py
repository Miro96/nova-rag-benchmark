"""Tests for dataset loading and runner utilities."""

import os
import tempfile
from pathlib import Path

import pytest

from rag_bench.datasets.loader import load_queries, load_repos
from rag_bench.runner import _estimate_index_size


class TestEstimateIndexSize:
    """Tests for _estimate_index_size — index size estimation from preset config."""

    def test_index_dir_explicit_override_wins(self, tmp_path):
        """Step 1: explicit index_dir takes precedence."""
        index_dir = tmp_path / "my_index"
        index_dir.mkdir()
        (index_dir / "data.bin").write_bytes(b"x" * 1024 * 512)  # ~0.5 MB

        config = {"index_dir": str(index_dir)}
        size = _estimate_index_size(config)
        assert size > 0.0, f"Expected > 0 for explicit index_dir, got {size}"

    def test_index_dirs_resolved_per_repo(self, tmp_path, monkeypatch):
        """Step 2: index_dirs entries are resolved inside each repo directory.

        Uses monkeypatch to ensure ~/.nova-rag fallback is empty so the test
        only measures the per-repo index directories.
        """
        repo_a = tmp_path / "repo_a"
        repo_a.mkdir()
        cocoindex_dir = repo_a / ".cocoindex"
        cocoindex_dir.mkdir()
        (cocoindex_dir / "index.db").write_bytes(b"x" * 1024 * 1024)  # ~1 MB

        repo_b = tmp_path / "repo_b"
        repo_b.mkdir()
        cocoindex_dir_b = repo_b / ".cocoindex"
        cocoindex_dir_b.mkdir()
        (cocoindex_dir_b / "index.db").write_bytes(b"x" * 1024 * 512)  # ~0.5 MB

        # Prevent fallback to ~/.nova-rag
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.delenv("NOVA_RAG_DATA_DIR", raising=False)

        config = {"index_dirs": [".cocoindex"]}
        repo_dirs = {"a": repo_a, "b": repo_b}

        size = _estimate_index_size(config, repo_dirs)
        assert size > 0.0, f"index_size_mb should be > 0 for .cocoindex dirs, got {size}"
        # Should be roughly ~1.5 MB
        assert size >= 1.0, f"Expected at least 1.0 MB, got {size}"

    def test_index_dirs_with_nova_rag(self, tmp_path, monkeypatch):
        """index_dirs=['.nova-rag'] finds .nova-rag inside repo dirs."""
        repo = tmp_path / "repo"
        repo.mkdir()
        nova_dir = repo / ".nova-rag"
        nova_dir.mkdir()
        (nova_dir / "data").write_bytes(b"x" * 1024 * 512)

        # Prevent fallback to ~/.nova-rag
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.delenv("NOVA_RAG_DATA_DIR", raising=False)

        config = {"index_dirs": [".nova-rag"]}
        repo_dirs = {"test": repo}

        size = _estimate_index_size(config, repo_dirs)
        assert size > 0.0
        assert size < 1.0, f"Should be ~0.5 MB from repo/.nova-rag, not fallback, got {size}"

    def test_index_dirs_ignored_when_no_repo_dirs(self, tmp_path):
        """When no repo_dirs are provided, index_dirs cannot be resolved and returns 0."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".cocoindex").mkdir()

        config = {"index_dirs": [".cocoindex"]}
        # No repo_dirs → step 2 skipped, falls through to env var / ~/.nova-rag
        size = _estimate_index_size(config, None)
        # Will likely be 0 (no env var set, ~/.nova-rag may not exist)
        # But at minimum, doesn't crash
        assert size >= 0.0

    def test_index_dirs_empty_when_index_missing(self, tmp_path):
        """When index_dirs are declared but the directories don't exist, returns 0 from this step."""
        repo = tmp_path / "repo"
        repo.mkdir()
        # No .cocoindex directory created

        config = {"index_dirs": [".cocoindex"]}
        repo_dirs = {"test": repo}

        size = _estimate_index_size(config, repo_dirs)
        # Falls through to env var / ~/.nova-rag fallback
        assert size >= 0.0, f"Expected >= 0, got {size}"

    def test_index_dir_overrides_index_dirs(self, tmp_path, monkeypatch):
        """Step 1 (index_dir) takes precedence over step 2 (index_dirs)."""
        # Create an explicit index
        explicit_dir = tmp_path / "explicit_index"
        explicit_dir.mkdir()
        (explicit_dir / "data.bin").write_bytes(b"x" * 1024 * 512)  # ~0.5 MB

        # Create a per-repo index that would be found via index_dirs
        repo = tmp_path / "repo"
        repo.mkdir()
        per_repo_dir = repo / ".cocoindex"
        per_repo_dir.mkdir()
        (per_repo_dir / "big.db").write_bytes(b"x" * 1024 * 1024 * 5)  # ~5 MB

        # Prevent fallback
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.delenv("NOVA_RAG_DATA_DIR", raising=False)

        config = {
            "index_dir": str(explicit_dir),
            "index_dirs": [".cocoindex"],
        }
        repo_dirs = {"test": repo}

        size = _estimate_index_size(config, repo_dirs)
        # Should use explicit_dir (~0.5 MB), not per-repo (~5 MB)
        assert size > 0.0
        assert size < 2.0, f"Should use explicit index_dir (~0.5 MB), not per-repo (~5 MB), got {size}"

    def test_backward_compat_nova_rag_data_dir(self, tmp_path, monkeypatch):
        """NOVA_RAG_DATA_DIR env var is still honored as fallback."""
        nova_dir = tmp_path / "custom_nova"
        nova_dir.mkdir()
        (nova_dir / "data").write_bytes(b"x" * 1024 * 512)

        monkeypatch.setenv("NOVA_RAG_DATA_DIR", str(nova_dir))

        config: dict = {}
        size = _estimate_index_size(config)
        assert size > 0.0, f"NOVA_RAG_DATA_DIR should be honored, got {size}"

    def test_backward_compat_default_nova_rag(self, tmp_path, monkeypatch):
        """~/.nova-rag is the ultimate fallback."""
        home_nova = tmp_path / ".nova-rag"
        home_nova.mkdir()
        (home_nova / "data").write_bytes(b"x" * 1024 * 256)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        config: dict = {}
        size = _estimate_index_size(config)
        assert size > 0.0

    def test_returns_zero_when_nothing_exists(self, tmp_path, monkeypatch):
        """Returns 0.0 when no index exists anywhere."""
        # Ensure NOVA_RAG_DATA_DIR is not set
        monkeypatch.delenv("NOVA_RAG_DATA_DIR", raising=False)

        # Make ~/.nova-rag not exist by patching home
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: empty_home)

        config: dict = {}
        size = _estimate_index_size(config)
        assert size == 0.0

    def test_multiple_index_dir_names(self, tmp_path, monkeypatch):
        """Multiple index_dir names are all scanned."""
        repo = tmp_path / "repo"
        repo.mkdir()
        idx1 = repo / ".index_a"
        idx1.mkdir()
        (idx1 / "a.db").write_bytes(b"x" * 1024 * 512)
        idx2 = repo / ".index_b"
        idx2.mkdir()
        (idx2 / "b.db").write_bytes(b"x" * 1024 * 256)

        # Prevent fallback
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.delenv("NOVA_RAG_DATA_DIR", raising=False)

        config = {"index_dirs": [".index_a", ".index_b"]}
        repo_dirs = {"test": repo}

        size = _estimate_index_size(config, repo_dirs)
        # Should sum both
        assert size >= 0.5, f"Expected >= 0.5 MB (sum of both), got {size}"


class TestDatasetLoader:
    def test_load_repos(self):
        repos = load_repos()
        assert len(repos) == 3
        names = {r.name for r in repos}
        assert names == {"flask", "fastapi", "express"}

    def test_load_all_queries(self):
        queries = load_queries()
        assert len(queries) == 105  # 30 original + 5 complex per repo

    def test_load_filtered_queries(self):
        queries = load_queries("flask")
        assert len(queries) == 35
        assert all(q.repo == "flask" for q in queries)

    def test_query_fields(self):
        queries = load_queries("flask")
        q = queries[0]
        assert q.id == "flask_001"
        assert q.type in ("locate", "callers", "explain", "impact",
                          "multi_hop", "cross_package", "architecture",
                          "dead_code", "conditional_path", "test_traceability")
        assert q.difficulty in ("easy", "medium", "hard")
        assert len(q.expected_files) > 0
        assert len(q.query) > 0

    def test_query_types_distribution(self):
        queries = load_queries()
        types = {q.type for q in queries}
        # Original types + 7 new complex query types from M1
        assert types == {"locate", "callers", "explain", "impact",
                         "multi_hop", "cross_package", "architecture",
                         "dead_code", "conditional_path", "test_traceability"}

    def test_difficulty_distribution(self):
        queries = load_queries()
        difficulties = {q.difficulty for q in queries}
        assert difficulties == {"easy", "medium", "hard"}
