"""Tests for dataset loading."""

from pathlib import Path

from rag_bench.datasets.loader import load_queries, load_repos


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
