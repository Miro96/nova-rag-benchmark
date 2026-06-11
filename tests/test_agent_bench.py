"""Tests for the end-to-end agent A/B benchmark (no real claude calls)."""

from pathlib import Path
from unittest.mock import patch

from rag_bench.agent_bench import (
    AgentCondition,
    BASELINE_TOOLS,
    _build_claude_cmd,
    aggregate,
    grade,
    grade_files,
    grade_symbols,
    render_markdown,
    run_agent_benchmark,
)
from rag_bench.datasets.loader import Query, RepoInfo


def _query(**kw):
    base = dict(
        id="q1", type="locate", query="where is the app class?",
        expected_files=["src/flask/app.py"], expected_symbols=["Flask"],
        difficulty="easy", repo="flask", expected_content=[],
    )
    base.update(kw)
    return Query(**base)


class TestGrading:
    def test_file_match_two_components(self):
        assert grade_files("It's defined in src/flask/app.py", ["src/flask/app.py"])
        assert grade_files("See flask/app.py line 50", ["src/flask/app.py"])
        assert grade_files("Look at /abs/path/src/flask/app.py", ["src/flask/app.py"])

    def test_bare_basename_does_not_count(self):
        # "app.py" alone is too generic — parent dir required
        assert not grade_files("It's in app.py somewhere", ["src/flask/app.py"])

    def test_windows_separators_normalized(self):
        assert grade_files(r"Defined in src\flask\app.py", ["src/flask/app.py"])

    def test_symbol_word_boundary(self):
        assert grade_symbols("The Flask class handles it", ["Flask"])
        assert not grade_symbols("Use flask_restful instead", ["Flask"])

    def test_no_expected_symbols_passes(self):
        assert grade_symbols("anything", [])

    def test_combined_grade(self):
        q = _query()
        file_hit, symbol_hit, correct = grade(
            "The Flask class lives in src/flask/app.py", q
        )
        assert file_hit and symbol_hit and correct
        _, _, wrong = grade("It is in flask/app.py", q)  # file yes, symbol no
        assert not wrong


class TestCommandConstruction:
    def test_baseline_has_no_mcp(self):
        cond = AgentCondition("baseline", list(BASELINE_TOOLS))
        cmd = _build_claude_cmd("q", cond, model="sonnet", max_turns=10)
        assert "--mcp-config" not in cmd
        # Hermetic flags (NOT --bare, which breaks headless auth)
        assert "--strict-mcp-config" in cmd
        assert "--setting-sources" in cmd
        assert "--bare" not in cmd
        joined = " ".join(cmd)
        assert "Grep,Glob,Read" in joined
        assert "dontAsk" in joined
        assert "--model" in cmd and "sonnet" in cmd

    def test_treatment_attaches_mcp(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        cfg.write_text("{}")
        cond = AgentCondition(
            "nova-rag", list(BASELINE_TOOLS) + ["mcp__nova-rag"], mcp_config=cfg
        )
        cmd = _build_claude_cmd("q", cond, model=None, max_turns=10)
        assert "--mcp-config" in cmd and str(cfg) in cmd
        assert "mcp__nova-rag" in " ".join(cmd)
        assert "--model" not in cmd


def _fake_runner_factory(answers: dict):
    """Returns a runner that answers by condition name."""

    def runner(prompt, condition, cwd, model, max_turns, timeout):
        text = answers.get(condition.name, "no idea")
        return {
            "result": text,
            "usage": {"input_tokens": 1000, "output_tokens": 100},
            "cost": {"total_cost_usd": 0.01},
            "num_turns": 3 if condition.name == "nova-rag" else 6,
            "duration_ms": 2000,
        }

    return runner


class TestOrchestration:
    def _run(self, tmp_path, answers):
        repo = RepoInfo(name="flask", git_url="x", ref="1", language="python", size="small")
        queries = [
            _query(id="q1"),
            _query(id="q2", expected_files=["src/flask/wrappers.py"],
                   expected_symbols=["Request"]),
        ]
        with patch("rag_bench.agent_bench.load_repos", return_value=[repo]), \
             patch("rag_bench.agent_bench.clone_repo", return_value=tmp_path), \
             patch("rag_bench.agent_bench.load_queries", return_value=queries):
            return run_agent_benchmark(
                runner=_fake_runner_factory(answers),
                on_progress=lambda m: None,
            )

    def test_ab_run_aggregates_both_conditions(self, tmp_path):
        doc = self._run(tmp_path, {
            "baseline": "no clue",
            "nova-rag": "Flask is in src/flask/app.py; Request in src/flask/wrappers.py",
        })
        s = doc["summary"]
        assert s["baseline"]["accuracy"] == 0.0
        assert s["nova-rag"]["accuracy"] == 1.0
        assert s["delta"]["accuracy_pp"] == 100.0
        assert s["delta"]["turns_change"] == -3.0
        # Every query ran in both conditions
        assert len(doc["queries"]) == 4

    def test_results_keep_raw_answers_for_audit(self, tmp_path):
        doc = self._run(tmp_path, {"baseline": "x", "nova-rag": "y"})
        assert all("answer" in q for q in doc["queries"])

    def test_render_markdown(self, tmp_path):
        doc = self._run(tmp_path, {
            "baseline": "no clue",
            "nova-rag": "Flask is in src/flask/app.py; Request in src/flask/wrappers.py",
        })
        md = render_markdown(doc)
        assert "| Accuracy (file+symbol) | 0.0% | 100.0% |" in md
        assert "accuracy +100.0 pp" in md

    def test_runner_errors_recorded_not_fatal(self, tmp_path):
        def broken(prompt, condition, cwd, model, max_turns, timeout):
            if condition.name == "baseline":
                raise RuntimeError("claude exploded")
            return {"result": "ok", "usage": {}, "num_turns": 1}

        repo = RepoInfo(name="flask", git_url="x", ref="1", language="python", size="small")
        with patch("rag_bench.agent_bench.load_repos", return_value=[repo]), \
             patch("rag_bench.agent_bench.clone_repo", return_value=tmp_path), \
             patch("rag_bench.agent_bench.load_queries", return_value=[_query()]):
            doc = run_agent_benchmark(runner=broken, on_progress=lambda m: None)
        assert doc["summary"]["baseline"]["errors"] == 1
        assert doc["summary"]["nova-rag"]["queries"] == 1


class TestAggregate:
    def test_empty_condition(self):
        from rag_bench.agent_bench import AgentQueryResult

        rows = [AgentQueryResult(
            query_id="q", repo="r", query_type="locate", difficulty="easy",
            condition="baseline", error="boom",
        )]
        s = aggregate(rows)
        assert s["baseline"]["queries"] == 0
        assert s["baseline"]["errors"] == 1
