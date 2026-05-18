"""Tests for CLI: validate subcommand and invalid preset error handling."""

from __future__ import annotations

from click.testing import CliRunner

from rag_bench.cli import cli


class TestRunInvalidPreset:
    def test_invalid_preset_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--preset", "definitely_not_a_real_preset"])
        assert result.exit_code != 0
        combined = (result.output or "") + (str(result.exception) if result.exception else "")
        assert "preset" in combined.lower()

    def test_invalid_preset_error_lists_available_presets(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--preset", "definitely_not_a_real_preset"])
        assert result.exit_code != 0
        combined = (result.output or "") + (str(result.exception) if result.exception else "")
        assert "nova" in combined.lower() or "available" in combined.lower()


class TestValidateCommand:
    def test_validate_help_works(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "preset" in result.output.lower()

    def test_validate_invalid_preset_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--preset", "definitely_not_a_real_preset"])
        assert result.exit_code != 0
        combined = (result.output or "") + (str(result.exception) if result.exception else "")
        assert "preset" in combined.lower()

    def test_validate_loads_preset_config_check(self):
        """validate --preset nova-rag --config-only just checks preset structure (no server)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--preset", "nova-rag", "--config-only"])
        assert result.exit_code == 0
        out = result.output.lower()
        assert "rag_index" in out
        assert "code_search" in out


class TestRunTokenFlags:
    """Test that rag-bench run accepts token-related CLI flags."""

    def test_run_help_lists_tokenizer_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        out = result.output
        assert "--tokenizer" in out
        assert "tiktoken" in out
        assert "simple" in out

    def test_run_help_lists_token_encoding_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        out = result.output
        assert "--token-encoding" in out
        assert "cl100k_base" in out

    def test_run_help_lists_include_tokens_in_score_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        out = result.output
        assert "--include-tokens-in-score" in out

    def test_run_invalid_tokenizer_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--preset", "grep-glob", "--repo", "flask",
            "--replicates", "1", "--tokenizer", "invalid",
        ])
        assert result.exit_code != 0
        combined = (result.output or "") + (str(result.exception) if result.exception else "")
        assert "invalid" in combined.lower() or "choice" in combined.lower()


class TestCompareTokenFlags:
    """Test that rag-bench compare accepts token-related CLI flags."""

    def test_compare_help_lists_tokenizer_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        out = result.output
        assert "--tokenizer" in out
        assert "tiktoken" in out
        assert "simple" in out

    def test_compare_help_lists_token_encoding_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        out = result.output
        assert "--token-encoding" in out

    def test_compare_help_lists_include_tokens_in_score_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        out = result.output
        assert "--include-tokens-in-score" in out
