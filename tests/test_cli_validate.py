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
