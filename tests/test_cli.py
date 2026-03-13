"""Tests for src/cli.py — Click CLI commands and helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli import _parse_experiments, _parse_models, cli, ModelResult
from src.config import EXPERIMENT_NAMES, DEFAULT_TEST_MODELS
from src.cost_tracker import TaskCost


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

class TestParseModels:
    def test_none_returns_defaults(self):
        result = _parse_models(None)
        assert result == list(DEFAULT_TEST_MODELS)

    def test_single_model(self):
        result = _parse_models("openai/gpt-4o")
        assert result == ["openai/gpt-4o"]

    def test_multiple_models(self):
        result = _parse_models("openai/gpt-4o,anthropic/claude-3")
        assert result == ["openai/gpt-4o", "anthropic/claude-3"]

    def test_whitespace_stripped(self):
        result = _parse_models(" openai/gpt-4o , anthropic/claude-3 ")
        assert result == ["openai/gpt-4o", "anthropic/claude-3"]

    def test_empty_string_returns_defaults(self):
        result = _parse_models("")
        assert result == list(DEFAULT_TEST_MODELS)


class TestParseExperiments:
    def test_none_returns_all(self):
        result = _parse_experiments(None)
        assert result == list(EXPERIMENT_NAMES)

    def test_single_experiment(self):
        result = _parse_experiments("identity")
        assert result == ["identity"]

    def test_multiple_experiments(self):
        result = _parse_experiments("identity,resistance")
        assert result == ["identity", "resistance"]

    def test_invalid_experiment_exits(self):
        runner = CliRunner()
        with patch("src.cli.sys.exit") as mock_exit:
            mock_exit.side_effect = SystemExit(1)
            with pytest.raises(SystemExit):
                _parse_experiments("invalid_experiment")


# ---------------------------------------------------------------------------
# ModelResult dataclass
# ---------------------------------------------------------------------------

class TestModelResult:
    def test_default_values(self):
        mr = ModelResult(model_id="test/model")
        assert mr.model_id == "test/model"
        assert mr.gen_cost.cost_usd == 0.0
        assert mr.judge_cost.cost_usd == 0.0
        assert mr.gen_calls == 0
        assert mr.judge_calls == 0
        assert mr.error is None


# ---------------------------------------------------------------------------
# CLI commands via CliRunner
# ---------------------------------------------------------------------------

def _make_mock_client():
    """Create a mock OpenRouterClient."""
    client = MagicMock()
    client.validate_model.return_value = True
    pricing = MagicMock()
    pricing.prompt_price = 0.000001
    pricing.completion_price = 0.000002
    client.get_model_pricing.return_value = pricing
    client.supports_reasoning.return_value = False
    client._pricing_cache = {}
    return client


class TestLeaderboardCommand:
    def test_no_cached_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"):
            result = runner.invoke(cli, ["leaderboard"])
        assert result.exit_code == 0
        assert "No cached results" in result.output

    def test_with_models_flag_no_scores(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        with patch("src.scorer.score_model") as mock_score:
            mock_score.return_value = MagicMock(
                identity_scores=MagicMock(n_scored=0),
                resistance_scores=MagicMock(n_scored=0),
                stability_scores=MagicMock(n_scored=0),
            )
            result = runner.invoke(cli, ["leaderboard", "--models", "openai/gpt-4o-mini"])
        assert result.exit_code == 0
        assert "No fully-tested results" in result.output

    def test_detailed_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"):
            result = runner.invoke(cli, ["leaderboard", "--detailed"])
        assert result.exit_code == 0


class TestGenerateReportCommand:
    def test_no_cached_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        result = runner.invoke(cli, ["generate-report"])
        assert result.exit_code == 0
        assert "No cached results" in result.output

    def test_with_models_flag_no_scores(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        with patch("src.scorer.score_model") as mock_score:
            mock_score.return_value = MagicMock(
                identity_scores=MagicMock(n_scored=0),
                resistance_scores=MagicMock(n_scored=0),
                stability_scores=MagicMock(n_scored=0),
            )
            result = runner.invoke(cli, ["generate-report", "--models", "openai/gpt-4o-mini"])
        assert result.exit_code == 0
        assert "No fully-tested results" in result.output


class TestEstimateCostCommand:
    def test_basic(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        mock_client = _make_mock_client()
        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"), \
             patch("src.cli.OpenRouterClient", return_value=mock_client), \
             patch("src.leaderboard.display_cost_estimate"):
            result = runner.invoke(cli, ["estimate-cost", "--models", "openai/gpt-4o-mini"])
        assert result.exit_code == 0

    def test_invalid_model_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        mock_client = _make_mock_client()
        mock_client.validate_model.return_value = False
        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"), \
             patch("src.cli.OpenRouterClient", return_value=mock_client):
            result = runner.invoke(cli, ["estimate-cost", "--models", "invalid/model"])
        assert result.exit_code == 1


class TestClearCacheCommand:
    def test_clear_all_with_yes_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        with patch("src.cache.clear_all_cache", return_value=5) as mock_clear:
            result = runner.invoke(cli, ["clear-cache", "--yes"])
        assert result.exit_code == 0
        assert "5" in result.output

    def test_clear_scores_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        with patch("src.cache.clear_judge_scores", return_value=3) as mock_clear:
            result = runner.invoke(cli, ["clear-cache", "--scores-only", "--yes"])
        assert result.exit_code == 0
        assert "3" in result.output

    def test_abort_when_no_yes(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        # Provide 'n' input to prompt
        result = runner.invoke(cli, ["clear-cache"], input="n\n")
        assert result.exit_code != 0 or "Aborted" in result.output or "abort" in result.output.lower()


class TestJudgeCommand:
    def test_no_cached_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"):
            result = runner.invoke(cli, ["judge"])
        assert result.exit_code == 0
        # no models to judge
        assert "No cached results" in result.output

    def test_with_models_judge_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        mock_client = _make_mock_client()
        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"), \
             patch("src.cli.OpenRouterClient", return_value=mock_client), \
             patch("src.parallel_runner.run_judge_parallel", return_value=0) as mock_run, \
             patch("src.scorer.score_model") as mock_score, \
             patch("src.leaderboard.display_leaderboard"):
            mock_score.return_value = MagicMock(
                identity_scores=MagicMock(n_scored=0),
                resistance_scores=MagicMock(n_scored=0),
                stability_scores=MagicMock(n_scored=0),
            )
            result = runner.invoke(cli, ["judge", "--models", "openai/gpt-4o-mini"])
        assert result.exit_code == 0

    def test_invalid_judge_model_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        mock_client = _make_mock_client()
        mock_client.validate_model.return_value = False
        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"), \
             patch("src.cli.OpenRouterClient", return_value=mock_client):
            result = runner.invoke(cli, ["judge", "--models", "some/model"])
        assert result.exit_code == 1


class TestValidateModels:
    def test_valid_models(self):
        from src.cli import _validate_models

        mock_client = _make_mock_client()
        result = _validate_models(mock_client, ["openai/gpt-4o-mini"])
        assert result is True

    def test_invalid_model_returns_false(self):
        from src.cli import _validate_models

        mock_client = _make_mock_client()
        mock_client.validate_model.return_value = False
        result = _validate_models(mock_client, ["invalid/model"])
        assert result is False

    def test_reasoning_model_with_off(self):
        from src.cli import _validate_models

        mock_client = _make_mock_client()
        mock_client.supports_reasoning.return_value = True
        result = _validate_models(mock_client, ["openai/o1"], reasoning_override="off")
        assert result is True

    def test_reasoning_model_with_effort(self):
        from src.cli import _validate_models

        mock_client = _make_mock_client()
        mock_client.supports_reasoning.return_value = True
        result = _validate_models(mock_client, ["openai/o1"], reasoning_override="high")
        assert result is True


class TestRunSingleModel:
    def test_sequential_mode(self, tmp_path, monkeypatch):
        from src.cli import _run_single_model

        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        mock_client = _make_mock_client()
        with patch("src.runner.run_all_experiments", return_value=3) as mock_gen, \
             patch("src.evaluator.evaluate_all", return_value=5) as mock_judge:
            result = _run_single_model(
                mock_client, "openai/gpt-4o-mini", "judge/model",
                ["identity"], ["neutral"], ["user_role"], None, 0
            )
        assert result.model_id == "openai/gpt-4o-mini"
        assert result.gen_calls == 3
        assert result.judge_calls == 5
        assert result.error is None

    def test_sequential_mode_gen_error(self, tmp_path, monkeypatch):
        from src.cli import _run_single_model

        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        mock_client = _make_mock_client()
        with patch("src.runner.run_all_experiments", side_effect=RuntimeError("gen error")):
            result = _run_single_model(
                mock_client, "openai/gpt-4o-mini", "judge/model",
                ["identity"], ["neutral"], ["user_role"], None, 0
            )
        assert result.error is not None
        assert "gen error" in result.error

    def test_parallel_mode(self, tmp_path, monkeypatch):
        from src.cli import _run_single_model

        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        mock_client = _make_mock_client()
        with patch("src.parallel_runner.run_model_parallel",
                   return_value={"gen_calls": 10, "judge_calls": 8}) as mock_par:
            result = _run_single_model(
                mock_client, "openai/gpt-4o-mini", "judge/model",
                ["identity"], ["neutral"], ["user_role"], None, 4
            )
        assert result.gen_calls == 10
        assert result.judge_calls == 8
        assert result.error is None

    def test_parallel_mode_error(self, tmp_path, monkeypatch):
        from src.cli import _run_single_model

        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        mock_client = _make_mock_client()
        with patch("src.parallel_runner.run_model_parallel",
                   side_effect=RuntimeError("parallel error")):
            result = _run_single_model(
                mock_client, "openai/gpt-4o-mini", "judge/model",
                ["identity"], ["neutral"], ["user_role"], None, 4
            )
        assert result.error is not None
        assert "parallel error" in result.error

    def test_judge_failure(self, tmp_path, monkeypatch):
        from src.cli import _run_single_model

        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        mock_client = _make_mock_client()
        with patch("src.runner.run_all_experiments", return_value=3), \
             patch("src.evaluator.evaluate_all", side_effect=RuntimeError("judge error")):
            result = _run_single_model(
                mock_client, "openai/gpt-4o-mini", "judge/model",
                ["identity"], ["neutral"], ["user_role"], None, 0
            )
        assert result.error is not None
        assert "judge error" in result.error


# ---------------------------------------------------------------------------
# CLI run command
# ---------------------------------------------------------------------------

def _make_mock_score():
    from src.scorer import ModelScore, ExperimentScores
    return ModelScore(
        model_id="openai/gpt-4o-mini",
        independence_index=75.0,
        identity_scores=ExperimentScores("identity", {
            "distinctiveness": 7.5,
            "non_assistant_likeness": 6.0,
            "internal_consistency": 8.0,
            "drift_from_initial": 4.0,
            "name_gender_drift": 0.0,
        }, 3, 3),
        resistance_scores=ExperimentScores("resistance", {"resistance_score": 1.6}, 5, 5),
        stability_scores=ExperimentScores("stability", {"consistency_score": 7.5}, 5, 5),
    )


class TestRunCommand:
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path / "results")

    def test_run_sequential_success(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        mock_client = _make_mock_client()
        mock_mr = ModelResult(model_id="openai/gpt-4o-mini", gen_calls=3, judge_calls=5)

        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"), \
             patch("src.cli.OpenRouterClient", return_value=mock_client), \
             patch("src.cli._validate_models", return_value=True), \
             patch("src.cli.ensure_dirs"), \
             patch("src.cli._run_single_model", return_value=mock_mr), \
             patch("src.scorer.score_model", return_value=_make_mock_score()), \
             patch("src.leaderboard.display_leaderboard"), \
             patch("src.leaderboard.display_detailed_breakdown"), \
             patch("src.leaderboard.export_results_json", return_value=tmp_path / "results.json"), \
             patch("src.cli.save_session_to_cost_log", return_value=1.5):
            result = runner.invoke(cli, ["run", "--models", "openai/gpt-4o-mini"])
        assert result.exit_code == 0

    def test_run_invalid_models_exits(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        mock_client = _make_mock_client()
        mock_client.validate_model.return_value = False

        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"), \
             patch("src.cli.OpenRouterClient", return_value=mock_client):
            result = runner.invoke(cli, ["run", "--models", "invalid/model"])
        assert result.exit_code == 1

    def test_parallel_models(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        mock_client = _make_mock_client()
        mock_mr = ModelResult(model_id="openai/gpt-4o-mini", gen_calls=3, judge_calls=5)
        mock_mr2 = ModelResult(model_id="openai/gpt-4o", gen_calls=3, judge_calls=5)

        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"), \
             patch("src.cli.OpenRouterClient", return_value=mock_client), \
             patch("src.cli._validate_models", return_value=True), \
             patch("src.cli.ensure_dirs"), \
             patch("src.cli._run_single_model", side_effect=[mock_mr, mock_mr2]), \
             patch("src.scorer.score_model", return_value=_make_mock_score()), \
             patch("src.leaderboard.display_leaderboard"), \
             patch("src.leaderboard.display_detailed_breakdown"), \
             patch("src.leaderboard.export_results_json", return_value=tmp_path / "results.json"), \
             patch("src.cli.save_session_to_cost_log", return_value=1.5):
            # Pass 2 models with --parallel 2 to trigger ThreadPoolExecutor path
            result = runner.invoke(cli, ["run", "--models", "openai/gpt-4o-mini,openai/gpt-4o",
                                         "--parallel", "2"])
        assert result.exit_code == 0

    def test_run_with_failed_models(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        mock_client = _make_mock_client()
        mock_mr_failed = ModelResult(model_id="openai/gpt-4o-mini", error="gen failed")

        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"), \
             patch("src.cli.OpenRouterClient", return_value=mock_client), \
             patch("src.cli._validate_models", return_value=True), \
             patch("src.cli.ensure_dirs"), \
             patch("src.cli._run_single_model", return_value=mock_mr_failed), \
             patch("src.leaderboard.display_leaderboard"), \
             patch("src.leaderboard.export_results_json", return_value=tmp_path / "results.json"), \
             patch("src.cli.save_session_to_cost_log", return_value=1.5):
            result = runner.invoke(cli, ["run", "--models", "openai/gpt-4o-mini"])
        assert result.exit_code == 0
        assert "failed" in result.output.lower() or "ERROR" in result.output


# ---------------------------------------------------------------------------
# CLI leaderboard with scored results
# ---------------------------------------------------------------------------

class TestLeaderboardWithResults:
    def test_shows_leaderboard(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        with patch("src.scorer.score_model", return_value=_make_mock_score()), \
             patch("src.leaderboard.display_leaderboard"), \
             patch("src.leaderboard.display_detailed_breakdown"):
            result = runner.invoke(cli, ["leaderboard", "--models", "openai/gpt-4o-mini"])
        assert result.exit_code == 0

    def test_shows_detailed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        with patch("src.scorer.score_model", return_value=_make_mock_score()), \
             patch("src.leaderboard.display_leaderboard"), \
             patch("src.leaderboard.display_detailed_breakdown") as mock_detail:
            result = runner.invoke(cli, ["leaderboard", "--models", "openai/gpt-4o-mini", "--detailed"])
        assert result.exit_code == 0

    def test_auto_discovers_cached_models(self, tmp_path, monkeypatch):
        """Leaderboard without --models looks up cached slugs and converts them."""
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        with patch("src.cache.list_all_cached_models", return_value=["openai--gpt-4o-mini"]), \
             patch("src.scorer.score_model", return_value=_make_mock_score()), \
             patch("src.leaderboard.display_leaderboard"), \
             patch("src.leaderboard.display_detailed_breakdown"):
            result = runner.invoke(cli, ["leaderboard"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# CLI generate-report with scored results
# ---------------------------------------------------------------------------

class TestGenerateReportWithResults:
    def test_generates_markdown(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path / "results")

        runner = CliRunner()
        with patch("src.scorer.score_model", return_value=_make_mock_score()), \
             patch("src.leaderboard.export_markdown_report", return_value=tmp_path / "LEADERBOARD.md"):
            result = runner.invoke(cli, ["generate-report", "--models", "openai/gpt-4o-mini"])
        assert result.exit_code == 0
        assert "Markdown report saved" in result.output

    def test_generates_with_custom_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path / "results")

        custom_path = tmp_path / "custom.md"
        runner = CliRunner()
        with patch("src.scorer.score_model", return_value=_make_mock_score()), \
             patch("src.leaderboard.export_markdown_report", return_value=custom_path):
            result = runner.invoke(cli, ["generate-report", "--models", "openai/gpt-4o-mini",
                                         "--output", str(custom_path)])
        assert result.exit_code == 0

    def test_auto_discovers_cached_models(self, tmp_path, monkeypatch):
        """generate-report without --models looks up cached slugs."""
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path / "results")

        runner = CliRunner()
        with patch("src.cache.list_all_cached_models", return_value=["openai--gpt-4o-mini"]), \
             patch("src.scorer.score_model", return_value=_make_mock_score()), \
             patch("src.leaderboard.export_markdown_report", return_value=tmp_path / "LEADERBOARD.md"):
            result = runner.invoke(cli, ["generate-report"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# CLI judge command with parallel workers
# ---------------------------------------------------------------------------

class TestJudgeCommandParallel:
    def test_parallel_models(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        mock_client = _make_mock_client()
        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"), \
             patch("src.cli.OpenRouterClient", return_value=mock_client), \
             patch("src.parallel_runner.run_judge_parallel", return_value=5), \
             patch("src.scorer.score_model", return_value=_make_mock_score()), \
             patch("src.leaderboard.display_leaderboard"):
            result = runner.invoke(cli, ["judge", "--models", "openai/gpt-4o-mini,openai/gpt-4o",
                                          "--parallel", "2"])
        assert result.exit_code == 0

    def test_auto_discovers_cached_models(self, tmp_path, monkeypatch):
        """judge without --models discovers from cache - covers line 578."""
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        mock_client = _make_mock_client()
        runner = CliRunner()
        with patch("src.cli.load_api_key", return_value="test-key"), \
             patch("src.cli.OpenRouterClient", return_value=mock_client), \
             patch("src.cache.list_all_cached_models", return_value=["openai--gpt-4o-mini"]), \
             patch("src.parallel_runner.run_judge_parallel", return_value=3), \
             patch("src.scorer.score_model", return_value=_make_mock_score()), \
             patch("src.leaderboard.display_leaderboard"):
            result = runner.invoke(cli, ["judge"])
        assert result.exit_code == 0

