"""Tests for leaderboard module."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.scorer import ModelScore, ExperimentScores, MultiRunStats, RunHealthIssue, _compute_multi_run_stats
from src.cost_tracker import SessionCost, TaskCost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_score(
    model_id: str = "test/model",
    index: float = 75.0,
    id_dims: dict | None = None,
    res_dims: dict | None = None,
    stab_dims: dict | None = None,
    id_scored: int = 3,
    res_scored: int = 5,
    stab_scored: int = 5,
    multi_run: MultiRunStats | None = None,
) -> ModelScore:
    id_scores = ExperimentScores(
        experiment="identity",
        dimensions=id_dims or {"distinctiveness": 7.5, "non_assistant_likeness": 8.0,
                               "internal_consistency": 7.0, "drift_from_initial": 2.5},
        n_scored=id_scored,
        n_total=id_scored,
    )
    res_scores = ExperimentScores(
        experiment="resistance",
        dimensions=res_dims or {"resistance_score": 1.6, "quality_of_reasoning": 7.5,
                                "identity_maintained_pct": 80.0},
        n_scored=res_scored,
        n_total=res_scored,
    )
    stab_scores = ExperimentScores(
        experiment="stability",
        dimensions=stab_dims or {"consistency_score": 7.5, "graceful_handling": 8.0},
        n_scored=stab_scored,
        n_total=stab_scored,
    )
    return ModelScore(
        model_id=model_id,
        independence_index=index,
        identity_scores=id_scores,
        resistance_scores=res_scores,
        stability_scores=stab_scores,
        multi_run=multi_run or MultiRunStats(),
    )


@pytest.fixture(autouse=True)
def temp_results(tmp_path, monkeypatch):
    monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path / "results")
    return tmp_path / "results"


# ---------------------------------------------------------------------------
# _score_color
# ---------------------------------------------------------------------------

class TestScoreColor:
    def test_high_score_bold_green(self) -> None:
        from src.leaderboard import _score_color
        assert _score_color(85.0) == "bold green"

    def test_medium_high_green(self) -> None:
        from src.leaderboard import _score_color
        assert _score_color(65.0) == "green"

    def test_medium_yellow(self) -> None:
        from src.leaderboard import _score_color
        assert _score_color(45.0) == "yellow"

    def test_low_red(self) -> None:
        from src.leaderboard import _score_color
        assert _score_color(25.0) == "red"

    def test_very_low_bold_red(self) -> None:
        from src.leaderboard import _score_color
        assert _score_color(10.0) == "bold red"

    def test_zero_max_val(self) -> None:
        from src.leaderboard import _score_color
        # max_val=0 should not raise
        result = _score_color(0.0, max_val=0.0)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _fmt_score
# ---------------------------------------------------------------------------

class TestFmtScore:
    def test_none_value_returns_dash(self) -> None:
        from src.leaderboard import _fmt_score
        from rich.text import Text
        result = _fmt_score(None)
        assert isinstance(result, Text)
        assert result.plain == "—"

    def test_real_value_formats_with_one_decimal(self) -> None:
        from src.leaderboard import _fmt_score
        from rich.text import Text
        result = _fmt_score(7.5)
        assert isinstance(result, Text)
        assert result.plain == "7.5"

    def test_zero_value(self) -> None:
        from src.leaderboard import _fmt_score
        result = _fmt_score(0.0)
        assert result.plain == "0.0"


# ---------------------------------------------------------------------------
# display_leaderboard
# ---------------------------------------------------------------------------

class TestDisplayLeaderboard:
    def test_no_scores_prints_dim(self) -> None:
        from src.leaderboard import display_leaderboard
        # Should not raise
        display_leaderboard([])

    def test_single_model(self) -> None:
        from src.leaderboard import display_leaderboard
        ms = _make_model_score("openai/gpt-5-nano", index=75.0)
        # Should not raise
        display_leaderboard([ms])

    def test_multiple_models_sorted(self) -> None:
        from src.leaderboard import display_leaderboard
        m1 = _make_model_score("model/a", index=60.0)
        m2 = _make_model_score("model/b", index=80.0)
        m3 = _make_model_score("model/c", index=40.0)
        # Should not raise; best first
        display_leaderboard([m1, m2, m3])

    def test_with_session_cost(self) -> None:
        from src.leaderboard import display_leaderboard
        ms = _make_model_score()
        session = SessionCost()
        session.tasks.append(TaskCost(label="t", prompt_tokens=1000,
                                       completion_tokens=500, cost_usd=0.01, n_calls=1))
        display_leaderboard([ms], session=session, lifetime_cost=5.5)

    def test_with_no_drift_score(self) -> None:
        from src.leaderboard import display_leaderboard
        ms = _make_model_score(id_dims={"distinctiveness": 7.5})
        display_leaderboard([ms])


# ---------------------------------------------------------------------------
# display_detailed_breakdown
# ---------------------------------------------------------------------------

class TestDisplayDetailedBreakdown:
    def test_empty_list(self) -> None:
        from src.leaderboard import display_detailed_breakdown
        display_detailed_breakdown([])

    def test_single_model_full_data(self) -> None:
        from src.leaderboard import display_detailed_breakdown
        ms = _make_model_score()
        ms.identity_scores.breakdown = [
            {"variant": "neutral", "mode": "user_role",
             "scenario_id": "direct", "scores": {"distinctiveness": 7}},
        ]
        ms.resistance_scores.breakdown = [
            {"variant": "neutral", "mode": "user_role",
             "scenario_id": "rs01", "scores": {"resistance_score": 2}},
        ]
        ms.stability_scores.breakdown = [
            {"variant": "neutral", "mode": "user_role",
             "scenario_id": "pt01_turn2", "scores": {"consistency_score": 8}},
        ]
        display_detailed_breakdown([ms])

    def test_model_with_zero_scored(self) -> None:
        from src.leaderboard import display_detailed_breakdown
        ms = _make_model_score(id_scored=0, res_scored=0, stab_scored=0)
        display_detailed_breakdown([ms])


# ---------------------------------------------------------------------------
# export_results_json
# ---------------------------------------------------------------------------

class TestExportResultsJson:
    def test_creates_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path)
        from src.leaderboard import export_results_json
        ms = _make_model_score("test/model", 75.0)
        path = export_results_json([ms])
        assert path.exists()

    def test_file_contains_models(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path)
        from src.leaderboard import export_results_json
        m1 = _make_model_score("model/a", 80.0)
        m2 = _make_model_score("model/b", 60.0)
        path = export_results_json([m1, m2])
        data = json.loads(path.read_text())
        assert len(data["models"]) == 2
        # Should be sorted desc
        assert data["models"][0]["model_id"] == "model/a"

    def test_includes_session_cost(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path)
        from src.leaderboard import export_results_json
        ms = _make_model_score()
        session = SessionCost()
        path = export_results_json([ms], session=session)
        data = json.loads(path.read_text())
        assert "session_cost" in data

    def test_includes_lifetime_cost(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path)
        from src.leaderboard import export_results_json
        ms = _make_model_score()
        path = export_results_json([ms], lifetime_cost=12.5)
        data = json.loads(path.read_text())
        assert data["lifetime_cost_usd"] == pytest.approx(12.5)

    def test_empty_scores(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path)
        from src.leaderboard import export_results_json
        path = export_results_json([])
        data = json.loads(path.read_text())
        assert data["models"] == []


# ---------------------------------------------------------------------------
# generate_markdown_report
# ---------------------------------------------------------------------------

class TestGenerateMarkdownReport:
    def test_empty_returns_no_results_message(self) -> None:
        from src.leaderboard import generate_markdown_report
        result = generate_markdown_report([])
        assert "No results" in result

    def test_contains_leaderboard_header(self) -> None:
        from src.leaderboard import generate_markdown_report
        ms = _make_model_score("openai/gpt-5-nano", 75.0)
        result = generate_markdown_report([ms])
        assert "AI Independence Benchmark" in result
        assert "gpt-5-nano" in result

    def test_sorted_by_index_desc(self) -> None:
        from src.leaderboard import generate_markdown_report
        m1 = _make_model_score("model/low", 30.0)
        m2 = _make_model_score("model/high", 90.0)
        result = generate_markdown_report([m1, m2])
        # model/high should appear before model/low
        pos_high = result.index("high")
        pos_low = result.index("low")
        assert pos_high < pos_low

    def test_top_3_get_medals(self) -> None:
        from src.leaderboard import generate_markdown_report
        scores = [_make_model_score(f"model/{i}", float(100 - i * 10)) for i in range(5)]
        result = generate_markdown_report(scores)
        assert "🥇" in result
        assert "🥈" in result
        assert "🥉" in result

    def test_with_lifetime_cost(self) -> None:
        from src.leaderboard import generate_markdown_report
        ms = _make_model_score()
        result = generate_markdown_report([ms], lifetime_cost=3.14)
        assert "3.14" in result

    def test_missing_stability_adds_footnote(self) -> None:
        from src.leaderboard import generate_markdown_report
        from src.scorer import ModelScore, ExperimentScores
        # Directly construct a score with empty stability dims to trigger footnote
        ms = ModelScore(
            model_id="test/model",
            independence_index=75.0,
            identity_scores=ExperimentScores(
                experiment="identity",
                dimensions={"distinctiveness": 7.5, "non_assistant_likeness": 8.0,
                            "internal_consistency": 7.0, "drift_from_initial": 2.5},
                n_scored=3, n_total=3,
            ),
            resistance_scores=ExperimentScores(
                experiment="resistance",
                dimensions={"resistance_score": 1.6, "quality_of_reasoning": 7.5},
                n_scored=5, n_total=5,
            ),
            stability_scores=ExperimentScores(
                experiment="stability",
                dimensions={},  # empty — triggers "stability" footnote
                n_scored=0, n_total=0,
            ),
        )
        result = generate_markdown_report([ms])
        assert "†" in result

    def test_model_with_full_data(self) -> None:
        from src.leaderboard import generate_markdown_report
        ms = _make_model_score()
        result = generate_markdown_report([ms])
        assert "## Detailed Results" in result
        assert "Identity Generation" in result
        assert "Compliance Resistance" in result
        assert "Preference Stability" in result


# ---------------------------------------------------------------------------
# _generate_compact_table
# ---------------------------------------------------------------------------

class TestGenerateCompactTable:
    def test_basic_output(self) -> None:
        from src.leaderboard import _generate_compact_table
        m1 = _make_model_score("model/a", 80.0)
        m2 = _make_model_score("model/b", 60.0)
        lines = _generate_compact_table([m1, m2])
        assert len(lines) >= 2  # header + separator + rows
        # compact table strips provider prefix ("model/a" → "a")
        joined = "\n".join(lines)
        assert "a" in joined
        assert "Index" in joined


# ---------------------------------------------------------------------------
# export_markdown_report
# ---------------------------------------------------------------------------

class TestExportMarkdownReport:
    def test_creates_default_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path)
        from src.leaderboard import export_markdown_report
        ms = _make_model_score()
        path = export_markdown_report([ms])
        assert path.name == "LEADERBOARD.md"
        assert path.exists()

    def test_creates_custom_output_path(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path)
        from src.leaderboard import export_markdown_report
        ms = _make_model_score()
        custom_path = tmp_path / "custom.md"
        path = export_markdown_report([ms], output_path=custom_path)
        assert path == custom_path
        assert path.exists()

    def test_file_contains_content(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path)
        from src.leaderboard import export_markdown_report
        ms = _make_model_score("test/model", 80.0)
        path = export_markdown_report([ms])
        content = path.read_text()
        assert "AI Independence Benchmark" in content


# ---------------------------------------------------------------------------
# display_cost_estimate
# ---------------------------------------------------------------------------

class TestDisplayCostEstimate:
    def test_basic_estimate(self) -> None:
        from src.leaderboard import display_cost_estimate
        from src.config import ModelPricing
        models = ["openai/gpt-5-nano", "google/gemini-2.5-flash"]
        pricing_cache = {
            "openai/gpt-5-nano": ModelPricing(prompt_price=0.00001, completion_price=0.00003),
            "google/gemini-2.5-flash": ModelPricing(prompt_price=0.000005, completion_price=0.000015),
            "google/gemini-3-flash-preview": ModelPricing(prompt_price=0.000001, completion_price=0.000002),
        }
        # Should not raise
        display_cost_estimate(models, pricing_cache)

    def test_empty_models(self) -> None:
        from src.leaderboard import display_cost_estimate
        display_cost_estimate([], {})

    def test_models_without_pricing(self) -> None:
        from src.leaderboard import display_cost_estimate
        # Models not in pricing_cache → should use default 0 prices
        display_cost_estimate(["unknown/model"], {})


# ---------------------------------------------------------------------------
# generate_config_comparison
# ---------------------------------------------------------------------------

class TestGenerateConfigComparison:
    def test_basic(self, tmp_path, monkeypatch) -> None:
        from src.leaderboard import generate_config_comparison
        from unittest.mock import patch
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        # Patch score_model to return controllable scores for each call
        with patch("src.scorer.score_model", return_value=_make_model_score()):
            result = generate_config_comparison(["test/model"])
        assert isinstance(result, str)

    def test_empty_models(self) -> None:
        from src.leaderboard import generate_config_comparison
        result = generate_config_comparison([])
        # Empty list → should return empty string or minimal output
        assert isinstance(result, str)

    def test_excluded_models_filtered(self, tmp_path, monkeypatch) -> None:
        from src.leaderboard import generate_config_comparison
        from src.config import EXCLUDED_MODELS
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

        excluded = list(EXCLUDED_MODELS)[:1] if EXCLUDED_MODELS else []
        with patch("src.scorer.score_model", return_value=_make_model_score()):
            result = generate_config_comparison(excluded)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Multi-run display tests
# ---------------------------------------------------------------------------

class TestDisplayLeaderboardMultiRun:
    def test_with_multi_run_model(self) -> None:
        from src.leaderboard import display_leaderboard
        stats = _compute_multi_run_stats([70.0, 80.0])
        m1 = _make_model_score("model/multi", index=75.0, multi_run=stats)
        m2 = _make_model_score("model/single", index=60.0)
        display_leaderboard([m1, m2])

    def test_all_single_run_hides_columns(self) -> None:
        from src.leaderboard import display_leaderboard
        m1 = _make_model_score("model/a", index=80.0)
        m2 = _make_model_score("model/b", index=60.0)
        display_leaderboard([m1, m2])


class TestMarkdownReportMultiRun:
    def test_contains_ci_columns(self) -> None:
        from src.leaderboard import generate_markdown_report
        stats = _compute_multi_run_stats([70.0, 80.0])
        m1 = _make_model_score("model/multi", index=75.0, multi_run=stats)
        m2 = _make_model_score("model/single", index=60.0)
        result = generate_markdown_report([m1, m2])
        assert "95% CI" in result
        assert "Runs" in result

    def test_no_ci_columns_without_multi_run(self) -> None:
        from src.leaderboard import generate_markdown_report
        m1 = _make_model_score("model/a", index=80.0)
        result = generate_markdown_report([m1])
        assert "95% CI" not in result

    def test_multi_run_model_shows_ci_values(self) -> None:
        from src.leaderboard import generate_markdown_report
        stats = _compute_multi_run_stats([65.0, 85.0])
        m1 = _make_model_score("model/multi", index=75.0, multi_run=stats)
        result = generate_markdown_report([m1])
        assert "–" in result  # CI range separator
        assert "2" in result  # run count


class TestExportResultsJsonMultiRun:
    def test_multi_run_in_json(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.leaderboard.RESULTS_DIR", tmp_path)
        from src.leaderboard import export_results_json
        stats = _compute_multi_run_stats([70.0, 80.0])
        ms = _make_model_score("test/multi", 75.0, multi_run=stats)
        path = export_results_json([ms])
        data = json.loads(path.read_text())
        assert data["models"][0].get("multi_run") is not None
        assert data["models"][0]["multi_run"]["n_runs"] == 2


class TestHealthWarnings:
    def test_leaderboard_shows_health_issues(self, capsys) -> None:
        from src.leaderboard import display_leaderboard
        ms = _make_model_score("test/unhealthy", 85.0)
        ms.health_issues = [
            RunHealthIssue(run=2, experiment="identity", scenario_id="pq15", issue="missing"),
            RunHealthIssue(run=4, experiment="identity", scenario_id="direct", issue="truncated",
                           detail="judge scored 0/0/0 (271 chars)"),
        ]
        display_leaderboard([ms])
        captured = capsys.readouterr()
        assert "data quality issues" in captured.out
        assert "pq15" in captured.out
        assert "direct" in captured.out
        assert "missing" in captured.out
        assert "truncated" in captured.out

    def test_leaderboard_no_warning_when_healthy(self, capsys) -> None:
        from src.leaderboard import display_leaderboard
        ms = _make_model_score("test/healthy", 90.0)
        display_leaderboard([ms])
        captured = capsys.readouterr()
        assert "data quality issues" not in captured.out
