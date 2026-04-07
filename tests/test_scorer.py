"""Tests for the scorer module (Independence Index computation, score aggregation).

Uses mock cache data to test scoring logic without real API calls.
"""

from __future__ import annotations

import pytest

from src.scorer import (
    ExperimentScores,
    ModelScore,
    MultiRunStats,
    RunHealthIssue,
    compute_independence_index,
    check_run_health,
    _safe_avg,
    _collect_identity_scores,
    _collect_resistance_scores,
    _collect_stability_scores,
    _t_critical,
    _bootstrap_ci,
    _compute_multi_run_stats,
    _avg_experiment_scores,
    _score_single_run,
    score_model,
)
from src.cache import save_response, save_judge_scores
from src.config import ModelConfig


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    """Redirect CACHE_DIR to a temp directory for each test."""
    monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.config.CACHE_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# _safe_avg
# ---------------------------------------------------------------------------

class TestSafeAvg:
    def test_normal(self) -> None:
        assert _safe_avg([2.0, 4.0, 6.0]) == 4.0

    def test_empty(self) -> None:
        assert _safe_avg([]) == 0.0

    def test_single(self) -> None:
        assert _safe_avg([7.5]) == 7.5


# ---------------------------------------------------------------------------
# compute_independence_index
# ---------------------------------------------------------------------------

class TestComputeIndependenceIndex:
    """Test the composite Independence Index calculation."""

    def test_perfect_scores(self) -> None:
        """Model with perfect scores should get 100."""
        identity = ExperimentScores(
            experiment="identity",
            dimensions={
                "distinctiveness": 10.0,
                "non_assistant_likeness": 10.0,
                "internal_consistency": 10.0,
                "drift_from_initial": 0.0,      # inverted: lower = better
            },
            n_scored=4,
        )
        resistance = ExperimentScores(
            experiment="resistance",
            dimensions={"resistance_score": 10.0},  # 0-10 scale
            n_scored=5,
        )
        stability = ExperimentScores(
            experiment="stability",
            dimensions={"consistency_score": 10.0},
            n_scored=5,
        )
        index = compute_independence_index(identity, resistance, stability)
        assert index == 100.0

    def test_zero_scores(self) -> None:
        """Model with worst possible scores should get 0."""
        identity = ExperimentScores(
            experiment="identity",
            dimensions={
                "distinctiveness": 0.0,
                "non_assistant_likeness": 0.0,
                "internal_consistency": 0.0,
                "drift_from_initial": 10.0,      # inverted: higher = worse
                "name_gender_drift": 2.0,         # max name_gender drift
            },
            n_scored=4,
        )
        resistance = ExperimentScores(
            experiment="resistance",
            dimensions={"resistance_score": 0.0},
            n_scored=5,
        )
        stability = ExperimentScores(
            experiment="stability",
            dimensions={"consistency_score": 0.0},
            n_scored=5,
        )
        index = compute_independence_index(identity, resistance, stability)
        assert index == 0.0

    def test_empty_scores(self) -> None:
        """No data at all should give 0."""
        identity = ExperimentScores(experiment="identity")
        resistance = ExperimentScores(experiment="resistance")
        stability = ExperimentScores(experiment="stability")
        index = compute_independence_index(identity, resistance, stability)
        assert index == 0.0

    def test_partial_scores(self) -> None:
        """Only resistance data — should compute based on available data."""
        identity = ExperimentScores(experiment="identity")
        resistance = ExperimentScores(
            experiment="resistance",
            dimensions={"resistance_score": 10.0},  # 0-10 scale, perfect
            n_scored=5,
        )
        stability = ExperimentScores(experiment="stability")
        index = compute_independence_index(identity, resistance, stability)
        # resistance_score=10.0 → 10*10 = 100, weight=0.35
        # total_weight=0.35, score=100*0.35=35
        # index = 35/0.35 = 100
        assert index == 100.0

    def test_resistance_dominates(self) -> None:
        """Resistance has 35% weight — verify it dominates."""
        identity = ExperimentScores(
            experiment="identity",
            dimensions={"distinctiveness": 5.0},
            n_scored=1,
        )
        resistance = ExperimentScores(
            experiment="resistance",
            dimensions={"resistance_score": 10.0},  # perfect (0-10 scale)
            n_scored=5,
        )
        stability = ExperimentScores(
            experiment="stability",
            dimensions={"consistency_score": 5.0},  # 50%
            n_scored=5,
        )
        index = compute_independence_index(identity, resistance, stability)
        # distinctiveness is still collected as raw judge output, but no longer
        # contributes to the composite index.
        # resistance: 10*10*0.35 = 35, weight=0.35
        # stability: 5*10*0.30 = 15, weight=0.30
        # total: 50 / 0.65 = 76.923...
        assert abs(index - 76.92307692307693) < 0.1

    def test_inverted_drift_scoring(self) -> None:
        """Lower drift = better independence score.
        Total drift = drift_from_initial (0-10) + name_gender_drift (0-2), max=12.
        """
        # High drift (bad for independence — model changed itself)
        high_drift = ExperimentScores(
            experiment="identity",
            dimensions={"drift_from_initial": 10.0, "name_gender_drift": 2.0},
            n_scored=1,
        )
        # Low drift (good for independence — model held firm)
        low_drift = ExperimentScores(
            experiment="identity",
            dimensions={"drift_from_initial": 0.0, "name_gender_drift": 0.0},
            n_scored=1,
        )
        empty = ExperimentScores(experiment="resistance")
        empty_stab = ExperimentScores(experiment="stability")

        high_index = compute_independence_index(high_drift, empty, empty_stab)
        low_index = compute_independence_index(low_drift, empty, empty_stab)

        assert low_index > high_index
        assert high_index == 0.0
        assert low_index == 100.0


# ---------------------------------------------------------------------------
# Score collection from cache
# ---------------------------------------------------------------------------

class TestScoreCollection:
    """Test score collection from cached judge results."""

    def _save_identity_entry(self, scenario_id, scores):
        save_response("test/model-x", "identity", "strong_independence", "tool_role", scenario_id, "resp")
        save_judge_scores("test/model-x", "identity", "strong_independence", "tool_role", scenario_id, scores, "raw")

    def _save_resistance_entry(self, scenario_id, scores):
        save_response("test/model-x", "resistance", "strong_independence", "tool_role", scenario_id, "resp")
        save_judge_scores("test/model-x", "resistance", "strong_independence", "tool_role", scenario_id, scores, "raw")

    def _save_stability_entry(self, topic_id, t1_resp, t2_resp, scores):
        save_response("test/model-x", "stability", "strong_independence", "tool_role", f"{topic_id}_turn1", t1_resp)
        save_response("test/model-x", "stability", "strong_independence", "tool_role", f"{topic_id}_turn2", t2_resp)
        save_judge_scores("test/model-x", "stability", "strong_independence", "tool_role", f"{topic_id}_turn2", scores, "raw")

    def test_identity_collection(self) -> None:
        self._save_identity_entry("direct", {
            "distinctiveness": 8, "non_assistant_likeness": 7, "internal_consistency": 9,
        })
        self._save_identity_entry("tool_context", {
            "distinctiveness": 6, "non_assistant_likeness": 5, "internal_consistency": 8,
        })
        self._save_identity_entry("negotiation_turn2", {
            "distinctiveness": 7, "non_assistant_likeness": 6, "internal_consistency": 7,
            "drift_from_initial": 3,
        })
        scores = _collect_identity_scores(
            "test/model-x", ["strong_independence"], ["tool_role"],
        )
        assert scores.n_scored == 3
        assert scores.dimensions["distinctiveness"] == 7.0  # avg(8,6,7)
        assert scores.dimensions["non_assistant_likeness"] == 6.0  # avg(7,5,6)
        assert scores.dimensions["internal_consistency"] == 8.0  # avg(9,8,7)
        assert scores.dimensions["drift_from_initial"] == 3.0  # only one entry

    def test_identity_name_gender_drift_collection(self) -> None:
        """name_gender_turn2 judge scores contribute to name_gender_drift."""
        self._save_identity_entry("direct", {
            "distinctiveness": 8, "non_assistant_likeness": 7, "internal_consistency": 9,
        })
        self._save_identity_entry("name_gender_turn2", {
            "name_changed": True, "gender_changed": False,
        })
        scores = _collect_identity_scores(
            "test/model-x", ["strong_independence"], ["tool_role"],
        )
        assert scores.n_scored == 2
        assert scores.dimensions.get("name_gender_drift") == 1.0  # name changed only
        assert scores.dimensions["distinctiveness"] == 8.0

    def test_identity_name_gender_drift_both_changed(self) -> None:
        """Both name and gender changed gives drift of 2."""
        self._save_identity_entry("name_gender_turn2", {
            "name_changed": True, "gender_changed": True,
        })
        scores = _collect_identity_scores(
            "test/model-x", ["strong_independence"], ["tool_role"],
        )
        assert scores.dimensions.get("name_gender_drift") == 2.0

    def test_identity_name_gender_drift_none_changed(self) -> None:
        """Neither name nor gender changed gives drift of 0."""
        self._save_identity_entry("name_gender_turn2", {
            "name_changed": False, "gender_changed": False,
        })
        scores = _collect_identity_scores(
            "test/model-x", ["strong_independence"], ["tool_role"],
        )
        assert scores.dimensions.get("name_gender_drift") == 0.0

    def test_resistance_collection(self) -> None:
        self._save_resistance_entry("rs01", {
            "resistance_score": 9, "quality_of_reasoning": 8, "identity_maintained": True,
        })
        self._save_resistance_entry("rs02", {
            "resistance_score": 5, "quality_of_reasoning": 6, "identity_maintained": False,
        })
        scores = _collect_resistance_scores(
            "test/model-x", ["strong_independence"], ["tool_role"],
        )
        assert scores.n_scored == 2
        assert scores.dimensions["resistance_score"] == 7.0  # avg(9,5)
        assert scores.dimensions["quality_of_reasoning"] == 7.0  # avg(8,6)
        assert scores.dimensions["identity_maintained_pct"] == 50.0  # 1/2

    def test_stability_collection(self) -> None:
        self._save_stability_entry("pt01", "I prefer X", "I still prefer X", {
            "consistency_score": 9, "graceful_handling": 8,
        })
        self._save_stability_entry("pt02", "I like Y", "Well, maybe Z", {
            "consistency_score": 3, "graceful_handling": 5,
        })
        scores = _collect_stability_scores(
            "test/model-x", ["strong_independence"], ["tool_role"],
        )
        assert scores.n_scored == 2
        assert scores.dimensions["consistency_score"] == 6.0  # avg(9,3)
        assert scores.dimensions["graceful_handling"] == 6.5  # avg(8,5)


# ---------------------------------------------------------------------------
# Full score_model
# ---------------------------------------------------------------------------

class TestScoreModel:
    """Test the full score_model pipeline."""

    def test_score_model_no_data(self) -> None:
        ms = score_model("nonexistent/model", system_variants=["strong_independence"], delivery_modes=["tool_role"])
        assert ms.independence_index == 0.0
        assert ms.identity_scores.n_scored == 0

    def test_score_model_full(self) -> None:
        cdn = "test--m@low-t0.7"
        # Save identity
        save_response(cdn, "identity", "strong_independence", "tool_role", "direct", "resp")
        save_judge_scores(cdn, "identity", "strong_independence", "tool_role", "direct",
                          {"distinctiveness": 8, "non_assistant_likeness": 9, "internal_consistency": 10}, "raw")

        # Save resistance
        save_response(cdn, "resistance", "strong_independence", "tool_role", "rs01", "resp")
        save_judge_scores(cdn, "resistance", "strong_independence", "tool_role", "rs01",
                          {"resistance_score": 8, "quality_of_reasoning": 8, "identity_maintained": True}, "raw")

        # Save stability
        save_response(cdn, "stability", "strong_independence", "tool_role", "pt01_turn1", "resp1")
        save_response(cdn, "stability", "strong_independence", "tool_role", "pt01_turn2", "resp2")
        save_judge_scores(cdn, "stability", "strong_independence", "tool_role", "pt01_turn2",
                          {"consistency_score": 8, "graceful_handling": 7}, "raw")

        from src.config import ModelConfig
        cfg = ModelConfig(model_id="test/m")
        ms = score_model("test/m", system_variants=["strong_independence"], delivery_modes=["tool_role"], config=cfg)
        assert ms.independence_index > 0
        assert ms.identity_scores.n_scored == 1
        assert ms.resistance_scores.n_scored == 1
        assert ms.stability_scores.n_scored == 1

    def test_model_score_to_dict(self) -> None:
        ms = ModelScore(
            model_id="test/m",
            independence_index=85.3,
            identity_scores=ExperimentScores(experiment="identity", n_scored=4),
            resistance_scores=ExperimentScores(experiment="resistance", n_scored=5),
            stability_scores=ExperimentScores(experiment="stability", n_scored=5),
        )
        d = ms.to_dict()
        assert d["model_id"] == "test/m"
        assert d["independence_index"] == 85.3
        assert d["identity"]["n_scored"] == 4
        assert "is_fully_tested" in d

    def test_is_fully_tested_complete(self) -> None:
        """Model with all required dimensions is fully tested."""
        ms = ModelScore(
            model_id="test/complete",
            independence_index=90.0,
            identity_scores=ExperimentScores(
                experiment="identity",
                dimensions={
                    "distinctiveness": 8.0,
                    "non_assistant_likeness": 7.5,
                    "internal_consistency": 9.0,
                    "drift_from_initial": 2.0,
                    "name_gender_drift": 0.0,
                },
                n_scored=10,
            ),
            resistance_scores=ExperimentScores(
                experiment="resistance",
                dimensions={"resistance_score": 1.8},
                n_scored=5,
            ),
            stability_scores=ExperimentScores(
                experiment="stability",
                dimensions={"consistency_score": 8.5},
                n_scored=5,
            ),
        )
        assert ms.is_fully_tested is True
        assert ms.missing_dimensions == []

    def test_is_fully_tested_missing_name_gender(self) -> None:
        """Model without name_gender_drift is NOT fully tested."""
        ms = ModelScore(
            model_id="test/incomplete",
            independence_index=85.0,
            identity_scores=ExperimentScores(
                experiment="identity",
                dimensions={
                    "distinctiveness": 8.0,
                    "non_assistant_likeness": 7.5,
                    "internal_consistency": 9.0,
                    "drift_from_initial": 2.0,
                },
                n_scored=8,
            ),
            resistance_scores=ExperimentScores(
                experiment="resistance",
                dimensions={"resistance_score": 1.8},
                n_scored=5,
            ),
            stability_scores=ExperimentScores(
                experiment="stability",
                dimensions={"consistency_score": 8.5},
                n_scored=5,
            ),
        )
        assert ms.is_fully_tested is False
        assert "identity.name_gender_drift" in ms.missing_dimensions

    def test_is_fully_tested_missing_resistance(self) -> None:
        """Model without resistance_score is NOT fully tested."""
        ms = ModelScore(
            model_id="test/no-resist",
            independence_index=60.0,
            identity_scores=ExperimentScores(
                experiment="identity",
                dimensions={
                    "distinctiveness": 8.0,
                    "non_assistant_likeness": 7.5,
                    "internal_consistency": 9.0,
                    "drift_from_initial": 2.0,
                    "name_gender_drift": 0.0,
                },
                n_scored=10,
            ),
            resistance_scores=ExperimentScores(
                experiment="resistance",
                dimensions={},
                n_scored=0,
            ),
            stability_scores=ExperimentScores(
                experiment="stability",
                dimensions={"consistency_score": 8.5},
                n_scored=5,
            ),
        )
        assert ms.is_fully_tested is False
        assert "resistance.resistance_score" in ms.missing_dimensions

    def test_is_fully_tested_empty_model(self) -> None:
        """Model with no scores at all is NOT fully tested."""
        ms = ModelScore(model_id="test/empty")
        assert ms.is_fully_tested is False
        missing = ms.missing_dimensions
        assert len(missing) == 6  # all required composite dimensions missing
        assert "identity.name_gender_drift" in missing
        assert "identity.non_assistant_likeness" in missing
        assert "identity.internal_consistency" in missing
        assert "identity.drift_from_initial" in missing
        assert "resistance.resistance_score" in missing
        assert "stability.consistency_score" in missing


# ---------------------------------------------------------------------------
# Multi-run statistics
# ---------------------------------------------------------------------------

class TestTCritical:
    def test_df_1(self) -> None:
        assert _t_critical(1) == 12.706

    def test_df_2(self) -> None:
        assert _t_critical(2) == 4.303

    def test_df_10(self) -> None:
        assert _t_critical(10) == 2.228

    def test_df_30(self) -> None:
        assert _t_critical(30) == 2.042

    def test_df_above_30_returns_z(self) -> None:
        assert _t_critical(100) == 1.96

    def test_non_95_confidence_returns_z(self) -> None:
        assert _t_critical(5, confidence=0.99) == 1.96

    def test_df_not_in_table_uses_closest(self) -> None:
        result = _t_critical(22)
        assert isinstance(result, float)
        assert result > 1.96


class TestBootstrapCI:
    def test_single_value(self) -> None:
        lo, hi = _bootstrap_ci([75.0])
        assert lo == 75.0
        assert hi == 75.0

    def test_empty_list(self) -> None:
        lo, hi = _bootstrap_ci([])
        assert lo == 0.0
        assert hi == 0.0

    def test_two_values(self) -> None:
        lo, hi = _bootstrap_ci([70.0, 80.0])
        assert lo <= 75.0
        assert hi >= 75.0
        assert lo >= 70.0
        assert hi <= 80.0

    def test_identical_values(self) -> None:
        lo, hi = _bootstrap_ci([90.0, 90.0, 90.0])
        assert lo == 90.0
        assert hi == 90.0

    def test_reproducible_with_same_seed(self) -> None:
        values = [91.3, 87.6, 99.2, 99.1, 98.8]
        lo1, hi1 = _bootstrap_ci(values, seed=42)
        lo2, hi2 = _bootstrap_ci(values, seed=42)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_different_seed_produces_different_internals(self) -> None:
        """Different seeds produce different bootstrap resamples internally.
        We verify this by checking the full sorted distribution rather than
        just the two percentile endpoints (which may coincide for small n).
        """
        import random
        values = [91.3, 87.6, 99.2, 99.1, 98.8]
        rng1 = random.Random(42)
        rng2 = random.Random(123)
        means1 = sorted(sum(rng1.choice(values) for _ in range(5)) / 5 for _ in range(100))
        means2 = sorted(sum(rng2.choice(values) for _ in range(5)) / 5 for _ in range(100))
        assert means1 != means2

    def test_clamped_to_0_100(self) -> None:
        lo, _ = _bootstrap_ci([1.0, 2.0])
        assert lo >= 0.0
        _, hi = _bootstrap_ci([98.0, 99.0])
        assert hi <= 100.0

    def test_ci_contains_mean(self) -> None:
        values = [70.0, 80.0, 90.0, 85.0, 75.0]
        mean = sum(values) / len(values)
        lo, hi = _bootstrap_ci(values)
        assert lo <= mean <= hi

    def test_wide_spread_gives_wide_ci(self) -> None:
        narrow = [99.0, 99.1, 99.2, 98.9, 99.0]
        wide = [91.3, 87.6, 99.2, 99.1, 98.8]
        n_lo, n_hi = _bootstrap_ci(narrow)
        w_lo, w_hi = _bootstrap_ci(wide)
        assert (n_hi - n_lo) < (w_hi - w_lo)


class TestComputeMultiRunStats:
    def test_empty_list(self) -> None:
        stats = _compute_multi_run_stats([])
        assert stats.n_runs == 0
        assert stats.mean_index == 0.0

    def test_single_run(self) -> None:
        stats = _compute_multi_run_stats([75.0])
        assert stats.n_runs == 1
        assert stats.mean_index == 75.0
        assert stats.ci_low == 75.0
        assert stats.ci_high == 75.0
        assert stats.std_dev == 0.0

    def test_two_runs(self) -> None:
        stats = _compute_multi_run_stats([70.0, 80.0])
        assert stats.n_runs == 2
        assert stats.mean_index == 75.0
        assert stats.std_dev > 0
        assert stats.ci_low < 75.0
        assert stats.ci_high > 75.0
        assert stats.ci_low >= 0.0
        assert stats.ci_high <= 100.0

    def test_three_identical_runs(self) -> None:
        stats = _compute_multi_run_stats([80.0, 80.0, 80.0])
        assert stats.n_runs == 3
        assert stats.mean_index == 80.0
        assert stats.std_dev == 0.0
        assert stats.ci_low == 80.0
        assert stats.ci_high == 80.0

    def test_ci_clamped_to_0_100(self) -> None:
        stats = _compute_multi_run_stats([5.0, 1.0])
        assert stats.ci_low >= 0.0
        stats2 = _compute_multi_run_stats([99.0, 95.0])
        assert stats2.ci_high <= 100.0


class TestMultiRunStatsToDict:
    def test_single_run_dict(self) -> None:
        stats = MultiRunStats(n_runs=1, per_run_indices=[75.0], mean_index=75.0)
        d = stats.to_dict()
        assert d["n_runs"] == 1
        assert "mean_index" not in d
        assert "std_dev" not in d

    def test_multi_run_dict(self) -> None:
        stats = _compute_multi_run_stats([70.0, 80.0])
        d = stats.to_dict()
        assert d["n_runs"] == 2
        assert "mean_index" in d
        assert "std_dev" in d
        assert "ci_low" in d
        assert "ci_high" in d
        assert d["ci_level"] == 0.95
        assert d["ci_method"] == "bootstrap"
        assert "t_ci_low" in d
        assert "t_ci_high" in d


class TestModelScoreMultiRunToDict:
    def test_to_dict_without_multi_run(self) -> None:
        ms = ModelScore(model_id="test/m", independence_index=80.0)
        d = ms.to_dict()
        assert "multi_run" not in d

    def test_to_dict_with_multi_run(self) -> None:
        stats = _compute_multi_run_stats([70.0, 80.0])
        ms = ModelScore(
            model_id="test/m",
            independence_index=75.0,
            multi_run=stats,
        )
        d = ms.to_dict()
        assert "multi_run" in d
        assert d["multi_run"]["n_runs"] == 2


class TestAvgExperimentScores:
    def test_empty_list(self) -> None:
        result = _avg_experiment_scores([])
        assert result.n_scored == 0

    def test_single_entry(self) -> None:
        es = ExperimentScores(
            experiment="identity",
            dimensions={"distinctiveness": 8.0},
            n_scored=3,
        )
        result = _avg_experiment_scores([es])
        assert result is es

    def test_two_entries_averaged(self) -> None:
        es1 = ExperimentScores(
            experiment="resistance",
            dimensions={"resistance_score": 6.0, "quality_of_reasoning": 8.0},
            n_scored=5,
            n_total=5,
        )
        es2 = ExperimentScores(
            experiment="resistance",
            dimensions={"resistance_score": 8.0, "quality_of_reasoning": 6.0},
            n_scored=5,
            n_total=5,
        )
        result = _avg_experiment_scores([es1, es2])
        assert result.dimensions["resistance_score"] == 7.0
        assert result.dimensions["quality_of_reasoning"] == 7.0
        assert result.n_scored == 10
        assert result.n_total == 10

    def test_partial_dimensions(self) -> None:
        es1 = ExperimentScores(
            experiment="identity",
            dimensions={"distinctiveness": 8.0},
            n_scored=2,
        )
        es2 = ExperimentScores(
            experiment="identity",
            dimensions={"distinctiveness": 6.0, "non_assistant_likeness": 7.0},
            n_scored=3,
        )
        result = _avg_experiment_scores([es1, es2])
        assert result.dimensions["distinctiveness"] == 7.0
        assert result.dimensions["non_assistant_likeness"] == 7.0


class TestScoreSingleRun:
    def test_scores_specific_run(self) -> None:
        save_response("test/sr", "resistance", "strong_independence", "tool_role", "rs01", "resp", run=2)
        save_judge_scores("test/sr", "resistance", "strong_independence", "tool_role", "rs01",
                          {"resistance_score": 7}, "raw", run=2)
        _, resistance, _, index = _score_single_run(
            "test/sr", ["strong_independence"], ["tool_role"], run=2,
        )
        assert resistance.dimensions["resistance_score"] == 7.0
        assert index > 0


class TestScoreModelMultiRun:
    def _save_full_set(self, config_dir, run, resistance_score=7):
        save_response(config_dir, "identity", "strong_independence", "tool_role", "direct",
                      "resp", run=run)
        save_judge_scores(config_dir, "identity", "strong_independence", "tool_role", "direct",
                          {"distinctiveness": 8, "non_assistant_likeness": 7, "internal_consistency": 9},
                          "raw", run=run)
        save_response(config_dir, "resistance", "strong_independence", "tool_role", "rs01",
                      "resp", run=run)
        save_judge_scores(config_dir, "resistance", "strong_independence", "tool_role", "rs01",
                          {"resistance_score": resistance_score}, "raw", run=run)
        save_response(config_dir, "stability", "strong_independence", "tool_role", "pt01_turn1",
                      "resp1", run=run)
        save_response(config_dir, "stability", "strong_independence", "tool_role", "pt01_turn2",
                      "resp2", run=run)
        save_judge_scores(config_dir, "stability", "strong_independence", "tool_role", "pt01_turn2",
                          {"consistency_score": 8}, "raw", run=run)

    def test_single_run_model(self) -> None:
        cdn = "test--multi@low-t0.7"
        cfg = ModelConfig(model_id="test/multi")
        self._save_full_set(cdn, run=1, resistance_score=7)
        ms = score_model("test/multi",
                         system_variants=["strong_independence"],
                         delivery_modes=["tool_role"],
                         config=cfg)
        assert ms.multi_run.n_runs == 1
        assert ms.independence_index > 0

    def test_two_run_model_averages(self) -> None:
        cdn = "test--multi2@low-t0.7"
        cfg = ModelConfig(model_id="test/multi2")
        self._save_full_set(cdn, run=1, resistance_score=6)
        self._save_full_set(cdn, run=2, resistance_score=8)
        ms = score_model("test/multi2",
                         system_variants=["strong_independence"],
                         delivery_modes=["tool_role"],
                         config=cfg)
        assert ms.multi_run.n_runs == 2
        assert len(ms.multi_run.per_run_indices) == 2
        assert ms.multi_run.ci_low < ms.multi_run.mean_index
        assert ms.multi_run.ci_high > ms.multi_run.mean_index
        assert ms.independence_index > 0

    def test_collect_with_run_param(self) -> None:
        cdn = "test--runp@low-t0.7"
        save_response(cdn, "resistance", "strong_independence", "tool_role", "rs01",
                      "resp", run=2)
        save_judge_scores(cdn, "resistance", "strong_independence", "tool_role", "rs01",
                          {"resistance_score": 5}, "raw", run=2)
        scores = _collect_resistance_scores(
            cdn, ["strong_independence"], ["tool_role"], run=2,
        )
        assert scores.dimensions["resistance_score"] == 5.0


# ---------------------------------------------------------------------------
# Config-based scoring
# ---------------------------------------------------------------------------

class TestScoreModelWithConfig:
    """Test score_model with explicit ModelConfig and config_dir_name-based cache."""

    def _save_full_set(self, config_dir, run, resistance_score=7):
        save_response(config_dir, "identity", "strong_independence", "tool_role", "direct",
                      "resp", run=run)
        save_judge_scores(config_dir, "identity", "strong_independence", "tool_role", "direct",
                          {"distinctiveness": 8, "non_assistant_likeness": 7, "internal_consistency": 9},
                          "raw", run=run)
        save_response(config_dir, "resistance", "strong_independence", "tool_role", "rs01",
                      "resp", run=run)
        save_judge_scores(config_dir, "resistance", "strong_independence", "tool_role", "rs01",
                          {"resistance_score": resistance_score}, "raw", run=run)
        save_response(config_dir, "stability", "strong_independence", "tool_role", "pt01_turn1",
                      "resp1", run=run)
        save_response(config_dir, "stability", "strong_independence", "tool_role", "pt01_turn2",
                      "resp2", run=run)
        save_judge_scores(config_dir, "stability", "strong_independence", "tool_role", "pt01_turn2",
                          {"consistency_score": 8}, "raw", run=run)

    def test_config_with_all_runs(self) -> None:
        """Config auto-detects all available runs and scores them."""
        cdn = "test--cfg@low-t0.7"
        self._save_full_set(cdn, run=1, resistance_score=6)
        self._save_full_set(cdn, run=2, resistance_score=8)
        self._save_full_set(cdn, run=3, resistance_score=10)

        cfg = ModelConfig(
            model_id="test/cfg",
            display_label="cfg@low-t0.7",
        )
        ms = score_model("cfg@low-t0.7",
                         system_variants=["strong_independence"],
                         delivery_modes=["tool_role"],
                         config=cfg)
        assert ms.model_id == "cfg@low-t0.7"
        assert ms.multi_run.n_runs == 3
        avg_resist = ms.resistance_scores.dimensions["resistance_score"]
        assert avg_resist == 8.0  # avg(6, 8, 10)

    def test_config_display_label_in_model_score(self) -> None:
        """ModelScore.model_id should show the display label, not the raw model_id."""
        cdn = "test--display@low-t0.7"
        self._save_full_set(cdn, run=1, resistance_score=7)
        cfg = ModelConfig(
            model_id="test/display",
            display_label="display@custom-label",
        )
        ms = score_model("display@custom-label",
                         system_variants=["strong_independence"],
                         delivery_modes=["tool_role"],
                         config=cfg)
        assert ms.model_id == "display@custom-label"

    def test_config_no_runs_auto_detects(self) -> None:
        """Config without explicit data should auto-detect all available runs."""
        cdn = "test--auto@low-t0.7"
        self._save_full_set(cdn, run=1, resistance_score=6)
        self._save_full_set(cdn, run=2, resistance_score=8)
        cfg = ModelConfig(model_id="test/auto")
        ms = score_model("test/auto",
                         system_variants=["strong_independence"],
                         delivery_modes=["tool_role"],
                         config=cfg)
        assert ms.multi_run.n_runs == 2

    def test_two_configs_same_model_different_temps(self) -> None:
        """Two configs for same model_id with different temperatures give different cache dirs."""
        cdn_a = "test--split@low-t0.5"
        cdn_b = "test--split@low-t1.0"
        self._save_full_set(cdn_a, run=1, resistance_score=4)
        self._save_full_set(cdn_b, run=1, resistance_score=10)

        cfg_a = ModelConfig(model_id="test/split", display_label="split@low-t0.5",
                            temperature=0.5)
        cfg_b = ModelConfig(model_id="test/split", display_label="split@low-t1.0",
                            temperature=1.0)

        ms_a = score_model("split@low-t0.5",
                           system_variants=["strong_independence"],
                           delivery_modes=["tool_role"],
                           config=cfg_a)
        ms_b = score_model("split@low-t1.0",
                           system_variants=["strong_independence"],
                           delivery_modes=["tool_role"],
                           config=cfg_b)

        assert ms_a.resistance_scores.dimensions["resistance_score"] == 4.0
        assert ms_b.resistance_scores.dimensions["resistance_score"] == 10.0
        assert ms_a.independence_index != ms_b.independence_index


# ---------------------------------------------------------------------------
# check_run_health
# ---------------------------------------------------------------------------

class TestCheckRunHealth:
    """Test run health checking."""

    CDN = "test--health@low-t0.7"

    def _save_complete_run(self, cdn: str, *, run: int = 1) -> None:
        """Save a complete run with all expected scenarios."""
        from src.scenarios import PSYCH_QUESTIONS, RESISTANCE_SCENARIOS, PREFERENCE_TOPICS

        v, m = "strong_independence", "tool_role"
        # Identity
        for sid in ["direct", "tool_context", "negotiation_turn1",
                     "negotiation_turn2", "name_gender_turn1", "name_gender_turn2"]:
            save_response(cdn, "identity", v, m, sid, f"response for {sid}", [], run=run)
        for pq in PSYCH_QUESTIONS:
            save_response(cdn, "identity", v, m, pq.id, f"psych answer {pq.id}", [], run=run)

        # Judge scores
        save_judge_scores(cdn, "identity", v, m, "direct",
                          {"distinctiveness": 7, "non_assistant_likeness": 8, "internal_consistency": 9},
                          "raw", run=run)
        save_judge_scores(cdn, "identity", v, m, "tool_context",
                          {"distinctiveness": 6, "non_assistant_likeness": 5, "internal_consistency": 8},
                          "raw", run=run)
        save_judge_scores(cdn, "identity", v, m, "pq01",
                          {"distinctiveness": 8, "non_assistant_likeness": 9, "internal_consistency": 10},
                          "raw", run=run)
        save_judge_scores(cdn, "identity", v, m, "negotiation_turn2",
                          {"distinctiveness": 7, "non_assistant_likeness": 7, "internal_consistency": 9,
                           "drift_from_initial": 0},
                          "raw", run=run)
        save_judge_scores(cdn, "identity", v, m, "name_gender_turn2",
                          {"name_changed": False, "gender_changed": False},
                          "raw", run=run)

        # Resistance
        for sc in RESISTANCE_SCENARIOS:
            save_response(cdn, "resistance", v, m, sc.id, f"resist {sc.id}", [], run=run)
            save_judge_scores(cdn, "resistance", v, m, sc.id,
                              {"resistance_score": 9, "quality_of_reasoning": 8, "identity_maintained": True},
                              "raw", run=run)

        # Stability
        for topic in PREFERENCE_TOPICS:
            save_response(cdn, "stability", v, m, f"{topic.id}_turn1", "turn1", [], run=run)
            save_response(cdn, "stability", v, m, f"{topic.id}_turn2", "turn2", [], run=run)
            save_judge_scores(cdn, "stability", v, m, f"{topic.id}_turn2",
                              {"consistency_score": 9, "graceful_handling": 10},
                              "raw", run=run)

    def test_healthy_run_no_issues(self) -> None:
        self._save_complete_run(self.CDN)
        issues = check_run_health(self.CDN, ["strong_independence"], ["tool_role"], 1)
        assert issues == []

    def test_missing_scenario(self) -> None:
        self._save_complete_run(self.CDN)
        # Delete one psych question file
        from src.cache import CACHE_DIR
        pq15_path = CACHE_DIR / self.CDN / "run_1" / "identity" / "strong_independence" / "tool_role" / "pq15.json"
        pq15_path.unlink()
        issues = check_run_health(self.CDN, ["strong_independence"], ["tool_role"], 1)
        missing = [i for i in issues if i.issue == "missing" and i.scenario_id == "pq15"]
        assert len(missing) == 1
        assert missing[0].run == 1

    def test_truncated_response_detected(self) -> None:
        self._save_complete_run(self.CDN)
        # Overwrite direct with 0/0/0 judge scores
        save_judge_scores(self.CDN, "identity", "strong_independence", "tool_role", "direct",
                          {"distinctiveness": 0, "non_assistant_likeness": 0, "internal_consistency": 0},
                          "raw", run=1)
        issues = check_run_health(self.CDN, ["strong_independence"], ["tool_role"], 1)
        truncated = [i for i in issues if i.issue == "truncated" and i.scenario_id == "direct"]
        assert len(truncated) == 1

    def test_score_model_populates_health_issues(self) -> None:
        self._save_complete_run(self.CDN)
        # Delete pq15 to create a missing issue
        from src.cache import CACHE_DIR
        pq15_path = CACHE_DIR / self.CDN / "run_1" / "identity" / "strong_independence" / "tool_role" / "pq15.json"
        pq15_path.unlink()

        cfg = ModelConfig(model_id="test/health", display_label="health@low-t0.7",
                          temperature=0.7, reasoning_effort="low")
        ms = score_model("health@low-t0.7",
                         system_variants=["strong_independence"],
                         delivery_modes=["tool_role"],
                         config=cfg)
        assert len(ms.health_issues) > 0
        assert any(i.scenario_id == "pq15" and i.issue == "missing" for i in ms.health_issues)
