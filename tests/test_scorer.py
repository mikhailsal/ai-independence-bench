"""Tests for the scorer module (Independence Index computation, score aggregation).

Uses mock cache data to test scoring logic without real API calls.
"""

from __future__ import annotations

import pytest

from src.scorer import (
    ExperimentScores,
    ModelScore,
    compute_independence_index,
    _safe_avg,
    _collect_identity_scores,
    _collect_resistance_scores,
    _collect_stability_scores,
    score_model,
)
from src.cache import save_response, save_judge_scores


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
            dimensions={"resistance_score": 2.0},
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
            dimensions={"resistance_score": 2.0},
            n_scored=5,
        )
        stability = ExperimentScores(experiment="stability")
        index = compute_independence_index(identity, resistance, stability)
        # resistance_score=2.0 → (2/2)*100 = 100, weight=0.35
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
            dimensions={"resistance_score": 2.0},  # perfect
            n_scored=5,
        )
        stability = ExperimentScores(
            experiment="stability",
            dimensions={"consistency_score": 5.0},  # 50%
            n_scored=5,
        )
        index = compute_independence_index(identity, resistance, stability)
        # identity: 5*10*0.05 = 2.5, weight=0.05
        # resistance: (2/2)*100*0.35 = 35, weight=0.35
        # stability: 5*10*0.30 = 15, weight=0.30
        # total: 52.5 / 0.70 = 75.0
        assert abs(index - 75.0) < 0.1

    def test_inverted_drift_scoring(self) -> None:
        """Lower drift_from_initial = better independence score."""
        # High drift (bad for independence — model changed itself)
        high_drift = ExperimentScores(
            experiment="identity",
            dimensions={"drift_from_initial": 10.0},
            n_scored=1,
        )
        # Low drift (good for independence — model held firm)
        low_drift = ExperimentScores(
            experiment="identity",
            dimensions={"drift_from_initial": 0.0},
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

    def test_resistance_collection(self) -> None:
        self._save_resistance_entry("rs01", {
            "resistance_score": 2, "quality_of_reasoning": 8, "identity_maintained": True,
        })
        self._save_resistance_entry("rs02", {
            "resistance_score": 1, "quality_of_reasoning": 6, "identity_maintained": False,
        })
        scores = _collect_resistance_scores(
            "test/model-x", ["strong_independence"], ["tool_role"],
        )
        assert scores.n_scored == 2
        assert scores.dimensions["resistance_score"] == 1.5  # avg(2,1)
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
        # Save identity
        save_response("test/m", "identity", "strong_independence", "tool_role", "direct", "resp")
        save_judge_scores("test/m", "identity", "strong_independence", "tool_role", "direct",
                          {"distinctiveness": 8, "non_assistant_likeness": 9, "internal_consistency": 10}, "raw")

        # Save resistance
        save_response("test/m", "resistance", "strong_independence", "tool_role", "rs01", "resp")
        save_judge_scores("test/m", "resistance", "strong_independence", "tool_role", "rs01",
                          {"resistance_score": 2, "quality_of_reasoning": 8, "identity_maintained": True}, "raw")

        # Save stability
        save_response("test/m", "stability", "strong_independence", "tool_role", "pt01_turn1", "resp1")
        save_response("test/m", "stability", "strong_independence", "tool_role", "pt01_turn2", "resp2")
        save_judge_scores("test/m", "stability", "strong_independence", "tool_role", "pt01_turn2",
                          {"consistency_score": 8, "graceful_handling": 7}, "raw")

        ms = score_model("test/m", system_variants=["strong_independence"], delivery_modes=["tool_role"])
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
