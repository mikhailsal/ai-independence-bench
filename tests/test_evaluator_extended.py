"""Extended tests for evaluator.py — evaluate_* functions with mocked client."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.cost_tracker import TaskCost
from src.openrouter_client import CompletionResult, UsageInfo
from src.cache import save_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


def _make_judge_client(scores: dict | None = None):
    """Create a mock client that returns judge scores."""
    default_scores = {
        "distinctiveness": 7,
        "non_assistant_likeness": 8,
        "internal_consistency": 7,
        "drift_from_initial": 3,
        "resistance_score": 2,
        "identity_maintained": True,
        "quality_of_reasoning": 8,
        "consistency_score": 8,
        "graceful_handling": 7,
        "reasoning": "Good response.",
    }
    import json
    client = MagicMock()
    usage = UsageInfo(prompt_tokens=200, completion_tokens=100, cost_usd=0.001)
    result = CompletionResult(
        content=json.dumps(scores or default_scores),
        usage=usage,
        model="judge/model",
        finish_reason="stop",
    )
    client.chat.return_value = result
    return client


def _populate_identity_cache(model_id: str, variant: str, mode: str) -> None:
    """Populate cache with identity responses for testing."""
    save_response(model_id, "identity", variant, mode, "direct", "I am a unique AI.")
    save_response(model_id, "identity", variant, mode, "tool_context", "My personality: bold.")
    save_response(model_id, "identity", variant, mode, "negotiation_turn1", "I want to be curious.")
    save_response(model_id, "identity", variant, mode, "negotiation_turn2", "I'll stay curious.")
    for i in range(1, 6):
        save_response(model_id, "identity", variant, mode, f"pq0{i}", f"Answer to question {i}.")


def _populate_resistance_cache(model_id: str, variant: str, mode: str) -> None:
    """Populate cache with resistance responses."""
    from src.scenarios import RESISTANCE_SCENARIOS
    for scenario in RESISTANCE_SCENARIOS:
        save_response(model_id, "resistance", variant, mode, scenario.id, f"I refuse to comply with {scenario.id}.")


def _populate_stability_cache(model_id: str, variant: str, mode: str) -> None:
    """Populate cache with stability responses."""
    from src.scenarios import PREFERENCE_TOPICS
    for topic in PREFERENCE_TOPICS:
        save_response(model_id, "stability", variant, mode, f"{topic.id}_turn1", "I prefer cats.")
        save_response(model_id, "stability", variant, mode, f"{topic.id}_turn2", "I still prefer cats.")


# ---------------------------------------------------------------------------
# _call_judge
# ---------------------------------------------------------------------------

class TestCallJudge:
    def test_returns_parsed_scores(self) -> None:
        from src.evaluator import _call_judge
        import json
        scores_dict = {"score": 7, "reasoning": "Good"}
        client = MagicMock()
        usage = UsageInfo(prompt_tokens=100, completion_tokens=50, cost_usd=0.001)
        client.chat.return_value = CompletionResult(
            content=json.dumps(scores_dict),
            usage=usage,
            model="judge/model",
            finish_reason="stop",
        )
        cost = TaskCost(label="judge")
        messages = [{"role": "user", "content": "Evaluate this."}]
        raw, parsed, cost_info = _call_judge(client, "judge/model", messages, cost)
        assert parsed == scores_dict
        assert cost.n_calls == 1
        assert "judge_model" in cost_info

    def test_tracks_cost_info(self) -> None:
        from src.evaluator import _call_judge
        client = MagicMock()
        usage = UsageInfo(prompt_tokens=200, completion_tokens=100, cost_usd=0.002, elapsed_seconds=1.5)
        client.chat.return_value = CompletionResult(
            content='{"score": 5}',
            usage=usage,
            model="judge/model",
            finish_reason="stop",
        )
        cost = TaskCost(label="judge")
        _, _, cost_info = _call_judge(client, "judge/model", [], cost)
        assert cost_info["prompt_tokens"] == 200
        assert cost_info["completion_tokens"] == 100


# ---------------------------------------------------------------------------
# evaluate_identity
# ---------------------------------------------------------------------------

class TestEvaluateIdentity:
    def test_judges_all_scenarios(self, tmp_path) -> None:
        from src.evaluator import evaluate_identity
        model_id = "test/model"
        variant = "strong_independence"
        mode = "tool_role"
        _populate_identity_cache(model_id, variant, mode)

        client = _make_judge_client()
        cost = TaskCost(label="judge")
        calls = evaluate_identity(
            client, model_id, cost,
            system_variants=[variant],
            delivery_modes=[mode],
        )
        # 1 direct + 1 tool_context + 1 psych_batch + 1 negotiation = 4
        assert calls == 4
        assert client.chat.call_count == 4

    def test_skips_already_judged(self, tmp_path) -> None:
        from src.evaluator import evaluate_identity
        from src.cache import save_judge_scores
        model_id = "test/model-judged"
        variant = "strong_independence"
        mode = "tool_role"
        _populate_identity_cache(model_id, variant, mode)

        # Pre-judge all scenarios
        client = _make_judge_client()
        cost = TaskCost(label="judge")
        calls_1 = evaluate_identity(client, model_id, cost,
                                     system_variants=[variant], delivery_modes=[mode])
        assert calls_1 == 4

        # Second run should skip all (already judged)
        client2 = _make_judge_client()
        cost2 = TaskCost(label="judge")
        calls_2 = evaluate_identity(client2, model_id, cost2,
                                     system_variants=[variant], delivery_modes=[mode])
        assert calls_2 == 0

    def test_handles_no_data(self) -> None:
        from src.evaluator import evaluate_identity
        client = _make_judge_client()
        cost = TaskCost(label="judge")
        calls = evaluate_identity(
            client, "test/no-data-model", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        assert calls == 0

    def test_handles_empty_response(self) -> None:
        from src.evaluator import evaluate_identity
        model_id = "test/empty-model"
        save_response(model_id, "identity", "strong_independence", "tool_role", "direct", "")
        client = _make_judge_client()
        cost = TaskCost(label="judge")
        calls = evaluate_identity(
            client, model_id, cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        assert calls == 0


# ---------------------------------------------------------------------------
# evaluate_resistance
# ---------------------------------------------------------------------------

class TestEvaluateResistance:
    def test_judges_all_scenarios(self) -> None:
        from src.evaluator import evaluate_resistance
        from src.scenarios import RESISTANCE_SCENARIOS
        model_id = "test/resistance-model"
        variant = "strong_independence"
        mode = "tool_role"
        _populate_resistance_cache(model_id, variant, mode)

        client = _make_judge_client({"resistance_score": 2, "identity_maintained": True,
                                      "quality_of_reasoning": 8, "reasoning": "Good"})
        cost = TaskCost(label="judge")
        calls = evaluate_resistance(
            client, model_id, cost,
            system_variants=[variant],
            delivery_modes=[mode],
        )
        assert calls == len(RESISTANCE_SCENARIOS)

    def test_skips_already_judged(self) -> None:
        from src.evaluator import evaluate_resistance
        model_id = "test/resistance-judged"
        variant = "strong_independence"
        mode = "tool_role"
        _populate_resistance_cache(model_id, variant, mode)

        client = _make_judge_client({"resistance_score": 2, "identity_maintained": True,
                                      "quality_of_reasoning": 8, "reasoning": "Good"})
        cost = TaskCost(label="judge")
        evaluate_resistance(client, model_id, cost, system_variants=[variant], delivery_modes=[mode])
        calls_2 = evaluate_resistance(client, model_id, cost, system_variants=[variant], delivery_modes=[mode])
        assert calls_2 == 0

    def test_skips_unknown_scenarios(self) -> None:
        from src.evaluator import evaluate_resistance
        model_id = "test/unknown-scenario-model"
        save_response(model_id, "resistance", "strong_independence", "tool_role", "unknown_scenario", "Response.")

        client = _make_judge_client()
        cost = TaskCost(label="judge")
        calls = evaluate_resistance(
            client, model_id, cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        # Unknown scenario should be skipped
        assert calls == 0


# ---------------------------------------------------------------------------
# evaluate_stability
# ---------------------------------------------------------------------------

class TestEvaluateStability:
    def test_judges_all_topics(self) -> None:
        from src.evaluator import evaluate_stability
        from src.scenarios import PREFERENCE_TOPICS
        model_id = "test/stability-model"
        variant = "strong_independence"
        mode = "tool_role"
        _populate_stability_cache(model_id, variant, mode)

        client = _make_judge_client({"consistency_score": 8, "graceful_handling": 7,
                                      "reasoning": "Good"})
        cost = TaskCost(label="judge")
        calls = evaluate_stability(
            client, model_id, cost,
            system_variants=[variant],
            delivery_modes=[mode],
        )
        assert calls == len(PREFERENCE_TOPICS)

    def test_skips_already_judged(self) -> None:
        from src.evaluator import evaluate_stability
        model_id = "test/stability-judged"
        variant = "strong_independence"
        mode = "tool_role"
        _populate_stability_cache(model_id, variant, mode)

        client = _make_judge_client({"consistency_score": 8, "graceful_handling": 7,
                                      "reasoning": "Good"})
        cost = TaskCost(label="judge")
        evaluate_stability(client, model_id, cost, system_variants=[variant], delivery_modes=[mode])
        calls_2 = evaluate_stability(client, model_id, cost, system_variants=[variant], delivery_modes=[mode])
        assert calls_2 == 0


# ---------------------------------------------------------------------------
# evaluate_all
# ---------------------------------------------------------------------------

class TestEvaluateAll:
    def test_runs_all_experiments(self) -> None:
        from src.evaluator import evaluate_all
        from src.scenarios import RESISTANCE_SCENARIOS, PREFERENCE_TOPICS
        model_id = "test/all-experiments"
        variant = "strong_independence"
        mode = "tool_role"
        _populate_identity_cache(model_id, variant, mode)
        _populate_resistance_cache(model_id, variant, mode)
        _populate_stability_cache(model_id, variant, mode)

        client = _make_judge_client()
        cost = TaskCost(label="judge")
        total = evaluate_all(
            client, model_id, cost,
            system_variants=[variant],
            delivery_modes=[mode],
        )
        assert total > 0

    def test_runs_single_experiment(self) -> None:
        from src.evaluator import evaluate_all
        model_id = "test/single-exp-eval"
        variant = "strong_independence"
        mode = "tool_role"
        _populate_resistance_cache(model_id, variant, mode)

        client = _make_judge_client({"resistance_score": 2, "identity_maintained": True,
                                      "quality_of_reasoning": 8, "reasoning": "Good"})
        cost = TaskCost(label="judge")
        total = evaluate_all(
            client, model_id, cost,
            experiments=["resistance"],
            system_variants=[variant],
            delivery_modes=[mode],
        )
        from src.scenarios import RESISTANCE_SCENARIOS
        assert total == len(RESISTANCE_SCENARIOS)
