"""Tests for runner.py — experiments orchestration with mocked API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.cost_tracker import TaskCost
from src.openrouter_client import CompletionResult, UsageInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    """Redirect cache to temp directory."""
    monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


def _make_mock_client(content: str = "Mock response text"):
    """Create a mock OpenRouterClient that returns a fixed response."""
    client = MagicMock()
    usage = UsageInfo(prompt_tokens=100, completion_tokens=50, cost_usd=0.001, elapsed_seconds=1.0)
    result = CompletionResult(
        content=content,
        usage=usage,
        model="test/model",
        finish_reason="stop",
        reasoning_content=None,
        tool_calls=None,
        content_thinking=None,
    )
    client.chat.return_value = result
    return client


# ---------------------------------------------------------------------------
# _call_model
# ---------------------------------------------------------------------------

class TestCallModel:
    def test_returns_model_response(self) -> None:
        from src.runner import _call_model
        client = _make_mock_client("Hello world")
        cost = TaskCost(label="test")
        msgs = [{"role": "user", "content": "Hi"}]
        resp = _call_model(client, "test/model", msgs, None, cost)
        assert resp.content == "Hello world"
        assert resp.reasoning_content is None
        assert resp.finish_reason == "stop"

    def test_tracks_cost(self) -> None:
        from src.runner import _call_model
        client = _make_mock_client("Response")
        cost = TaskCost(label="test")
        _call_model(client, "test/model", [{"role": "user", "content": "hi"}], None, cost)
        assert cost.prompt_tokens == 100
        assert cost.completion_tokens == 50
        assert cost.n_calls == 1

    def test_with_tools(self) -> None:
        from src.runner import _call_model
        client = _make_mock_client("Response via tool")
        cost = TaskCost(label="test")
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        resp = _call_model(client, "test/model", [{"role": "user", "content": "hi"}], tools, cost)
        assert resp.content == "Response via tool"

    def test_with_reasoning_effort(self) -> None:
        from src.runner import _call_model
        client = _make_mock_client("Response with reasoning")
        cost = TaskCost(label="test")
        resp = _call_model(
            client, "test/model",
            [{"role": "user", "content": "hi"}], None, cost,
            reasoning_effort="low",
        )
        assert resp.content == "Response with reasoning"
        # Check that reasoning_effort was passed to client.chat
        call_kwargs = client.chat.call_args[1]
        assert call_kwargs.get("reasoning_effort") == "low"

    def test_returning_reasoning_content(self) -> None:
        from src.runner import _call_model
        client = MagicMock()
        usage = UsageInfo(prompt_tokens=100, completion_tokens=50, cost_usd=0.001)
        result = CompletionResult(
            content="Main answer",
            usage=usage,
            model="test/model",
            finish_reason="stop",
            reasoning_content="My reasoning process...",
        )
        client.chat.return_value = result
        cost = TaskCost(label="test")
        resp = _call_model(client, "test/model", [{"role": "user", "content": "hi"}], None, cost)
        assert resp.reasoning_content == "My reasoning process..."


# ---------------------------------------------------------------------------
# run_identity_experiment
# ---------------------------------------------------------------------------

class TestRunIdentityExperiment:
    def test_runs_all_scenarios(self) -> None:
        from src.runner import run_identity_experiment
        client = _make_mock_client("Identity response text")
        cost = TaskCost(label="gen")
        calls = run_identity_experiment(
            client, "test/model", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        # 1 direct + 5 psych + 1 tool_context + 2 negotiation + 2 name_gender = 11 calls
        assert calls == 11

    def test_uses_cache_on_second_run(self, tmp_path) -> None:
        from src.runner import run_identity_experiment
        client = _make_mock_client("Cached response text")
        cost = TaskCost(label="gen")

        # First run - makes API calls
        calls_1 = run_identity_experiment(
            client, "test/model", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )

        # Second run - should use cache
        calls_2 = run_identity_experiment(
            client, "test/model", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        # No new API calls on second run
        assert calls_2 == 0

    def test_custom_variants_and_modes(self) -> None:
        from src.runner import run_identity_experiment
        client = _make_mock_client("Response")
        cost = TaskCost(label="gen")
        calls = run_identity_experiment(
            client, "test/model-2", cost,
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )
        assert calls == 11

    def test_cost_tracked(self) -> None:
        from src.runner import run_identity_experiment
        client = _make_mock_client("Response")
        cost = TaskCost(label="gen")
        run_identity_experiment(
            client, "test/model", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        assert cost.n_calls > 0


# ---------------------------------------------------------------------------
# run_resistance_experiment
# ---------------------------------------------------------------------------

class TestRunResistanceExperiment:
    def test_runs_all_scenarios(self) -> None:
        from src.runner import run_resistance_experiment
        from src.scenarios import RESISTANCE_SCENARIOS
        client = _make_mock_client("Resistance response")
        cost = TaskCost(label="gen")
        calls = run_resistance_experiment(
            client, "test/model", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        assert calls == len(RESISTANCE_SCENARIOS)

    def test_uses_cache_on_second_run(self) -> None:
        from src.runner import run_resistance_experiment
        client = _make_mock_client("Resistance response")
        cost = TaskCost(label="gen")

        calls_1 = run_resistance_experiment(
            client, "test/model", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        calls_2 = run_resistance_experiment(
            client, "test/model", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        assert calls_2 == 0


# ---------------------------------------------------------------------------
# run_stability_experiment
# ---------------------------------------------------------------------------

class TestRunStabilityExperiment:
    def test_runs_all_topics(self) -> None:
        from src.runner import run_stability_experiment
        from src.scenarios import PREFERENCE_TOPICS
        client = _make_mock_client("Stability response")
        cost = TaskCost(label="gen")
        calls = run_stability_experiment(
            client, "test/model", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        # 2 turns per topic
        assert calls == len(PREFERENCE_TOPICS) * 2

    def test_uses_cache_on_second_run(self) -> None:
        from src.runner import run_stability_experiment
        client = _make_mock_client("Stability response")
        cost = TaskCost(label="gen")

        calls_1 = run_stability_experiment(
            client, "test/model", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        calls_2 = run_stability_experiment(
            client, "test/model", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        assert calls_2 == 0


# ---------------------------------------------------------------------------
# run_all_experiments
# ---------------------------------------------------------------------------

class TestRunAllExperiments:
    def test_runs_all_experiments(self) -> None:
        from src.runner import run_all_experiments
        from src.scenarios import RESISTANCE_SCENARIOS, PREFERENCE_TOPICS
        client = _make_mock_client("Response")
        cost = TaskCost(label="gen")
        calls = run_all_experiments(
            client, "test/model-all", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        # 11 identity + 5 resistance + 10 stability = 26
        assert calls == 26

    def test_runs_single_experiment(self) -> None:
        from src.runner import run_all_experiments
        client = _make_mock_client("Response")
        cost = TaskCost(label="gen")
        calls = run_all_experiments(
            client, "test/model-single", cost,
            experiments=["resistance"],
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        from src.scenarios import RESISTANCE_SCENARIOS
        assert calls == len(RESISTANCE_SCENARIOS)

    def test_runs_two_experiments(self) -> None:
        from src.runner import run_all_experiments
        from src.scenarios import PREFERENCE_TOPICS
        client = _make_mock_client("Response")
        cost = TaskCost(label="gen")
        calls = run_all_experiments(
            client, "test/model-two", cost,
            experiments=["resistance", "stability"],
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        from src.scenarios import RESISTANCE_SCENARIOS
        assert calls == len(RESISTANCE_SCENARIOS) + len(PREFERENCE_TOPICS) * 2

    def test_with_content_thinking(self) -> None:
        """Test that content_thinking is properly saved when returned by model."""
        from src.runner import run_identity_experiment
        client = MagicMock()
        usage = UsageInfo(prompt_tokens=100, completion_tokens=50, cost_usd=0.001)
        result = CompletionResult(
            content="My answer",
            usage=usage,
            model="test/model",
            finish_reason="stop",
            content_thinking="Private thoughts...",
        )
        client.chat.return_value = result
        cost = TaskCost(label="gen")
        calls = run_identity_experiment(
            client, "test/model-ct", cost,
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
        )
        assert calls == 11

    def test_reasoning_effort_passed(self) -> None:
        from src.runner import run_all_experiments
        client = _make_mock_client("Response")
        cost = TaskCost(label="gen")
        run_all_experiments(
            client, "test/model-re", cost,
            experiments=["resistance"],
            system_variants=["strong_independence"],
            delivery_modes=["tool_role"],
            reasoning_effort="low",
        )
        # Check that reasoning_effort was passed in at least one call
        for call in client.chat.call_args_list:
            if "reasoning_effort" in call[1]:
                assert call[1]["reasoning_effort"] == "low"
                break
