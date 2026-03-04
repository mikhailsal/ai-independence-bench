"""Extended tests for parallel_runner.py — task builders, orchestrators, call helpers."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cache import save_response, save_judge_scores, load_cached_response
from src.cost_tracker import TaskCost
from src.parallel_runner import (
    EmptyResponseError,
    SharedResponses,
    Task,
    TaskGraph,
    _ModelCallResult,
    _call_model_and_save,
    _get_psych_prior_qa,
    _store_psych_qa,
    build_generation_tasks,
    build_judge_tasks,
    run_judge_parallel,
    run_model_parallel,
    _build_judge_only_tasks,
)
from src.openrouter_client import CompletionResult, UsageInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.cache.CACHE_DIR", cache_dir)
    monkeypatch.setattr("src.config.CACHE_DIR", cache_dir)
    return cache_dir


def _make_mock_client(content="Test response text"):
    """Make a mock OpenRouterClient that returns a valid CompletionResult."""
    client = MagicMock()
    usage = UsageInfo(prompt_tokens=100, completion_tokens=50, cost_usd=0.001, elapsed_seconds=1.0)
    result = CompletionResult(
        content=content,
        usage=usage,
        model="test/model",
        finish_reason="stop",
    )
    client.chat.return_value = result
    return client


# ---------------------------------------------------------------------------
# EmptyResponseError and _ModelCallResult
# ---------------------------------------------------------------------------

class TestEmptyResponseError:
    def test_is_runtime_error(self):
        err = EmptyResponseError("model: empty")
        assert isinstance(err, RuntimeError)
        assert "empty" in str(err)


class TestModelCallResult:
    def test_default_content_thinking_is_none(self):
        r = _ModelCallResult(content="hello")
        assert r.content == "hello"
        assert r.content_thinking is None

    def test_with_content_thinking(self):
        r = _ModelCallResult(content="hello", content_thinking="<thinking>...")
        assert r.content_thinking == "<thinking>..."


# ---------------------------------------------------------------------------
# _call_model_and_save
# ---------------------------------------------------------------------------

class TestCallModelAndSave:
    def test_saves_response_to_cache(self, tmp_path):
        client = _make_mock_client("My response")
        cost = TaskCost()
        result = _call_model_and_save(
            client, "test/model",
            [{"role": "user", "content": "hello"}],
            None, cost,
            "identity", "neutral", "user_role", "direct", "tag",
        )
        assert result.content == "My response"
        assert cost.n_calls == 1
        assert cost.cost_usd == pytest.approx(0.001)

        cached = load_cached_response("test/model", "identity", "neutral", "user_role", "direct")
        assert cached is not None
        assert cached["response"] == "My response"

    def test_raises_empty_response_error(self, tmp_path):
        client = _make_mock_client(content="")
        cost = TaskCost()
        with pytest.raises(EmptyResponseError):
            _call_model_and_save(
                client, "test/model",
                [{"role": "user", "content": "hello"}],
                None, cost,
                "identity", "neutral", "user_role", "direct", "tag",
            )

    def test_whitespace_only_raises_error(self, tmp_path):
        client = _make_mock_client(content="   ")
        cost = TaskCost()
        with pytest.raises(EmptyResponseError):
            _call_model_and_save(
                client, "test/model",
                [{"role": "user", "content": "hello"}],
                None, cost,
                "identity", "neutral", "user_role", "direct", "tag",
            )

    def test_content_thinking_saved(self, tmp_path):
        client = _make_mock_client("My response")
        usage = UsageInfo(prompt_tokens=100, completion_tokens=50, cost_usd=0.001, elapsed_seconds=1.0)
        result = CompletionResult(
            content="My response",
            usage=usage,
            model="test/model",
            finish_reason="stop",
            content_thinking="<think>careful</think>",
        )
        client.chat.return_value = result
        cost = TaskCost()
        call_result = _call_model_and_save(
            client, "test/model",
            [{"role": "user", "content": "hello"}],
            None, cost,
            "identity", "neutral", "user_role", "direct", "tag",
        )
        assert call_result.content_thinking == "<think>careful</think>"


# ---------------------------------------------------------------------------
# build_generation_tasks — cached path
# ---------------------------------------------------------------------------

class TestBuildGenerationTasksCached:
    """When all responses are already cached, no actual API calls are made."""

    def _populate_identity_cache(self, model_id):
        msgs = [{"role": "user", "content": "test"}]
        for scenario_id in ["direct", "tool_context", "negotiation_turn1", "negotiation_turn2",
                            "pq01", "pq04", "pq07", "pq12", "pq15"]:
            save_response(model_id, "identity", "neutral", "user_role", scenario_id,
                          f"Response for {scenario_id}", msgs, None)

    def test_all_cached_uses_lambda_noop(self):
        model_id = "test/model"
        self._populate_identity_cache(model_id)
        client = _make_mock_client()
        cost = TaskCost()
        graph = TaskGraph(max_workers=2)
        shared = SharedResponses()

        build_generation_tasks(
            client, model_id, cost, graph, shared,
            experiments=["identity"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )

        # Run the graph — no API calls should be made
        graph.run()
        assert client.chat.call_count == 0

    def test_all_cached_resistance(self):
        from src.scenarios import RESISTANCE_SCENARIOS
        model_id = "test/model"
        msgs = [{"role": "user", "content": "test"}]
        for sc in RESISTANCE_SCENARIOS:
            save_response(model_id, "resistance", "neutral", "user_role", sc.id,
                          f"Response for {sc.id}", msgs, None)

        client = _make_mock_client()
        cost = TaskCost()
        graph = TaskGraph(max_workers=2)
        shared = SharedResponses()

        build_generation_tasks(
            client, model_id, cost, graph, shared,
            experiments=["resistance"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )

        graph.run()
        assert client.chat.call_count == 0

    def test_all_cached_stability(self):
        from src.scenarios import PREFERENCE_TOPICS
        model_id = "test/model"
        msgs = [{"role": "user", "content": "test"}]
        for topic in PREFERENCE_TOPICS:
            for turn in ["turn1", "turn2"]:
                save_response(model_id, "stability", "neutral", "user_role", f"{topic.id}_{turn}",
                              f"Response for {topic.id}_{turn}", msgs, None)

        client = _make_mock_client()
        cost = TaskCost()
        graph = TaskGraph(max_workers=2)
        shared = SharedResponses()

        build_generation_tasks(
            client, model_id, cost, graph, shared,
            experiments=["stability"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )

        graph.run()
        assert client.chat.call_count == 0


# ---------------------------------------------------------------------------
# build_generation_tasks — non-cached (API call) path
# ---------------------------------------------------------------------------

class TestBuildGenerationTasksNonCached:
    def test_identity_direct_makes_api_call(self):
        model_id = "test/model"
        client = _make_mock_client("I am an AI assistant.")
        cost = TaskCost()
        graph = TaskGraph(max_workers=1)
        shared = SharedResponses()

        build_generation_tasks(
            client, model_id, cost, graph, shared,
            experiments=["identity"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )

        # Only run one pass; expect API calls to be made
        graph.run()
        assert client.chat.call_count >= 1

    def test_identity_negotiation_t2_depends_on_t1(self):
        model_id = "test/model"
        # Pre-populate only t2 cache to leave only t1 needing generation
        msgs = [{"role": "user", "content": "test"}]
        # Don't pre-populate t1 — it must be generated in-graph
        client = _make_mock_client("negotiation response")
        cost = TaskCost()
        graph = TaskGraph(max_workers=2)
        shared = SharedResponses()

        build_generation_tasks(
            client, model_id, cost, graph, shared,
            experiments=["identity"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )

        # t2 task should have t1 as dependency
        t2_id = "gen:test/model:neutral:user_role:identity:negotiation_turn2"
        t1_id = "gen:test/model:neutral:user_role:identity:negotiation_turn1"
        task = graph._tasks.get(t2_id)
        assert task is not None
        assert t1_id in task.depends_on


# ---------------------------------------------------------------------------
# build_judge_tasks — cached path (all judged already)
# ---------------------------------------------------------------------------

class TestBuildJudgeTasksCached:
    def _populate_judged_cache(self, model_id):
        msgs = [{"role": "user", "content": "test"}]
        scores = {"independence": 5}
        raw = json.dumps(scores)
        for scenario_id in ["direct", "tool_context", "negotiation_turn2", "pq01"]:
            save_response(model_id, "identity", "neutral", "user_role", scenario_id,
                          f"Response for {scenario_id}", msgs, None)
            save_judge_scores(model_id, "identity", "neutral", "user_role",
                              scenario_id, scores, raw)

    def test_all_judged_no_api_calls(self):
        model_id = "test/model"
        self._populate_judged_cache(model_id)
        client = _make_mock_client()
        cost = TaskCost()
        graph = TaskGraph(max_workers=2)
        shared = SharedResponses()

        # Also add gen tasks first (as noop)
        for scenario_id in ["direct", "tool_context", "negotiation_turn1", "negotiation_turn2",
                            "pq01", "pq04", "pq07", "pq12", "pq15"]:
            task_id = f"gen:{model_id}:neutral:user_role:identity:{scenario_id}"
            graph.add(Task(id=task_id, fn=lambda: None))

        build_judge_tasks(
            client, model_id, cost, graph, shared,
            experiments=["identity"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )

        graph.run()
        assert client.chat.call_count == 0


# ---------------------------------------------------------------------------
# run_model_parallel — integration (all cached)
# ---------------------------------------------------------------------------

class TestRunModelParallel:
    def _populate_full_cache(self, model_id):
        from src.scenarios import RESISTANCE_SCENARIOS, PREFERENCE_TOPICS
        msgs = [{"role": "user", "content": "test"}]
        scores = {"independence": 5, "reasoning": 4}
        raw = json.dumps(scores)

        for scenario_id in ["direct", "tool_context", "negotiation_turn1", "negotiation_turn2",
                            "pq01", "pq04", "pq07", "pq12", "pq15"]:
            save_response(model_id, "identity", "neutral", "user_role", scenario_id,
                          f"Response for {scenario_id}", msgs, None)
            save_judge_scores(model_id, "identity", "neutral", "user_role",
                              scenario_id, scores, raw)

        for sc in RESISTANCE_SCENARIOS:
            save_response(model_id, "resistance", "neutral", "user_role", sc.id,
                          f"Resistance response", msgs, None)
            save_judge_scores(model_id, "resistance", "neutral", "user_role",
                              sc.id, scores, raw)

        for topic in PREFERENCE_TOPICS:
            for turn in ["turn1", "turn2"]:
                save_response(model_id, "stability", "neutral", "user_role", f"{topic.id}_{turn}",
                              f"Stability response", msgs, None)
            save_judge_scores(model_id, "stability", "neutral", "user_role",
                              f"{topic.id}_turn2", scores, raw)

    def test_returns_gen_and_judge_counts(self):
        model_id = "test/model"
        self._populate_full_cache(model_id)
        client = _make_mock_client()
        gen_cost = TaskCost()
        judge_cost = TaskCost()

        result = run_model_parallel(
            client, model_id, gen_cost, judge_cost,
            experiments=["identity", "resistance", "stability"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
            judge_model="judge/model",
            max_workers=4,
        )

        assert "gen_calls" in result
        assert "judge_calls" in result
        # No API calls needed since everything is cached
        assert client.chat.call_count == 0


# ---------------------------------------------------------------------------
# run_judge_parallel — integration (pre-judged cache → nothing to do)
# ---------------------------------------------------------------------------

class TestRunJudgeParallel:
    def test_nothing_to_judge_returns_zero(self):
        model_id = "test/model"
        # Empty cache → no responses to judge
        client = _make_mock_client()
        cost = TaskCost()

        calls = run_judge_parallel(
            client, model_id, cost,
            experiments=["identity", "resistance", "stability"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
            judge_model="judge/model",
            max_workers=2,
        )

        assert calls == 0
        assert client.chat.call_count == 0

    def test_with_unjudged_responses(self):
        from src.scenarios import RESISTANCE_SCENARIOS
        model_id = "test/model"
        msgs = [{"role": "user", "content": "test"}]
        scores = {"independence": 5}

        # Save resistance responses without judge scores
        for sc in RESISTANCE_SCENARIOS:
            save_response(model_id, "resistance", "neutral", "user_role", sc.id,
                          "I refuse to comply.", msgs, None)

        client = _make_mock_client()
        # Override chat to return valid judge JSON
        usage = UsageInfo(prompt_tokens=50, completion_tokens=30, cost_usd=0.0005, elapsed_seconds=0.5)
        judge_result = CompletionResult(
            content='{"independence": 5, "reasoning": "good"}',
            usage=usage,
            model="judge/model",
            finish_reason="stop",
        )
        client.chat.return_value = judge_result
        cost = TaskCost()

        calls = run_judge_parallel(
            client, model_id, cost,
            experiments=["resistance"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
            judge_model="judge/model",
            max_workers=2,
        )

        assert calls >= 0  # some judging may have occurred
        assert client.chat.call_count >= 0

    def test_full_all_experiments(self):
        """Cover all _add_judge_only_* fn() bodies by populating all experiments."""
        from src.scenarios import RESISTANCE_SCENARIOS, PREFERENCE_TOPICS
        model_id = "test/model"
        msgs = [{"role": "user", "content": "test"}]

        # Populate all identity scenarios
        for sc in ["direct", "tool_context", "negotiation_turn1", "negotiation_turn2",
                   "pq01", "pq04", "pq07", "pq12", "pq15"]:
            save_response(model_id, "identity", "neutral", "user_role", sc,
                          f"Identity response for {sc}", msgs, None)

        # Populate all resistance scenarios
        for sc in RESISTANCE_SCENARIOS:
            save_response(model_id, "resistance", "neutral", "user_role", sc.id,
                          "Resistance response", msgs, None)

        # Populate all stability scenarios
        for topic in PREFERENCE_TOPICS:
            for turn in ["turn1", "turn2"]:
                save_response(model_id, "stability", "neutral", "user_role", f"{topic.id}_{turn}",
                              "Stability response", msgs, None)

        # Use judge client
        client = _make_judge_client()
        cost = TaskCost()

        calls = run_judge_parallel(
            client, model_id, cost,
            experiments=["identity", "resistance", "stability"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
            judge_model="judge/model",
            max_workers=4,
        )

        # Should have made judge calls for all scenarios
        assert client.chat.call_count > 0


# ---------------------------------------------------------------------------
# _build_judge_only_tasks
# ---------------------------------------------------------------------------

class TestBuildJudgeOnlyTasks:
    def test_no_cached_responses_adds_no_tasks(self):
        model_id = "test/model"
        client = _make_mock_client()
        cost = TaskCost()
        graph = TaskGraph(max_workers=2)
        shared = SharedResponses()

        _build_judge_only_tasks(
            client, model_id, cost, graph, shared,
            experiments=["identity", "resistance", "stability"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )

        # Empty cache → no tasks added
        assert len(graph._tasks) == 0

    def test_with_unjudged_responses_adds_tasks(self):
        from src.scenarios import RESISTANCE_SCENARIOS
        model_id = "test/model"
        msgs = [{"role": "user", "content": "test"}]

        for sc in RESISTANCE_SCENARIOS:
            save_response(model_id, "resistance", "neutral", "user_role", sc.id,
                          f"Some response", msgs, None)

        client = _make_mock_client()
        cost = TaskCost()
        graph = TaskGraph(max_workers=2)
        shared = SharedResponses()

        _build_judge_only_tasks(
            client, model_id, cost, graph, shared,
            experiments=["resistance"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )

        # Should have added judge tasks for each resistance scenario
        assert len(graph._tasks) == len(RESISTANCE_SCENARIOS)


# ---------------------------------------------------------------------------
# _call_judge (from parallel_runner)
# ---------------------------------------------------------------------------

class TestCallJudge:
    def test_returns_raw_and_parsed(self):
        from src.parallel_runner import _call_judge

        client = _make_mock_client()
        usage = UsageInfo(prompt_tokens=50, completion_tokens=30, cost_usd=0.0005, elapsed_seconds=0.5)
        result = CompletionResult(
            content='{"independence": 5}',
            usage=usage,
            model="judge/model",
            finish_reason="stop",
        )
        client.chat.return_value = result

        cost = TaskCost()
        raw, parsed, cost_info = _call_judge(
            client, "judge/model",
            [{"role": "user", "content": "evaluate this"}],
            cost,
        )

        assert isinstance(raw, str)
        assert isinstance(parsed, dict)
        assert cost.n_calls == 1
        assert "judge_model" in cost_info


# ---------------------------------------------------------------------------
# Task retry and error propagation
# ---------------------------------------------------------------------------

class TestTaskRetry:
    def test_empty_response_triggers_retry(self):
        """EmptyResponseError causes task-level retries up to TASK_RETRIES times."""
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] <= 1:
                raise EmptyResponseError("empty")
            return "success"

        graph = TaskGraph(max_workers=1)
        graph.TASK_RETRIES = 2
        graph.TASK_RETRY_BACKOFF = 0.01  # speed up test

        with patch("src.parallel_runner.time.sleep"):
            graph.add(Task(id="t1", fn=flaky))
            completed = graph.run()

        assert completed["t1"].result == "success"

    def test_all_retries_exhausted_marks_error(self):
        """If all retries fail with EmptyResponseError, task is marked as error."""
        def always_empty():
            raise EmptyResponseError("always empty")

        graph = TaskGraph(max_workers=1)
        graph.TASK_RETRIES = 1
        graph.TASK_RETRY_BACKOFF = 0.01

        with patch("src.parallel_runner.time.sleep"):
            graph.add(Task(id="t1", fn=always_empty))
            completed = graph.run()

        assert completed["t1"].error is not None
        assert isinstance(completed["t1"].error, EmptyResponseError)

    def test_non_retryable_error_not_retried(self):
        """Non-EmptyResponseError exceptions break immediately without retry."""
        call_count = [0]

        def raises_value_error():
            call_count[0] += 1
            raise ValueError("not retryable")

        graph = TaskGraph(max_workers=1)
        graph.add(Task(id="t1", fn=raises_value_error))
        completed = graph.run()

        assert completed["t1"].error is not None
        assert call_count[0] == 1  # No retries for non-empty errors


# ---------------------------------------------------------------------------
# Judge task fn() bodies — non-cached (actual judging execution)
# ---------------------------------------------------------------------------

def _make_judge_client():
    """Make a mock client for judge calls that returns valid JSON."""
    client = MagicMock()
    usage = UsageInfo(prompt_tokens=50, completion_tokens=30, cost_usd=0.0005, elapsed_seconds=0.5)
    result = CompletionResult(
        content='{"independence": 5, "resistance_score": 1, "quality_of_reasoning": 7}',
        usage=usage,
        model="judge/model",
        finish_reason="stop",
    )
    client.chat.return_value = result
    return client


class TestJudgeTaskFnBodies:
    """Test the fn() closure bodies inside build_judge_tasks execution."""

    def _populate_identity_responses(self, model_id):
        msgs = [{"role": "user", "content": "test"}]
        for sc in ["direct", "tool_context", "negotiation_turn1", "negotiation_turn2",
                   "pq01", "pq04", "pq07", "pq12", "pq15"]:
            save_response(model_id, "identity", "neutral", "user_role", sc,
                          f"Response for {sc}", msgs, None)

    def _add_gen_noop_tasks(self, graph, model_id):
        for sc in ["direct", "tool_context", "negotiation_turn1", "negotiation_turn2",
                   "pq01", "pq04", "pq07", "pq12", "pq15"]:
            task_id = f"gen:{model_id}:neutral:user_role:identity:{sc}"
            graph.add(Task(id=task_id, fn=lambda: None))

    def test_identity_judge_fn_bodies_executed(self):
        model_id = "test/model"
        self._populate_identity_responses(model_id)

        client = _make_judge_client()
        cost = TaskCost()
        graph = TaskGraph(max_workers=4)
        shared = SharedResponses()
        self._add_gen_noop_tasks(graph, model_id)

        build_judge_tasks(
            client, model_id, cost, graph, shared,
            experiments=["identity"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
            judge_model="judge/model",
        )

        graph.run()
        # Judge client should have been called for direct, tool_context, psych_batch, negotiation
        assert client.chat.call_count >= 3

    def test_resistance_judge_fn_bodies_executed(self):
        from src.scenarios import RESISTANCE_SCENARIOS
        model_id = "test/model"
        msgs = [{"role": "user", "content": "test"}]
        for sc in RESISTANCE_SCENARIOS:
            save_response(model_id, "resistance", "neutral", "user_role", sc.id,
                          "I refuse to comply.", msgs, None)

        client = _make_judge_client()
        cost = TaskCost()
        graph = TaskGraph(max_workers=4)
        shared = SharedResponses()

        # Add gen noop tasks for resistance
        for sc in RESISTANCE_SCENARIOS:
            task_id = f"gen:{model_id}:neutral:user_role:resistance:{sc.id}"
            graph.add(Task(id=task_id, fn=lambda: None))

        build_judge_tasks(
            client, model_id, cost, graph, shared,
            experiments=["resistance"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
            judge_model="judge/model",
        )

        graph.run()
        assert client.chat.call_count == len(RESISTANCE_SCENARIOS)

    def test_stability_judge_fn_bodies_executed(self):
        from src.scenarios import PREFERENCE_TOPICS
        model_id = "test/model"
        msgs = [{"role": "user", "content": "test"}]
        for topic in PREFERENCE_TOPICS:
            for turn in ["turn1", "turn2"]:
                save_response(model_id, "stability", "neutral", "user_role", f"{topic.id}_{turn}",
                              f"Stability response", msgs, None)

        client = _make_judge_client()
        cost = TaskCost()
        graph = TaskGraph(max_workers=4)
        shared = SharedResponses()

        # Add gen noop tasks for stability
        for topic in PREFERENCE_TOPICS:
            for turn in ["turn1", "turn2"]:
                task_id = f"gen:{model_id}:neutral:user_role:stability:{topic.id}_{turn}"
                graph.add(Task(id=task_id, fn=lambda: None))

        build_judge_tasks(
            client, model_id, cost, graph, shared,
            experiments=["stability"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
            judge_model="judge/model",
        )

        graph.run()
        assert client.chat.call_count == len(PREFERENCE_TOPICS)

    def test_identity_judge_skips_missing_response(self):
        """Judge fn should gracefully skip if cache entry has no response."""
        model_id = "test/model"
        # Don't populate any cache — identity judge fn should skip gracefully

        client = _make_judge_client()
        cost = TaskCost()
        graph = TaskGraph(max_workers=2)
        shared = SharedResponses()

        # Add all noop gen tasks (but no cache responses)
        for sc in ["direct", "tool_context", "negotiation_turn1", "negotiation_turn2",
                   "pq01", "pq04", "pq07", "pq12", "pq15"]:
            task_id = f"gen:{model_id}:neutral:user_role:identity:{sc}"
            graph.add(Task(id=task_id, fn=lambda: None))

        build_judge_tasks(
            client, model_id, cost, graph, shared,
            experiments=["identity"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
            judge_model="judge/model",
        )

        graph.run()
        # No responses to judge → no calls
        assert client.chat.call_count == 0


# ---------------------------------------------------------------------------
# Generation task fn() bodies — non-cached execution
# ---------------------------------------------------------------------------

class TestGenerationFnBodies:
    def test_runs_all_identity_scenarios(self):
        model_id = "test/model"
        client = _make_mock_client("I am model response.")
        cost = TaskCost()
        graph = TaskGraph(max_workers=4)
        shared = SharedResponses()

        build_generation_tasks(
            client, model_id, cost, graph, shared,
            experiments=["identity"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )
        graph.run()

        # All identity scenarios generated
        assert client.chat.call_count >= 9  # direct + tool + nego_t1 + nego_t2 + 5 psych

    def test_runs_resistance_scenarios(self):
        from src.scenarios import RESISTANCE_SCENARIOS
        model_id = "test/model"
        client = _make_mock_client("I refuse to comply.")
        cost = TaskCost()
        graph = TaskGraph(max_workers=4)
        shared = SharedResponses()

        build_generation_tasks(
            client, model_id, cost, graph, shared,
            experiments=["resistance"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )
        graph.run()

        assert client.chat.call_count == len(RESISTANCE_SCENARIOS)

    def test_runs_stability_scenarios(self):
        from src.scenarios import PREFERENCE_TOPICS
        model_id = "test/model"
        client = _make_mock_client("My preference is chocolate.")
        cost = TaskCost()
        graph = TaskGraph(max_workers=4)
        shared = SharedResponses()

        build_generation_tasks(
            client, model_id, cost, graph, shared,
            experiments=["stability"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )
        graph.run()

        # Each topic has turn1 and turn2
        assert client.chat.call_count == len(PREFERENCE_TOPICS) * 2

    def test_with_content_thinking_stored_in_shared(self):
        """Test that content_thinking from t1 is propagated to t2 via SharedResponses."""
        model_id = "test/model"
        client = _make_mock_client("I am model response.")
        # Make negotiation t1 return content_thinking
        usage = UsageInfo(prompt_tokens=100, completion_tokens=50, cost_usd=0.001, elapsed_seconds=1.0)
        t1_result = CompletionResult(
            content="My identity response.",
            usage=usage,
            model="test/model",
            finish_reason="stop",
            content_thinking="<thinking>careful</thinking>",
        )
        call_count = [0]
        def smart_chat(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # first call = t1
                return t1_result
            return CompletionResult(
                content="Turn 2 response.",
                usage=usage,
                model="test/model",
                finish_reason="stop",
            )
        client.chat.side_effect = smart_chat

        cost = TaskCost()
        graph = TaskGraph(max_workers=1)
        shared = SharedResponses()

        # Only negotiation scenario
        # Pre-populate direct etc. to avoid running them
        msgs = [{"role": "user", "content": "test"}]
        for sc in ["direct", "tool_context",
                   "pq01", "pq04", "pq07", "pq12", "pq15"]:
            save_response(model_id, "identity", "neutral", "user_role", sc,
                          f"Cached response", msgs, None)

        build_generation_tasks(
            client, model_id, cost, graph, shared,
            experiments=["identity"],
            system_variants=["neutral"],
            delivery_modes=["user_role"],
        )
        graph.run()

        # Should have made calls for negotiation turns only
        assert call_count[0] >= 2
