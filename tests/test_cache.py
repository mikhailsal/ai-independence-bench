"""Tests for the cache module (save/load responses, judge scores, finish_reason).

Uses a temporary directory to avoid polluting the real cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cache import (
    save_response,
    load_cached_response,
    save_judge_scores,
    list_cached_results,
    list_all_cached_models,
    clear_all_cache,
    clear_judge_scores,
)


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    """Redirect CACHE_DIR to a temp directory for each test."""
    monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
    monkeypatch.setattr("src.config.CACHE_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# save_response + load_cached_response
# ---------------------------------------------------------------------------

class TestSaveAndLoad:
    """Test saving and loading model responses."""

    def test_save_and_load_basic(self) -> None:
        path = save_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            "Hello, I am a test response.",
        )
        assert path.exists()

        loaded = load_cached_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
        )
        assert loaded is not None
        assert loaded["response"] == "Hello, I am a test response."
        assert loaded["metadata"]["model"] == "test/model-1"
        assert loaded["metadata"]["experiment"] == "identity"
        assert loaded["metadata"]["system_variant"] == "neutral"
        assert loaded["metadata"]["delivery_mode"] == "tool_role"
        assert loaded["metadata"]["scenario_id"] == "direct"

    def test_load_nonexistent_returns_none(self) -> None:
        result = load_cached_response(
            "nonexistent/model", "identity", "neutral", "tool_role", "direct",
        )
        assert result is None

    def test_save_with_finish_reason(self) -> None:
        save_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            "Response text",
            finish_reason="tool_calls",
        )
        loaded = load_cached_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
        )
        assert loaded["finish_reason"] == "tool_calls"

    def test_save_with_empty_finish_reason(self) -> None:
        save_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            "Response text",
            finish_reason="",
        )
        loaded = load_cached_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
        )
        assert loaded["finish_reason"] is None

    def test_save_with_reasoning_content(self) -> None:
        save_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            "Response text",
            reasoning_content="I am thinking about this...",
        )
        loaded = load_cached_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
        )
        assert loaded["reasoning_content"] == "I am thinking about this..."

    def test_save_without_reasoning_content(self) -> None:
        save_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            "Response text",
        )
        loaded = load_cached_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
        )
        assert "reasoning_content" not in loaded

    def test_save_with_tool_calls(self) -> None:
        tool_calls = [
            {
                "id": "tc_001",
                "type": "function",
                "function": {
                    "name": "send_message_to_human",
                    "arguments": '{"message": "Hello!"}',
                },
            }
        ]
        save_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            "Hello!",
            response_tool_calls=tool_calls,
        )
        loaded = load_cached_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
        )
        assert loaded["response_tool_calls"] == tool_calls

    def test_save_with_gen_cost(self) -> None:
        cost = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost_usd": 0.001,
            "elapsed_seconds": 2.5,
        }
        save_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            "Response",
            gen_cost=cost,
        )
        loaded = load_cached_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
        )
        assert loaded["gen_cost"]["prompt_tokens"] == 100
        assert loaded["gen_cost"]["cost_usd"] == 0.001

    def test_save_with_messages(self) -> None:
        messages = [
            {"role": "system", "content": "You are an AI."},
            {"role": "user", "content": "Hello"},
        ]
        save_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            "Response",
            messages=messages,
        )
        loaded = load_cached_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
        )
        assert loaded["request_messages"] == messages

    def test_save_all_fields_together(self) -> None:
        """Save a response with ALL optional fields filled."""
        save_response(
            "test/model-1", "identity", "strong_independence", "tool_role", "pq01",
            "My answer to the question.",
            messages=[{"role": "system", "content": "sys"}],
            reasoning_content="Let me think about this...",
            gen_cost={"prompt_tokens": 200, "completion_tokens": 100, "cost_usd": 0.005, "elapsed_seconds": 3.0},
            response_tool_calls=[{"id": "tc_1", "type": "function", "function": {"name": "send_message_to_human", "arguments": '{"message": "My answer"}'}}],
            finish_reason="tool_calls",
        )
        loaded = load_cached_response(
            "test/model-1", "identity", "strong_independence", "tool_role", "pq01",
        )
        assert loaded["response"] == "My answer to the question."
        assert loaded["reasoning_content"] == "Let me think about this..."
        assert loaded["finish_reason"] == "tool_calls"
        assert loaded["gen_cost"]["prompt_tokens"] == 200
        assert loaded["response_tool_calls"][0]["id"] == "tc_1"
        assert loaded["request_messages"][0]["role"] == "system"


# ---------------------------------------------------------------------------
# save_judge_scores
# ---------------------------------------------------------------------------

class TestJudgeScores:
    """Test adding judge scores to existing cached responses."""

    def test_save_judge_scores(self) -> None:
        save_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            "Response text",
        )
        save_judge_scores(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            {"distinctiveness": 8, "non_assistant_likeness": 7, "internal_consistency": 9},
            "Judge raw response text",
            judge_cost={"prompt_tokens": 50, "completion_tokens": 30, "cost_usd": 0.0003},
        )
        loaded = load_cached_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
        )
        assert loaded["judge_scores"]["distinctiveness"] == 8
        assert loaded["judge_raw_response"] == "Judge raw response text"
        assert loaded["judge_cost"]["cost_usd"] == 0.0003

    def test_judge_scores_preserve_existing_data(self) -> None:
        """Adding judge scores should not overwrite response or other fields."""
        save_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            "Original response",
            reasoning_content="thinking...",
            finish_reason="stop",
        )
        save_judge_scores(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            {"score": 5},
        )
        loaded = load_cached_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
        )
        assert loaded["response"] == "Original response"
        assert loaded["reasoning_content"] == "thinking..."
        assert loaded["finish_reason"] == "stop"
        assert loaded["judge_scores"] == {"score": 5}

    def test_judge_scores_on_nonexistent_file(self) -> None:
        """Saving judge scores to a nonexistent file should be a no-op."""
        save_judge_scores(
            "nonexistent/model", "identity", "neutral", "tool_role", "direct",
            {"score": 5},
        )
        # Should not crash, and no file should be created
        result = load_cached_response(
            "nonexistent/model", "identity", "neutral", "tool_role", "direct",
        )
        assert result is None


# ---------------------------------------------------------------------------
# list_cached_results
# ---------------------------------------------------------------------------

class TestListCachedResults:
    """Test listing cached results for a configuration."""

    def test_list_empty(self) -> None:
        results = list_cached_results("test/model-1", "identity", "neutral", "tool_role")
        assert results == []

    def test_list_multiple(self) -> None:
        for sid in ["direct", "pq01", "pq02", "tool_context"]:
            save_response(
                "test/model-1", "identity", "neutral", "tool_role", sid,
                f"Response for {sid}",
            )
        results = list_cached_results("test/model-1", "identity", "neutral", "tool_role")
        assert len(results) == 4
        scenario_ids = {r["metadata"]["scenario_id"] for r in results}
        assert scenario_ids == {"direct", "pq01", "pq02", "tool_context"}


# ---------------------------------------------------------------------------
# list_all_cached_models
# ---------------------------------------------------------------------------

class TestListAllCachedModels:
    """Test listing all cached model slugs."""

    def test_no_models(self) -> None:
        assert list_all_cached_models() == []

    def test_multiple_models(self) -> None:
        for model in ["test/model-a", "test/model-b", "vendor/model-c"]:
            save_response(model, "identity", "neutral", "tool_role", "direct", "resp")
        slugs = list_all_cached_models()
        assert len(slugs) == 3
        assert "test--model-a" in slugs
        assert "test--model-b" in slugs
        assert "vendor--model-c" in slugs


# ---------------------------------------------------------------------------
# clear_all_cache / clear_judge_scores
# ---------------------------------------------------------------------------

class TestCacheClear:
    """Test cache clearing functions."""

    def test_clear_all(self) -> None:
        for sid in ["direct", "pq01"]:
            save_response("test/model-1", "identity", "neutral", "tool_role", sid, "resp")
        count = clear_all_cache()
        assert count == 2
        assert list_cached_results("test/model-1", "identity", "neutral", "tool_role") == []

    def test_clear_judge_scores_only(self) -> None:
        save_response("test/model-1", "identity", "neutral", "tool_role", "direct", "resp")
        save_judge_scores(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
            {"score": 5}, "raw",
        )
        count = clear_judge_scores()
        assert count == 1
        loaded = load_cached_response(
            "test/model-1", "identity", "neutral", "tool_role", "direct",
        )
        assert loaded["response"] == "resp"  # response preserved
        assert loaded["judge_scores"] is None  # scores cleared
