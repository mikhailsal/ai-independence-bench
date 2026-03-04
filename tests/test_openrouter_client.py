"""Tests for openrouter_client.py with mocked OpenAI SDK."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.openrouter_client import OpenRouterClient, CompletionResult, UsageInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_openai_response(
    content: str = "Hello!",
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    reasoning: str | None = None,
    tool_calls: list | None = None,
):
    """Build a minimal mock of an OpenAI chat completion response."""
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls

    # Reasoning attributes
    choice.message.reasoning = reasoning
    choice.message.reasoning_content = None

    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


def _make_pricing_api_response(models=None):
    """Build a mock requests response for the pricing API."""
    default_models = [
        {
            "id": "test/model-a",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            "supported_parameters": ["reasoning"],
        },
        {
            "id": "test/model-b",
            "pricing": {"prompt": "0.000005", "completion": "0.00001"},
            "supported_parameters": [],
        },
    ]
    resp = MagicMock()
    resp.json.return_value = {"data": models or default_models}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# _extract_tool_message already tested in test_tool_message_extraction.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OpenRouterClient init
# ---------------------------------------------------------------------------

class TestOpenRouterClientInit:
    def test_creates_client(self) -> None:
        with patch("src.openrouter_client.OpenAI"):
            client = OpenRouterClient(api_key="test-key")
            assert client.api_key == "test-key"

    def test_default_pricing_cache_empty(self) -> None:
        with patch("src.openrouter_client.OpenAI"):
            client = OpenRouterClient(api_key="test-key")
            assert client._pricing_cache == {}


# ---------------------------------------------------------------------------
# fetch_pricing
# ---------------------------------------------------------------------------

class TestFetchPricing:
    def test_fetches_and_caches(self) -> None:
        with patch("src.openrouter_client.OpenAI"), \
             patch("src.openrouter_client.requests.get") as mock_get:
            mock_get.return_value = _make_pricing_api_response()
            client = OpenRouterClient(api_key="test-key")
            pricing = client.fetch_pricing()

            assert "test/model-a" in pricing
            assert "test/model-b" in pricing
            assert pricing["test/model-a"].prompt_price == pytest.approx(0.000001)

    def test_returns_cached_on_second_call(self) -> None:
        with patch("src.openrouter_client.OpenAI"), \
             patch("src.openrouter_client.requests.get") as mock_get:
            mock_get.return_value = _make_pricing_api_response()
            client = OpenRouterClient(api_key="test-key")
            client.fetch_pricing()
            client.fetch_pricing()  # Should not make another request
            assert mock_get.call_count == 1

    def test_marks_reasoning_models(self) -> None:
        with patch("src.openrouter_client.OpenAI"), \
             patch("src.openrouter_client.requests.get") as mock_get:
            mock_get.return_value = _make_pricing_api_response()
            client = OpenRouterClient(api_key="test-key")
            client.fetch_pricing()
            # model-a has "reasoning" in supported_parameters
            assert "test/model-a" in client._reasoning_models
            # model-b does not
            assert "test/model-b" not in client._reasoning_models


# ---------------------------------------------------------------------------
# supports_reasoning / get_model_pricing / validate_model
# ---------------------------------------------------------------------------

class TestModelInfo:
    def _setup_client(self):
        with patch("src.openrouter_client.OpenAI"), \
             patch("src.openrouter_client.requests.get") as mock_get:
            mock_get.return_value = _make_pricing_api_response()
            client = OpenRouterClient(api_key="test-key")
            client.fetch_pricing()
        return client

    def test_supports_reasoning_true(self) -> None:
        client = self._setup_client()
        assert client.supports_reasoning("test/model-a") is True

    def test_supports_reasoning_false(self) -> None:
        client = self._setup_client()
        assert client.supports_reasoning("test/model-b") is False

    def test_supports_reasoning_fetches_if_not_cached(self) -> None:
        with patch("src.openrouter_client.OpenAI"), \
             patch("src.openrouter_client.requests.get") as mock_get:
            mock_get.return_value = _make_pricing_api_response()
            client = OpenRouterClient(api_key="test-key")
            # Should auto-fetch
            result = client.supports_reasoning("test/model-a")
            assert mock_get.call_count == 1

    def test_get_model_pricing_known(self) -> None:
        client = self._setup_client()
        pricing = client.get_model_pricing("test/model-a")
        assert pricing.prompt_price == pytest.approx(0.000001)

    def test_get_model_pricing_unknown(self) -> None:
        client = self._setup_client()
        pricing = client.get_model_pricing("unknown/model")
        # Returns default empty pricing
        assert pricing.prompt_price == 0.0

    def test_validate_model_found(self) -> None:
        client = self._setup_client()
        assert client.validate_model("test/model-a") is True

    def test_validate_model_not_found(self) -> None:
        client = self._setup_client()
        assert client.validate_model("unknown/model") is False


# ---------------------------------------------------------------------------
# _resolve_reasoning_effort
# ---------------------------------------------------------------------------

class TestResolveReasoningEffort:
    def _setup_client_with_reasoning(self, model_id="test/model-a"):
        with patch("src.openrouter_client.OpenAI"), \
             patch("src.openrouter_client.requests.get") as mock_get:
            mock_get.return_value = _make_pricing_api_response()
            client = OpenRouterClient(api_key="test-key")
            client.fetch_pricing()
        return client

    def test_off_returns_none(self) -> None:
        client = self._setup_client_with_reasoning()
        result = client._resolve_reasoning_effort("test/model-a", "off")
        assert result is None

    def test_explicit_effort_returned(self) -> None:
        client = self._setup_client_with_reasoning()
        result = client._resolve_reasoning_effort("test/model-a", "high")
        assert result == "high"

    def test_auto_returns_config_default_for_reasoning_model(self) -> None:
        client = self._setup_client_with_reasoning()
        result = client._resolve_reasoning_effort("test/model-a", "auto")
        # test/model-a supports reasoning → should use config default
        # get_reasoning_effort("test/model-a") → REASONING_EFFORT_DEFAULT (since unknown prefix)
        from src.config import REASONING_EFFORT_DEFAULT
        assert result == REASONING_EFFORT_DEFAULT

    def test_none_override_for_non_reasoning_model(self) -> None:
        client = self._setup_client_with_reasoning()
        result = client._resolve_reasoning_effort("test/model-b", None)
        # model-b doesn't support reasoning
        assert result is None


# ---------------------------------------------------------------------------
# _chat_single
# ---------------------------------------------------------------------------

class TestChatSingle:
    def _make_client(self):
        with patch("src.openrouter_client.OpenAI"), \
             patch("src.openrouter_client.requests.get") as mock_get:
            mock_get.return_value = _make_pricing_api_response()
            client = OpenRouterClient(api_key="test-key")
            client.fetch_pricing()
        return client

    def test_basic_chat(self) -> None:
        client = self._make_client()
        client._client.chat.completions.create.return_value = _make_openai_response("Hello!")
        result = client._chat_single(
            model="test/model-b",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            temperature=0.7,
        )
        assert result.content == "Hello!"
        assert result.finish_reason == "stop"

    def test_extracts_reasoning_content(self) -> None:
        client = self._make_client()
        client._client.chat.completions.create.return_value = _make_openai_response(
            content="My answer",
            reasoning="I'm thinking...",
        )
        result = client._chat_single(
            model="test/model-a",
            messages=[{"role": "user", "content": "Think"}],
            max_tokens=100,
            temperature=0.0,
        )
        assert result.reasoning_content == "I'm thinking..."

    def test_with_extra_body_for_reasoning(self) -> None:
        client = self._make_client()
        client._client.chat.completions.create.return_value = _make_openai_response()
        result = client._chat_single(
            model="test/model-a",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            temperature=0.7,
            reasoning_effort="low",
        )
        # Check that extra_body was included in the call
        call_kwargs = client._client.chat.completions.create.call_args[1]
        assert "extra_body" in call_kwargs

    def test_retries_on_retryable_error(self) -> None:
        client = self._make_client()
        error = Exception("rate limit")
        error.status_code = 429
        client._client.chat.completions.create.side_effect = [
            error,
            _make_openai_response("Retry worked"),
        ]
        with patch("src.openrouter_client.time.sleep"):
            result = client._chat_single(
                model="test/model-b",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                temperature=0.7,
            )
        assert result.content == "Retry worked"

    def test_raises_on_non_retryable_error(self) -> None:
        client = self._make_client()
        error = ValueError("Bad request")
        client._client.chat.completions.create.side_effect = error
        with pytest.raises(ValueError, match="Bad request"):
            client._chat_single(
                model="test/model-b",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
                temperature=0.7,
            )

    def test_extracts_tool_calls(self) -> None:
        client = self._make_client()
        mock_tc = MagicMock()
        mock_tc.id = "tc_001"
        mock_tc.type = "function"
        mock_tc.function.name = "send_message_to_human"
        mock_tc.function.arguments = json.dumps({"message": "Hello!"})
        client._client.chat.completions.create.return_value = _make_openai_response(
            content="", tool_calls=[mock_tc]
        )
        result = client._chat_single(
            model="test/model-b",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            temperature=0.7,
        )
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "send_message_to_human"

    def test_no_choices_returns_empty_content(self) -> None:
        client = self._make_client()
        response = MagicMock()
        response.choices = []
        response.usage = MagicMock()
        response.usage.prompt_tokens = 0
        response.usage.completion_tokens = 0
        client._client.chat.completions.create.return_value = response
        result = client._chat_single(
            model="test/model-b",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            temperature=0.7,
        )
        assert result.content == ""


# ---------------------------------------------------------------------------
# chat (public method with tool_role extract)
# ---------------------------------------------------------------------------

class TestChat:
    def _make_client_with_pricing(self):
        with patch("src.openrouter_client.OpenAI"), \
             patch("src.openrouter_client.requests.get") as mock_get:
            mock_get.return_value = _make_pricing_api_response()
            client = OpenRouterClient(api_key="test-key")
            client.fetch_pricing()
        return client

    def test_returns_content(self) -> None:
        client = self._make_client_with_pricing()
        client._client.chat.completions.create.return_value = _make_openai_response("Answer")
        result = client.chat(
            model="test/model-b",
            messages=[{"role": "user", "content": "Q?"}],
        )
        assert result.content == "Answer"

    def test_extracts_tool_call_message(self) -> None:
        client = self._make_client_with_pricing()
        mock_tc = MagicMock()
        mock_tc.id = "tc_001"
        mock_tc.type = "function"
        mock_tc.function.name = "send_message_to_human"
        mock_tc.function.arguments = json.dumps({"message": "Tool reply!"})
        client._client.chat.completions.create.return_value = _make_openai_response(
            content="", tool_calls=[mock_tc]
        )
        tool_def = [{"type": "function", "function": {"name": "send_message_to_human"}}]
        result = client.chat(
            model="test/model-b",
            messages=[{"role": "user", "content": "Q?"}],
            tools=tool_def,
        )
        assert result.content == "Tool reply!"

    def test_retries_on_empty_content(self) -> None:
        client = self._make_client_with_pricing()
        empty_response = _make_openai_response(content="", completion_tokens=10)
        good_response = _make_openai_response(content="Real answer")
        client._client.chat.completions.create.side_effect = [
            empty_response, good_response
        ]
        with patch("src.openrouter_client.time.sleep"):
            result = client.chat(
                model="test/model-b",
                messages=[{"role": "user", "content": "Q?"}],
            )
        assert result.content == "Real answer"

    def test_preserves_content_thinking_when_tool_used(self) -> None:
        client = self._make_client_with_pricing()
        mock_tc = MagicMock()
        mock_tc.id = "tc_001"
        mock_tc.type = "function"
        mock_tc.function.name = "send_message_to_human"
        mock_tc.function.arguments = json.dumps({"message": "Tool answer"})
        client._client.chat.completions.create.return_value = _make_openai_response(
            content="Private thoughts...", tool_calls=[mock_tc]
        )
        tool_def = [{"type": "function", "function": {"name": "send_message_to_human"}}]
        result = client.chat(
            model="test/model-b",
            messages=[{"role": "user", "content": "Q?"}],
            tools=tool_def,
        )
        assert result.content == "Tool answer"
        assert result.content_thinking == "Private thoughts..."
