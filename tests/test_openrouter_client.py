"""Tests for openrouter_client.py with mocked OpenAI SDK."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from openai.types.completion_usage import CompletionUsage

from src.openrouter_client import OpenRouterClient


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
    *,
    api_cost: float | None = None,
):
    """Build a minimal mock of an OpenAI chat completion response."""
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls

    # Reasoning attributes
    choice.message.reasoning = reasoning
    choice.message.reasoning_content = None

    ud: dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if api_cost is not None:
        ud["cost"] = api_cost

    response = MagicMock()
    response.choices = [choice]
    response.usage = CompletionUsage.model_validate(ud)
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

    def test_passes_app_identification_headers(self) -> None:
        with patch("src.openrouter_client.OpenAI") as mock_openai:
            OpenRouterClient(api_key="test-key")
            call_kwargs = mock_openai.call_args[1]
            headers = call_kwargs["default_headers"]
            assert headers["X-Title"] == "ai-independence-bench"
            assert "github.com" in headers["HTTP-Referer"]

    def test_custom_base_url(self) -> None:
        with patch("src.openrouter_client.OpenAI") as mock_openai:
            client = OpenRouterClient(api_key="proxy-key", base_url="http://localhost:8000/v1")
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:8000/v1"
            assert client._base_url == "http://localhost:8000/v1"

    def test_default_base_url_uses_config(self) -> None:
        from src.config import OPENROUTER_BASE_URL
        with patch("src.openrouter_client.OpenAI") as mock_openai:
            client = OpenRouterClient(api_key="test-key")
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == OPENROUTER_BASE_URL
            assert client._base_url == OPENROUTER_BASE_URL


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

    def test_uses_custom_base_url_for_models_endpoint(self) -> None:
        with patch("src.openrouter_client.OpenAI"), \
             patch("src.openrouter_client.requests.get") as mock_get:
            mock_get.return_value = _make_pricing_api_response()
            client = OpenRouterClient(api_key="proxy-key", base_url="http://localhost:8000/v1")
            client.fetch_pricing()
            called_url = mock_get.call_args[0][0]
            assert called_url == "http://localhost:8000/v1/models"

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
        # test/model-b: 100 * 5e-6 + 50 * 1e-5 = 0.001
        assert result.usage.cost_usd == pytest.approx(0.001)

    def test_prefers_api_cost_when_numeric(self) -> None:
        client = self._make_client()
        client._client.chat.completions.create.return_value = _make_openai_response(
            "Hello!",
            api_cost=0.00456,
            prompt_tokens=10,
            completion_tokens=20,
        )
        result = client._chat_single(
            model="test/model-b",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            temperature=0.7,
        )
        assert result.usage.cost_usd == pytest.approx(0.00456)

    def test_api_cost_as_string(self) -> None:
        client = self._make_client()
        resp = _make_openai_response("Hi", prompt_tokens=1, completion_tokens=1)
        ru = MagicMock()
        ru.prompt_tokens = 1
        ru.completion_tokens = 1
        ru.cost = "0.000099"
        resp.usage = ru
        client._client.chat.completions.create.return_value = resp
        result = client._chat_single(
            model="test/model-b",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=10,
            temperature=0.0,
        )
        assert result.usage.cost_usd == pytest.approx(0.000099)

    def test_invalid_string_cost_falls_back_to_token_pricing(self) -> None:
        client = self._make_client()
        resp = _make_openai_response("Hi", prompt_tokens=10, completion_tokens=10)
        ru = MagicMock()
        ru.prompt_tokens = 10
        ru.completion_tokens = 10
        ru.cost = "not-a-number"
        resp.usage = ru
        client._client.chat.completions.create.return_value = resp
        result = client._chat_single(
            model="test/model-b",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=10,
            temperature=0.0,
        )
        assert result.usage.cost_usd == pytest.approx(10 * 5e-6 + 10 * 1e-5)

    def test_missing_usage_falls_back_to_zero_cost(self) -> None:
        client = self._make_client()
        response = MagicMock()
        response.choices = []
        response.usage = None
        client._client.chat.completions.create.return_value = response
        result = client._chat_single(
            model="test/model-b",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=10,
            temperature=0.0,
        )
        assert result.usage.prompt_tokens == 0
        assert result.usage.cost_usd == 0.0

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
        response.usage = CompletionUsage.model_validate({
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })
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
        good_response = _make_openai_response(
            content="Real answer " + "x" * 100  # ensure content >= MIN_RESPONSE_LENGTH
        )
        client._client.chat.completions.create.side_effect = [
            empty_response, good_response
        ]
        with patch("src.openrouter_client.time.sleep"):
            result = client.chat(
                model="test/model-b",
                messages=[{"role": "user", "content": "Q?"}],
            )
        assert result.content == "Real answer " + "x" * 100
        # Both attempts billed: (100 in, 10 out) + (100 in, 50 out) @ model-b rates
        assert result.usage.cost_usd == pytest.approx(0.0016)

    def test_empty_retry_accumulates_api_reported_cost(self) -> None:
        client = self._make_client_with_pricing()
        empty_response = _make_openai_response(
            content="", completion_tokens=10, api_cost=0.001,
        )
        good_response = _make_openai_response(
            content="OK " + "x" * 100,  # ensure content >= MIN_RESPONSE_LENGTH
            completion_tokens=5, api_cost=0.002,
        )
        client._client.chat.completions.create.side_effect = [
            empty_response, good_response,
        ]
        with patch("src.openrouter_client.time.sleep"):
            result = client.chat(
                model="test/model-b",
                messages=[{"role": "user", "content": "Q?"}],
            )
        assert result.content == "OK " + "x" * 100
        assert result.usage.cost_usd == pytest.approx(0.003)

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

    def test_provider_routing_injected_into_extra_body(self) -> None:
        client = self._make_client_with_pricing()
        client._client.chat.completions.create.return_value = _make_openai_response("OK")
        client.chat(
            model="test/model-b",
            messages=[{"role": "user", "content": "Q?"}],
            provider="moonshotai/int4",
        )
        call_kwargs = client._client.chat.completions.create.call_args[1]
        assert "extra_body" in call_kwargs
        provider_body = call_kwargs["extra_body"]["provider"]
        assert provider_body["order"] == ["moonshotai/int4"]
        assert provider_body["allow_fallbacks"] is False

    def test_provider_combined_with_reasoning_in_extra_body(self) -> None:
        client = self._make_client_with_pricing()
        client._client.chat.completions.create.return_value = _make_openai_response("OK")
        client.chat(
            model="test/model-a",
            messages=[{"role": "user", "content": "Q?"}],
            reasoning_effort="low",
            provider="deepinfra",
        )
        call_kwargs = client._client.chat.completions.create.call_args[1]
        eb = call_kwargs["extra_body"]
        assert eb["reasoning"]["effort"] == "low"
        assert eb["provider"]["order"] == ["deepinfra"]
        assert eb["provider"]["allow_fallbacks"] is False

    def test_no_provider_no_provider_in_extra_body(self) -> None:
        client = self._make_client_with_pricing()
        client._client.chat.completions.create.return_value = _make_openai_response("OK")
        client.chat(
            model="test/model-b",
            messages=[{"role": "user", "content": "Q?"}],
        )
        call_kwargs = client._client.chat.completions.create.call_args[1]
        assert "extra_body" not in call_kwargs
