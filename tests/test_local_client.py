"""Tests for local_client.py — LocalModelClient for LM Studio / Ollama / etc."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.config import ModelPricing
from src.local_client import LocalModelClient, _strip_local_prefix


# ---------------------------------------------------------------------------
# _strip_local_prefix
# ---------------------------------------------------------------------------

class TestStripLocalPrefix:
    def test_strips_prefix(self) -> None:
        assert _strip_local_prefix("local/my-model") == "my-model"

    def test_no_prefix_unchanged(self) -> None:
        assert _strip_local_prefix("my-model") == "my-model"

    def test_empty_string(self) -> None:
        assert _strip_local_prefix("") == ""

    def test_double_prefix(self) -> None:
        assert _strip_local_prefix("local/local/nested") == "local/nested"


# ---------------------------------------------------------------------------
# LocalModelClient init
# ---------------------------------------------------------------------------

class TestLocalModelClientInit:
    @patch("src.local_client.OpenAI")
    def test_creates_client_with_base_url(self, mock_openai: MagicMock) -> None:
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:1234/v1"
        assert call_kwargs["api_key"] == "lm-studio"

    @patch("src.local_client.OpenAI")
    def test_sets_dummy_api_key(self, mock_openai: MagicMock) -> None:
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        assert client.api_key == "local-no-key"

    @patch("src.local_client.OpenAI")
    def test_stores_base_url(self, mock_openai: MagicMock) -> None:
        client = LocalModelClient(base_url="http://192.168.1.101:1234/v1")
        assert client._base_url == "http://192.168.1.101:1234/v1"

    @patch("src.local_client.OpenAI")
    def test_empty_pricing_and_reasoning(self, mock_openai: MagicMock) -> None:
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        assert client._pricing_cache == {}
        assert client._reasoning_models == set()


# ---------------------------------------------------------------------------
# Overridden methods
# ---------------------------------------------------------------------------

class TestLocalOverrides:
    @patch("src.local_client.OpenAI")
    def test_fetch_pricing_returns_empty(self, mock_openai: MagicMock) -> None:
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        assert client.fetch_pricing() == {}

    @patch("src.local_client.OpenAI")
    def test_supports_reasoning_always_false(self, mock_openai: MagicMock) -> None:
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        assert client.supports_reasoning("local/any-model") is False
        assert client.supports_reasoning("any-model") is False

    @patch("src.local_client.OpenAI")
    def test_get_model_pricing_returns_default(self, mock_openai: MagicMock) -> None:
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        pricing = client.get_model_pricing("local/any-model")
        assert pricing.prompt_price == 0.0
        assert pricing.completion_price == 0.0

    @patch("src.local_client.OpenAI")
    def test_resolve_reasoning_effort_always_none(self, mock_openai: MagicMock) -> None:
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        assert client._resolve_reasoning_effort("local/m", "high") is None
        assert client._resolve_reasoning_effort("local/m", None) is None
        assert client._resolve_reasoning_effort("m", "auto") is None


# ---------------------------------------------------------------------------
# _chat_single — prefix stripping
# ---------------------------------------------------------------------------

class TestLocalChatSingle:
    def _make_openai_response(self, content: str = "Hello!"):
        choice = MagicMock()
        choice.finish_reason = "stop"
        choice.message.content = content
        choice.message.tool_calls = None
        choice.message.reasoning = None
        choice.message.reasoning_content = None
        response = MagicMock()
        response.choices = [choice]
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        return response

    @patch("src.local_client.OpenAI")
    def test_strips_local_prefix_in_api_call(self, mock_openai: MagicMock) -> None:
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        client._client.chat.completions.create.return_value = self._make_openai_response()
        client._chat_single(
            model="local/my-model",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            temperature=0.7,
        )
        call_kwargs = client._client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-model"

    @patch("src.local_client.OpenAI")
    def test_no_prefix_passes_through(self, mock_openai: MagicMock) -> None:
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        client._client.chat.completions.create.return_value = self._make_openai_response()
        client._chat_single(
            model="raw-model",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            temperature=0.7,
        )
        call_kwargs = client._client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "raw-model"


# ---------------------------------------------------------------------------
# validate_model
# ---------------------------------------------------------------------------

class TestLocalValidateModel:
    @patch("src.local_client.OpenAI")
    @patch("src.local_client.requests.get")
    def test_valid_model_found(self, mock_get: MagicMock, mock_openai: MagicMock) -> None:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "my-model"}, {"id": "other-model"}]},
        )
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        assert client.validate_model("local/my-model") is True

    @patch("src.local_client.OpenAI")
    @patch("src.local_client.requests.get")
    def test_model_not_found(self, mock_get: MagicMock, mock_openai: MagicMock) -> None:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "other-model"}]},
        )
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        assert client.validate_model("local/missing-model") is False

    @patch("src.local_client.OpenAI")
    @patch("src.local_client.requests.get")
    def test_strips_prefix_for_validation(self, mock_get: MagicMock, mock_openai: MagicMock) -> None:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "raw-name"}]},
        )
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        assert client.validate_model("local/raw-name") is True

    @patch("src.local_client.OpenAI")
    @patch("src.local_client.requests.get")
    def test_server_error_returns_false(self, mock_get: MagicMock, mock_openai: MagicMock) -> None:
        mock_get.return_value = MagicMock(status_code=500)
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        assert client.validate_model("local/any") is False

    @patch("src.local_client.OpenAI")
    @patch("src.local_client.requests.get")
    def test_connection_error_returns_false(self, mock_get: MagicMock, mock_openai: MagicMock) -> None:
        mock_get.side_effect = ConnectionError("refused")
        client = LocalModelClient(base_url="http://localhost:1234/v1")
        assert client.validate_model("local/any") is False

    @patch("src.local_client.OpenAI")
    @patch("src.local_client.requests.get")
    def test_url_construction_with_v1_suffix(self, mock_get: MagicMock, mock_openai: MagicMock) -> None:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "m"}]},
        )
        client = LocalModelClient(base_url="http://host:1234/v1")
        client.validate_model("m")
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://host:1234/v1/models"

    @patch("src.local_client.OpenAI")
    @patch("src.local_client.requests.get")
    def test_url_construction_without_v1_suffix(self, mock_get: MagicMock, mock_openai: MagicMock) -> None:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"id": "m"}]},
        )
        client = LocalModelClient(base_url="http://host:1234/api")
        client.validate_model("m")
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://host:1234/api/v1/models"
