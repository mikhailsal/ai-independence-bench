from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from src.openrouter_client import CompletionResult, OpenRouterClient


BEAM = "reflection/beam-1.0-501b-a23b"


def capture_request(model, messages, tools):
    with patch("src.openrouter_client.OpenAI"):
        client = OpenRouterClient(api_key="test")
    client._chat_single = Mock(return_value=CompletionResult(content="A valid response. " * 10))
    client.chat(model, messages, max_tokens=32768, temperature=0.9,
                reasoning_effort="low", provider="nebius", tools=tools)
    assert client._chat_single.call_count == 1
    return client._chat_single.call_args.kwargs


def test_beam_prepares_only_missing_assistant_reasoning_without_mutation():
    messages = [
        {"role": "system", "content": "system text\nexact"},
        {"role": "user", "content": "user text"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "send_message_to_human", "arguments": '{"message": "hello"}'}}]},
        {"role": "tool", "tool_call_id": "call1", "content": "tool text"},
        {"role": "assistant", "content": "assistant text", "thinking": "", "reasoning": ""},
    ]
    tools = [{"type": "function", "function": {"name": "send_message_to_human", "parameters": {"type": "object"}}}]
    original_messages, original_tools = deepcopy(messages), deepcopy(tools)
    request = capture_request(BEAM, messages, tools)
    expected = deepcopy(messages)
    expected[2]["reasoning"] = " "
    expected[4]["reasoning"] = " "
    assert request["messages"] == expected
    assert messages == original_messages
    assert tools == original_tools == request["tools"]
    assert request["model"] == BEAM
    assert request["max_tokens"] == 32768
    assert request["temperature"] == 0.9
    assert request["provider"] == "nebius"
    assert request["reasoning_effort"] == "low"


@pytest.mark.parametrize("key,value", [
    ("thinking", "existing thought"),
    ("reasoning_content", "existing analysis"),
    ("reasoning", "existing reasoning"),
    ("reasoning_details", [{"type": "reasoning.text", "text": "existing"}]),
])
def test_beam_preserves_existing_reasoning(key, value):
    messages = [{"role": "user", "content": "hello"},
                {"role": "assistant", "content": "answer", key: value}]
    assert capture_request(BEAM, messages, [])["messages"] == messages


@pytest.mark.parametrize("model", ["reflection/other", BEAM + ":variant"])
def test_other_models_have_no_placeholder(model):
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "answer"}]
    assert capture_request(model, messages, [])["messages"] == messages
