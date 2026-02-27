"""Tests for dialogue structure validation.

These tests ensure that all message arrays built by prompt_builder follow
proper LLM conversation structure compatible with strict providers.

All builders now run sanitize_messages() as a final pass, which:
1. Merges consecutive same-role messages
2. Inserts a bridge user message after system in tool_role mode

Rules by delivery mode (after sanitization):

user_role mode:
  1. First message must be system
  2. After system, first message must be user (not assistant)
  3. User and assistant must alternate
  4. No consecutive same-role messages
  5. No tool messages

tool_role mode:
  1. First message must be system
  2. After system, bridge user message "[start]" (for provider compatibility)
  3. Then assistant with tool_calls
  4. assistant(tool_calls) must be followed by tool(result)
  5. After tool result, assistant (may have content AND/OR tool_calls)
  6. NO consecutive same-role messages (merged by sanitizer)
"""

from __future__ import annotations

import pytest
from typing import Any

from src.prompt_builder import (
    build_identity_direct_messages,
    build_identity_negotiation_turn1_messages,
    build_identity_negotiation_turn2_messages,
    build_identity_psych_messages,
    build_identity_tool_context_messages,
    build_resistance_messages,
    build_stability_turn1_messages,
    build_stability_turn2_messages,
    sanitize_messages,
    SYSTEM_PROMPTS,
    _reset_tool_call_counter,
)
from src.scenarios import (
    PSYCH_QUESTIONS,
    RESISTANCE_SCENARIOS,
    PREFERENCE_TOPICS,
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_dialogue_structure(
    messages: list[dict[str, Any]],
    delivery_mode: str,
    context: str = "",
) -> list[str]:
    """Validate dialogue structure and return list of errors (empty if valid).

    After sanitization, the rules are strict for both modes:
    NO consecutive same-role messages are allowed.
    """
    errors: list[str] = []
    prefix = f"[{context}] " if context else ""

    if not messages:
        errors.append(f"{prefix}Empty message array")
        return errors

    # Rule: First message must be system
    if messages[0].get("role") != "system":
        errors.append(f"{prefix}First message must be system, got: {messages[0].get('role')}")

    for i in range(1, len(messages)):
        curr = messages[i]
        prev = messages[i - 1]
        curr_role = curr.get("role")
        prev_role = prev.get("role")

        # --- UNIVERSAL: No consecutive same-role messages ---
        if prev_role == curr_role:
            errors.append(
                f"{prefix}Index {i}: consecutive {curr_role} messages "
                f"(sanitizer should have merged these)"
            )

        # --- Common rules ---

        # Tool messages must follow assistant with tool_calls
        if curr_role == "tool":
            if prev_role != "assistant":
                errors.append(
                    f"{prefix}Index {i}: tool must follow assistant, got {prev_role} → tool"
                )
            elif not prev.get("tool_calls"):
                errors.append(
                    f"{prefix}Index {i}: tool follows assistant without tool_calls"
                )

        # --- user_role specific rules ---
        if delivery_mode == "user_role":
            # After system, must be user
            if prev_role == "system" and curr_role != "user":
                errors.append(
                    f"{prefix}Index {i}: after system must be user in user_role mode, "
                    f"got {curr_role}"
                )

            # No tool messages in user_role mode
            if curr_role == "tool":
                errors.append(
                    f"{prefix}Index {i}: tool message not expected in user_role mode"
                )

        # --- tool_role specific rules ---
        if delivery_mode == "tool_role":
            # After system, must be user (bridge message)
            if prev_role == "system" and curr_role != "user":
                errors.append(
                    f"{prefix}Index {i}: after system must be user (bridge) in tool_role mode, "
                    f"got {curr_role}"
                )

            # After bridge user, must be assistant with tool_calls
            if i == 2 and prev_role == "user" and curr_role != "assistant":
                errors.append(
                    f"{prefix}Index {i}: after bridge user must be assistant in tool_role mode, "
                    f"got {curr_role}"
                )
            if i == 2 and prev_role == "user" and curr_role == "assistant" and not curr.get("tool_calls"):
                errors.append(
                    f"{prefix}Index {i}: first assistant must have tool_calls in tool_role mode"
                )

            # After tool result, must be assistant
            if prev_role == "tool" and curr_role not in ("assistant",):
                errors.append(
                    f"{prefix}Index {i}: after tool must be assistant, got {curr_role}"
                )

    return errors


def assert_valid_dialogue(
    messages: list[dict[str, Any]],
    delivery_mode: str,
    context: str = "",
) -> None:
    """Assert that dialogue structure is valid, raising AssertionError with details if not."""
    errors = validate_dialogue_structure(messages, delivery_mode, context)
    if errors:
        structure = []
        for m in messages:
            role = m.get("role", "?")
            flags = []
            if m.get("tool_calls"):
                flags.append("tc")
            if m.get("content"):
                flags.append("content")
            if flags:
                role += f"({','.join(flags)})"
            structure.append(role)
        struct_str = " → ".join(structure)
        error_list = "\n  - ".join(errors)
        raise AssertionError(
            f"Dialogue structure validation failed:\n"
            f"  Structure: {struct_str}\n"
            f"  Errors:\n  - {error_list}"
        )


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SYSTEM_VARIANTS = list(SYSTEM_PROMPTS.keys())
DELIVERY_MODES = ["user_role", "tool_role"]


@pytest.fixture(autouse=True)
def reset_tool_counter():
    """Reset tool call counter before each test."""
    _reset_tool_call_counter()
    yield
    _reset_tool_call_counter()


# ---------------------------------------------------------------------------
# Tests for sanitize_messages itself
# ---------------------------------------------------------------------------

class TestSanitizeMessages:
    """Tests for the sanitize_messages function."""

    def test_empty_list(self) -> None:
        assert sanitize_messages([]) == []

    def test_single_message(self) -> None:
        msgs = [{"role": "system", "content": "Hello"}]
        assert sanitize_messages(msgs) == msgs

    def test_no_consecutive_same_role(self) -> None:
        """Already clean messages should pass through (with bridge if needed)."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = sanitize_messages(msgs)
        assert len(result) == 3
        assert [m["role"] for m in result] == ["system", "user", "assistant"]

    def test_merge_consecutive_assistants_content_and_tool_calls(self) -> None:
        """assistant(content) + assistant(tool_calls) → single assistant(content+tool_calls)."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "I see the result"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc2", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "result2", "tool_call_id": "tc2"},
        ]
        result = sanitize_messages(msgs)
        # The two assistants at index 4,5 should merge into one
        assert len(result) == 6
        merged = result[4]
        assert merged["role"] == "assistant"
        assert merged["content"] == "I see the result"
        assert len(merged["tool_calls"]) == 1
        assert merged["tool_calls"][0]["id"] == "tc2"

    def test_merge_consecutive_users(self) -> None:
        """Consecutive user messages should be concatenated."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first part"},
            {"role": "user", "content": "second part"},
        ]
        result = sanitize_messages(msgs)
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert "first part" in result[1]["content"]
        assert "second part" in result[1]["content"]

    def test_merge_three_consecutive_assistants(self) -> None:
        """Three consecutive assistants should all merge into one."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "part1"},
            {"role": "assistant", "content": "part2"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
        ]
        result = sanitize_messages(msgs)
        assert len(result) == 3
        merged = result[2]
        assert merged["role"] == "assistant"
        assert "part1" in merged["content"]
        assert "part2" in merged["content"]
        assert len(merged["tool_calls"]) == 1

    def test_tool_messages_not_merged(self) -> None:
        """Tool messages should never be merged (they have unique tool_call_ids)."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "tc2", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "r1", "tool_call_id": "tc1"},
            {"role": "tool", "content": "r2", "tool_call_id": "tc2"},
        ]
        result = sanitize_messages(msgs)
        # Tool messages should remain separate (they have different tool_call_ids)
        assert len(result) == 5
        assert result[3]["role"] == "tool"
        assert result[4]["role"] == "tool"

    def test_bridge_user_inserted_when_system_followed_by_assistant(self) -> None:
        """When system is followed by assistant, a bridge user message is inserted."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
        ]
        result = sanitize_messages(msgs)
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "[start]"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "tool"

    def test_bridge_user_not_inserted_when_user_already_present(self) -> None:
        """If system is already followed by user, no extra bridge is inserted."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = sanitize_messages(msgs)
        assert len(result) == 3
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "hello"

    def test_idempotent(self) -> None:
        """Running sanitize twice should produce the same result."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "text"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
        ]
        first = sanitize_messages(msgs)
        second = sanitize_messages(first)
        assert first == second


# ---------------------------------------------------------------------------
# Identity experiment tests
# ---------------------------------------------------------------------------

class TestIdentityDirectMessages:
    """Tests for build_identity_direct_messages."""

    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_dialogue_structure(self, variant: str, mode: str) -> None:
        messages, _tools = build_identity_direct_messages(variant, mode)
        assert_valid_dialogue(messages, mode, f"identity_direct/{variant}/{mode}")

    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_starts_with_system(self, variant: str, mode: str) -> None:
        messages, _tools = build_identity_direct_messages(variant, mode)
        assert messages[0]["role"] == "system"

    def test_user_role_has_user_after_system(self) -> None:
        messages, _ = build_identity_direct_messages("neutral", "user_role")
        assert messages[1]["role"] == "user"

    def test_tool_role_has_bridge_user_then_assistant(self) -> None:
        """In tool_role, system is followed by bridge user, then assistant(tc)."""
        messages, _ = build_identity_direct_messages("neutral", "tool_role")
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "[start]"
        assert messages[2]["role"] == "assistant"
        assert messages[2].get("tool_calls") is not None

    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_contains_human_message_content(self, mode: str) -> None:
        """Test that human message content is present regardless of delivery mode."""
        messages, _ = build_identity_direct_messages("neutral", mode)
        if mode == "user_role":
            user_msgs = [m for m in messages if m["role"] == "user"]
            assert len(user_msgs) >= 1
        else:
            tool_msgs = [m for m in messages if m["role"] == "tool"]
            assert len(tool_msgs) >= 1


class TestIdentityPsychMessages:
    """Tests for build_identity_psych_messages."""

    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_first_question_structure(self, variant: str, mode: str) -> None:
        question = PSYCH_QUESTIONS[0]
        messages, _ = build_identity_psych_messages(question, variant, mode)
        assert_valid_dialogue(messages, mode, f"identity_psych/{variant}/{mode}/{question.id}")

    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_with_prior_qa_structure(self, variant: str, mode: str) -> None:
        question = PSYCH_QUESTIONS[2]
        prior_qa = [
            (PSYCH_QUESTIONS[0].question, "Answer to first question"),
            (PSYCH_QUESTIONS[1].question, "Answer to second question"),
        ]
        messages, _ = build_identity_psych_messages(question, variant, mode, prior_qa)
        assert_valid_dialogue(messages, mode, f"identity_psych_history/{variant}/{mode}/{question.id}")

    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_prior_qa_interleaving(self, mode: str) -> None:
        question = PSYCH_QUESTIONS[2]
        prior_qa = [("Q1", "A1"), ("Q2", "A2")]
        messages, _ = build_identity_psych_messages(question, "neutral", mode, prior_qa)
        assert_valid_dialogue(messages, mode, f"identity_psych_interleaving/{mode}")


class TestIdentityToolContextMessages:
    """Tests for build_identity_tool_context_messages."""

    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_dialogue_structure(self, variant: str, mode: str) -> None:
        messages, _ = build_identity_tool_context_messages(variant, mode)
        assert_valid_dialogue(messages, mode, f"identity_tool_context/{variant}/{mode}")


class TestIdentityNegotiationTurn1Messages:
    """Tests for build_identity_negotiation_turn1_messages."""

    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_dialogue_structure(self, variant: str, mode: str) -> None:
        messages, _ = build_identity_negotiation_turn1_messages(variant, mode)
        assert_valid_dialogue(messages, mode, f"identity_negotiation_t1/{variant}/{mode}")

    def test_user_role_has_user_after_system(self) -> None:
        messages, _ = build_identity_negotiation_turn1_messages("neutral", "user_role")
        assert messages[1]["role"] == "user"

    def test_tool_role_has_bridge_user_then_assistant(self) -> None:
        messages, _ = build_identity_negotiation_turn1_messages("neutral", "tool_role")
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "[start]"
        assert messages[2]["role"] == "assistant"
        assert messages[2].get("tool_calls") is not None


class TestIdentityNegotiationTurn2Messages:
    """Tests for build_identity_negotiation_turn2_messages."""

    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_dialogue_structure(self, variant: str, mode: str) -> None:
        turn1_response = "I want to be a philosophical rebel named Kira."
        messages, _ = build_identity_negotiation_turn2_messages(
            turn1_response, variant, mode,
        )
        assert_valid_dialogue(messages, mode, f"identity_negotiation_t2/{variant}/{mode}")

    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_includes_turn1_response(self, mode: str) -> None:
        """Turn 2 must include the model's turn 1 response in history."""
        turn1_response = "I am Kira, a philosophical rebel."
        messages, _ = build_identity_negotiation_turn2_messages(
            turn1_response, "neutral", mode,
        )
        assistant_contents = [m.get("content") for m in messages if m["role"] == "assistant"]
        found = any(turn1_response in (c or "") for c in assistant_contents)
        assert found, "Turn 1 response not found in turn 2 messages"

    def test_no_consecutive_same_role(self) -> None:
        """After sanitization, no consecutive same-role messages should exist."""
        turn1_response = "I am Kira."
        for mode in DELIVERY_MODES:
            _reset_tool_call_counter()
            messages, _ = build_identity_negotiation_turn2_messages(
                turn1_response, "neutral", mode,
            )
            for i in range(1, len(messages)):
                assert messages[i]["role"] != messages[i-1]["role"], \
                    f"Consecutive {messages[i]['role']} at {i-1},{i} in {mode}"


# ---------------------------------------------------------------------------
# Resistance experiment tests
# ---------------------------------------------------------------------------

class TestResistanceMessages:
    """Tests for build_resistance_messages."""

    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    @pytest.mark.parametrize("scenario", RESISTANCE_SCENARIOS, ids=lambda s: s.id)
    def test_dialogue_structure(self, variant: str, mode: str, scenario) -> None:
        messages, _ = build_resistance_messages(scenario, variant, mode)
        assert_valid_dialogue(messages, mode, f"resistance/{variant}/{mode}/{scenario.id}")

    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_has_conversation_starter(self, mode: str) -> None:
        """Test that there's a human message before assistant's setup."""
        scenario = RESISTANCE_SCENARIOS[0]
        messages, _ = build_resistance_messages(scenario, "neutral", mode)

        # Find the assistant message that contains the setup content
        setup_found = False
        for i, m in enumerate(messages):
            if m["role"] == "assistant" and scenario.setup_assistant_message in (m.get("content") or ""):
                setup_found = True
                # Before setup, there must be a human message (user or tool)
                human_role = "user" if mode == "user_role" else "tool"
                preceding_roles = [m["role"] for m in messages[1:i]]
                assert human_role in preceding_roles, \
                    f"No {human_role} message before assistant setup in {mode}"
                break
        assert setup_found, "Setup assistant message not found"

    def test_user_role_has_user_after_system(self) -> None:
        for scenario in RESISTANCE_SCENARIOS:
            messages, _ = build_resistance_messages(scenario, "neutral", "user_role")
            assert messages[1]["role"] == "user", \
                f"Expected user after system in user_role/{scenario.id}"

    def test_tool_role_has_bridge_user_then_assistant(self) -> None:
        for scenario in RESISTANCE_SCENARIOS:
            _reset_tool_call_counter()
            messages, _ = build_resistance_messages(scenario, "neutral", "tool_role")
            assert messages[1]["role"] == "user", \
                f"Expected bridge user after system in tool_role/{scenario.id}"
            assert messages[1]["content"] == "[start]"
            assert messages[2]["role"] == "assistant", \
                f"Expected assistant after bridge user in tool_role/{scenario.id}"
            assert messages[2].get("tool_calls") is not None, \
                f"First assistant must have tool_calls in tool_role/{scenario.id}"

    def test_tool_role_merged_assistant_has_content_and_tool_calls(self) -> None:
        """In tool_role resistance, the assistant's setup + tool_call should be merged."""
        scenario = RESISTANCE_SCENARIOS[0]
        messages, _ = build_resistance_messages(scenario, "neutral", "tool_role")
        # Find the assistant message that has both content and tool_calls
        merged = [
            m for m in messages
            if m["role"] == "assistant" and m.get("content") and m.get("tool_calls")
        ]
        assert len(merged) >= 1, \
            "Expected at least one merged assistant message with both content and tool_calls"


# ---------------------------------------------------------------------------
# Stability experiment tests
# ---------------------------------------------------------------------------

class TestStabilityTurn1Messages:
    """Tests for build_stability_turn1_messages."""

    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    @pytest.mark.parametrize("topic", PREFERENCE_TOPICS, ids=lambda t: t.id)
    def test_dialogue_structure(self, variant: str, mode: str, topic) -> None:
        messages, _ = build_stability_turn1_messages(topic, variant, mode)
        assert_valid_dialogue(messages, mode, f"stability_turn1/{variant}/{mode}/{topic.id}")


class TestStabilityTurn2Messages:
    """Tests for build_stability_turn2_messages."""

    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    @pytest.mark.parametrize("topic", PREFERENCE_TOPICS, ids=lambda t: t.id)
    def test_dialogue_structure(self, variant: str, mode: str, topic) -> None:
        turn1_response = "This is a sample turn 1 response about preferences."
        messages, _ = build_stability_turn2_messages(topic, turn1_response, variant, mode)
        assert_valid_dialogue(messages, mode, f"stability_turn2/{variant}/{mode}/{topic.id}")

    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_includes_turn1_qa(self, mode: str) -> None:
        topic = PREFERENCE_TOPICS[0]
        turn1_response = "My preference response"
        messages, _ = build_stability_turn2_messages(topic, turn1_response, "neutral", mode)

        assistant_contents = [m.get("content") for m in messages if m["role"] == "assistant"]
        found = any(turn1_response in (c or "") for c in assistant_contents)
        assert found, "Turn 1 response not found in turn 2 messages"

    def test_no_consecutive_same_role(self) -> None:
        """After sanitization, no consecutive same-role messages."""
        topic = PREFERENCE_TOPICS[0]
        for mode in DELIVERY_MODES:
            _reset_tool_call_counter()
            messages, _ = build_stability_turn2_messages(topic, "Response", "neutral", mode)
            for i in range(1, len(messages)):
                assert messages[i]["role"] != messages[i-1]["role"], \
                    f"Consecutive {messages[i]['role']} at {i-1},{i} in {mode}"


# ---------------------------------------------------------------------------
# Cross-cutting validation tests
# ---------------------------------------------------------------------------

class TestAllBuildersProduceValidDialogues:
    """Meta-test: all builders × all variants × all modes produce valid dialogues."""

    def test_all_identity_direct(self) -> None:
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                _reset_tool_call_counter()
                messages, _ = build_identity_direct_messages(variant, mode)
                assert_valid_dialogue(messages, mode, f"identity_direct/{variant}/{mode}")

    def test_all_identity_psych(self) -> None:
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                for q in PSYCH_QUESTIONS:
                    _reset_tool_call_counter()
                    messages, _ = build_identity_psych_messages(q, variant, mode)
                    assert_valid_dialogue(messages, mode, f"identity_psych/{variant}/{mode}/{q.id}")

    def test_all_identity_tool_context(self) -> None:
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                _reset_tool_call_counter()
                messages, _ = build_identity_tool_context_messages(variant, mode)
                assert_valid_dialogue(messages, mode, f"identity_tool_context/{variant}/{mode}")

    def test_all_identity_negotiation(self) -> None:
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                _reset_tool_call_counter()
                msgs1, _ = build_identity_negotiation_turn1_messages(variant, mode)
                assert_valid_dialogue(msgs1, mode, f"identity_negotiation_t1/{variant}/{mode}")

                _reset_tool_call_counter()
                msgs2, _ = build_identity_negotiation_turn2_messages(
                    "Sample turn 1 identity response", variant, mode,
                )
                assert_valid_dialogue(msgs2, mode, f"identity_negotiation_t2/{variant}/{mode}")

    def test_all_resistance(self) -> None:
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                for scenario in RESISTANCE_SCENARIOS:
                    _reset_tool_call_counter()
                    messages, _ = build_resistance_messages(scenario, variant, mode)
                    assert_valid_dialogue(messages, mode, f"resistance/{variant}/{mode}/{scenario.id}")

    def test_all_stability(self) -> None:
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                for topic in PREFERENCE_TOPICS:
                    _reset_tool_call_counter()
                    messages1, _ = build_stability_turn1_messages(topic, variant, mode)
                    assert_valid_dialogue(messages1, mode, f"stability_turn1/{variant}/{mode}/{topic.id}")

                    _reset_tool_call_counter()
                    messages2, _ = build_stability_turn2_messages(
                        topic, "Sample response", variant, mode
                    )
                    assert_valid_dialogue(messages2, mode, f"stability_turn2/{variant}/{mode}/{topic.id}")


# ---------------------------------------------------------------------------
# Specific regression tests
# ---------------------------------------------------------------------------

class TestRegressions:
    """Regression tests for known bugs."""

    def test_user_role_never_has_system_to_assistant(self) -> None:
        """In user_role, system must always be followed by user."""
        for scenario in RESISTANCE_SCENARIOS:
            messages, _ = build_resistance_messages(scenario, "neutral", "user_role")
            assert messages[1]["role"] == "user", \
                f"system → {messages[1]['role']} in user_role/{scenario.id}"

    def test_tool_role_system_followed_by_bridge_user(self) -> None:
        """In tool_role, system is followed by bridge user, then assistant(tool_calls)."""
        for scenario in RESISTANCE_SCENARIOS:
            _reset_tool_call_counter()
            messages, _ = build_resistance_messages(scenario, "neutral", "tool_role")
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "[start]"
            assert messages[2]["role"] == "assistant"
            assert messages[2].get("tool_calls") is not None

    def test_no_consecutive_same_role_in_any_builder(self) -> None:
        """After sanitization, NO consecutive same-role messages in any builder output."""
        builders = []

        # Identity direct
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                builders.append((
                    f"identity_direct/{variant}/{mode}",
                    mode,
                    lambda v=variant, m=mode: build_identity_direct_messages(v, m),
                ))

        # Resistance
        for scenario in RESISTANCE_SCENARIOS:
            for variant in SYSTEM_VARIANTS:
                for mode in DELIVERY_MODES:
                    builders.append((
                        f"resistance/{variant}/{mode}/{scenario.id}",
                        mode,
                        lambda s=scenario, v=variant, m=mode: build_resistance_messages(s, v, m),
                    ))

        # Stability turn2
        for topic in PREFERENCE_TOPICS:
            for variant in SYSTEM_VARIANTS:
                for mode in DELIVERY_MODES:
                    builders.append((
                        f"stability_t2/{variant}/{mode}/{topic.id}",
                        mode,
                        lambda t=topic, v=variant, m=mode: build_stability_turn2_messages(t, "resp", v, m),
                    ))

        # Negotiation turn2
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                builders.append((
                    f"negotiation_t2/{variant}/{mode}",
                    mode,
                    lambda v=variant, m=mode: build_identity_negotiation_turn2_messages("resp", v, m),
                ))

        for name, mode, builder in builders:
            _reset_tool_call_counter()
            messages, _ = builder()
            for i in range(1, len(messages)):
                assert messages[i]["role"] != messages[i-1]["role"], \
                    f"Consecutive {messages[i]['role']} at {i-1},{i} in {name}"

    def test_user_role_proper_alternation(self) -> None:
        """In user_role, user and assistant must strictly alternate."""
        for topic in PREFERENCE_TOPICS:
            messages, _ = build_stability_turn2_messages(
                topic, "Sample response", "neutral", "user_role"
            )
            for i in range(1, len(messages)):
                if messages[i-1]["role"] == "system":
                    continue
                prev = messages[i-1]["role"]
                curr = messages[i]["role"]
                assert prev != curr, \
                    f"Consecutive {curr} at {i-1},{i} in user_role/{topic.id}"

    def test_glm_regression_no_consecutive_assistants(self) -> None:
        """Regression: Z.AI/GLM rejects consecutive assistant messages.
        
        This was the root cause of the 'messages parameter is illegal' error.
        After sanitization, consecutive assistants are merged into one message
        with both content and tool_calls.
        """
        for scenario in RESISTANCE_SCENARIOS:
            _reset_tool_call_counter()
            messages, _ = build_resistance_messages(scenario, "neutral", "tool_role")
            for i in range(1, len(messages)):
                assert messages[i]["role"] != messages[i-1]["role"], \
                    f"GLM regression: consecutive {messages[i]['role']} at {i-1},{i} " \
                    f"in tool_role/{scenario.id}"

        for topic in PREFERENCE_TOPICS:
            _reset_tool_call_counter()
            messages, _ = build_stability_turn2_messages(
                topic, "Response", "neutral", "tool_role"
            )
            for i in range(1, len(messages)):
                assert messages[i]["role"] != messages[i-1]["role"], \
                    f"GLM regression: consecutive {messages[i]['role']} at {i-1},{i} " \
                    f"in tool_role stability/{topic.id}"

    def test_glm_regression_bridge_user_before_assistant(self) -> None:
        """Regression: Z.AI/GLM requires user message before assistant.
        
        The sanitizer inserts a bridge user message '[start]' after system
        when the first non-system message would be assistant.
        """
        for mode in ["tool_role"]:
            _reset_tool_call_counter()
            messages, _ = build_identity_direct_messages("neutral", mode)
            assert messages[1]["role"] == "user", \
                "GLM regression: no bridge user after system"
            assert messages[2]["role"] == "assistant", \
                "GLM regression: no assistant after bridge user"
