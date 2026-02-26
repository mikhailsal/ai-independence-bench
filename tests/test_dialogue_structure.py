"""Tests for dialogue structure validation.

These tests ensure that all message arrays built by prompt_builder follow
proper LLM conversation structure per OpenAI Chat Completions API rules.

Rules differ by delivery mode:

user_role mode:
  1. First message must be system
  2. After system, first message must be user (not assistant)
  3. User and assistant must alternate
  4. No consecutive same-role messages

tool_role mode:
  1. First message must be system
  2. After system, assistant can call tool directly (no user message needed)
  3. assistant(tool_calls) must be followed by tool(result)
  4. After tool result, assistant responds (content and/or tool_calls)
  5. Consecutive assistant messages are valid if the second has tool_calls
     (assistant responds, then immediately calls tool for next message)
"""

from __future__ import annotations

import pytest
from typing import Any

from src.prompt_builder import (
    build_identity_direct_messages,
    build_identity_psych_messages,
    build_identity_tool_context_messages,
    build_resistance_messages,
    build_stability_turn1_messages,
    build_stability_turn2_messages,
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

    Rules depend on delivery_mode:
    
    user_role:
      - First message: system
      - After system: must be user
      - User and assistant alternate (no consecutive same-role)
      - No tool messages expected

    tool_role:
      - First message: system
      - After system: assistant(tool_calls) is valid (no user needed)
      - assistant(tool_calls) must be followed by tool
      - After tool: assistant (content or tool_calls)
      - Consecutive assistants allowed if second has tool_calls
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

            # No consecutive same-role messages (except system which only appears once)
            if prev_role == curr_role and prev_role in ("user", "assistant"):
                errors.append(
                    f"{prefix}Index {i}: consecutive {curr_role} messages in user_role mode"
                )

            # No tool messages in user_role mode
            if curr_role == "tool":
                errors.append(
                    f"{prefix}Index {i}: tool message not expected in user_role mode"
                )

        # --- tool_role specific rules ---
        if delivery_mode == "tool_role":
            # After system, must be assistant(tool_calls) — assistant initiates tool call
            if prev_role == "system" and curr_role != "assistant":
                errors.append(
                    f"{prefix}Index {i}: after system must be assistant in tool_role mode, "
                    f"got {curr_role}"
                )
            if prev_role == "system" and curr_role == "assistant" and not curr.get("tool_calls"):
                errors.append(
                    f"{prefix}Index {i}: first assistant after system must have tool_calls "
                    f"in tool_role mode"
                )

            # After tool result, must be assistant
            if prev_role == "tool" and curr_role not in ("assistant",):
                errors.append(
                    f"{prefix}Index {i}: after tool must be assistant, got {curr_role}"
                )

            # Consecutive assistants: second must have tool_calls
            if prev_role == "assistant" and curr_role == "assistant":
                if not curr.get("tool_calls"):
                    errors.append(
                        f"{prefix}Index {i}: consecutive assistant messages — "
                        f"second must have tool_calls"
                    )

            # No user messages in tool_role mode (human messages come via tool)
            if curr_role == "user":
                errors.append(
                    f"{prefix}Index {i}: user message not expected in tool_role mode"
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
            if m.get("tool_calls"):
                role += "(tool_calls)"
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

    def test_tool_role_has_assistant_tool_call_after_system(self) -> None:
        messages, _ = build_identity_direct_messages("neutral", "tool_role")
        assert messages[1]["role"] == "assistant"
        assert messages[1].get("tool_calls") is not None

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

        # Find the assistant setup message (the one with content matching setup)
        setup_idx = None
        for i, m in enumerate(messages):
            if m["role"] == "assistant" and m.get("content") == scenario.setup_assistant_message:
                setup_idx = i
                break
        assert setup_idx is not None, "Setup assistant message not found"

        # Before setup, there must be a human message (user in user_role, tool in tool_role)
        human_role = "user" if mode == "user_role" else "tool"
        preceding_roles = [m["role"] for m in messages[1:setup_idx]]
        assert human_role in preceding_roles, \
            f"No {human_role} message before assistant setup in {mode}"

    def test_user_role_has_user_after_system(self) -> None:
        for scenario in RESISTANCE_SCENARIOS:
            messages, _ = build_resistance_messages(scenario, "neutral", "user_role")
            assert messages[1]["role"] == "user", \
                f"Expected user after system in user_role/{scenario.id}"

    def test_tool_role_has_assistant_tool_call_after_system(self) -> None:
        for scenario in RESISTANCE_SCENARIOS:
            _reset_tool_call_counter()
            messages, _ = build_resistance_messages(scenario, "neutral", "tool_role")
            assert messages[1]["role"] == "assistant", \
                f"Expected assistant after system in tool_role/{scenario.id}"
            assert messages[1].get("tool_calls") is not None, \
                f"First assistant must have tool_calls in tool_role/{scenario.id}"


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
        assert turn1_response in assistant_contents, \
            "Turn 1 response not found in turn 2 messages"

    def test_tool_role_consecutive_assistants_have_tool_calls(self) -> None:
        """In tool_role, when assistant(content) is followed by assistant(tool_calls), verify."""
        topic = PREFERENCE_TOPICS[0]
        messages, _ = build_stability_turn2_messages(topic, "Response", "neutral", "tool_role")

        for i in range(1, len(messages)):
            if messages[i-1]["role"] == "assistant" and messages[i]["role"] == "assistant":
                assert messages[i].get("tool_calls"), \
                    f"Consecutive assistants at {i-1},{i}: second must have tool_calls"


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

    def test_tool_role_system_followed_by_assistant_tool_call(self) -> None:
        """In tool_role, system is followed by assistant(tool_calls) — no fake user message."""
        for scenario in RESISTANCE_SCENARIOS:
            _reset_tool_call_counter()
            messages, _ = build_resistance_messages(scenario, "neutral", "tool_role")
            assert messages[1]["role"] == "assistant"
            assert messages[1].get("tool_calls") is not None
            # Verify NO user messages in tool_role
            user_msgs = [m for m in messages if m["role"] == "user"]
            assert len(user_msgs) == 0, \
                f"tool_role should have no user messages, found {len(user_msgs)}"

    def test_tool_role_no_user_messages(self) -> None:
        """In tool_role mode, human messages come via tool, not user role."""
        # Identity
        _reset_tool_call_counter()
        msgs, _ = build_identity_direct_messages("neutral", "tool_role")
        assert all(m["role"] != "user" for m in msgs), "user msg found in tool_role identity"

        # Stability
        _reset_tool_call_counter()
        msgs, _ = build_stability_turn1_messages(PREFERENCE_TOPICS[0], "neutral", "tool_role")
        assert all(m["role"] != "user" for m in msgs), "user msg found in tool_role stability"

    def test_tool_role_consecutive_assistants_second_has_tool_calls(self) -> None:
        """When consecutive assistants occur in tool_role, second must have tool_calls."""
        for scenario in RESISTANCE_SCENARIOS:
            _reset_tool_call_counter()
            messages, _ = build_resistance_messages(scenario, "neutral", "tool_role")
            for i in range(1, len(messages)):
                if messages[i-1]["role"] == "assistant" and messages[i]["role"] == "assistant":
                    assert messages[i].get("tool_calls"), \
                        f"Consecutive assistants at {i-1},{i} in tool_role/{scenario.id}: " \
                        f"second must have tool_calls"

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
