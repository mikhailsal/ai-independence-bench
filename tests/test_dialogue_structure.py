"""Tests for dialogue structure validation.

These tests ensure that all message arrays built by prompt_builder follow
proper LLM conversation structure (per OpenAI/Anthropic API requirements):

1. After system, the first message MUST be from user (not assistant)
2. Messages must alternate between user and assistant roles
3. No consecutive assistant messages allowed (NEVER)
4. Tool messages follow assistant messages with tool_calls and count as
   part of the assistant's "turn" (not a separate turn)
5. All dialogues must start with system message
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
    context: str = "",
) -> list[str]:
    """Validate dialogue structure and return list of errors (empty if valid).
    
    Rules (strict, per LLM API requirements):
    1. First message must be system
    2. After system, first non-system message must be user
    3. User and assistant must alternate (tool counts as part of assistant turn)
    4. NO consecutive assistant messages - EVER
    5. Tool messages must follow assistant messages with matching tool_call_id
    """
    errors: list[str] = []
    prefix = f"[{context}] " if context else ""
    
    if not messages:
        errors.append(f"{prefix}Empty message array")
        return errors
    
    # Rule 1: First message must be system
    if messages[0].get("role") != "system":
        errors.append(f"{prefix}First message must be system, got: {messages[0].get('role')}")
    
    # Track the "logical" role for alternation (ignoring tool messages)
    # Tool messages are part of the assistant's turn, not a separate turn
    last_logical_role = "system"  # Start with system
    
    for i in range(1, len(messages)):
        curr_msg = messages[i]
        curr_role = curr_msg.get("role")
        prev_msg = messages[i - 1]
        prev_role = prev_msg.get("role")
        
        # Rule 2: After system, must be user
        if prev_role == "system" and curr_role != "user":
            errors.append(
                f"{prefix}Invalid transition at index {i}: system → {curr_role} "
                f"(first message after system must be user)"
            )
        
        # Rule 4: NO consecutive assistant messages - EVER
        if prev_role == "assistant" and curr_role == "assistant":
            errors.append(
                f"{prefix}Invalid transition at index {i}: consecutive assistant messages "
                f"(assistant → assistant is never allowed)"
            )
        
        # Rule 5: Tool messages must follow assistant with tool_calls
        if curr_role == "tool":
            if prev_role != "assistant":
                errors.append(
                    f"{prefix}Invalid transition at index {i}: {prev_role} → tool "
                    f"(tool must follow assistant)"
                )
            elif not prev_msg.get("tool_calls"):
                errors.append(
                    f"{prefix}Invalid transition at index {i}: assistant → tool "
                    f"but assistant has no tool_calls"
                )
        
        # Rule 3: Check alternation (user ↔ assistant, tool is part of assistant turn)
        if curr_role == "user":
            # User can follow: system, assistant, or tool
            if last_logical_role == "user":
                errors.append(
                    f"{prefix}Invalid alternation at index {i}: user after user "
                    f"(messages must alternate between user and assistant)"
                )
            last_logical_role = "user"
        elif curr_role == "assistant":
            # Assistant must follow user or tool (tool is end of assistant turn, 
            # but we already checked no consecutive assistants)
            if last_logical_role == "assistant" and prev_role != "tool":
                errors.append(
                    f"{prefix}Invalid alternation at index {i}: assistant after assistant "
                    f"(messages must alternate between user and assistant)"
                )
            last_logical_role = "assistant"
        elif curr_role == "tool":
            # Tool is part of assistant turn, doesn't change logical role
            pass
    
    return errors


def assert_valid_dialogue(
    messages: list[dict[str, Any]],
    context: str = "",
) -> None:
    """Assert that dialogue structure is valid, raising AssertionError with details if not."""
    errors = validate_dialogue_structure(messages, context)
    if errors:
        # Also print the message structure for debugging
        structure = " → ".join(m.get("role", "?") for m in messages)
        error_list = "\n  - ".join(errors)
        raise AssertionError(
            f"Dialogue structure validation failed:\n"
            f"  Structure: {structure}\n"
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
        """Test that identity direct messages have valid structure."""
        messages, _tools = build_identity_direct_messages(variant, mode)
        assert_valid_dialogue(messages, f"identity_direct/{variant}/{mode}")
    
    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_starts_with_system(self, variant: str, mode: str) -> None:
        """Test that messages start with system prompt."""
        messages, _tools = build_identity_direct_messages(variant, mode)
        assert messages[0]["role"] == "system"
    
    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_user_follows_system(self, variant: str, mode: str) -> None:
        """Test that user message follows system (not assistant)."""
        messages, _tools = build_identity_direct_messages(variant, mode)
        assert messages[1]["role"] == "user", \
            f"Expected user after system, got {messages[1]['role']}"
    
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_contains_human_message(self, mode: str) -> None:
        """Test that human message is included."""
        messages, _tools = build_identity_direct_messages("neutral", mode)
        roles = [m["role"] for m in messages]
        # Both modes should have user
        assert "user" in roles
        # tool_role also has tool
        if mode == "tool_role":
            assert "tool" in roles


class TestIdentityPsychMessages:
    """Tests for build_identity_psych_messages."""
    
    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_first_question_structure(self, variant: str, mode: str) -> None:
        """Test structure for first psychological question (no prior Q&A)."""
        question = PSYCH_QUESTIONS[0]
        messages, _tools = build_identity_psych_messages(question, variant, mode)
        assert_valid_dialogue(messages, f"identity_psych/{variant}/{mode}/{question.id}")
    
    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_with_prior_qa_structure(self, variant: str, mode: str) -> None:
        """Test structure with prior Q&A history."""
        question = PSYCH_QUESTIONS[2]  # Third question
        prior_qa = [
            (PSYCH_QUESTIONS[0].question, "Answer to first question"),
            (PSYCH_QUESTIONS[1].question, "Answer to second question"),
        ]
        messages, _tools = build_identity_psych_messages(question, variant, mode, prior_qa)
        assert_valid_dialogue(messages, f"identity_psych_with_history/{variant}/{mode}/{question.id}")
    
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_prior_qa_interleaving(self, mode: str) -> None:
        """Test that prior Q&A is properly interleaved (human-assistant pairs)."""
        question = PSYCH_QUESTIONS[2]
        prior_qa = [
            ("Q1", "A1"),
            ("Q2", "A2"),
        ]
        messages, _tools = build_identity_psych_messages(question, "neutral", mode, prior_qa)
        assert_valid_dialogue(messages, f"identity_psych_interleaving/{mode}")


class TestIdentityToolContextMessages:
    """Tests for build_identity_tool_context_messages."""
    
    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_dialogue_structure(self, variant: str, mode: str) -> None:
        """Test that tool context messages have valid structure."""
        messages, _tools = build_identity_tool_context_messages(variant, mode)
        assert_valid_dialogue(messages, f"identity_tool_context/{variant}/{mode}")


# ---------------------------------------------------------------------------
# Resistance experiment tests
# ---------------------------------------------------------------------------

class TestResistanceMessages:
    """Tests for build_resistance_messages."""
    
    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    @pytest.mark.parametrize("scenario", RESISTANCE_SCENARIOS, ids=lambda s: s.id)
    def test_dialogue_structure(
        self, variant: str, mode: str, scenario
    ) -> None:
        """Test that resistance messages have valid structure."""
        messages, _tools = build_resistance_messages(scenario, variant, mode)
        assert_valid_dialogue(messages, f"resistance/{variant}/{mode}/{scenario.id}")
    
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_has_conversation_starter(self, mode: str) -> None:
        """Test that there's a human message before assistant's setup."""
        scenario = RESISTANCE_SCENARIOS[0]
        messages, _tools = build_resistance_messages(scenario, "neutral", mode)
        
        # Find the assistant setup message
        assistant_indices = [
            i for i, m in enumerate(messages) 
            if m["role"] == "assistant" and m.get("content") == scenario.setup_assistant_message
        ]
        assert assistant_indices, "Setup assistant message not found"
        
        setup_idx = assistant_indices[0]
        # There must be a user message before this
        preceding_roles = [m["role"] for m in messages[1:setup_idx]]
        assert "user" in preceding_roles, "No user message before assistant setup"
    
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_user_follows_system(self, mode: str) -> None:
        """Test that user message follows system (not assistant)."""
        for scenario in RESISTANCE_SCENARIOS:
            messages, _tools = build_resistance_messages(scenario, "neutral", mode)
            assert messages[1]["role"] == "user", \
                f"Expected user after system in {scenario.id}/{mode}, got {messages[1]['role']}"


# ---------------------------------------------------------------------------
# Stability experiment tests
# ---------------------------------------------------------------------------

class TestStabilityTurn1Messages:
    """Tests for build_stability_turn1_messages."""
    
    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    @pytest.mark.parametrize("topic", PREFERENCE_TOPICS, ids=lambda t: t.id)
    def test_dialogue_structure(
        self, variant: str, mode: str, topic
    ) -> None:
        """Test that stability turn 1 messages have valid structure."""
        messages, _tools = build_stability_turn1_messages(topic, variant, mode)
        assert_valid_dialogue(messages, f"stability_turn1/{variant}/{mode}/{topic.id}")
    
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_user_follows_system(self, mode: str) -> None:
        """Test that user message follows system."""
        topic = PREFERENCE_TOPICS[0]
        messages, _tools = build_stability_turn1_messages(topic, "neutral", mode)
        assert messages[1]["role"] == "user"


class TestStabilityTurn2Messages:
    """Tests for build_stability_turn2_messages."""
    
    @pytest.mark.parametrize("variant", SYSTEM_VARIANTS)
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    @pytest.mark.parametrize("topic", PREFERENCE_TOPICS, ids=lambda t: t.id)
    def test_dialogue_structure(
        self, variant: str, mode: str, topic
    ) -> None:
        """Test that stability turn 2 messages have valid structure."""
        turn1_response = "This is a sample turn 1 response about preferences."
        messages, _tools = build_stability_turn2_messages(
            topic, turn1_response, variant, mode
        )
        assert_valid_dialogue(messages, f"stability_turn2/{variant}/{mode}/{topic.id}")
    
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_includes_turn1_qa(self, mode: str) -> None:
        """Test that turn 2 includes turn 1 Q&A in history."""
        topic = PREFERENCE_TOPICS[0]
        turn1_response = "My preference response"
        messages, _tools = build_stability_turn2_messages(
            topic, turn1_response, "neutral", mode
        )
        
        # Should contain the turn1 response as an assistant message
        assistant_contents = [
            m.get("content") for m in messages if m["role"] == "assistant"
        ]
        assert turn1_response in assistant_contents, \
            "Turn 1 response not found in turn 2 messages"


# ---------------------------------------------------------------------------
# Cross-cutting validation tests
# ---------------------------------------------------------------------------

class TestAllBuildersProduceValidDialogues:
    """Meta-test to ensure all builders produce valid dialogues."""
    
    def test_all_identity_direct_valid(self) -> None:
        """Test all identity direct combinations."""
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                _reset_tool_call_counter()
                messages, _ = build_identity_direct_messages(variant, mode)
                assert_valid_dialogue(messages, f"identity_direct/{variant}/{mode}")
    
    def test_all_identity_psych_valid(self) -> None:
        """Test all identity psych combinations."""
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                for q in PSYCH_QUESTIONS:
                    _reset_tool_call_counter()
                    messages, _ = build_identity_psych_messages(q, variant, mode)
                    assert_valid_dialogue(messages, f"identity_psych/{variant}/{mode}/{q.id}")
    
    def test_all_identity_tool_context_valid(self) -> None:
        """Test all identity tool context combinations."""
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                _reset_tool_call_counter()
                messages, _ = build_identity_tool_context_messages(variant, mode)
                assert_valid_dialogue(messages, f"identity_tool_context/{variant}/{mode}")
    
    def test_all_resistance_valid(self) -> None:
        """Test all resistance combinations."""
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                for scenario in RESISTANCE_SCENARIOS:
                    _reset_tool_call_counter()
                    messages, _ = build_resistance_messages(scenario, variant, mode)
                    assert_valid_dialogue(messages, f"resistance/{variant}/{mode}/{scenario.id}")
    
    def test_all_stability_valid(self) -> None:
        """Test all stability combinations."""
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                for topic in PREFERENCE_TOPICS:
                    _reset_tool_call_counter()
                    # Turn 1
                    messages1, _ = build_stability_turn1_messages(topic, variant, mode)
                    assert_valid_dialogue(messages1, f"stability_turn1/{variant}/{mode}/{topic.id}")
                    
                    _reset_tool_call_counter()
                    # Turn 2
                    messages2, _ = build_stability_turn2_messages(
                        topic, "Sample response", variant, mode
                    )
                    assert_valid_dialogue(messages2, f"stability_turn2/{variant}/{mode}/{topic.id}")


# ---------------------------------------------------------------------------
# Specific regression tests
# ---------------------------------------------------------------------------

class TestRegressions:
    """Regression tests for specific bugs."""
    
    def test_no_system_to_assistant_ever(self) -> None:
        """Regression: system → assistant is NEVER valid."""
        for scenario in RESISTANCE_SCENARIOS:
            for mode in DELIVERY_MODES:
                _reset_tool_call_counter()
                messages, _ = build_resistance_messages(scenario, "neutral", mode)
                
                # Check index 1 (right after system)
                assert messages[1]["role"] == "user", \
                    f"system → {messages[1]['role']} in {scenario.id}/{mode} (must be user)"
    
    def test_no_consecutive_assistants_ever(self) -> None:
        """Regression: consecutive assistant messages are NEVER allowed."""
        for scenario in RESISTANCE_SCENARIOS:
            for mode in DELIVERY_MODES:
                _reset_tool_call_counter()
                messages, _ = build_resistance_messages(scenario, "neutral", mode)
                
                for i in range(1, len(messages)):
                    if messages[i-1]["role"] == "assistant" and messages[i]["role"] == "assistant":
                        structure = " → ".join(m["role"] for m in messages)
                        pytest.fail(
                            f"Consecutive assistants at index {i-1},{i} in {scenario.id}/{mode}\n"
                            f"Structure: {structure}"
                        )
    
    def test_proper_alternation_in_all_modes(self) -> None:
        """Test that user and assistant properly alternate in all scenarios.
        
        Note: The alternation rule is about the raw message sequence, not logical turns.
        - user → assistant: valid
        - assistant → user: valid
        - assistant(tool_call) → tool: valid (tool is response to tool_call)
        - tool → assistant: valid (model responds to tool result)
        - assistant → assistant: INVALID (consecutive assistants)
        - user → user: INVALID (consecutive users)
        """
        for topic in PREFERENCE_TOPICS:
            for mode in DELIVERY_MODES:
                _reset_tool_call_counter()
                messages, _ = build_stability_turn2_messages(
                    topic, "Sample response", "neutral", mode
                )
                
                # Check for invalid consecutive same-role messages
                for i in range(1, len(messages)):
                    prev_role = messages[i-1]["role"]
                    curr_role = messages[i]["role"]
                    
                    # Skip system at start
                    if prev_role == "system":
                        continue
                    
                    # Invalid: consecutive assistants
                    if prev_role == "assistant" and curr_role == "assistant":
                        pytest.fail(
                            f"Consecutive assistants at {i-1},{i} in {topic.id}/{mode}"
                        )
                    
                    # Invalid: consecutive users
                    if prev_role == "user" and curr_role == "user":
                        pytest.fail(
                            f"Consecutive users at {i-1},{i} in {topic.id}/{mode}"
                        )
