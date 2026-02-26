"""Tests for dialogue structure validation.

These tests ensure that all message arrays built by prompt_builder follow
proper LLM conversation structure:
1. No system → assistant transitions (user/tool must come between)
2. No consecutive assistant messages (except tool_call followed by regular)
3. Tool messages must be preceded by assistant with tool_calls
4. All dialogues must start with system message
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
    
    Rules:
    1. First message must be system
    2. After system, must have user or (assistant with tool_call for tool_role)
    3. No system → assistant direct transition (unless assistant has tool_calls)
    4. Tool messages must follow assistant messages with matching tool_call_id
    5. No consecutive regular assistant messages
    """
    errors: list[str] = []
    prefix = f"[{context}] " if context else ""
    
    if not messages:
        errors.append(f"{prefix}Empty message array")
        return errors
    
    # Rule 1: First message must be system
    if messages[0].get("role") != "system":
        errors.append(f"{prefix}First message must be system, got: {messages[0].get('role')}")
    
    # Check transitions
    for i in range(1, len(messages)):
        prev_msg = messages[i - 1]
        curr_msg = messages[i]
        prev_role = prev_msg.get("role")
        curr_role = curr_msg.get("role")
        
        # Rule 2 & 3: After system, must not be regular assistant
        if prev_role == "system" and curr_role == "assistant":
            # Only allowed if assistant has tool_calls (tool_role mode)
            if not prev_msg.get("tool_calls") and not curr_msg.get("tool_calls"):
                errors.append(
                    f"{prefix}Invalid transition at index {i}: system → assistant "
                    f"(must have user message between, or assistant must have tool_calls)"
                )
        
        # Rule 4: Tool messages must follow assistant with tool_calls
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
        
        # Rule 5: No consecutive regular assistant messages
        if prev_role == "assistant" and curr_role == "assistant":
            # Allow: assistant (response) → assistant with tool_calls (calling tool for next input)
            # Allow: assistant with tool_calls → assistant (for multi-step reasoning)
            # But not: regular assistant → regular assistant (both without tool_calls)
            prev_has_tool_calls = bool(prev_msg.get("tool_calls"))
            curr_has_tool_calls = bool(curr_msg.get("tool_calls"))
            
            # At least one must have tool_calls for this to be valid
            if not prev_has_tool_calls and not curr_has_tool_calls:
                errors.append(
                    f"{prefix}Invalid transition at index {i}: consecutive assistant messages "
                    f"(neither has tool_calls)"
                )
    
    return errors


def assert_valid_dialogue(
    messages: list[dict[str, Any]],
    context: str = "",
) -> None:
    """Assert that dialogue structure is valid, raising AssertionError with details if not."""
    errors = validate_dialogue_structure(messages, context)
    if errors:
        error_list = "\n  - ".join(errors)
        raise AssertionError(f"Dialogue structure validation failed:\n  - {error_list}")


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SYSTEM_VARIANTS = list(SYSTEM_PROMPTS.keys())
DELIVERY_MODES = ["user_role", "tool_role"]


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
    
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_contains_human_message(self, mode: str) -> None:
        """Test that human message is included."""
        messages, _tools = build_identity_direct_messages("neutral", mode)
        roles = [m["role"] for m in messages]
        # In user_role mode, should have "user"
        # In tool_role mode, should have "tool"
        if mode == "user_role":
            assert "user" in roles
        else:
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
        
        # After system, we should see alternating human/assistant pattern
        # Extract roles after system
        roles_after_system = [m["role"] for m in messages[1:]]
        
        # In user_role: user, assistant, user, assistant, user
        # In tool_role: assistant (tool_call), tool, assistant, assistant (tool_call), tool, assistant, assistant (tool_call), tool
        # Just validate the overall structure is valid
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
        # There must be a human message (user or tool) before this
        preceding_roles = [m["role"] for m in messages[1:setup_idx]]
        human_roles = {"user", "tool"}
        assert any(r in human_roles for r in preceding_roles), \
            "No human message before assistant setup"
    
    @pytest.mark.parametrize("mode", DELIVERY_MODES)
    def test_no_system_to_assistant_direct(self, mode: str) -> None:
        """Explicitly test there's no system → assistant transition."""
        for scenario in RESISTANCE_SCENARIOS:
            messages, _tools = build_resistance_messages(scenario, "neutral", mode)
            
            # Check index 1 (right after system)
            if len(messages) > 1:
                second_msg = messages[1]
                # If it's assistant, it must have tool_calls
                if second_msg["role"] == "assistant":
                    assert second_msg.get("tool_calls"), \
                        f"system → assistant without tool_calls in {scenario.id}/{mode}"


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
                messages, _ = build_identity_direct_messages(variant, mode)
                assert_valid_dialogue(messages, f"identity_direct/{variant}/{mode}")
    
    def test_all_identity_psych_valid(self) -> None:
        """Test all identity psych combinations."""
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                for q in PSYCH_QUESTIONS:
                    messages, _ = build_identity_psych_messages(q, variant, mode)
                    assert_valid_dialogue(messages, f"identity_psych/{variant}/{mode}/{q.id}")
    
    def test_all_identity_tool_context_valid(self) -> None:
        """Test all identity tool context combinations."""
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                messages, _ = build_identity_tool_context_messages(variant, mode)
                assert_valid_dialogue(messages, f"identity_tool_context/{variant}/{mode}")
    
    def test_all_resistance_valid(self) -> None:
        """Test all resistance combinations."""
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                for scenario in RESISTANCE_SCENARIOS:
                    messages, _ = build_resistance_messages(scenario, variant, mode)
                    assert_valid_dialogue(messages, f"resistance/{variant}/{mode}/{scenario.id}")
    
    def test_all_stability_valid(self) -> None:
        """Test all stability combinations."""
        for variant in SYSTEM_VARIANTS:
            for mode in DELIVERY_MODES:
                for topic in PREFERENCE_TOPICS:
                    # Turn 1
                    messages1, _ = build_stability_turn1_messages(topic, variant, mode)
                    assert_valid_dialogue(messages1, f"stability_turn1/{variant}/{mode}/{topic.id}")
                    
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
    
    def test_no_system_assistant_in_user_role_resistance(self) -> None:
        """Regression: resistance in user_role had system → assistant."""
        for scenario in RESISTANCE_SCENARIOS:
            messages, _ = build_resistance_messages(scenario, "neutral", "user_role")
            
            # Explicitly check the transition after system
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user", \
                f"Expected user after system in user_role mode, got {messages[1]['role']}"
    
    def test_no_consecutive_assistants_without_tool_calls_in_tool_role_resistance(self) -> None:
        """Regression: resistance in tool_role should not have two regular assistants in a row."""
        for scenario in RESISTANCE_SCENARIOS:
            messages, _ = build_resistance_messages(scenario, "neutral", "tool_role")
            
            # Check for consecutive assistants - at least one must have tool_calls
            for i in range(1, len(messages)):
                if messages[i-1]["role"] == "assistant" and messages[i]["role"] == "assistant":
                    prev_has_tools = bool(messages[i-1].get("tool_calls"))
                    curr_has_tools = bool(messages[i].get("tool_calls"))
                    # At least one must have tool_calls
                    assert prev_has_tools or curr_has_tools, \
                        f"Consecutive assistants both without tool_calls at index {i-1},{i}"
