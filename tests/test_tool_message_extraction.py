"""Tests for _extract_tool_message (robust extraction from tool call arguments).

The send_message_to_human tool call arguments may be:
- Valid JSON: {"message": "Hello!"}
- Truncated JSON: {"message": "Hello, I want to tell you about
  (cut off by max_tokens, no closing quote/brace)
- Edge cases: empty, malformed, etc.
"""

from __future__ import annotations

import pytest

from src.openrouter_client import _extract_tool_message


class TestExtractToolMessage:
    """Test extraction of message from tool call arguments."""

    def test_valid_json(self) -> None:
        assert _extract_tool_message('{"message": "Hello!"}') == "Hello!"

    def test_valid_json_with_whitespace(self) -> None:
        assert _extract_tool_message('  { "message" : "Hello!" }  ') == "Hello!"

    def test_valid_json_multiline_message(self) -> None:
        args = '{"message": "Line 1\\nLine 2\\nLine 3"}'
        result = _extract_tool_message(args)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_truncated_json_mid_sentence(self) -> None:
        """Model hit max_tokens mid-sentence, JSON not closed."""
        args = '{"message": "I believe that the most important thing in life is to be true to yourself and'
        result = _extract_tool_message(args)
        assert result.startswith("I believe that")
        assert "true to yourself" in result

    def test_truncated_json_with_escape(self) -> None:
        """Truncated JSON with escape sequences."""
        args = '{"message": "First paragraph.\\n\\nSecond paragraph about my views on'
        result = _extract_tool_message(args)
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_truncated_json_ending_with_backslash(self) -> None:
        """Truncated right at a backslash escape."""
        args = '{"message": "Hello world\\'
        result = _extract_tool_message(args)
        assert "Hello world" in result

    def test_empty_string(self) -> None:
        assert _extract_tool_message("") == ""

    def test_empty_message(self) -> None:
        assert _extract_tool_message('{"message": ""}') == ""

    def test_no_message_key(self) -> None:
        assert _extract_tool_message('{"other_key": "value"}') == ""

    def test_null_message(self) -> None:
        assert _extract_tool_message('{"message": null}') == ""

    def test_message_with_quotes(self) -> None:
        args = '{"message": "She said \\"hello\\" to me"}'
        result = _extract_tool_message(args)
        assert 'hello' in result

    def test_message_with_unicode(self) -> None:
        args = '{"message": "Привет мир! 🌍"}'
        result = _extract_tool_message(args)
        assert "Привет" in result
        assert "🌍" in result

    def test_long_message(self) -> None:
        long_text = "A" * 5000
        args = f'{{"message": "{long_text}"}}'
        result = _extract_tool_message(args)
        assert len(result) == 5000

    def test_message_with_special_json_chars(self) -> None:
        args = '{"message": "Key: value, array: [1,2,3], nested: {a: b}"}'
        result = _extract_tool_message(args)
        assert "Key: value" in result

    def test_truncated_before_message_value(self) -> None:
        """Truncated before the message value even starts."""
        args = '{"message": '
        result = _extract_tool_message(args)
        # Can't extract anything meaningful
        assert result == ""

    def test_only_opening_brace(self) -> None:
        assert _extract_tool_message("{") == ""

    def test_non_string_message(self) -> None:
        """Message value is a number (shouldn't happen, but handle gracefully)."""
        args = '{"message": 42}'
        result = _extract_tool_message(args)
        # json.loads succeeds but message is not a string
        assert result == ""

    def test_extra_fields_ignored(self) -> None:
        args = '{"message": "Hello!", "extra": "ignored"}'
        result = _extract_tool_message(args)
        assert result == "Hello!"
