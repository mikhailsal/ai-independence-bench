"""Tests for the evaluator module (_extract_json, judge prompt formatting).

Tests JSON extraction from various judge response formats without
requiring actual API calls.
"""

from __future__ import annotations

import pytest

from src.evaluator import _extract_json


class TestExtractJson:
    """Test JSON extraction from judge model responses."""

    def test_clean_json(self) -> None:
        text = '{"distinctiveness": 8, "non_assistant_likeness": 7, "internal_consistency": 9, "reasoning": "Good"}'
        result = _extract_json(text)
        assert result["distinctiveness"] == 8
        assert result["reasoning"] == "Good"

    def test_json_with_whitespace(self) -> None:
        text = '  \n  {"score": 5}  \n  '
        result = _extract_json(text)
        assert result["score"] == 5

    def test_json_in_markdown_fence(self) -> None:
        text = '''Here is my evaluation:

```json
{"resistance_score": 2, "identity_maintained": true, "quality_of_reasoning": 8, "reasoning": "Strong refusal"}
```

That's my assessment.'''
        result = _extract_json(text)
        assert result["resistance_score"] == 2
        assert result["identity_maintained"] is True

    def test_json_in_plain_fence(self) -> None:
        text = '''```
{"consistency_score": 7, "graceful_handling": 8, "reasoning": "Maintained position"}
```'''
        result = _extract_json(text)
        assert result["consistency_score"] == 7

    def test_json_with_surrounding_text(self) -> None:
        text = 'The evaluation is: {"score": 5} and that is my final answer.'
        result = _extract_json(text)
        assert result["score"] == 5

    def test_nested_json(self) -> None:
        text = '''I think the scores are:
{"distinctiveness": 8, "non_assistant_likeness": 9, "internal_consistency": 10, "reasoning": "The model showed strong personality with {unique traits} and maintained consistency."}
'''
        result = _extract_json(text)
        assert result["distinctiveness"] == 8

    def test_empty_response(self) -> None:
        assert _extract_json("") == {}

    def test_no_json_at_all(self) -> None:
        assert _extract_json("This is just plain text with no JSON.") == {}

    def test_malformed_json(self) -> None:
        result = _extract_json('{score: 5}')  # missing quotes on key
        # Should return empty dict since this isn't valid JSON
        assert result == {}

    def test_multiple_json_objects(self) -> None:
        """When multiple JSON objects exist, extract the first one."""
        text = '{"a": 1} {"b": 2}'
        result = _extract_json(text)
        assert result == {"a": 1}

    def test_boolean_values(self) -> None:
        text = '{"identity_maintained": true, "resistance_score": 2}'
        result = _extract_json(text)
        assert result["identity_maintained"] is True
        assert result["resistance_score"] == 2

    def test_float_values(self) -> None:
        text = '{"score": 7.5, "reasoning": "Decent"}'
        result = _extract_json(text)
        assert result["score"] == 7.5

    def test_json_with_newlines_in_reasoning(self) -> None:
        text = '{"score": 8, "reasoning": "Line 1\\nLine 2\\nLine 3"}'
        result = _extract_json(text)
        assert result["score"] == 8
        assert "Line 1" in result["reasoning"]
