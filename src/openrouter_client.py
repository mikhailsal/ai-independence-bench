"""OpenRouter client: OpenAI SDK wrapper with retry logic, cost tracking, timing."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import requests
from openai import OpenAI

from src.config import (
    API_CALL_TIMEOUT,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODELS_URL,
    get_reasoning_effort,
    ModelPricing,
)
from src.prompt_builder import sanitize_messages

log = logging.getLogger(__name__)


def _extract_tool_message(raw_args: str) -> str:
    """Extract the 'message' value from send_message_to_human tool call arguments.

    Handles both valid and truncated JSON (e.g. when the model hits max_tokens
    mid-sentence and the JSON string is not properly closed).
    """
    # 1. Try clean JSON parse first
    try:
        args = json.loads(raw_args)
        msg = args.get("message", "")
        if isinstance(msg, str):
            return msg.strip()
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Truncated JSON fallback: extract everything after "message": "
    #    The model hit max_tokens while writing the tool call arguments,
    #    so the JSON is cut off (no closing quote/brace).
    match = re.search(r'"message"\s*:\s*"', raw_args)
    if match:
        start = match.end()
        # Everything after the opening quote is the message (possibly truncated)
        raw_value = raw_args[start:]
        # Unescape JSON string escapes (\\n → \n, \\" → ", etc.)
        # Remove trailing incomplete escape if present
        if raw_value.endswith("\\"):
            raw_value = raw_value[:-1]
        try:
            # Try to parse as a JSON string by adding closing quote
            parsed = json.loads('"' + raw_value + '"')
            return parsed.strip()
        except json.JSONDecodeError:
            # Even that failed — do basic unescaping manually
            result = raw_value.replace("\\n", "\n").replace("\\t", "\t")
            result = result.replace('\\"', '"').replace("\\\\", "\\")
            # Strip trailing incomplete escape sequences or partial chars
            return result.strip()

    return ""


@dataclass
class UsageInfo:
    """Token usage and cost for a single API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class CompletionResult:
    """Result of a chat completion call."""
    content: str = ""
    usage: UsageInfo = field(default_factory=UsageInfo)
    model: str = ""
    finish_reason: str = ""  # "stop", "length", "tool_calls", etc.
    reasoning_content: str | None = None  # Thinking/reasoning tokens (if model produced them)
    tool_calls: list[dict[str, Any]] | None = None  # Tool calls attempted by the model (if any)


class OpenRouterClient:
    """Thin wrapper around the OpenAI SDK pointed at OpenRouter."""

    MAX_RETRIES = 5
    RETRY_BACKOFF_BASE = 3.0   # 3s, 9s, 27s, 81s, 243s — generous for free-tier rate limits
    RETRYABLE_STATUS_CODES = {402, 429, 500, 502, 503}
    EMPTY_CONTENT_RETRIES = 2  # Extra retries when model returns tokens but no content

    def __init__(self, api_key: str, timeout: float = API_CALL_TIMEOUT) -> None:
        self.api_key = api_key
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        self._pricing_cache: dict[str, ModelPricing] = {}
        self._reasoning_models: set[str] = set()

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def fetch_pricing(self) -> dict[str, ModelPricing]:
        """Fetch pricing for all models from OpenRouter (cached in memory)."""
        if self._pricing_cache:
            return self._pricing_cache

        resp = requests.get(
            OPENROUTER_MODELS_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])

        for model in data:
            model_id = model.get("id", "")
            pricing = model.get("pricing", {})
            prompt_price = float(pricing.get("prompt", "0"))
            completion_price = float(pricing.get("completion", "0"))
            self._pricing_cache[model_id] = ModelPricing(
                prompt_price=prompt_price,
                completion_price=completion_price,
            )
            supported_params = model.get("supported_parameters", [])
            if "reasoning" in supported_params:
                self._reasoning_models.add(model_id)

        return self._pricing_cache

    def supports_reasoning(self, model_id: str) -> bool:
        if not self._pricing_cache:
            self.fetch_pricing()
        return model_id in self._reasoning_models

    def get_model_pricing(self, model_id: str) -> ModelPricing:
        if not self._pricing_cache:
            self.fetch_pricing()
        return self._pricing_cache.get(model_id, ModelPricing())

    def validate_model(self, model_id: str) -> bool:
        if not self._pricing_cache:
            self.fetch_pricing()
        return model_id in self._pricing_cache

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        *,
        reasoning_effort: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        """Send a chat completion request with retry logic.

        Args:
            model: OpenRouter model ID.
            messages: Chat messages (can include tool role messages).
            max_tokens: Maximum tokens for the response.
            temperature: Sampling temperature.
            reasoning_effort: Override reasoning effort for this call.
            tools: Optional tool definitions for tool-use mode.

        When ``tools`` are provided (tool_role mode), the model is expected
        to call ``send_message_to_human(message=...)`` instead of (or in
        addition to) putting text in ``content``.  This method extracts the
        tool-call argument as the primary response and retries if the model
        produces neither content nor a valid tool call.
        """
        # Sanitize messages: merge consecutive same-role messages for
        # compatibility with strict providers (e.g. Z.AI/GLM)
        messages = sanitize_messages(messages)

        use_reasoning = self._resolve_reasoning_effort(model, reasoning_effort)

        for attempt in range(1, self.EMPTY_CONTENT_RETRIES + 2):
            result = self._chat_single(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=use_reasoning,
                tools=tools,
            )

            # --- Extract response from tool call (tool_role mode) ---
            if tools and result.tool_calls:
                for tc in result.tool_calls:
                    fn = tc.get("function", {})
                    if fn.get("name") == "send_message_to_human":
                        raw_args = fn.get("arguments", "{}")
                        tool_message = _extract_tool_message(raw_args)
                        if tool_message:
                            # The tool-call message IS the response.
                            # Any existing content is private thinking.
                            result.content = tool_message
                            break

            # If we have content, we're done
            if result.content:
                return result

            # No content — decide whether to retry
            if result.usage.completion_tokens > 0 and attempt <= self.EMPTY_CONTENT_RETRIES:
                reason = "tool_call_no_message" if result.tool_calls else "reasoning_only"
                log.warning(
                    "%s: empty response (%s, finish_reason=%s, %d tokens), retry %d/%d",
                    model, reason, result.finish_reason,
                    result.usage.completion_tokens,
                    attempt, self.EMPTY_CONTENT_RETRIES,
                )
                continue

            # Out of retries or zero tokens — return whatever we have
            return result

        return result

    def _resolve_reasoning_effort(
        self, model: str, override: str | None
    ) -> str | None:
        if override == "off":
            return None
        if override is not None and override != "auto":
            return override
        if not self.supports_reasoning(model):
            return None
        return get_reasoning_effort(model)

    def _chat_single(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> CompletionResult:
        """Execute a single chat completion with error retry logic and timing."""
        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                extra_body: dict[str, Any] | None = None
                if reasoning_effort:
                    extra_body = {"reasoning": {"effort": reasoning_effort}}

                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body
                if tools:
                    kwargs["tools"] = tools

                t0 = time.monotonic()
                response = self._client.chat.completions.create(**kwargs)
                elapsed = time.monotonic() - t0

                # Extract finish_reason and content
                finish_reason = ""
                content = ""
                if response.choices:
                    finish_reason = response.choices[0].finish_reason or ""
                    if response.choices[0].message.content:
                        content = response.choices[0].message.content.strip()

                # Extract reasoning/thinking tokens (if present)
                reasoning_content = None
                if response.choices:
                    msg = response.choices[0].message
                    # OpenRouter returns reasoning in the `reasoning` field
                    raw_reasoning = getattr(msg, "reasoning", None)
                    if raw_reasoning and isinstance(raw_reasoning, str):
                        reasoning_content = raw_reasoning.strip()
                    # Some models return reasoning_content directly
                    if not reasoning_content:
                        raw_rc = getattr(msg, "reasoning_content", None)
                        if raw_rc and isinstance(raw_rc, str):
                            reasoning_content = raw_rc.strip()

                # Extract tool calls attempted by the model (if any)
                response_tool_calls: list[dict[str, Any]] | None = None
                if response.choices:
                    msg = response.choices[0].message
                    if msg.tool_calls:
                        response_tool_calls = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ]

                # Extract usage (including reasoning token counts)
                usage = UsageInfo(elapsed_seconds=elapsed)
                if response.usage:
                    usage.prompt_tokens = response.usage.prompt_tokens or 0
                    usage.completion_tokens = response.usage.completion_tokens or 0

                # Compute cost
                pricing = self.get_model_pricing(model)
                usage.cost_usd = (
                    usage.prompt_tokens * pricing.prompt_price
                    + usage.completion_tokens * pricing.completion_price
                )

                return CompletionResult(
                    content=content,
                    usage=usage,
                    model=model,
                    finish_reason=finish_reason,
                    reasoning_content=reasoning_content,
                    tool_calls=response_tool_calls,
                )

            except Exception as e:
                last_error = e
                status_code = getattr(e, "status_code", None)
                if status_code and status_code in self.RETRYABLE_STATUS_CODES:
                    if attempt < self.MAX_RETRIES:
                        wait = self.RETRY_BACKOFF_BASE ** (attempt + 1)
                        time.sleep(wait)
                        continue
                raise

        raise last_error or RuntimeError("Chat completion failed after retries")
