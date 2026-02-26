"""OpenRouter client: OpenAI SDK wrapper with retry logic, cost tracking, timing."""

from __future__ import annotations

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


class OpenRouterClient:
    """Thin wrapper around the OpenAI SDK pointed at OpenRouter."""

    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2.0
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

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
        """
        use_reasoning = self._resolve_reasoning_effort(model, reasoning_effort)
        return self._chat_single(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=use_reasoning,
            tools=tools,
        )

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

                # Extract content
                content = ""
                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content.strip()

                # Extract usage
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
