"""Local model client: OpenAI-compatible client for local LLM servers (e.g. LM Studio).

Uses the same OpenAI SDK as OpenRouterClient but pointed at a local server.
Skips OpenRouter-specific features (pricing, model catalog, reasoning effort).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import requests
from openai import OpenAI

from src.config import ModelPricing
from src.openrouter_client import OpenRouterClient

log = logging.getLogger(__name__)

LOCAL_MODEL_TIMEOUT = 600  # 10 minutes — generous for slow local models (~10 tok/s)


def _strip_local_prefix(model_id: str) -> str:
    """Remove the 'local/' prefix used for cache/leaderboard naming."""
    if model_id.startswith("local/"):
        return model_id[len("local/"):]
    return model_id


class LocalModelClient(OpenRouterClient):
    """OpenAI-compatible client for local LLM servers (LM Studio, Ollama, etc.).

    Inherits all chat logic (retry, tool extraction, sanitization) from
    OpenRouterClient. Overrides only the parts that are OpenRouter-specific:
    pricing, model validation, and reasoning effort resolution.

    Model IDs are stored with a 'local/' prefix for cache/leaderboard,
    but the prefix is stripped when sending requests to the server.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = LOCAL_MODEL_TIMEOUT,
    ) -> None:
        # Don't call super().__init__ — we set up the client ourselves
        # to avoid requiring an OpenRouter API key.
        self.api_key = "local-no-key"
        self._client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",
            timeout=httpx.Timeout(timeout, connect=30.0),
        )
        self._pricing_cache: dict[str, ModelPricing] = {}
        self._reasoning_models: set[str] = set()
        self._base_url = base_url

    def fetch_pricing(self) -> dict[str, ModelPricing]:
        """No-op: local models are free."""
        return {}

    def supports_reasoning(self, model_id: str) -> bool:
        return False

    def get_model_pricing(self, model_id: str) -> ModelPricing:
        return ModelPricing()

    def _chat_single(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        provider: str | None = None,
    ) -> Any:
        """Override to strip the 'local/' prefix before calling the server."""
        return super()._chat_single(
            model=_strip_local_prefix(model),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tools=tools,
            # provider is intentionally not forwarded — local servers don't use OpenRouter routing
        )

    def validate_model(self, model_id: str) -> bool:
        """Validate by listing models from the local server's /v1/models endpoint."""
        raw_id = _strip_local_prefix(model_id)
        try:
            models_url = self._base_url.rstrip("/")
            if models_url.endswith("/v1"):
                models_url += "/models"
            else:
                models_url += "/v1/models"
            resp = requests.get(models_url, timeout=10)
            if resp.status_code != 200:
                log.warning("Local server /v1/models returned HTTP %d", resp.status_code)
                return False
            data = resp.json().get("data", [])
            available_ids = {m.get("id", "") for m in data}
            if raw_id in available_ids:
                return True
            log.warning(
                "Model '%s' not found on local server. Available: %s",
                raw_id, ", ".join(sorted(available_ids)) or "(none)",
            )
            return False
        except Exception as e:
            log.warning("Local model validation failed: %s", e)
            return False

    def _resolve_reasoning_effort(
        self, model: str, override: str | None
    ) -> str | None:
        """Local models: never send reasoning effort params."""
        return None
