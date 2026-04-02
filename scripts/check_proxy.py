#!/usr/bin/env python3
"""Verify that the benchmark can connect through the configured proxy."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_MODELS_URL,
    _DEFAULT_OPENROUTER_BASE_URL,
    load_api_key,
)


def main() -> None:
    print("=== AI Independence Bench — Proxy Check ===\n")

    key = load_api_key(required=False)
    masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
    print(f"API key:     {masked_key}")
    print(f"Base URL:    {OPENROUTER_BASE_URL}")
    print(f"Models URL:  {OPENROUTER_MODELS_URL}")

    if OPENROUTER_BASE_URL == _DEFAULT_OPENROUTER_BASE_URL:
        print("\n[INFO] Using default OpenRouter URL (no proxy configured).")
    else:
        print(f"\n[INFO] Custom proxy detected: {OPENROUTER_BASE_URL}")

    print("\n--- Connectivity check ---")

    import requests

    try:
        resp = requests.get(
            OPENROUTER_MODELS_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
        print(f"GET {OPENROUTER_MODELS_URL} -> HTTP {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            print(f"Models available: {len(data)}")
            if data:
                sample = [m.get("id", "?") for m in data[:3]]
                print(f"Sample models: {', '.join(sample)}")
            print("\n[OK] Proxy connectivity verified.")
        else:
            print(f"\n[WARN] Non-200 response: {resp.text[:200]}")
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        sys.exit(1)

    print("\n--- Quick chat completion test ---")
    from src.openrouter_client import OpenRouterClient

    try:
        client = OpenRouterClient(api_key=key)
        result = client.chat(
            model="google/gemini-3-flash-preview",
            messages=[{"role": "user", "content": "Reply with just the word 'pong'."}],
            max_tokens=10,
            temperature=0.0,
        )
        print(f"Model response: {result.content!r}")
        print(f"Tokens: {result.usage.prompt_tokens} in / {result.usage.completion_tokens} out")
        print(f"Cost: ${result.usage.cost_usd:.6f}")
        print(f"Time: {result.usage.elapsed_seconds:.2f}s")

        if result.content:
            print("\n[OK] Chat completion through proxy works!")
        else:
            print("\n[WARN] Empty response received.")
    except Exception as e:
        print(f"\n[ERROR] Chat completion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
