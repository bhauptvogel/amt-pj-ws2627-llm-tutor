"""One opt-in call to a real provider per adapter; skipped when that provider is unreachable."""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

from llmtutor.cli import load_env
from llmtutor.models import create_model
from llmtutor.types import Message, ModelRequest

pytestmark = pytest.mark.live

NORMALISED_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "error"}
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:8b"


def test_openrouter_smoke_call() -> None:
    load_env()
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is not set")

    adapter = create_model(
        "openai_compatible",
        {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-oss-120b",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        os.environ,
    )
    response = adapter.generate(
        ModelRequest(
            messages=(Message("user", "Say hello in one word."),),
            temperature=0.0,
            max_tokens=128,
        )
    )

    assert isinstance(response.text, str)
    assert response.text
    assert response.finish_reason in NORMALISED_FINISH_REASONS
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens > 0
    assert response.raw is not None
    assert response.raw["id"]
    assert "cost" in response.raw["usage"]


def local_models() -> dict[str, Any] | None:
    """The daemon's model list, or None when no daemon is listening."""

    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def test_ollama_smoke_call() -> None:
    tags = local_models()
    if tags is None:
        pytest.skip(f"no Ollama daemon at {OLLAMA_URL}")
    names = {item.get("name") for item in tags.get("models", []) if isinstance(item, dict)}
    if OLLAMA_MODEL not in names:
        pytest.skip(f"{OLLAMA_MODEL} is not pulled; run: ollama pull {OLLAMA_MODEL}")

    adapter = create_model(
        "ollama",
        {"base_url": OLLAMA_URL, "model": OLLAMA_MODEL, "num_ctx": 8192, "think": False},
        os.environ,
    )
    response = adapter.generate(
        ModelRequest(
            messages=(Message("user", "Say hello in one word."),),
            temperature=0.0,
            max_tokens=128,
        )
    )

    assert isinstance(response.text, str)
    assert response.text
    assert response.finish_reason in {"stop", "length"}
    assert response.usage is not None
    assert 0 < response.usage.prompt_tokens < 8192
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )
    assert response.raw is not None
    assert response.raw["done"] is True

    provenance = adapter.provenance()  # type: ignore[attr-defined]
    assert provenance["ollama_version"]
    assert provenance["digest"]
    assert provenance["quantization_level"]
