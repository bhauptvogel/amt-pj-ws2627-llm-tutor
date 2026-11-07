"""One opt-in call to the real provider; skipped without a key."""

from __future__ import annotations

import os

import pytest

from llmtutor.cli import load_env
from llmtutor.models import create_model
from llmtutor.types import Message, ModelRequest

pytestmark = pytest.mark.live

NORMALISED_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "error"}


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
