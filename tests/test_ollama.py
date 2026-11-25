"""Tests for the native Ollama adapter."""

import httpx
import pytest

from llmtutor.models.ollama import OllamaAdapter, OllamaSettings
from llmtutor.types import ErrorKind, Message, ModelError, ModelRequest


def test_ollama_generates_response():
    """Check request formatting and response parsing."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)

        assert request.url.path == "/api/chat"
        assert payload["model"] == "qwen3.5:2b"
        assert payload["think"] is False
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0
        assert payload["options"]["num_predict"] == 512
        assert payload["options"]["seed"] == 42

        return httpx.Response(
            200,
            json={
                "model": "qwen3.5:2b",
                "message": {
                    "role": "assistant",
                    "content": "What happens when a function calls itself?",
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 53,
                "eval_count": 15,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OllamaAdapter(
        OllamaSettings(
            base_url="http://localhost:11434",
            model="qwen3.5:2b",
            think=False,
        ),
        client=client,
    )

    result = adapter.generate(
        ModelRequest(
            messages=(
                Message("system", "You are a tutor."),
                Message("user", "Explain recursion."),
            ),
            temperature=0,
            max_tokens=512,
            seed=42,
        )
    )

    assert result.text == "What happens when a function calls itself?"
    assert result.model == "qwen3.5:2b"
    assert result.finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 53
    assert result.usage.completion_tokens == 15
    assert result.usage.total_tokens == 68


def test_ollama_missing_model():
    """A missing Ollama model should produce a NOT_FOUND error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": "model not found"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OllamaAdapter(
        OllamaSettings(
            base_url="http://localhost:11434",
            model="missing-model",
        ),
        client=client,
    )

    with pytest.raises(ModelError) as exc:
        adapter.generate(
            ModelRequest(
                messages=(Message("user", "Hello"),),
                temperature=0,
            )
        )

    assert exc.value.kind == ErrorKind.NOT_FOUND
    assert exc.value.status == 404


def test_ollama_invalid_response():
    """Malformed provider responses should be classified correctly."""

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"unexpected": "data"},
            )
        )
    )

    adapter = OllamaAdapter(
        OllamaSettings(
            base_url="http://localhost:11434",
            model="qwen3.5:2b",
        ),
        client=client,
    )

    with pytest.raises(ModelError) as exc:
        adapter.generate(
            ModelRequest(
                messages=(Message("user", "Hello"),),
                temperature=0,
            )
        )

    assert exc.value.kind == ErrorKind.BAD_RESPONSE
