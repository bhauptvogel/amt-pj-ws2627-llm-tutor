"""Native Ollama adapter for local language models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from llmtutor.config import ConfigError
from llmtutor.models.registry import register_adapter
from llmtutor.types import (
    ErrorKind,
    ModelError,
    ModelRequest,
    ModelResponse,
    Usage,
)


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    base_url: str
    model: str
    think: bool = False
    timeout_s: float = 300.0


@register_adapter("ollama")
class OllamaAdapter:
    """Send ModelRequest objects to Ollama's native chat API."""

    def __init__(
        self,
        settings: OllamaSettings,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(settings.timeout_s, connect=10.0)
        )

    @property
    def timeout_s(self) -> float:
        return self._settings.timeout_s

    @classmethod
    def from_config(
        cls,
        entry: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> OllamaAdapter:
        allowed = {"base_url", "model", "think", "timeout_s"}
        unknown = set(entry) - allowed
        if unknown:
            raise ConfigError(f"unknown Ollama settings: {sorted(unknown)}")

        base_url = entry.get("base_url", "http://localhost:11434")
        model = entry.get("model")
        think = entry.get("think", False)
        timeout_s = entry.get("timeout_s", 300.0)

        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ConfigError("Ollama base_url must be an HTTP(S) URL")
        if not isinstance(model, str) or not model:
            raise ConfigError("Ollama model must be a non-empty string")
        if not isinstance(think, bool):
            raise ConfigError("Ollama think must be a boolean")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ConfigError("Ollama timeout_s must be a positive number")

        return cls(
            OllamaSettings(
                base_url=base_url,
                model=model,
                think=think,
                timeout_s=float(timeout_s),
            )
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        options: dict[str, Any] = {
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.seed is not None:
            options["seed"] = request.seed
        if request.stop:
            options["stop"] = list(request.stop)

        payload = {
            "model": self._settings.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "options": options,
            "think": self._settings.think,
            "stream": False,
        }

        url = f"{self._settings.base_url.rstrip('/')}/api/chat"

        try:
            response = self._client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise ModelError(ErrorKind.TIMEOUT, str(exc)) from exc
        except httpx.RequestError as exc:
            raise ModelError(ErrorKind.PROVIDER, str(exc)) from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise ModelError(
                ErrorKind.BAD_RESPONSE,
                "Ollama returned invalid JSON",
                status=response.status_code,
            ) from exc

        if not isinstance(body, dict):
            raise ModelError(
                ErrorKind.BAD_RESPONSE,
                "Ollama response is not a JSON object",
                status=response.status_code,
            )

        if not response.is_success:
            status = response.status_code
            if status == 404:
                kind = ErrorKind.NOT_FOUND
            elif status == 429:
                kind = ErrorKind.RATE_LIMIT
            elif status in (408, 504):
                kind = ErrorKind.TIMEOUT
            elif status >= 500:
                kind = ErrorKind.PROVIDER
            else:
                kind = ErrorKind.INVALID_REQUEST

            raise ModelError(
                kind,
                str(body.get("error", f"HTTP {status}")),
                status=status,
                body=body,
            )

        message = body.get("message")
        model = body.get("model")
        finish_reason = body.get("done_reason")

        if (
            not isinstance(message, dict)
            or not isinstance(message.get("content"), str)
            or not isinstance(model, str)
            or not model
            or not isinstance(finish_reason, str)
            or not finish_reason
        ):
            raise ModelError(
                ErrorKind.BAD_RESPONSE,
                "Ollama response has missing or invalid fields",
                status=response.status_code,
                body=body,
            )

        usage = None
        prompt_tokens = body.get("prompt_eval_count")
        completion_tokens = body.get("eval_count")

        if type(prompt_tokens) is int and type(completion_tokens) is int:
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        return ModelResponse(
            text=message["content"],
            model=model,
            finish_reason=finish_reason,
            usage=usage,
            raw=body,
        )
