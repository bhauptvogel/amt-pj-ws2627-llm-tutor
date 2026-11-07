"""The OpenAI-compatible chat-completions adapter, configured for OpenRouter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from llmtutor.config import ConfigError
from llmtutor.models.registry import register_adapter
from llmtutor.types import ErrorKind, ModelError, ModelRequest, ModelResponse, Usage

ALLOWED_SETTINGS = frozenset({"base_url", "model", "api_key_env", "timeout_s"})
REQUIRED_SETTINGS = ("base_url", "model", "api_key_env")
DEFAULT_TIMEOUT_S = 300.0
CONNECT_TIMEOUT_S = 10.0


@dataclass(frozen=True, slots=True)
class OpenAICompatibleSettings:
    base_url: str
    model: str
    api_key_env: str
    api_key: str = field(repr=False)
    timeout_s: float = DEFAULT_TIMEOUT_S


@register_adapter("openai_compatible")
class OpenAICompatibleAdapter:
    """A translator between our request/response types and the chat-completions wire format."""

    def __init__(
        self,
        settings: OpenAICompatibleSettings,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(settings.timeout_s, connect=CONNECT_TIMEOUT_S)
        )

    @property
    def timeout_s(self) -> float:
        return self._settings.timeout_s

    @classmethod
    def from_config(
        cls,
        entry: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> OpenAICompatibleAdapter:
        unknown = sorted(set(entry) - ALLOWED_SETTINGS)
        if unknown:
            raise ConfigError(
                f"unknown setting(s) for adapter 'openai_compatible': {', '.join(unknown)}"
            )
        missing = [key for key in REQUIRED_SETTINGS if key not in entry]
        if missing:
            raise ConfigError(
                f"missing required setting(s) for adapter 'openai_compatible': {', '.join(missing)}"
            )

        for key in ("base_url", "model", "api_key_env"):
            value = entry[key]
            if not isinstance(value, str) or not value:
                raise ConfigError(f"setting {key!r} must be a non-empty string")

        base_url = entry["base_url"]
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError(f"setting 'base_url' must be an http(s) URL, got {base_url!r}")

        timeout_s = entry.get("timeout_s", DEFAULT_TIMEOUT_S)
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ConfigError("setting 'timeout_s' must be a number")
        if timeout_s <= 0:
            raise ConfigError("setting 'timeout_s' must be positive")

        api_key_env = entry["api_key_env"]
        api_key = environ.get(api_key_env)
        if not api_key:
            raise ConfigError(f"environment variable {api_key_env} is not set")

        return cls(
            OpenAICompatibleSettings(
                base_url=base_url,
                model=entry["model"],
                api_key_env=api_key_env,
                api_key=api_key,
                timeout_s=float(timeout_s),
            )
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop:
            payload["stop"] = list(request.stop)

        url = f"{self._settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self._client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ModelError(ErrorKind.TIMEOUT, str(exc) or "request timed out") from exc
        except httpx.RequestError as exc:
            raise ModelError(ErrorKind.PROVIDER, str(exc)) from exc

        body = _decode_body(response)
        status = response.status_code

        if not response.is_success:
            kind = _status_kind(status)
            retry_after = _retry_after(response) if kind is ErrorKind.RATE_LIMIT else None
            raise ModelError(
                kind,
                _error_message(body) or f"HTTP {status}",
                status=status,
                body=body,
                retry_after=retry_after,
            )

        if body is None:
            raise ModelError(
                ErrorKind.BAD_RESPONSE,
                "response body is not a JSON object",
                status=status,
            )
        if "error" in body and "choices" not in body:
            raise ModelError(
                ErrorKind.PROVIDER,
                _error_message(body) or "provider error",
                status=status,
                body=body,
            )
        return _parse_response(body, status)


def _decode_body(response: httpx.Response) -> Mapping[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _error_message(body: Mapping[str, Any] | None) -> str | None:
    if body is None:
        return None
    error = body.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    return None


def _status_kind(status: int) -> ErrorKind:
    if status in (401, 403):
        return ErrorKind.AUTH
    if status == 402:
        return ErrorKind.PAYMENT
    if status == 404:
        return ErrorKind.NOT_FOUND
    if status in (408, 504):
        return ErrorKind.TIMEOUT
    if status == 429:
        return ErrorKind.RATE_LIMIT
    if 500 <= status <= 599:
        return ErrorKind.PROVIDER
    if 400 <= status <= 499:
        return ErrorKind.INVALID_REQUEST
    return ErrorKind.BAD_RESPONSE


def _retry_after(response: httpx.Response) -> int | None:
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    header = header.strip()
    return int(header) if header.isdigit() else None


def _parse_response(body: Mapping[str, Any], status: int) -> ModelResponse:
    def malformed(reason: str) -> ModelError:
        return ModelError(ErrorKind.BAD_RESPONSE, reason, status=status, body=body)

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise malformed("response field 'model' is missing or not a string")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise malformed("response field 'choices' is missing or empty")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise malformed("response field 'choices[0]' is not an object")

    message = choice.get("message")
    if not isinstance(message, dict):
        raise malformed("response field 'choices[0].message' is missing")
    text = message.get("content")
    if text is not None and not isinstance(text, str):
        raise malformed("response field 'choices[0].message.content' is not a string or null")

    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise malformed("response field 'choices[0].finish_reason' is missing or not a string")

    usage = _parse_usage(body.get("usage"), status, body)
    return ModelResponse(
        text=text,
        model=model,
        finish_reason=finish_reason,
        usage=usage,
        raw=body,
    )


def _parse_usage(value: Any, status: int, body: Mapping[str, Any]) -> Usage | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ModelError(
            ErrorKind.BAD_RESPONSE,
            "response field 'usage' is not an object",
            status=status,
            body=body,
        )
    counts = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        count = value.get(key)
        if isinstance(count, bool) or not isinstance(count, int):
            raise ModelError(
                ErrorKind.BAD_RESPONSE,
                f"response field 'usage.{key}' is missing or not an integer",
                status=status,
                body=body,
            )
        counts.append(count)
    return Usage(*counts)
