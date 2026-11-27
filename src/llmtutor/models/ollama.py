"""The native Ollama chat adapter: local weights, a pinned context window, attested provenance.

The OpenAI-compatible endpoint Ollama also serves cannot express a context size, and a run
that silently truncates a long prompt is a wrong result rather than a slow one. This adapter
speaks `/api/chat` for that reason, and because it reports the token counts the runtime
actually evaluated.

A model entry in a run config must state `num_ctx`: a run should never inherit whatever window
the daemon happens to pick. Constructing the adapter directly in a script may omit it and gets
`DEFAULT_NUM_CTX`, which is what the pilot scripts did before the setting existed.

Truncation is recorded, not blocked. The daemon cuts a prompt that does not fit down to
roughly half the window and reports the tokens it really evaluated, so a truncated call is
visible in its call record (`prompt_eval_count` in the raw payload) against the `num_ctx`
recorded in the manifest. Measured twice, because the offset is the model's chat template and
not a constant: a 451-token prompt on gemma3:1b against num_ctx 32, 64, 128 and 256 reported
19, 35, 67 and 131 prompt tokens; a 418-token prompt on qwen3:8b reported 18, 34, 66 and 130.
Both were independent of `num_predict`, which is why `prompt_eval_count >= num_ctx` is not a
usable signal. Re-measure before leaning on the half-window shape for another model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from llmtutor.config import ConfigError
from llmtutor.models.registry import register_adapter
from llmtutor.types import ErrorKind, ModelError, ModelRequest, ModelResponse, Usage

ALLOWED_SETTINGS = frozenset({"base_url", "model", "num_ctx", "think", "keep_alive", "timeout_s"})
REQUIRED_SETTINGS = ("model", "num_ctx")
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_NUM_CTX = 8192
DEFAULT_KEEP_ALIVE = "30m"
DEFAULT_TIMEOUT_S = 300.0
CONNECT_TIMEOUT_S = 10.0
CLOUD_SUFFIX = ":cloud"


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    """The typed parameters one model entry needs to reach a local daemon."""

    base_url: str
    model: str
    num_ctx: int = DEFAULT_NUM_CTX
    think: bool = False
    keep_alive: str = DEFAULT_KEEP_ALIVE
    timeout_s: float = DEFAULT_TIMEOUT_S


@register_adapter("ollama")
class OllamaAdapter:
    """A Model over `/api/chat`, translating our request into Ollama's options."""

    def __init__(
        self,
        settings: OllamaSettings,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(settings.timeout_s, connect=CONNECT_TIMEOUT_S)
        )
        self._provenance: dict[str, Any] | None = None

    @property
    def timeout_s(self) -> float:
        return self._settings.timeout_s

    @property
    def base_url(self) -> str:
        """The effective URL, so a defaulted daemon address still reaches the manifest."""

        return self._settings.base_url

    def provenance(self) -> Mapping[str, Any] | None:
        """What the daemon attested about the weights, once it has been asked."""

        return self._provenance

    @classmethod
    def from_config(
        cls,
        entry: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> OllamaAdapter:
        # `environ` is unused on purpose: a local daemon takes no key, which is why this
        # adapter has no `api_key_env` setting to get wrong.
        del environ

        unknown = sorted(set(entry) - ALLOWED_SETTINGS)
        if unknown:
            raise ConfigError(f"unknown setting(s) for adapter 'ollama': {', '.join(unknown)}")
        missing = [key for key in REQUIRED_SETTINGS if key not in entry]
        if missing:
            raise ConfigError(
                f"missing required setting(s) for adapter 'ollama': {', '.join(missing)}"
            )

        model = entry["model"]
        if not isinstance(model, str) or not model:
            raise ConfigError("setting 'model' must be a non-empty string")
        if model.endswith(CLOUD_SUFFIX):
            raise ConfigError(
                f"setting 'model' must name local weights, got {model!r}: ':cloud' models run "
                "on Ollama's servers, so the manifest would claim localhost falsely"
            )

        base_url = entry.get("base_url", DEFAULT_BASE_URL)
        if not isinstance(base_url, str) or not base_url:
            raise ConfigError("setting 'base_url' must be a non-empty string")
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError(f"setting 'base_url' must be an http(s) URL, got {base_url!r}")

        num_ctx = entry["num_ctx"]
        if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx < 1:
            raise ConfigError("setting 'num_ctx' must be a positive integer")

        think = entry.get("think", False)
        if not isinstance(think, bool):
            raise ConfigError("setting 'think' must be a boolean")

        keep_alive = entry.get("keep_alive", DEFAULT_KEEP_ALIVE)
        if not isinstance(keep_alive, str) or not keep_alive:
            raise ConfigError("setting 'keep_alive' must be a non-empty string such as '30m'")

        timeout_s = entry.get("timeout_s", DEFAULT_TIMEOUT_S)
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ConfigError("setting 'timeout_s' must be a number")
        if timeout_s <= 0:
            raise ConfigError("setting 'timeout_s' must be positive")

        return cls(
            OllamaSettings(
                base_url=base_url,
                model=model,
                num_ctx=num_ctx,
                think=think,
                keep_alive=keep_alive,
                timeout_s=float(timeout_s),
            )
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._probe()
        status, body = self._send("POST", "/api/chat", self._chat_payload(request))
        if "error" in body and "message" not in body:
            raise ModelError(
                ErrorKind.PROVIDER,
                _error_message(body) or "provider error",
                status=status,
                body=body,
            )
        return _parse_response(body, status)

    def _chat_payload(self, request: ModelRequest) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_ctx": self._settings.num_ctx,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.seed is not None:
            options["seed"] = request.seed
        if request.stop:
            options["stop"] = list(request.stop)

        return {
            "model": self._settings.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "stream": False,
            "think": self._settings.think,
            "keep_alive": self._settings.keep_alive,
            "options": options,
        }

    def _probe(self) -> None:
        """Resolve the weights once: the daemon's version, digest and quantisation.

        Missing weights fail here, as `NOT_FOUND`, before a generation is paid for.
        """

        if self._provenance is not None:
            return

        _, version = self._send("GET", "/api/version")
        _, tags = self._send("GET", "/api/tags")
        _, show = self._send("POST", "/api/show", {"model": self._settings.model})
        details = show.get("details")
        details = details if isinstance(details, dict) else {}

        self._provenance = {
            "ollama_version": _str_or_none(version.get("version")),
            "digest": _find_digest(tags, self._settings.model),
            "quantization_level": _str_or_none(details.get("quantization_level")),
            "parameter_size": _str_or_none(details.get("parameter_size")),
            "family": _str_or_none(details.get("family")),
            "modified_at": _str_or_none(show.get("modified_at")),
        }

    def _send(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> tuple[int, Mapping[str, Any]]:
        url = f"{self._settings.base_url.rstrip('/')}{path}"
        try:
            response = self._client.request(method, url, json=payload)
        except httpx.TimeoutException as exc:
            raise ModelError(ErrorKind.TIMEOUT, str(exc) or "request timed out") from exc
        except httpx.RequestError as exc:
            raise ModelError(ErrorKind.PROVIDER, str(exc)) from exc

        status = response.status_code
        body = _decode_body(response)

        if not response.is_success:
            raise ModelError(
                _status_kind(status),
                _error_message(body) or f"HTTP {status}",
                status=status,
                body=body,
            )
        if body is None:
            raise ModelError(
                ErrorKind.BAD_RESPONSE,
                "response body is not a JSON object",
                status=status,
            )
        return status, body


def _decode_body(response: httpx.Response) -> Mapping[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _error_message(body: Mapping[str, Any] | None) -> str | None:
    """Ollama reports `{"error": "text"}`, OpenAI-style payloads a nested message."""

    if body is None:
        return None
    error = body.get("error")
    if isinstance(error, str) and error:
        return error
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    return None


def _status_kind(status: int) -> ErrorKind:
    if status in (401, 403):
        return ErrorKind.AUTH
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


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _find_digest(tags: Mapping[str, Any], model: str) -> str | None:
    entries = tags.get("models")
    if not isinstance(entries, list):
        return None
    wanted = {model, f"{model}:latest"}
    for item in entries:
        if isinstance(item, dict) and item.get("name") in wanted:
            return _str_or_none(item.get("digest"))
    return None


def _parse_response(body: Mapping[str, Any], status: int) -> ModelResponse:
    def malformed(reason: str) -> ModelError:
        return ModelError(ErrorKind.BAD_RESPONSE, reason, status=status, body=body)

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise malformed("response field 'model' is missing or not a string")

    if body.get("done") is not True:
        raise malformed("response field 'done' is not true; streaming replies are not supported")

    message = body.get("message")
    if not isinstance(message, dict):
        raise malformed("response field 'message' is missing")
    text = message.get("content")
    if text is not None and not isinstance(text, str):
        raise malformed("response field 'message.content' is not a string or null")

    finish_reason = body.get("done_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise malformed("response field 'done_reason' is missing or not a string")

    return ModelResponse(
        text=text,
        model=model,
        finish_reason=finish_reason,
        usage=_parse_usage(body, status),
        raw=body,
    )


def _parse_usage(body: Mapping[str, Any], status: int) -> Usage | None:
    prompt = body.get("prompt_eval_count")
    completion = body.get("eval_count")
    if prompt is None and completion is None:
        return None

    for key, count in (("prompt_eval_count", prompt), ("eval_count", completion)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ModelError(
                ErrorKind.BAD_RESPONSE,
                f"response field {key!r} is missing or not a non-negative integer",
                status=status,
                body=body,
            )
    return Usage(prompt, completion, prompt + completion)
