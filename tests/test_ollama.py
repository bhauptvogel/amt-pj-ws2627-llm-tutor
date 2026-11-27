"""The native Ollama adapter over `httpx.MockTransport`."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llmtutor.config import ConfigError
from llmtutor.models.ollama import (
    DEFAULT_BASE_URL,
    DEFAULT_NUM_CTX,
    OllamaAdapter,
    OllamaSettings,
)
from llmtutor.types import ErrorKind, Message, ModelError, ModelRequest, Usage

BASE_URL = "http://daemon.test:11434"
DIGEST = "sha256:" + "a" * 64

VERSION = {"version": "0.40.12"}
TAGS = {
    "models": [
        {
            "name": "qwen3:8b",
            "model": "qwen3:8b",
            "digest": DIGEST,
            "size": 5_200_000_000,
            "details": {"quantization_level": "Q4_K_M"},
        }
    ]
}
SHOW = {
    "details": {
        "format": "gguf",
        "family": "qwen3",
        "families": ["qwen3"],
        "parameter_size": "8.2B",
        "quantization_level": "Q4_K_M",
    },
    "modified_at": "2026-11-01T10:00:00Z",
    "model_info": {},
    "parameters": "",
}
CHAT: dict[str, Any] = {
    "model": "qwen3:8b",
    "created_at": "2026-11-27T12:25:47.047909Z",
    "message": {"role": "assistant", "content": "hi"},
    "done": True,
    "done_reason": "stop",
    "total_duration": 481173125,
    "load_duration": 6175708,
    "prompt_eval_count": 11,
    "prompt_eval_cached_count": 0,
    "prompt_eval_duration": 72360000,
    "eval_count": 31,
    "eval_duration": 397238000,
}
PROVENANCE = {
    "ollama_version": "0.40.12",
    "digest": DIGEST,
    "quantization_level": "Q4_K_M",
    "parameter_size": "8.2B",
    "family": "qwen3",
    "modified_at": "2026-11-01T10:00:00Z",
}
REQUEST = ModelRequest(
    messages=(Message("system", "be nice"), Message("user", "hello")),
    temperature=0.0,
    max_tokens=16,
    seed=7,
    stop=("END",),
)


def ok(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=body)


def chat(**overrides: object) -> dict[str, Any]:
    body = json.loads(json.dumps(CHAT))
    body.update(overrides)
    return body


def build(
    overrides: dict[str, httpx.Response] | None = None,
    **settings: object,
) -> tuple[OllamaAdapter, list[httpx.Request]]:
    table: dict[str, httpx.Response] = {
        "/api/version": ok(VERSION),
        "/api/tags": ok(TAGS),
        "/api/show": ok(SHOW),
        "/api/chat": ok(CHAT),
    }
    table.update(overrides or {})
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        entry = table.get(request.url.path)
        if isinstance(entry, Exception):
            raise entry
        if entry is None:
            return httpx.Response(404, json={"error": f"no route for {request.url.path}"})
        return entry

    values: dict[str, object] = {
        "base_url": BASE_URL,
        "model": "qwen3:8b",
        "num_ctx": 8192,
        "think": False,
        "keep_alive": "30m",
        "timeout_s": 300.0,
    }
    values.update(settings)
    adapter = OllamaAdapter(
        OllamaSettings(**values),  # type: ignore[arg-type]
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return adapter, seen


def paths(seen: list[httpx.Request]) -> list[str]:
    return [request.url.path for request in seen]


def test_request_contract() -> None:
    adapter, seen = build()

    adapter.generate(REQUEST)

    sent = next(request for request in seen if request.url.path == "/api/chat")
    assert sent.method == "POST"
    assert str(sent.url) == f"{BASE_URL}/api/chat"
    assert sent.headers["content-type"] == "application/json"
    assert json.loads(sent.content) == {
        "model": "qwen3:8b",
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello"},
        ],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "num_ctx": 8192,
            "temperature": 0.0,
            "num_predict": 16,
            "seed": 7,
            "stop": ["END"],
        },
    }


def test_think_and_keep_alive_are_sent_as_configured() -> None:
    adapter, seen = build(think=True, keep_alive="5m")

    adapter.generate(REQUEST)

    sent = json.loads(next(r for r in seen if r.url.path == "/api/chat").content)
    assert sent["think"] is True
    assert sent["keep_alive"] == "5m"


def test_optional_fields_are_omitted() -> None:
    adapter, seen = build()
    request = ModelRequest(messages=(Message("user", "hello"),), temperature=0.0)

    adapter.generate(request)

    sent = json.loads(next(r for r in seen if r.url.path == "/api/chat").content)
    assert sent["options"] == {"num_ctx": 8192, "temperature": 0.0}


def test_response_maps_to_the_contract() -> None:
    adapter, _ = build()

    response = adapter.generate(REQUEST)

    assert response.text == "hi"
    assert response.model == "qwen3:8b"
    assert response.finish_reason == "stop"
    assert response.usage == Usage(prompt_tokens=11, completion_tokens=31, total_tokens=42)
    assert response.raw == CHAT


def test_thinking_stays_in_the_raw_payload_and_out_of_the_text() -> None:
    body = chat(message={"role": "assistant", "content": "hi", "thinking": "let me think"})
    adapter, _ = build({"/api/chat": ok(body)})

    response = adapter.generate(REQUEST)

    assert response.text == "hi"
    assert response.raw["message"]["thinking"] == "let me think"


def test_a_truncated_prompt_is_recorded_rather_than_rejected() -> None:
    """Ollama cuts a too-long prompt to about half the window and reports what it evaluated.

    The count is kept as evidence next to `num_ctx` in the manifest; nothing blocks the call,
    so `prompt_eval_count` at the half-window mark is how a truncated run is found afterwards.
    """

    body = chat(prompt_eval_count=19)
    adapter, _ = build({"/api/chat": ok(body)}, num_ctx=32)

    response = adapter.generate(REQUEST)

    assert response.usage == Usage(prompt_tokens=19, completion_tokens=31, total_tokens=50)
    assert response.raw["prompt_eval_count"] == 19


def test_missing_usage_is_none() -> None:
    body = chat()
    del body["prompt_eval_count"]
    del body["eval_count"]
    adapter, _ = build({"/api/chat": ok(body)})

    assert adapter.generate(REQUEST).usage is None


def test_provenance_is_none_until_the_first_call() -> None:
    adapter, _ = build()

    assert adapter.provenance() is None


def test_provenance_reports_what_the_daemon_attested() -> None:
    adapter, _ = build()

    adapter.generate(REQUEST)

    assert adapter.provenance() == PROVENANCE


def test_provenance_is_probed_once_per_adapter() -> None:
    adapter, seen = build()

    adapter.generate(REQUEST)
    adapter.generate(REQUEST)

    assert paths(seen).count("/api/version") == 1
    assert paths(seen).count("/api/tags") == 1
    assert paths(seen).count("/api/show") == 1
    assert paths(seen).count("/api/chat") == 2


def test_provenance_tolerates_a_model_missing_from_the_tags_list() -> None:
    tags = {"models": [{"name": "other:1b", "digest": DIGEST}]}
    adapter, _ = build({"/api/tags": ok(tags)})

    adapter.generate(REQUEST)

    assert adapter.provenance()["digest"] is None
    assert adapter.provenance()["ollama_version"] == "0.40.12"


def test_a_model_the_daemon_lacks_fails_before_any_generation() -> None:
    missing = httpx.Response(
        404, json={"error": "model 'qwen3:8b' not found, try pulling it first"}
    )
    adapter, seen = build({"/api/show": missing})

    with pytest.raises(ModelError) as info:
        adapter.generate(REQUEST)

    assert info.value.kind is ErrorKind.NOT_FOUND
    assert str(info.value) == "model 'qwen3:8b' not found, try pulling it first"
    assert "/api/chat" not in paths(seen)
    assert adapter.provenance() is None


def test_error_envelope_on_http_200_is_a_provider_failure() -> None:
    envelope = {"error": "runner crashed"}
    adapter, _ = build({"/api/chat": ok(envelope)})

    with pytest.raises(ModelError) as info:
        adapter.generate(REQUEST)

    assert info.value.kind is ErrorKind.PROVIDER
    assert str(info.value) == "runner crashed"
    assert info.value.body == envelope


@pytest.mark.parametrize(
    "status, kind",
    [
        (401, ErrorKind.AUTH),
        (403, ErrorKind.AUTH),
        (404, ErrorKind.NOT_FOUND),
        (400, ErrorKind.INVALID_REQUEST),
        (408, ErrorKind.TIMEOUT),
        (429, ErrorKind.RATE_LIMIT),
        (500, ErrorKind.PROVIDER),
        (503, ErrorKind.PROVIDER),
    ],
)
def test_http_failure_table(status: int, kind: ErrorKind) -> None:
    adapter, _ = build({"/api/chat": httpx.Response(status, json={"error": "nope"})})

    with pytest.raises(ModelError) as info:
        adapter.generate(REQUEST)

    assert (info.value.kind, info.value.status, str(info.value)) == (kind, status, "nope")
    assert info.value.body == {"error": "nope"}
    assert info.value.retry_after is None


def test_non_json_error_body_keeps_http_status_as_message() -> None:
    adapter, _ = build({"/api/chat": httpx.Response(500, text="oops")})

    with pytest.raises(ModelError) as info:
        adapter.generate(REQUEST)

    assert info.value.kind is ErrorKind.PROVIDER
    assert info.value.body is None
    assert str(info.value) == "HTTP 500"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda body: body.pop("model"), "field 'model'"),
        (lambda body: body.update(done=False), "field 'done'"),
        (lambda body: body.pop("message"), "field 'message'"),
        (lambda body: body["message"].update(content=1), "message.content"),
        (lambda body: body.pop("done_reason"), "field 'done_reason'"),
        (lambda body: body.update(eval_count="lots"), "eval_count"),
        (lambda body: body.update(prompt_eval_count=-1), "prompt_eval_count"),
    ],
)
def test_malformed_success_body_is_bad_response(mutate: Any, message: str) -> None:
    body = chat()
    mutate(body)
    adapter, _ = build({"/api/chat": ok(body)})

    with pytest.raises(ModelError) as info:
        adapter.generate(REQUEST)

    assert info.value.kind is ErrorKind.BAD_RESPONSE
    assert info.value.status == 200
    assert message in str(info.value)


def test_non_json_success_body_is_bad_response() -> None:
    adapter, _ = build({"/api/chat": httpx.Response(200, text="not json")})

    with pytest.raises(ModelError) as info:
        adapter.generate(REQUEST)

    assert info.value.kind is ErrorKind.BAD_RESPONSE
    assert info.value.body is None


def test_timeout_exception_maps_to_timeout() -> None:
    adapter, _ = build({"/api/chat": httpx.ReadTimeout("timed out")})

    with pytest.raises(ModelError) as info:
        adapter.generate(REQUEST)

    assert info.value.kind is ErrorKind.TIMEOUT
    assert info.value.status is None
    assert str(info.value) == "timed out"


def test_unreachable_daemon_maps_to_provider() -> None:
    adapter, _ = build({"/api/chat": httpx.ConnectError("no route")})

    with pytest.raises(ModelError) as info:
        adapter.generate(REQUEST)

    assert info.value.kind is ErrorKind.PROVIDER
    assert str(info.value) == "no route"


def test_from_config_applies_the_documented_defaults() -> None:
    built = OllamaAdapter.from_config({"model": "qwen3:8b", "num_ctx": 8192}, {})

    assert (built.timeout_s, built._client.timeout.read) == (300.0, 300.0)
    assert built._client.timeout.connect == 10.0
    assert built.base_url == DEFAULT_BASE_URL
    assert built._settings.num_ctx == 8192
    assert built._settings.think is False
    assert built._settings.keep_alive == "30m"
    assert built.provenance() is None


def test_direct_construction_falls_back_to_the_default_window() -> None:
    """A script that builds settings by hand still gets a pinned window, not the daemon's."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        table = {
            "/api/version": ok(VERSION),
            "/api/tags": ok(TAGS),
            "/api/show": ok(SHOW),
            "/api/chat": ok(CHAT),
        }
        return table[request.url.path]

    adapter = OllamaAdapter(
        OllamaSettings(base_url=BASE_URL, model="qwen3:8b"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    adapter.generate(REQUEST)

    sent = json.loads(next(r for r in seen if r.url.path == "/api/chat").content)
    assert sent["options"]["num_ctx"] == DEFAULT_NUM_CTX
    assert sent["keep_alive"] == "30m"


def test_from_config_reads_every_setting() -> None:
    built = OllamaAdapter.from_config(
        {
            "base_url": BASE_URL,
            "model": "qwen3:8b",
            "num_ctx": 16384,
            "think": True,
            "keep_alive": "5m",
            "timeout_s": 12,
        },
        {},
    )

    assert built.timeout_s == 12.0
    assert built._client.timeout.read == 12.0
    assert built._settings.num_ctx == 16384
    assert built._settings.think is True
    assert built._settings.keep_alive == "5m"


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"model": "qwen3:8b", "num_ctx": 8192, "extra": 1}, "unknown"),
        ({"model": "qwen3:8b"}, "missing required"),
        ({"num_ctx": 8192}, "missing required"),
        ({"model": "qwen3:cloud", "num_ctx": 8192}, "local weights"),
        ({"model": "", "num_ctx": 8192}, "non-empty string"),
        ({"model": "qwen3:8b", "num_ctx": 0}, "positive integer"),
        ({"model": "qwen3:8b", "num_ctx": "big"}, "positive integer"),
        ({"model": "qwen3:8b", "num_ctx": True}, "positive integer"),
        ({"model": "qwen3:8b", "num_ctx": 8192, "think": "yes"}, "boolean"),
        ({"model": "qwen3:8b", "num_ctx": 8192, "keep_alive": ""}, "keep_alive"),
        ({"model": "qwen3:8b", "num_ctx": 8192, "base_url": "daemon.test"}, "http"),
        ({"model": "qwen3:8b", "num_ctx": 8192, "timeout_s": 0}, "positive"),
        ({"model": "qwen3:8b", "num_ctx": 8192, "timeout_s": "soon"}, "number"),
    ],
)
def test_from_config_rejects_bad_settings(entry: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        OllamaAdapter.from_config(entry, {})
