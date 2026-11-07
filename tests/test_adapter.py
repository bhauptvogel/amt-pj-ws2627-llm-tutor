"""The real OpenAI-compatible adapter over `httpx.MockTransport`."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from llmtutor.config import ConfigError
from llmtutor.models.api import OpenAICompatibleAdapter, OpenAICompatibleSettings
from llmtutor.types import ErrorKind, Message, ModelError, ModelRequest, Usage

BASE_URL = "https://example.test/api/v1"
BODY = {
    "id": "gen-1",
    "object": "chat.completion",
    "model": "m/1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
        "cost": 0.001,
        "is_byok": False,
    },
}
REQUEST = ModelRequest(
    messages=(Message("system", "be nice"), Message("user", "hello")),
    temperature=0.0,
    max_tokens=16,
    seed=7,
    stop=("END",),
)


def settings(**overrides: object) -> OpenAICompatibleSettings:
    values: dict[str, object] = {
        "base_url": BASE_URL,
        "model": "m/1",
        "api_key_env": "KEY",
        "api_key": "secret",
        "timeout_s": 300.0,
    }
    values.update(overrides)
    return OpenAICompatibleSettings(**values)  # type: ignore[arg-type]


def adapter(handler: Callable[[httpx.Request], httpx.Response]) -> OpenAICompatibleAdapter:
    transport = httpx.MockTransport(handler)
    return OpenAICompatibleAdapter(settings(), client=httpx.Client(transport=transport))


def test_request_contract() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json=BODY)

    adapter(handler).generate(REQUEST)

    sent = seen["request"]
    assert sent.method == "POST"
    assert str(sent.url) == f"{BASE_URL}/chat/completions"
    assert sent.headers["authorization"] == "Bearer secret"
    assert sent.headers["content-type"] == "application/json"
    assert json.loads(sent.content) == {
        "model": "m/1",
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello"},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
        "seed": 7,
        "stop": ["END"],
    }


def test_optional_fields_are_omitted() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json=BODY)

    request = ModelRequest(messages=(Message("user", "hello"),), temperature=0.0)
    adapter(handler).generate(request)

    assert json.loads(seen["request"].content) == {
        "model": "m/1",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.0,
    }


def test_response_maps_to_the_contract() -> None:
    response = adapter(lambda request: httpx.Response(200, json=BODY)).generate(REQUEST)

    assert response.text == "hi"
    assert response.model == "m/1"
    assert response.finish_reason == "stop"
    assert response.usage == Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7)
    assert response.raw == BODY


def test_null_content_stays_none() -> None:
    body = json.loads(json.dumps(BODY))
    body["choices"][0]["message"]["content"] = None

    response = adapter(lambda request: httpx.Response(200, json=body)).generate(REQUEST)

    assert response.text is None


def test_missing_usage_is_none() -> None:
    body = json.loads(json.dumps(BODY))
    del body["usage"]

    response = adapter(lambda request: httpx.Response(200, json=body)).generate(REQUEST)

    assert response.usage is None


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda body: body.pop("model"), "field 'model'"),
        (lambda body: body.update(choices=[]), "field 'choices'"),
        (lambda body: body["choices"][0].pop("message"), "choices[0].message"),
        (lambda body: body["choices"][0]["message"].update(content=1), "content"),
        (lambda body: body["choices"][0].pop("finish_reason"), "finish_reason"),
        (lambda body: body["usage"].pop("total_tokens"), "usage.total_tokens"),
        (lambda body: body.update(usage="lots"), "usage"),
    ],
)
def test_malformed_success_body_is_bad_response(
    mutate: Callable[[dict], object],
    message: str,
) -> None:
    body = json.loads(json.dumps(BODY))
    mutate(body)

    with pytest.raises(ModelError) as info:
        adapter(lambda request: httpx.Response(200, json=body)).generate(REQUEST)

    assert info.value.kind is ErrorKind.BAD_RESPONSE
    assert info.value.status == 200
    assert info.value.retry_after is None
    assert message in str(info.value)


def test_non_json_success_body_is_bad_response() -> None:
    with pytest.raises(ModelError) as info:
        adapter(lambda request: httpx.Response(200, text="not json")).generate(REQUEST)

    assert info.value.kind is ErrorKind.BAD_RESPONSE
    assert info.value.status == 200
    assert info.value.body is None


def test_error_envelope_on_http_200_is_a_provider_failure() -> None:
    envelope = {"error": {"code": 502, "message": "upstream went away"}}

    with pytest.raises(ModelError) as info:
        adapter(lambda request: httpx.Response(200, json=envelope)).generate(REQUEST)

    assert info.value.kind is ErrorKind.PROVIDER
    assert info.value.status == 200
    assert info.value.body == envelope
    assert str(info.value) == "upstream went away"


@pytest.mark.parametrize(
    "status, headers, kind, retry_after",
    [
        (401, {}, ErrorKind.AUTH, None),
        (403, {}, ErrorKind.AUTH, None),
        (402, {}, ErrorKind.PAYMENT, None),
        (404, {}, ErrorKind.NOT_FOUND, None),
        (400, {}, ErrorKind.INVALID_REQUEST, None),
        (422, {}, ErrorKind.INVALID_REQUEST, None),
        (408, {}, ErrorKind.TIMEOUT, None),
        (504, {}, ErrorKind.TIMEOUT, None),
        (429, {"Retry-After": "7"}, ErrorKind.RATE_LIMIT, 7),
        (429, {}, ErrorKind.RATE_LIMIT, None),
        (429, {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}, ErrorKind.RATE_LIMIT, None),
        (500, {}, ErrorKind.PROVIDER, None),
        (503, {}, ErrorKind.PROVIDER, None),
    ],
)
def test_http_failure_table(
    status: int,
    headers: dict[str, str],
    kind: ErrorKind,
    retry_after: int | None,
) -> None:
    envelope = {"error": {"code": status, "message": "nope"}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=envelope, headers=headers)

    with pytest.raises(ModelError) as info:
        adapter(handler).generate(REQUEST)

    assert (info.value.kind, info.value.status, info.value.retry_after, str(info.value)) == (
        kind,
        status,
        retry_after,
        "nope",
    )
    assert info.value.body == envelope


def test_non_json_error_body_keeps_http_status_as_message() -> None:
    with pytest.raises(ModelError) as info:
        adapter(lambda request: httpx.Response(500, text="oops")).generate(REQUEST)

    assert info.value.kind is ErrorKind.PROVIDER
    assert info.value.body is None
    assert str(info.value) == "HTTP 500"


def test_timeout_exception_maps_to_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(ModelError) as info:
        adapter(handler).generate(REQUEST)

    assert info.value.kind is ErrorKind.TIMEOUT
    assert info.value.status is None
    assert str(info.value) == "timed out"


def test_other_transport_error_maps_to_provider() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(ModelError) as info:
        adapter(handler).generate(REQUEST)

    assert info.value.kind is ErrorKind.PROVIDER
    assert info.value.status is None
    assert str(info.value) == "no route"


def test_from_config_builds_settings_and_timeouts() -> None:
    built = OpenAICompatibleAdapter.from_config(
        {
            "base_url": BASE_URL,
            "model": "m/1",
            "api_key_env": "KEY",
        },
        {"KEY": "secret"},
    )

    assert built.timeout_s == 300.0
    assert built._client.timeout.read == 300.0
    assert built._client.timeout.connect == 10.0


def test_from_config_reads_a_custom_timeout() -> None:
    built = OpenAICompatibleAdapter.from_config(
        {"base_url": BASE_URL, "model": "m/1", "api_key_env": "KEY", "timeout_s": 12},
        {"KEY": "secret"},
    )

    assert built.timeout_s == 12.0
    assert built._client.timeout.read == 12.0


@pytest.mark.parametrize(
    "entry, environ, message",
    [
        (
            {"base_url": BASE_URL, "model": "m/1", "api_key_env": "KEY", "extra": 1},
            {"KEY": "k"},
            "unknown",
        ),
        ({"model": "m/1", "api_key_env": "KEY"}, {"KEY": "k"}, "missing required"),
        ({"base_url": BASE_URL, "api_key_env": "KEY"}, {"KEY": "k"}, "missing required"),
        ({"base_url": BASE_URL, "model": "m/1"}, {"KEY": "k"}, "missing required"),
        ({"base_url": BASE_URL, "model": "m/1", "api_key_env": "KEY"}, {}, "KEY is not set"),
        (
            {"base_url": BASE_URL, "model": "m/1", "api_key_env": "KEY"},
            {"KEY": ""},
            "KEY is not set",
        ),
        ({"base_url": "example.test", "model": "m/1", "api_key_env": "KEY"}, {"KEY": "k"}, "http"),
        (
            {"base_url": BASE_URL, "model": "m/1", "api_key_env": "KEY", "timeout_s": 0},
            {"KEY": "k"},
            "positive",
        ),
        (
            {"base_url": BASE_URL, "model": "m/1", "api_key_env": "KEY", "timeout_s": "soon"},
            {"KEY": "k"},
            "must be a number",
        ),
    ],
)
def test_from_config_rejects_bad_settings(
    entry: dict[str, object],
    environ: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=message):
        OpenAICompatibleAdapter.from_config(entry, environ)
