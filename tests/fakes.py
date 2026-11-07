"""A scripted `Model` implementation, registered as the `fake` adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llmtutor.config import ConfigError
from llmtutor.models import register_adapter
from llmtutor.types import ErrorKind, ModelError, ModelRequest, ModelResponse, Usage

OK = ModelResponse(
    text="hello",
    model="fake-model",
    finish_reason="stop",
    usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    raw={"id": "fake-1"},
)

RATE_LIMIT = ModelError(ErrorKind.RATE_LIMIT, "slow down", status=429)

SCENARIOS: dict[str, tuple[ModelResponse | BaseException, ...]] = {
    "ok": (OK,),
    "no_content": (
        ModelResponse(
            text=None,
            model="fake-model",
            finish_reason="length",
            usage=None,
            raw=None,
        ),
    ),
    "retry_then_ok": (RATE_LIMIT, OK),
    "rate_limit_twice_then_ok": (RATE_LIMIT, RATE_LIMIT, OK),
    "retry_after_5": (ModelError(ErrorKind.RATE_LIMIT, "slow down", status=429, retry_after=5), OK),
    "retry_after_120": (
        ModelError(ErrorKind.RATE_LIMIT, "slow down", status=429, retry_after=120),
        OK,
    ),
    "auth": (ModelError(ErrorKind.AUTH, "bad key", status=401),),
    "provider": (ModelError(ErrorKind.PROVIDER, "overloaded", status=503),) * 3,
    "contract_bug": (AssertionError("boom"),),
    "interrupt": (KeyboardInterrupt(),),
}

SETTINGS = frozenset({"scenario"})


@register_adapter("fake")
class FakeAdapter:
    """Plays one registered scenario, one outcome per attempt."""

    @classmethod
    def from_config(
        cls,
        entry: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> FakeAdapter:
        unknown = sorted(set(entry) - SETTINGS)
        if unknown:
            raise ConfigError(f"unknown key(s) for adapter 'fake': {', '.join(unknown)}")
        if "scenario" not in entry:
            raise ConfigError("missing required setting for adapter 'fake': scenario")
        scenario = entry["scenario"]
        if not isinstance(scenario, str) or scenario not in SCENARIOS:
            names = ", ".join(sorted(SCENARIOS))
            raise ConfigError(f"unknown scenario {scenario!r}; known: {names}")
        return cls(scenario)

    def __init__(self, scenario: str) -> None:
        self._scenario = scenario
        self._cursor = 0
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        outcomes = SCENARIOS[self._scenario]
        if self._cursor >= len(outcomes):
            raise AssertionError(f"scenario {self._scenario!r} exhausted")
        outcome = outcomes[self._cursor]
        self._cursor += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
