"""Core data types shared by the model layer, the runner and the artefacts.

The vocabulary is the one in `GLOSSARY.md`: a Model turns one `ModelRequest` into one
`ModelResponse`, and failures raise `ModelError` classified by `ErrorKind`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, NamedTuple, Protocol


class Message(NamedTuple):
    """One turn in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """The conversation and sampling settings that make up a single call."""

    messages: tuple[Message, ...]
    temperature: float
    max_tokens: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Usage:
    """The token counts a provider reports, never invented."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One answer together with what the provider reported about how it produced it."""

    text: str | None
    model: str
    finish_reason: str
    usage: Usage | None
    raw: Mapping[str, Any] | None


class ErrorKind(StrEnum):
    """The classification of a failed call that retry and abort policy act on."""

    AUTH = "auth"
    NOT_FOUND = "not_found"
    INVALID_REQUEST = "invalid_request"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    PROVIDER = "provider"
    BAD_RESPONSE = "bad_response"
    PAYMENT = "payment"


class ModelError(Exception):
    """One failed call: kind, optional HTTP status, decoded body and Retry-After."""

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        status: int | None = None,
        body: Mapping[str, Any] | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.body = body
        self.retry_after = retry_after


class Model(Protocol):
    """The seam the runner calls: one conversation in, one answer or `ModelError` out."""

    def generate(self, request: ModelRequest) -> ModelResponse: ...
