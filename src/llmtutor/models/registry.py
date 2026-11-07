"""Adapter selection by name: registration, lookup and construction from config."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, Self

from llmtutor.registry import Registry
from llmtutor.types import Model

ADAPTERS: Registry[Model] = Registry("adapter")


class FromConfig(Protocol):
    """The builder contract every adapter implements."""

    @classmethod
    def from_config(
        cls,
        entry: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> Self: ...


def register_adapter(name: str) -> Callable[[type[FromConfig]], type[FromConfig]]:
    """Register an adapter class under a config name."""

    def decorator(cls: type[FromConfig]) -> type[FromConfig]:
        ADAPTERS.register(name, cls)  # type: ignore[arg-type]
        return cls

    return decorator


def adapter_names() -> tuple[str, ...]:
    return ADAPTERS.names()


def create_model(
    adapter: str,
    entry: Mapping[str, Any],
    environ: Mapping[str, str],
) -> Model:
    """Build one model instance from its entry settings and the injected environment."""

    cls = ADAPTERS.get(adapter)
    return cls.from_config(entry, environ)
