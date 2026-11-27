"""The generic registry and the adapter registry facade."""

import pytest

from llmtutor.config import ConfigError
from llmtutor.models import adapter_names, create_model
from llmtutor.registry import Registry, RegistryError


class Thing:
    pass


def test_register_get_and_names() -> None:
    registry: Registry[Thing] = Registry("thing")
    registry.register("a", Thing)
    assert registry.get("a") is Thing
    assert registry.names() == ("a",)


def test_duplicate_name_fails() -> None:
    registry: Registry[Thing] = Registry("thing")
    registry.register("a", Thing)
    with pytest.raises(RegistryError, match="duplicate thing name 'a'"):
        registry.register("a", Thing)


def test_unknown_name_lists_registered_names() -> None:
    registry: Registry[Thing] = Registry("thing")
    registry.register("a", Thing)
    with pytest.raises(RegistryError, match="unknown thing 'b'; registered: a"):
        registry.get("b")


def test_unknown_name_in_empty_registry() -> None:
    registry: Registry[Thing] = Registry("thing")
    with pytest.raises(RegistryError, match=r"registered: \(none\)"):
        registry.get("b")


def test_builtin_and_fake_adapters_are_registered() -> None:
    assert adapter_names() == ("fake", "ollama", "openai_compatible")


def test_create_model_unknown_adapter() -> None:
    with pytest.raises(RegistryError, match="unknown adapter 'nope'"):
        create_model("nope", {}, {})


def test_create_model_rejects_unknown_settings() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        create_model("fake", {"scenario": "ok", "extra": 1}, {})
