"""The model layer: one `Model` protocol, a name-based adapter registry and its adapters."""

from llmtutor.models.api import OpenAICompatibleAdapter
from llmtutor.models.registry import (
    ADAPTERS,
    adapter_names,
    create_model,
    register_adapter,
)

__all__ = [
    "ADAPTERS",
    "OpenAICompatibleAdapter",
    "adapter_names",
    "create_model",
    "register_adapter",
]
