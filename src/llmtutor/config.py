"""Strict loading and resolution of one run config into a run plan."""

from __future__ import annotations

import hashlib
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmtutor.types import Message, Model

SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
TOP_LEVEL_KEYS = frozenset({"prompt", "sampling", "dataset", "models"})
PROMPT_KEYS = frozenset({"id", "version", "system", "user"})
SAMPLING_KEYS = frozenset({"temperature", "max_tokens", "seed", "stop"})
DATASET_KEYS = frozenset({"path"})


class ConfigError(Exception):
    """The run config is malformed or cannot be resolved into a run plan."""


@dataclass(frozen=True, slots=True)
class Prompt:
    id: str
    version: int
    system: str | None
    user: str

    def messages(self) -> tuple[Message, ...]:
        messages: list[Message] = []
        if self.system is not None:
            messages.append(Message("system", self.system))
        messages.append(Message("user", self.user))
        return tuple(messages)


@dataclass(frozen=True, slots=True)
class Sampling:
    temperature: float = 0.0
    max_tokens: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Dataset:
    """The dataset file named by a run config, read once and hashed at load time.

    Its item schema belongs to the task layer; the core only guarantees which bytes a
    run was pointed at. A task loader reads `raw` rather than the path, so it parses
    exactly the bytes `sha256` covers.
    """

    path: Path
    raw: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelEntry:
    name: str
    adapter: str
    settings: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    entry: ModelEntry
    model: Model


@dataclass(frozen=True, slots=True)
class RunPlan:
    path: Path
    raw: bytes
    sha256: str
    prompt: Prompt
    sampling: Sampling
    dataset: Dataset | None
    models: tuple[ResolvedModel, ...]


def load_plan(path: Path, environ: Mapping[str, str]) -> RunPlan:
    """Read, validate and resolve a config before any run artefact exists."""

    raw = path.read_bytes()
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{path}: config is not valid UTF-8: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc

    if not isinstance(document, dict):
        raise ConfigError(f"{path}: config must be a TOML table")

    unknown = sorted(set(document) - TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"{path}: unknown top-level key(s): {', '.join(unknown)}")

    prompt = _parse_prompt(document.get("prompt"), path)
    sampling = _parse_sampling(document.get("sampling"), path)
    dataset = _parse_dataset(document.get("dataset"), path)
    entries = _parse_models(document.get("models"), path)

    from llmtutor.models.registry import create_model
    from llmtutor.registry import RegistryError

    resolved: list[ResolvedModel] = []
    for entry in entries:
        try:
            model = create_model(entry.adapter, entry.settings, environ)
        except (ConfigError, RegistryError) as exc:
            raise ConfigError(f"{path}: model entry {entry.name!r}: {exc}") from exc
        resolved.append(ResolvedModel(entry=entry, model=model))

    return RunPlan(
        path=path,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        prompt=prompt,
        sampling=sampling,
        dataset=dataset,
        models=tuple(resolved),
    )


def _parse_prompt(value: Any, path: Path) -> Prompt:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: [prompt] table is required")
    _reject_unknown(value, PROMPT_KEYS, f"{path}: [prompt]")

    prompt_id = _require_str(value, "id", f"{path}: [prompt]")
    version = _require_positive_int(value, "version", f"{path}: [prompt]")
    user = _require_str(value, "user", f"{path}: [prompt]")

    system = value.get("system")
    if system is not None and not isinstance(system, str):
        raise ConfigError(f"{path}: [prompt] key 'system' must be a string")

    return Prompt(id=prompt_id, version=version, system=system, user=user)


def _parse_sampling(value: Any, path: Path) -> Sampling:
    if value is None:
        return Sampling()
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: [sampling] must be a table")
    _reject_unknown(value, SAMPLING_KEYS, f"{path}: [sampling]")

    temperature = value.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ConfigError(f"{path}: [sampling] key 'temperature' must be a number")
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError(f"{path}: [sampling] key 'temperature' must be between 0 and 2")

    max_tokens = value.get("max_tokens")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ConfigError(f"{path}: [sampling] key 'max_tokens' must be an integer")
        if max_tokens < 1:
            raise ConfigError(f"{path}: [sampling] key 'max_tokens' must be positive")

    seed = value.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ConfigError(f"{path}: [sampling] key 'seed' must be an integer")

    stop = _parse_stop(value.get("stop"), path)
    return Sampling(
        temperature=float(temperature),
        max_tokens=max_tokens,
        seed=seed,
        stop=stop,
    )


def _parse_dataset(value: Any, path: Path) -> Dataset | None:
    """Read the optional [dataset] block; item shape and loading belong to the task layer."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: [dataset] must be a table")
    _reject_unknown(value, DATASET_KEYS, f"{path}: [dataset]")

    dataset_path = Path(_require_str(value, "path", f"{path}: [dataset]"))
    try:
        raw = dataset_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path}: cannot read dataset {dataset_path}: {exc}") from exc
    if not raw:
        raise ConfigError(f"{path}: dataset {dataset_path} is empty")
    return Dataset(
        path=dataset_path,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _parse_stop(value: Any, path: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{path}: [sampling] key 'stop' must be a string or a list of strings")
    return tuple(value)


def _parse_models(value: Any, path: Path) -> list[ModelEntry]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path}: at least one [[models]] entry is required")

    entries: list[ModelEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(value, start=1):
        where = f"{path}: [[models]] entry {index}"
        if not isinstance(item, dict):
            raise ConfigError(f"{where} must be a table")

        name = _require_str(item, "name", where)
        if not SLUG.fullmatch(name):
            raise ConfigError(f"{where}: 'name' must match {SLUG.pattern}, got {name!r}")
        if name in seen:
            raise ConfigError(f"{where}: duplicate model entry name {name!r}")
        seen.add(name)

        adapter = _require_str(item, "adapter", where)
        settings = {key: value for key, value in item.items() if key not in ("name", "adapter")}
        entries.append(ModelEntry(name=name, adapter=adapter, settings=settings))

    return entries


def _reject_unknown(value: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(unknown)}")


def _require_str(value: Mapping[str, Any], key: str, where: str) -> str:
    if key not in value:
        raise ConfigError(f"{where}: missing required key {key!r}")
    result = value[key]
    if not isinstance(result, str) or not result:
        raise ConfigError(f"{where}: key {key!r} must be a non-empty string")
    return result


def _require_positive_int(value: Mapping[str, Any], key: str, where: str) -> int:
    if key not in value:
        raise ConfigError(f"{where}: missing required key {key!r}")
    result = value[key]
    if isinstance(result, bool) or not isinstance(result, int) or result < 1:
        raise ConfigError(f"{where}: key {key!r} must be a positive integer")
    return result
