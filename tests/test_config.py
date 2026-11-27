"""Strict run-config parsing and plan resolution."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from llmtutor.config import ConfigError, load_plan

VALID = """\
[prompt]
id = "p"
version = 2
system = "be nice"
user = "hello"

[sampling]
temperature = 0.5
max_tokens = 16
seed = 7
stop = ["END"]

[[models]]
name = "a"
adapter = "fake"
scenario = "ok"
"""


def write(tmp_path: Path, text: str = VALID, name: str = "smoke.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_plan_resolves_prompt_sampling_and_models(tmp_path: Path) -> None:
    path = write(tmp_path)

    plan = load_plan(path, {})

    assert plan.path == path
    assert plan.raw == VALID.encode()
    assert plan.sha256 == hashlib.sha256(VALID.encode()).hexdigest()
    assert plan.prompt.id == "p"
    assert plan.prompt.version == 2
    assert [message.role for message in plan.prompt.messages()] == ["system", "user"]
    assert plan.prompt.messages()[0].content == "be nice"
    assert plan.sampling.temperature == 0.5
    assert plan.sampling.max_tokens == 16
    assert plan.sampling.seed == 7
    assert plan.sampling.stop == ("END",)
    assert [resolved.entry.name for resolved in plan.models] == ["a"]
    assert plan.models[0].entry.adapter == "fake"
    assert plan.models[0].entry.settings == {"scenario": "ok"}


def test_sampling_defaults_and_missing_system(tmp_path: Path) -> None:
    text = """\
[prompt]
id = "p"
version = 1
user = "hello"

[[models]]
name = "a"
adapter = "fake"
scenario = "ok"
"""
    plan = load_plan(write(tmp_path, text), {})

    assert plan.sampling.temperature == 0.0
    assert plan.sampling.max_tokens is None
    assert plan.sampling.seed is None
    assert plan.sampling.stop == ()
    assert plan.prompt.messages() == (("user", "hello"),)


def test_stop_may_be_a_single_string(tmp_path: Path) -> None:
    text = VALID.replace('stop = ["END"]', 'stop = "END"')
    assert load_plan(write(tmp_path, text), {}).sampling.stop == ("END",)


def test_dataset_is_optional(tmp_path: Path) -> None:
    assert load_plan(write(tmp_path), {}).dataset is None


def test_dataset_is_read_and_hashed(tmp_path: Path) -> None:
    items = tmp_path / "items_v1.jsonl"
    body = b'{"item_id": "fb-001"}\n'
    items.write_bytes(body)

    plan = load_plan(write(tmp_path, VALID + f'\n[dataset]\npath = "{items}"\n'), {})

    assert plan.dataset is not None
    assert plan.dataset.path == items
    assert plan.dataset.raw == body
    assert plan.dataset.sha256 == hashlib.sha256(body).hexdigest()


@pytest.mark.parametrize(
    "text, message",
    [
        (
            "[prompt]\nid='p'\nversion=1\nuser='u'\n[dataset]\n",
            "missing required key 'path'",
        ),
        (
            "[prompt]\nid='p'\nversion=1\nuser='u'\n[dataset]\npath=''\n",
            "non-empty string",
        ),
        (
            "[prompt]\nid='p'\nversion=1\nuser='u'\n[dataset]\nname='x'\n",
            "unknown key\\(s\\)",
        ),
        (
            "dataset='x'\n[prompt]\nid='p'\nversion=1\nuser='u'\n",
            "must be a table",
        ),
    ],
)
def test_invalid_dataset_blocks_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_plan(write(tmp_path, text), {})


def test_missing_dataset_file_is_a_config_error(tmp_path: Path) -> None:
    text = VALID + f'\n[dataset]\npath = "{tmp_path / "absent.jsonl"}"\n'
    with pytest.raises(ConfigError, match="cannot read dataset"):
        load_plan(write(tmp_path, text), {})


def test_empty_dataset_is_a_config_error(tmp_path: Path) -> None:
    items = tmp_path / "empty.jsonl"
    items.write_bytes(b"")
    text = VALID + f'\n[dataset]\npath = "{items}"\n'
    with pytest.raises(ConfigError, match="is empty"):
        load_plan(write(tmp_path, text), {})


@pytest.mark.parametrize(
    "text, message",
    [
        ("[prompt]\nid='p'\nversion=1\nuser='u'\n[extra]\nfoo=1\n", "unknown top-level"),
        ("[prompt]\nid='p'\nversion=1\nuser='u'\nfoo=1\n", "unknown key"),
        ("[prompt]\nid='p'\nversion=1\n", "missing required key 'user'"),
        ("[prompt]\nid='p'\nversion=0\nuser='u'\n", "positive integer"),
        ("[prompt]\nid='p'\nversion=true\nuser='u'\n", "positive integer"),
        ("[prompt]\nid=''\nversion=1\nuser='u'\n", "non-empty string"),
        ("[prompt]\nid='p'\nversion=1\nuser='u'\nsystem=1\n", "'system' must be a string"),
        (
            "[prompt]\nid='p'\nversion=1\nuser='u'\n[sampling]\ntemperature=3.0\n",
            "between 0 and 2",
        ),
        (
            "[prompt]\nid='p'\nversion=1\nuser='u'\n[sampling]\ntemperature='hot'\n",
            "must be a number",
        ),
        (
            "[prompt]\nid='p'\nversion=1\nuser='u'\n[sampling]\nmax_tokens=0\n",
            "must be positive",
        ),
        (
            "[prompt]\nid='p'\nversion=1\nuser='u'\n[sampling]\nstop=[1]\n",
            "string or a list of strings",
        ),
    ],
)
def test_invalid_shapes_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_plan(write(tmp_path, text), {})


def test_models_are_required(tmp_path: Path) -> None:
    text = "[prompt]\nid='p'\nversion=1\nuser='u'\n"
    with pytest.raises(ConfigError, match="at least one"):
        load_plan(write(tmp_path, text), {})


def test_empty_models_are_rejected(tmp_path: Path) -> None:
    text = "models=[]\n[prompt]\nid='p'\nversion=1\nuser='u'\n"
    with pytest.raises(ConfigError, match="at least one"):
        load_plan(write(tmp_path, text), {})


def test_duplicate_model_names_are_rejected(tmp_path: Path) -> None:
    text = (
        VALID
        + """
[[models]]
name = "a"
adapter = "fake"
scenario = "ok"
"""
    )
    with pytest.raises(ConfigError, match="duplicate model entry name 'a'"):
        load_plan(write(tmp_path, text), {})


def test_model_name_must_be_a_slug(tmp_path: Path) -> None:
    text = VALID.replace('name = "a"', 'name = "Not A Slug"')
    with pytest.raises(ConfigError, match="must match"):
        load_plan(write(tmp_path, text), {})


def test_unknown_adapter_names_the_entry(tmp_path: Path) -> None:
    text = VALID.replace('adapter = "fake"', 'adapter = "nope"')
    with pytest.raises(ConfigError, match=r"model entry 'a': unknown adapter 'nope'"):
        load_plan(write(tmp_path, text), {})


def test_missing_api_key_names_the_variable_and_entry(tmp_path: Path) -> None:
    text = VALID.replace(
        'adapter = "fake"\nscenario = "ok"',
        'adapter = "openai_compatible"\n'
        'base_url = "https://example.test/v1"\n'
        'model = "m/1"\n'
        'api_key_env = "MISSING_KEY"',
    )
    with pytest.raises(
        ConfigError, match=r"model entry 'a': environment variable MISSING_KEY is not set"
    ):
        load_plan(write(tmp_path, text), {})


def test_adapter_settings_stay_strict(tmp_path: Path) -> None:
    text = VALID.replace('scenario = "ok"', 'scenaro = "ok"')
    with pytest.raises(ConfigError, match=r"model entry 'a': unknown key\(s\)"):
        load_plan(write(tmp_path, text), {})


def test_invalid_toml_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_plan(write(tmp_path, "[prompt\n"), {})
