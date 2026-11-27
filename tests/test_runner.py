"""The runner: retry policy, call records, manifests and failure handling."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from llmtutor import runner
from llmtutor.config import ModelEntry, ResolvedModel, load_plan

FIXED = datetime(2026, 11, 7, 15, 30, 12, 481000, tzinfo=UTC)
ISO = "2026-11-07T15:30:12.481Z"

CONFIG = """\
[prompt]
id = "t"
version = 1
system = "be nice"
user = "hello"

[sampling]
temperature = 0.0
max_tokens = 16
seed = 7
stop = ["END"]

[[models]]
name = "a"
adapter = "fake"
scenario = "{scenario}"
"""

REQUEST = {
    "messages": [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello"},
    ],
    "temperature": 0.0,
    "max_tokens": 16,
    "seed": 7,
    "stop": ["END"],
}

MANIFEST_KEYS = (
    "artefact_version",
    "run_id",
    "status",
    "started_at",
    "finished_at",
    "config",
    "prompt",
    "model",
    "sampling",
    "dataset",
    "policy",
    "counts",
    "error",
)

POLICY = {
    "max_attempts": 3,
    "retryable": ["rate_limit", "timeout", "provider"],
    "backoff": {"base_s": 1.0, "cap_s": 60.0, "jitter": "full"},
}


def write_config(tmp_path: Path, scenario: str, extra: str = "") -> Path:
    path = tmp_path / "smoke.toml"
    path.write_text(CONFIG.format(scenario=scenario) + extra, encoding="utf-8")
    return path


def run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    *,
    extra: str = "",
) -> tuple[list[runner.RunResult], list[float]]:
    sleeps: list[float] = []
    monkeypatch.setattr(runner, "utcnow", lambda: FIXED)
    monkeypatch.setattr(runner, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(runner, "uniform", lambda low, high: high)
    monkeypatch.setattr(runner, "sleep", sleeps.append)
    plan = load_plan(write_config(tmp_path, scenario, extra), {})
    return runner.run_plan(plan, tmp_path / "runs"), sleeps


def calls(artefact: Path) -> list[dict]:
    text = (artefact / "calls.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def manifest(artefact: Path) -> dict:
    return json.loads((artefact / "manifest.json").read_text(encoding="utf-8"))


def summary(full: dict) -> dict:
    return {key: full[key] for key in MANIFEST_KEYS}


def expected_manifest(**overrides: object) -> dict:
    base = {
        "artefact_version": 1,
        "run_id": "20261107T153012Z-smoke-a",
        "status": "completed",
        "started_at": ISO,
        "finished_at": ISO,
        "config": {
            "path": "CONFIG_PATH",
            "sha256": "CONFIG_HASH",
        },
        "prompt": {"id": "t", "version": 1},
        "model": {
            "entry": "a",
            "adapter": "fake",
            "requested_model": None,
            "base_url": None,
            "api_key_env": None,
            "timeout_s": None,
            "settings": {"scenario": "ok"},
            "provenance": None,
        },
        "sampling": {"temperature": 0.0, "max_tokens": 16, "seed": 7, "stop": ["END"]},
        "dataset": None,
        "policy": POLICY,
        "counts": {"attempts": 1, "ok": 1, "errors": 0},
        "error": None,
    }
    base.update(overrides)
    return base


def test_success_writes_config_copy_call_record_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, sleeps = run(tmp_path, monkeypatch, "ok")

    assert sleeps == []
    assert len(results) == 1
    result = results[0]
    assert (result.run_id, result.status, result.text) == (
        "20261107T153012Z-smoke-a",
        "completed",
        "hello",
    )
    assert result.error_kind is None

    artefact = result.artefact
    assert artefact == tmp_path / "runs" / "20261107T153012Z-smoke-a"
    assert sorted(path.name for path in artefact.iterdir()) == [
        "calls.jsonl",
        "config.toml",
        "manifest.json",
    ]
    assert (artefact / "config.toml").read_bytes() == (tmp_path / "smoke.toml").read_bytes()

    assert calls(artefact) == [
        {
            "call_id": "call-0001",
            "item_id": None,
            "attempt": 1,
            "started_at": ISO,
            "duration_ms": 0,
            "request": REQUEST,
            "outcome": "ok",
            "response": {
                "text": "hello",
                "model": "fake-model",
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                "raw": {"id": "fake-1"},
            },
        }
    ]

    full = manifest(artefact)
    expected = expected_manifest()
    expected["config"] = {
        "path": str(tmp_path / "smoke.toml"),
        "sha256": load_plan(tmp_path / "smoke.toml", {}).sha256,
    }
    assert summary(full) == expected
    assert full["llmtutor_version"] == "0.1.0"
    assert isinstance(full["python_version"], str)
    assert set(full["git"]) == {"commit", "dirty"}


def test_retries_use_the_pinned_backoff_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, sleeps = run(tmp_path, monkeypatch, "rate_limit_twice_then_ok")

    assert sleeps == [1.0, 2.0]
    assert results[0].status == "completed"
    records = calls(results[0].artefact)
    assert [record["attempt"] for record in records] == [1, 2, 3]
    assert [record["outcome"] for record in records] == ["error", "error", "ok"]
    assert {record["call_id"] for record in records} == {"call-0001"}
    assert all(record["request"] == REQUEST for record in records)
    assert [record["error"]["kind"] for record in records[:2]] == ["rate_limit", "rate_limit"]

    full = manifest(results[0].artefact)
    assert full["counts"] == {"attempts": 3, "ok": 1, "errors": 2}
    assert full["status"] == "completed"
    assert full["error"] is None


def test_retry_after_wins_when_larger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, sleeps = run(tmp_path, monkeypatch, "retry_after_5")
    assert sleeps == [5.0]


def test_retry_after_is_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, sleeps = run(tmp_path, monkeypatch, "retry_after_120")
    assert sleeps == [60.0]


def test_non_retryable_failure_stops_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, sleeps = run(tmp_path, monkeypatch, "auth")

    assert sleeps == []
    result = results[0]
    assert (result.status, result.text, result.error_kind) == ("failed", None, "auth")
    assert result.error_message == "bad key"
    assert len(calls(result.artefact)) == 1

    full = manifest(result.artefact)
    assert full["status"] == "failed"
    assert full["error"] == {"kind": "auth", "status": 401, "message": "bad key"}
    assert full["counts"] == {"attempts": 1, "ok": 0, "errors": 1}


def test_exhausted_retries_fail_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results, sleeps = run(tmp_path, monkeypatch, "provider")

    assert sleeps == [1.0, 2.0]
    result = results[0]
    assert (result.status, result.error_kind) == ("failed", "provider")
    assert manifest(result.artefact)["counts"] == {"attempts": 3, "ok": 0, "errors": 3}


def test_non_model_error_is_recorded_as_bad_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, sleeps = run(tmp_path, monkeypatch, "contract_bug")

    assert sleeps == []
    result = results[0]
    assert (result.status, result.error_kind) == ("failed", "bad_response")
    record = calls(result.artefact)[0]
    assert record["error"] == {
        "kind": "bad_response",
        "status": None,
        "body": None,
        "message": "AssertionError: boom",
    }
    assert manifest(result.artefact)["error"] == {
        "kind": "bad_response",
        "status": None,
        "message": "AssertionError: boom",
    }


def test_keyboard_interrupt_aborts_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, monkeypatch, "interrupt")

    artefact = tmp_path / "runs" / "20261107T153012Z-smoke-a"
    full = manifest(artefact)
    assert full["status"] == "aborted"
    assert full["error"] == {"kind": "aborted", "status": None, "message": "KeyboardInterrupt"}
    assert full["counts"] == {"attempts": 0, "ok": 0, "errors": 0}


def test_collision_never_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = tmp_path / "runs" / "20261107T153012Z-smoke-a"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("earlier run", encoding="utf-8")

    results, _ = run(tmp_path, monkeypatch, "ok")

    assert results[0].run_id == "20261107T153012Z-smoke-a-2"
    assert results[0].artefact == tmp_path / "runs" / "20261107T153012Z-smoke-a-2"
    assert manifest(results[0].artefact)["run_id"] == "20261107T153012Z-smoke-a-2"
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "earlier run"


def test_a_failed_sibling_does_not_stop_the_next_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra = """
[[models]]
name = "b"
adapter = "fake"
scenario = "ok"
"""
    results, _ = run(tmp_path, monkeypatch, "auth", extra=extra)

    assert [result.status for result in results] == ["failed", "completed"]
    assert [result.entry for result in results] == ["a", "b"]
    assert results[1].text == "hello"
    assert calls(results[0].artefact)[0]["outcome"] == "error"
    assert calls(results[1].artefact)[0]["outcome"] == "ok"


def test_dataset_runs_are_refused_until_the_item_loop(tmp_path: Path) -> None:
    items = tmp_path / "items_v1.jsonl"
    items.write_text('{"item_id": "fb-001"}\n', encoding="utf-8")
    config = write_config(
        tmp_path,
        "ok",
        extra=f'\n[dataset]\npath = "{items}"\n',
    )
    plan = load_plan(config, {})

    with pytest.raises(runner.RunError, match="not implemented"):
        runner.run_plan(plan, tmp_path / "runs")

    assert not (tmp_path / "runs").exists()


def test_calls_are_flushed_before_backoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines_at_sleep: list[int] = []
    artefact = tmp_path / "runs" / "20261107T153012Z-smoke-a"

    def observe(delay: float) -> None:
        lines_at_sleep.append(len(calls(artefact)))

    monkeypatch.setattr(runner, "utcnow", lambda: FIXED)
    monkeypatch.setattr(runner, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(runner, "uniform", lambda low, high: high)
    monkeypatch.setattr(runner, "sleep", observe)
    plan = load_plan(write_config(tmp_path, "rate_limit_twice_then_ok"), {})
    runner.run_plan(plan, tmp_path / "runs")

    assert lines_at_sleep == [1, 2]


class Attesting:
    """A model that can vouch for the weights behind it."""

    timeout_s = 12.0
    base_url = "http://daemon.test:11434"

    def provenance(self) -> dict[str, Any]:
        return {"digest": "sha256:abc", "ollama_version": "0.40.12"}


def ollama_entry() -> ModelEntry:
    return ModelEntry(
        name="q",
        adapter="ollama",
        settings={"model": "qwen3:8b", "num_ctx": 8192, "think": True},
    )


def test_model_block_records_every_setting_and_the_attested_provenance() -> None:
    block = runner._model_block(ResolvedModel(entry=ollama_entry(), model=Attesting()))

    assert block == {
        "entry": "q",
        "adapter": "ollama",
        "requested_model": "qwen3:8b",
        "base_url": "http://daemon.test:11434",
        "api_key_env": None,
        "timeout_s": 12.0,
        "settings": {"model": "qwen3:8b", "num_ctx": 8192, "think": True},
        "provenance": {"digest": "sha256:abc", "ollama_version": "0.40.12"},
    }


def test_model_block_records_no_provenance_for_an_adapter_that_cannot_attest() -> None:
    class Silent:
        pass

    block = runner._model_block(ResolvedModel(entry=ollama_entry(), model=Silent()))

    assert block["provenance"] is None


def test_model_block_survives_a_provenance_probe_that_raises() -> None:
    class Broken:
        def provenance(self) -> dict[str, Any]:
            raise RuntimeError("daemon went away")

    block = runner._model_block(ResolvedModel(entry=ollama_entry(), model=Broken()))

    assert block["provenance"] is None


def test_the_manifest_carries_provenance_learned_during_the_run(tmp_path: Path) -> None:
    class Late:
        def __init__(self) -> None:
            self.attested = False

        def provenance(self) -> dict[str, Any] | None:
            return {"digest": "sha256:abc"} if self.attested else None

    model = Late()
    resolved = ResolvedModel(entry=ollama_entry(), model=model)
    record: dict[str, Any] = {"model": runner._model_block(resolved)}
    assert record["model"]["provenance"] is None

    model.attested = True
    runner._finalise(tmp_path, record, resolved, 1, 1, 0, "completed")

    assert record["model"]["provenance"] == {"digest": "sha256:abc"}
    assert manifest(tmp_path)["model"]["provenance"] == {"digest": "sha256:abc"}
