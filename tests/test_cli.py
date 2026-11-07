"""The CLI: exit codes and output, driven in-process."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmtutor import cli, runner

FIXED = datetime(2026, 11, 7, 15, 30, 12, 481000, tzinfo=UTC)

CONFIG = """\
[prompt]
id = "t"
version = 1
user = "hello"

[[models]]
name = "a"
adapter = "fake"
scenario = "{scenario}"
"""

EXTRA = """
[[models]]
name = "b"
adapter = "fake"
scenario = "ok"
"""


def write_config(tmp_path: Path, scenario: str, extra: str = "") -> Path:
    path = tmp_path / "smoke.toml"
    path.write_text(CONFIG.format(scenario=scenario) + extra, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "utcnow", lambda: FIXED)
    monkeypatch.setattr(runner, "perf_counter", lambda: 0.0)
    monkeypatch.setattr(runner, "uniform", lambda low, high: high)
    monkeypatch.setattr(runner, "sleep", lambda delay: None)


@pytest.fixture(autouse=True)
def work_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def test_success_prints_header_and_answer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, "ok")

    exit_code = cli.main(["run", "--config", str(config)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "== a\nhello\n"
    assert captured.err == ""
    assert (tmp_path / "runs" / "20261107T153012Z-smoke-a").is_dir()


def test_null_text_prints_no_content(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = write_config(tmp_path, "no_content")

    exit_code = cli.main(["run", "--config", str(config)])

    assert exit_code == 0
    assert capsys.readouterr().out == "== a\n(no content)\n"


def test_failed_run_exits_one_and_names_the_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, "auth")

    exit_code = cli.main(["run", "--config", str(config)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == "== a\n"
    assert captured.err == "20261107T153012Z-smoke-a: auth: bad key\n"


def test_one_failed_sibling_still_runs_the_next(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, "auth", extra=EXTRA)

    exit_code = cli.main(["run", "--config", str(config)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == "== a\n== b\nhello\n"
    assert captured.err == "20261107T153012Z-smoke-a: auth: bad key\n"


def test_config_error_exits_two_before_any_artefact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, "ok")
    config.write_text(
        config.read_text(encoding="utf-8").replace('adapter = "fake"', 'adapter = "nope"')
    )

    exit_code = cli.main(["run", "--config", str(config)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "unknown adapter 'nope'" in captured.err
    assert not (tmp_path / "runs").exists()


def test_missing_config_file_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["run", "--config", str(tmp_path / "absent.toml")])

    assert exit_code == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_keyboard_interrupt_exits_one_thirty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, "interrupt")

    exit_code = cli.main(["run", "--config", str(config)])

    assert exit_code == 130
    artefact = tmp_path / "runs" / "20261107T153012Z-smoke-a"
    manifest = json.loads((artefact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "aborted"


def test_exit_one_when_a_retryable_failure_exhausts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, "provider")

    exit_code = cli.main(["run", "--config", str(config)])

    assert exit_code == 1
    assert capsys.readouterr().err == "20261107T153012Z-smoke-a: provider: overloaded\n"


def test_load_env_reads_dotenv_without_overriding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("LLMTUTOR_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLMTUTOR_TEST_KEY", raising=False)

    try:
        cli.load_env()
        assert os.environ["LLMTUTOR_TEST_KEY"] == "from_file"
    finally:
        os.environ.pop("LLMTUTOR_TEST_KEY", None)


def test_load_env_keeps_the_real_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("LLMTUTOR_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLMTUTOR_TEST_KEY", "from_process")

    cli.load_env()

    assert os.environ["LLMTUTOR_TEST_KEY"] == "from_process"
