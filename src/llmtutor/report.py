"""Run artefacts: manifest, config copy and call records."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from llmtutor.types import ModelError, ModelRequest, ModelResponse


def config_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def package_version() -> str:
    try:
        return version("llmtutor")
    except PackageNotFoundError:
        return "0.0.0"


def python_version() -> str:
    return platform.python_version()


def git_info() -> dict[str, Any] | None:
    """Best effort: the short commit and whether the worktree is dirty."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if commit.returncode != 0 or status.returncode != 0:
        return None
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def create_run_dir(runs_root: Path, run_id: str) -> tuple[Path, str]:
    """Create `runs/<run-id>/` without ever overwriting an earlier run."""

    runs_root.mkdir(parents=True, exist_ok=True)
    candidate = run_id
    suffix = 2
    while True:
        path = runs_root / candidate
        try:
            path.mkdir()
        except FileExistsError:
            candidate = f"{run_id}-{suffix}"
            suffix += 1
            continue
        return path, candidate


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_call(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def request_record(request: ModelRequest) -> dict[str, Any]:
    return {
        "messages": [
            {"role": message.role, "content": message.content} for message in request.messages
        ],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "seed": request.seed,
        "stop": list(request.stop),
    }


def response_record(response: ModelResponse) -> dict[str, Any]:
    usage = None
    if response.usage is not None:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return {
        "text": response.text,
        "model": response.model,
        "finish_reason": response.finish_reason,
        "usage": usage,
        "raw": response.raw,
    }


def error_record(error: ModelError) -> dict[str, Any]:
    return {
        "kind": str(error.kind),
        "status": error.status,
        "body": error.body,
        "message": str(error),
    }


def write_config_copy(artefact: Path, raw: bytes) -> None:
    (artefact / "config.toml").write_bytes(raw)
