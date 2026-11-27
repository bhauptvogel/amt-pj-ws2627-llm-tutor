"""Sequential execution of a run plan, with the retry policy and artefact writing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from random import uniform
from time import perf_counter, sleep
from typing import Any

from llmtutor import report
from llmtutor.config import ResolvedModel, RunPlan
from llmtutor.types import ErrorKind, Model, ModelError, ModelRequest, ModelResponse

MAX_ATTEMPTS = 3
RETRYABLE = frozenset({ErrorKind.RATE_LIMIT, ErrorKind.TIMEOUT, ErrorKind.PROVIDER})
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0
MANIFEST_VERSION = 1
CALL_ID = "call-0001"


class RunError(Exception):
    """A valid plan this runner cannot execute yet."""


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    entry: str
    requested_model: str | None
    status: str
    text: str | None
    error_kind: str | None
    error_message: str | None
    artefact: Path


def run_plan(plan: RunPlan, runs_root: Path) -> list[RunResult]:
    """Run every model entry in config order; a failed run does not stop its siblings."""

    if plan.dataset is not None:
        raise RunError(
            "dataset runs are not implemented yet: the config is valid, but the runner "
            "still makes one call per model. The task layer's item loop comes next."
        )

    started = utcnow()
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    results = []
    for resolved in plan.models:
        results.append(_run_one(plan, resolved, runs_root, stamp, started))
    return results


def _run_one(
    plan: RunPlan,
    resolved: ResolvedModel,
    runs_root: Path,
    stamp: str,
    started: datetime,
) -> RunResult:
    entry = resolved.entry
    run_id = f"{stamp}-{plan.path.stem}-{entry.name}"
    artefact, run_id = report.create_run_dir(runs_root, run_id)
    report.write_config_copy(artefact, plan.raw)
    calls_path = artefact / "calls.jsonl"

    manifest: dict[str, Any] = {
        "artefact_version": MANIFEST_VERSION,
        "run_id": run_id,
        "status": "running",
        "started_at": _iso(started),
        "finished_at": None,
        "llmtutor_version": report.package_version(),
        "python_version": report.python_version(),
        "git": report.git_info(),
        "config": {"path": str(plan.path), "sha256": plan.sha256},
        "prompt": {"id": plan.prompt.id, "version": plan.prompt.version},
        "model": _model_block(resolved),
        "sampling": {
            "temperature": plan.sampling.temperature,
            "max_tokens": plan.sampling.max_tokens,
            "seed": plan.sampling.seed,
            "stop": list(plan.sampling.stop),
        },
        "dataset": None,
        "policy": {
            "max_attempts": MAX_ATTEMPTS,
            "retryable": [
                ErrorKind.RATE_LIMIT.value,
                ErrorKind.TIMEOUT.value,
                ErrorKind.PROVIDER.value,
            ],
            "backoff": {"base_s": BACKOFF_BASE_S, "cap_s": BACKOFF_CAP_S, "jitter": "full"},
        },
        "counts": {"attempts": 0, "ok": 0, "errors": 0},
        "error": None,
    }
    report.write_manifest(artefact / "manifest.json", manifest)

    records = 0
    ok_count = 0
    error_count = 0
    response = None
    last_error: ModelError | None = None

    request = ModelRequest(
        messages=plan.prompt.messages(),
        temperature=plan.sampling.temperature,
        max_tokens=plan.sampling.max_tokens,
        seed=plan.sampling.seed,
        stop=plan.sampling.stop,
    )

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            call_started = utcnow()
            started_clock = perf_counter()
            try:
                response = resolved.model.generate(request)
            except ModelError as exc:
                error = exc
            except Exception as exc:
                error = ModelError(ErrorKind.BAD_RESPONSE, f"{type(exc).__name__}: {exc}")
            else:
                report.append_call(
                    calls_path,
                    _call_record(request, attempt, call_started, started_clock, response=response),
                )
                records += 1
                ok_count += 1
                last_error = None
                break

            report.append_call(
                calls_path,
                _call_record(request, attempt, call_started, started_clock, error=error),
            )
            records += 1
            error_count += 1
            last_error = error

            if error.kind not in RETRYABLE or attempt == MAX_ATTEMPTS:
                break
            sleep(_backoff(attempt, error))
    except KeyboardInterrupt:
        _finalise(
            artefact,
            manifest,
            resolved,
            records,
            ok_count,
            error_count,
            "aborted",
            message="KeyboardInterrupt",
        )
        raise
    except Exception as exc:
        _finalise(
            artefact,
            manifest,
            resolved,
            records,
            ok_count,
            error_count,
            "aborted",
            message=f"{type(exc).__name__}: {exc}",
        )
        return RunResult(
            run_id=run_id,
            entry=entry.name,
            requested_model=entry.settings.get("model"),
            status="aborted",
            text=None,
            error_kind="aborted",
            error_message=f"{type(exc).__name__}: {exc}",
            artefact=artefact,
        )

    status = "completed" if last_error is None else "failed"
    _finalise(
        artefact,
        manifest,
        resolved,
        records,
        ok_count,
        error_count,
        status,
        last_error=last_error,
    )

    text = response.text if response is not None and status == "completed" else None
    return RunResult(
        run_id=run_id,
        entry=entry.name,
        requested_model=entry.settings.get("model"),
        status=status,
        text=text,
        error_kind=str(last_error.kind) if last_error is not None else None,
        error_message=str(last_error) if last_error is not None else None,
        artefact=artefact,
    )


def _backoff(attempt: int, error: ModelError) -> float:
    delay = uniform(0.0, BACKOFF_BASE_S * 2 ** (attempt - 1))
    if error.retry_after is not None and error.retry_after > delay:
        return min(float(error.retry_after), BACKOFF_CAP_S)
    return delay


def _call_record(
    request: ModelRequest,
    attempt: int,
    started: datetime,
    started_clock: float,
    *,
    response: ModelResponse | None = None,
    error: ModelError | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "call_id": CALL_ID,
        "item_id": None,
        "attempt": attempt,
        "started_at": _iso(started),
        "duration_ms": round((perf_counter() - started_clock) * 1000),
        "request": report.request_record(request),
    }
    if error is None:
        record["outcome"] = "ok"
        record["response"] = report.response_record(response)
    else:
        record["outcome"] = "error"
        record["error"] = report.error_record(error)
    return record


def _finalise(
    artefact: Path,
    manifest: dict[str, Any],
    resolved: ResolvedModel,
    attempts: int,
    ok_count: int,
    error_count: int,
    status: str,
    *,
    last_error: ModelError | None = None,
    message: str | None = None,
) -> None:
    manifest["status"] = status
    manifest["finished_at"] = _iso(utcnow())
    manifest["counts"] = {"attempts": attempts, "ok": ok_count, "errors": error_count}
    # The adapter learns the digest and daemon version on the first call, so the block is
    # written again here, where it can carry what the run actually ran against.
    manifest["model"] = _model_block(resolved)
    if status == "completed":
        manifest["error"] = None
    elif status == "failed" and last_error is not None:
        manifest["error"] = {
            "kind": str(last_error.kind),
            "status": last_error.status,
            "message": str(last_error),
        }
    else:
        manifest["error"] = {"kind": "aborted", "status": None, "message": message}
    report.write_manifest(artefact / "manifest.json", manifest)


def _model_block(resolved: ResolvedModel) -> dict[str, Any]:
    """Provenance of the call: what we asked for, every setting verbatim, and what the adapter
    could attest about the weights behind it.

    The named keys are kept because the evaluation layer reads them; `settings` carries the
    rest, so a setting no reader knows about is recorded rather than dropped.
    """

    settings = dict(resolved.entry.settings)
    timeout_s = getattr(resolved.model, "timeout_s", None)
    if timeout_s is None:
        timeout_s = settings.get("timeout_s")
    base_url = getattr(resolved.model, "base_url", None) or settings.get("base_url")
    return {
        "entry": resolved.entry.name,
        "adapter": resolved.entry.adapter,
        "requested_model": settings.get("model"),
        "base_url": base_url,
        "api_key_env": settings.get("api_key_env"),
        "timeout_s": timeout_s,
        "settings": settings,
        "provenance": _provenance(resolved.model),
    }


def _provenance(model: Model) -> dict[str, Any] | None:
    """Whatever the adapter can attest about the weights, or None when it cannot attest."""

    attested = getattr(model, "provenance", None)
    if not callable(attested):
        return None
    try:
        facts = attested()
    except Exception:
        return None
    return dict(facts) if facts else None


def _iso(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"
