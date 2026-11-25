"""The judge: label every tutor response in a responses file against one task's rubric.

A judge config names the task, the prompt version, how many criteria one call asks for,
the sampling and one model entry, built through the model layer's adapter registry. Each
judge call is retried on transport failures with jittered backoff and, separately, at
most `OUTPUT_RETRIES` times on output that is not schema-valid JSON; after that the
response is marked `judge_error`, never dropped.

A judge run writes `runs/<run-id>/` with `manifest.json`, a copy of the config,
`calls.jsonl` (one record per attempt) and `judgements.jsonl` (one record per response).
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from random import uniform
from time import perf_counter, sleep
from typing import Any

from llmtutor import report
from llmtutor.config import SLUG, ConfigError, ModelEntry, Sampling, _parse_sampling
from llmtutor.evaluation.criteria import RUBRICS, Criterion, Rubric
from llmtutor.evaluation.prompts import PromptError, PromptTemplate, load_template
from llmtutor.evaluation.responses import ResponseSet, TutorResponse
from llmtutor.evaluation.schemas import (
    Schema,
    feedback_schema,
    key_order_issues,
    labels_of,
    schema_id,
    schema_issues,
    schema_sha256,
)
from llmtutor.models.registry import create_model
from llmtutor.registry import RegistryError
from llmtutor.runner import BACKOFF_CAP_S, RETRYABLE, utcnow
from llmtutor.types import ErrorKind, Model, ModelError, ModelRequest, ModelResponse

ARTEFACT_VERSION = 1
OUTPUT_RETRIES = 2
# More patient than the runner's single call: a judge run makes hundreds of calls, and
# providers throttle them (429) even on paid models. Waits grow 0-2, 0-4, ... 0-32 s.
TRANSPORT_ATTEMPTS = 6
BACKOFF_BASE_S = 2.0
CRITERIA_PER_CALL = ("all", "one")
# Failures that would repeat for every response: stop the run instead of marking each one.
ABORTING = frozenset({ErrorKind.AUTH, ErrorKind.PAYMENT, ErrorKind.NOT_FOUND})

TOP_LEVEL_KEYS = frozenset({"judge", "sampling", "model"})
JUDGE_KEYS = frozenset({"task", "prompt_version", "criteria_per_call", "optional_criteria"})
FENCE = re.compile(r"^```(?:json)?[ \t]*\n(.*)\n```$", re.DOTALL)

# Schema builders per task; the explanation and MCQ judges need their templates first.
SCHEMAS: Mapping[str, Callable[[Sequence[str]], Schema]] = {"feedback": feedback_schema}


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    path: Path
    raw: bytes
    sha256: str
    rubric: Rubric
    template: PromptTemplate
    criteria_per_call: str
    criteria: tuple[Criterion, ...]
    sampling: Sampling
    entry: ModelEntry
    model: Model

    def units(self) -> tuple[tuple[Criterion, ...], ...]:
        """The criteria each judge call asks for, in rubric order."""

        if self.criteria_per_call == "all":
            return (self.criteria,)
        return tuple((criterion,) for criterion in self.criteria)

    def schema(self, criteria: Sequence[Criterion]) -> Schema:
        return SCHEMAS[self.rubric.task]([criterion.name for criterion in criteria])


class JudgeAbort(Exception):
    """A failure that would repeat for every response, such as a rejected API key."""

    def __init__(self, error: ModelError) -> None:
        super().__init__(f"{error.kind}: {error}")
        self.error = error


def load_judge_config(path: Path, environ: Mapping[str, str]) -> JudgeConfig:
    raw = path.read_bytes()
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: invalid config: {exc}") from exc

    unknown = sorted(set(document) - TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"{path}: unknown top-level key(s): {', '.join(unknown)}")

    judge = document.get("judge")
    if not isinstance(judge, dict):
        raise ConfigError(f"{path}: [judge] table is required")
    unknown = sorted(set(judge) - JUDGE_KEYS)
    if unknown:
        raise ConfigError(f"{path}: [judge] unknown key(s): {', '.join(unknown)}")

    task = judge.get("task")
    if task not in RUBRICS:
        raise ConfigError(f"{path}: [judge] 'task' must be one of {', '.join(sorted(RUBRICS))}")
    if task not in SCHEMAS:
        raise ConfigError(f"{path}: [judge] task {task!r} has no judge yet")
    rubric = RUBRICS[task]

    version = judge.get("prompt_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ConfigError(f"{path}: [judge] 'prompt_version' must be a positive integer")
    try:
        template = load_template(rubric, version)
    except PromptError as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    per_call = judge.get("criteria_per_call", "all")
    if per_call not in CRITERIA_PER_CALL:
        raise ConfigError(f"{path}: [judge] 'criteria_per_call' must be 'all' or 'one'")

    optional = judge.get("optional_criteria", [])
    allowed = {criterion.name for criterion in rubric.criteria if criterion.optional}
    if not isinstance(optional, list) or not set(optional) <= allowed:
        raise ConfigError(
            f"{path}: [judge] 'optional_criteria' may name only {sorted(allowed) or 'nothing'}"
        )
    criteria = tuple(c for c in rubric.criteria if not c.optional or c.name in optional)

    sampling = _parse_sampling(document.get("sampling"), path)
    entry = _parse_model(document.get("model"), path)
    try:
        model = create_model(entry.adapter, entry.settings, environ)
    except (ConfigError, RegistryError) as exc:
        raise ConfigError(f"{path}: [model] {exc}") from exc

    return JudgeConfig(
        path=path,
        raw=raw,
        sha256=report.config_sha256(raw),
        rubric=rubric,
        template=template,
        criteria_per_call=per_call,
        criteria=criteria,
        sampling=sampling,
        entry=entry,
        model=model,
    )


def _parse_model(value: Any, path: Path) -> ModelEntry:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: [model] table is required")
    name = value.get("name")
    if not isinstance(name, str) or not SLUG.fullmatch(name):
        raise ConfigError(f"{path}: [model] 'name' must match {SLUG.pattern}")
    adapter = value.get("adapter")
    if not isinstance(adapter, str) or not adapter:
        raise ConfigError(f"{path}: [model] 'adapter' must be a non-empty string")
    settings = {key: item for key, item in value.items() if key not in ("name", "adapter")}
    return ModelEntry(name=name, adapter=adapter, settings=settings)


def check_responses(config: JudgeConfig, responses: ResponseSet) -> None:
    """Fail before any call if a response belongs to another task or lacks an item field."""

    for response in responses.responses:
        if response.task != config.rubric.task:
            raise ConfigError(
                f"{responses.path}: item {response.item_id!r} is task {response.task!r}, "
                f"the judge config is {config.rubric.task!r}"
            )
        missing = config.template.missing(response.values())
        if missing:
            raise ConfigError(
                f"{responses.path}: item {response.item_id!r} lacks field(s): {', '.join(missing)}"
            )


def parse_output(text: str | None) -> tuple[Any, list[str]]:
    """The judge's JSON object, or the reasons it is not one. One code fence is tolerated."""

    if text is None:
        return None, ["output is empty"]
    stripped = text.strip()
    fenced = FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        value = json.loads(stripped, object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as exc:
        return None, [f"output is not JSON: {exc}"]
    except ValueError as exc:
        return None, [str(exc)]
    if not isinstance(value, dict):
        return None, ["output is not a JSON object"]
    return value, []


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"output repeats key(s): {', '.join(duplicates)}")
    return dict(pairs)


@dataclass(slots=True)
class CallOutcome:
    """The result of one judge call after all its attempts."""

    criteria: tuple[str, ...]
    status: str
    attempts: int
    output_retries: int
    raw_output: str | None = None
    parsed: dict[str, Any] | None = None
    issues: list[str] = field(default_factory=list)
    key_order_issues: list[str] = field(default_factory=list)
    reported_model: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class JudgeRunResult:
    run_id: str
    status: str
    counts: Mapping[str, int]
    artefact: Path
    error: str | None = None


def run_judge(
    config: JudgeConfig,
    responses: ResponseSet,
    runs_root: Path,
    *,
    kind: str = "judge",
) -> JudgeRunResult:
    """Judge every response in order; `kind` names the run, e.g. "sanity" for sanity tests."""

    check_responses(config, responses)
    started = utcnow()
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{kind}-{config.path.stem}-{config.entry.name}"
    artefact, run_id = report.create_run_dir(runs_root, run_id)
    report.write_config_copy(artefact, config.raw)

    counts = {"responses": 0, "ok": 0, "retried": 0, "judge_error": 0, "attempts": 0}
    reported: set[str] = set()
    manifest = _manifest(config, responses, run_id, started, kind)
    report.write_manifest(artefact / "manifest.json", manifest)

    def finish(status: str, error: str | None = None) -> JudgeRunResult:
        manifest.update(
            status=status,
            finished_at=_iso(utcnow()),
            counts=dict(counts),
            reported_models=sorted(reported),
            error=error,
        )
        report.write_manifest(artefact / "manifest.json", manifest)
        return JudgeRunResult(run_id, status, dict(counts), artefact, error)

    try:
        for response in responses.responses:
            outcomes = [
                _judge_call(config, response, unit, artefact / "calls.jsonl")
                for unit in config.units()
            ]
            record = _judgement_record(config, response, outcomes)
            report.append_call(artefact / "judgements.jsonl", record)
            counts["responses"] += 1
            counts[record["status"]] += 1
            counts["attempts"] += sum(outcome.attempts for outcome in outcomes)
            reported.update(o.reported_model for o in outcomes if o.reported_model)
    except JudgeAbort as exc:
        return finish("aborted", str(exc))
    except KeyboardInterrupt:
        finish("aborted", "KeyboardInterrupt")
        raise
    return finish("completed")


def _judge_call(
    config: JudgeConfig,
    response: TutorResponse,
    criteria: tuple[Criterion, ...],
    calls_path: Path,
) -> CallOutcome:
    names = tuple(criterion.name for criterion in criteria)
    schema = config.schema(criteria)
    request = ModelRequest(
        messages=config.template.messages(criteria, response.values()),
        temperature=config.sampling.temperature,
        max_tokens=config.sampling.max_tokens,
        seed=config.sampling.seed,
        stop=config.sampling.stop,
    )
    call_id = f"{response.item_id}/{response.tutor_model}/{'+'.join(names)}"
    outcome = CallOutcome(criteria=names, status="judge_error", attempts=0, output_retries=0)
    transport_failures = 0

    while True:
        outcome.attempts += 1
        started, clock = utcnow(), perf_counter()
        try:
            answer = config.model.generate(request)
        except Exception as exc:
            error = exc if isinstance(exc, ModelError) else _bad_response(exc)
            _append(calls_path, call_id, response, request, outcome.attempts, started, clock, error)
            if error.kind in ABORTING:
                raise JudgeAbort(error) from exc
            outcome.error = f"{error.kind}: {error}"
            if error.kind in RETRYABLE:
                transport_failures += 1
                if transport_failures >= TRANSPORT_ATTEMPTS:
                    return outcome
                sleep(_backoff(transport_failures, error))
                continue
            if error.kind is not ErrorKind.BAD_RESPONSE or not _retry_output(outcome):
                return outcome
            continue

        outcome.raw_output = answer.text
        outcome.reported_model = answer.model
        parsed, issues = parse_output(answer.text)
        if parsed is not None:
            issues = schema_issues(schema, parsed)
        _append(
            calls_path, call_id, response, request, outcome.attempts, started, clock, answer, issues
        )
        if issues:
            outcome.issues, outcome.error = issues, "invalid output"
            if not _retry_output(outcome):
                return outcome
            continue

        outcome.parsed = parsed
        outcome.issues, outcome.error = [], None
        outcome.key_order_issues = key_order_issues(schema, parsed)
        outcome.status = "retried" if outcome.output_retries else "ok"
        return outcome


def _retry_output(outcome: CallOutcome) -> bool:
    if outcome.output_retries >= OUTPUT_RETRIES:
        return False
    outcome.output_retries += 1
    return True


def _bad_response(exc: Exception) -> ModelError:
    return ModelError(ErrorKind.BAD_RESPONSE, f"{type(exc).__name__}: {exc}")


def _backoff(failures: int, error: ModelError) -> float:
    """Full jitter like the runner, raised to Retry-After up to the runner's cap."""

    delay = uniform(0.0, BACKOFF_BASE_S * 2 ** (failures - 1))
    if error.retry_after is not None and error.retry_after > delay:
        return min(float(error.retry_after), BACKOFF_CAP_S)
    return delay


def _append(
    path: Path,
    call_id: str,
    response: TutorResponse,
    request: ModelRequest,
    attempt: int,
    started: datetime,
    clock: float,
    result: ModelResponse | ModelError,
    issues: list[str] | None = None,
) -> None:
    record: dict[str, Any] = {
        "call_id": call_id,
        "item_id": response.item_id,
        "attempt": attempt,
        "started_at": _iso(started),
        "duration_ms": round((perf_counter() - clock) * 1000),
        "request": report.request_record(request),
    }
    if isinstance(result, ModelError):
        record["outcome"] = "error"
        record["error"] = report.error_record(result)
    else:
        record["outcome"] = "invalid_output" if issues else "ok"
        record["response"] = report.response_record(result)
        record["output_issues"] = issues or []
    report.append_call(path, record)


def _judgement_record(
    config: JudgeConfig,
    response: TutorResponse,
    outcomes: Sequence[CallOutcome],
) -> dict[str, Any]:
    failed = [outcome for outcome in outcomes if outcome.status == "judge_error"]
    if failed:
        status = "judge_error"
    elif any(outcome.status == "retried" for outcome in outcomes):
        status = "retried"
    else:
        status = "ok"

    labels: dict[str, str] = {}
    for outcome in outcomes:
        if outcome.parsed is not None:
            labels.update(labels_of(config.rubric, outcome.parsed))

    return {
        "item_id": response.item_id,
        "task": config.rubric.task,
        "tutor_model": response.tutor_model,
        "status": status,
        "labels": labels,
        "rubric_version": config.rubric.version,
        "prompt_version": config.template.version,
        "prompt_sha256": config.template.sha256,
        "calls": [
            {
                "criteria": list(outcome.criteria),
                "status": outcome.status,
                "attempts": outcome.attempts,
                "output_retries": outcome.output_retries,
                "schema_id": schema_id(config.rubric),
                "schema_sha256": schema_sha256(
                    config.schema([config.rubric.criterion(n) for n in outcome.criteria])
                ),
                "reported_model": outcome.reported_model,
                "raw_output": outcome.raw_output,
                "parsed": outcome.parsed,
                "issues": outcome.issues,
                "key_order_issues": outcome.key_order_issues,
                "error": outcome.error,
            }
            for outcome in outcomes
        ],
    }


def _manifest(
    config: JudgeConfig,
    responses: ResponseSet,
    run_id: str,
    started: datetime,
    kind: str,
) -> dict[str, Any]:
    settings = config.entry.settings
    return {
        "artefact_version": ARTEFACT_VERSION,
        "kind": kind,
        "run_id": run_id,
        "status": "running",
        "started_at": _iso(started),
        "finished_at": None,
        "llmtutor_version": report.package_version(),
        "python_version": report.python_version(),
        "git": report.git_info(),
        "config": {"path": str(config.path), "sha256": config.sha256},
        "responses": {
            "path": str(responses.path),
            "sha256": responses.sha256,
            "count": len(responses.responses),
        },
        "rubric": {
            "task": config.rubric.task,
            "version": config.rubric.version,
            "criteria": [criterion.name for criterion in config.criteria],
        },
        "prompt": {"version": config.template.version, "sha256": config.template.sha256},
        "schema_id": schema_id(config.rubric),
        "criteria_per_call": config.criteria_per_call,
        "judge": {
            "entry": config.entry.name,
            "adapter": config.entry.adapter,
            "requested_model": settings.get("model"),
            "base_url": settings.get("base_url"),
            "api_key_env": settings.get("api_key_env"),
        },
        "reported_models": [],
        "sampling": {
            "temperature": config.sampling.temperature,
            "max_tokens": config.sampling.max_tokens,
            "seed": config.sampling.seed,
            "stop": list(config.sampling.stop),
        },
        "policy": {
            "transport_max_attempts": TRANSPORT_ATTEMPTS,
            "retryable": sorted(kind.value for kind in RETRYABLE),
            "aborting": sorted(kind.value for kind in ABORTING),
            "output_retries": OUTPUT_RETRIES,
            "backoff": {"base_s": BACKOFF_BASE_S, "cap_s": BACKOFF_CAP_S, "jitter": "full"},
        },
        "counts": {},
        "error": None,
    }


def _iso(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"
