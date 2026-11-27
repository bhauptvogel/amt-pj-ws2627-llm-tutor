"""Tutor responses to judge, read from JSONL.

One line per tutor response:

    {"item_id": "t1-0042", "task": "feedback", "tutor_model": "oss-120b",
     "tutor_response": "...", "item": {"question": "...", "student_answer": "...", ...}}

`item` carries the item fields the task's judge template names, judge-only ones included.
`tutor_model` is recorded with the judgement but never shown to the judge. This is the
evaluation layer's input contract until the task layer and the runner produce it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KEYS = frozenset({"item_id", "task", "tutor_model", "tutor_response", "item"})


class ResponsesError(Exception):
    """The responses file is malformed."""


@dataclass(frozen=True, slots=True)
class TutorResponse:
    item_id: str
    task: str
    tutor_model: str
    text: str
    item: Mapping[str, str]

    def values(self) -> dict[str, str]:
        """Placeholder values for the judge template."""

        return {**self.item, "tutor_response": self.text}


@dataclass(frozen=True, slots=True)
class ResponseSet:
    path: Path
    sha256: str
    responses: tuple[TutorResponse, ...]


def load_responses(path: Path) -> ResponseSet:
    raw = path.read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ResponsesError(f"{path}: not valid UTF-8: {exc}") from exc

    responses: list[TutorResponse] = []
    seen: set[tuple[str, str]] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResponsesError(f"{where}: invalid JSON: {exc}") from exc
        response = _parse(record, where)
        key = (response.item_id, response.tutor_model)
        if key in seen:
            raise ResponsesError(f"{where}: duplicate item {key[0]!r} for tutor {key[1]!r}")
        seen.add(key)
        responses.append(response)

    if not responses:
        raise ResponsesError(f"{path}: no responses")
    return ResponseSet(
        path=path, sha256=hashlib.sha256(raw).hexdigest(), responses=tuple(responses)
    )


def _parse(record: Any, where: str) -> TutorResponse:
    if not isinstance(record, dict):
        raise ResponsesError(f"{where}: each line must be a JSON object")
    unknown = sorted(set(record) - KEYS)
    if unknown:
        raise ResponsesError(f"{where}: unknown key(s): {', '.join(unknown)}")
    for key in ("item_id", "task", "tutor_model"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ResponsesError(f"{where}: {key!r} must be a non-empty string")
    if not isinstance(record.get("tutor_response"), str):
        raise ResponsesError(f"{where}: 'tutor_response' must be a string")
    item = record.get("item")
    if not isinstance(item, dict):
        raise ResponsesError(f"{where}: 'item' must be an object")
    if "tutor_response" in item:
        raise ResponsesError(f"{where}: 'item' must not carry 'tutor_response'")
    return TutorResponse(
        item_id=record["item_id"],
        task=record["task"],
        tutor_model=record["tutor_model"],
        text=record["tutor_response"],
        item={key: _field(value, f"{where}: item.{key}") for key, value in item.items()},
    )


def _field(value: Any, where: str) -> str:
    """Strings as they are; lists and objects as JSON, e.g. key points or a level scale."""

    if isinstance(value, str):
        return value
    if isinstance(value, bool) or value is None:
        raise ResponsesError(f"{where} must be a string, number, list or object")
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    raise ResponsesError(f"{where} must be a string, number, list or object")
