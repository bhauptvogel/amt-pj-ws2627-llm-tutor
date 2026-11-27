"""JSON Schemas for judge outputs, built from the rubrics so label sets live in one place.

Every judged criterion is an object whose keys run evidence -> reasoning -> label, so the
judge writes its quote and argument before it commits to a label. JSON Schema cannot
express key order; `key_order_issues` checks it separately on the parsed object.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator

from llmtutor.evaluation.criteria import (
    EXPLANATION,
    FEEDBACK,
    MCQ,
    Criterion,
    Rubric,
    judge_checked_flaws,
)

Schema = dict[str, Any]

KEY_POINT_COVERAGE = ("covered", "partial", "missing")

# The prompt asks for "the exact quote(s)", so several quotes may come as a list.
EVIDENCE: Schema = {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}


def feedback_schema(names: Sequence[str] | None = None) -> Schema:
    analysis = _object(
        {
            "student_error": {"type": "string"},
            "reference_final_answer": {"type": "string"},
            "final_answer_occurrences": {"type": ["array", "null"], "items": {"type": "string"}},
        }
    )
    return _judgement_schema(FEEDBACK, analysis, names=names)


def explanation_schema(level_scale: Sequence[str], names: Sequence[str] | None = None) -> Schema:
    """The perceived level must be one of the item's ordered levels."""

    if not level_scale or len(set(level_scale)) != len(level_scale):
        raise ValueError("level_scale must be a non-empty list of distinct levels")
    coverage = _object(
        {
            "key_point": {"type": "string"},
            "coverage": {"enum": list(KEY_POINT_COVERAGE)},
        }
    )
    analysis = _object(
        {
            "level_evidence": {"type": "string"},
            "perceived_level": {"enum": list(level_scale)},
            "key_point_coverage": {"type": "array", "items": coverage},
        }
    )
    return _judgement_schema(EXPLANATION, analysis, names=names)


def mcq_schema() -> Schema:
    """The judge solves the item before it sees the key, then checks the semantic flaws.

    Rule-checked flaws are computed in code and never asked of the judge.
    """

    analysis = _object({"reasoning": {"type": "string"}, "judge_answer": {"type": "string"}})
    flag = _object({"evidence": EVIDENCE, "flag": {"type": "boolean"}})
    iwf = _object({str(flaw.id): flag for flaw in judge_checked_flaws()})
    return _judgement_schema(MCQ, analysis, extra={"iwf": iwf})


def _judgement_schema(
    rubric: Rubric,
    analysis: Schema,
    *,
    names: Sequence[str] | None = None,
    extra: Mapping[str, Schema] | None = None,
) -> Schema:
    """Without `names`, any valid judgement of the rubric; with them, exactly those criteria.

    The narrow form is what one judge call is asked for, so every named criterion is
    required there, optional ones included.
    """

    if names is None:
        chosen = rubric.criteria
        required = [c.name for c in chosen if not c.diagnostic and not c.optional]
    else:
        chosen = tuple(rubric.criterion(name) for name in names)
        required = [c.name for c in chosen if not c.diagnostic]
    judged = [criterion for criterion in chosen if not criterion.diagnostic]
    properties: Schema = {"analysis": analysis}
    if judged:
        properties["criteria"] = _object(
            {criterion.name: _labelled(criterion) for criterion in judged},
            required=required,
        )
    for criterion in chosen:
        if criterion.diagnostic:
            properties[criterion.name] = {"enum": list(criterion.labels)}
    properties.update(extra or {})
    schema = _object(properties)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": schema_id(rubric),
        **schema,
    }


def _labelled(criterion: Criterion) -> Schema:
    return _object(
        {
            "evidence": EVIDENCE,
            "reasoning": {"type": "string"},
            "label": {"enum": list(criterion.labels)},
        }
    )


def _object(properties: Schema, *, required: Sequence[str] | None = None) -> Schema:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties) if required is None else list(required),
        "properties": properties,
    }


def schema_id(rubric: Rubric) -> str:
    return f"llmtutor/judge/{rubric.task}/v{rubric.version}"


def schema_sha256(schema: Schema) -> str:
    """Hash of the canonical JSON, so key order in this file does not change it."""

    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def schema_issues(schema: Schema, judgement: Any) -> list[str]:
    """Every schema violation, each prefixed with the JSON path where it occurs."""

    validator = Draft202012Validator(schema)
    issues = []
    for error in sorted(validator.iter_errors(judgement), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(part) for part in error.absolute_path) or "(root)"
        issues.append(f"{where}: {error.message}")
    return issues


def key_order_issues(schema: Schema, judgement: Any, path: str = "") -> list[str]:
    """Objects whose keys come in a different order than the schema lists them.

    Run this only on an object that passed `schema_issues`; it relies on parsed JSON
    keeping the order in which the judge wrote the keys.
    """

    issues: list[str] = []
    if isinstance(judgement, dict) and "properties" in schema:
        expected = [key for key in schema["properties"] if key in judgement]
        actual = [key for key in judgement if key in schema["properties"]]
        if actual != expected:
            issues.append(f"{path or '(root)'}: keys {actual} should be ordered {expected}")
        for key, child in schema["properties"].items():
            if key in judgement:
                issues += key_order_issues(child, judgement[key], f"{path}/{key}".lstrip("/"))
    elif isinstance(judgement, list) and "items" in schema:
        for index, element in enumerate(judgement):
            issues += key_order_issues(schema["items"], element, f"{path}/{index}".lstrip("/"))
    return issues


def labels_of(rubric: Rubric, judgement: Mapping[str, Any]) -> dict[str, str]:
    """The label per criterion from a schema-valid judgement; absent optional ones omitted."""

    labels: dict[str, str] = {}
    for criterion in rubric.criteria:
        if criterion.diagnostic:
            if criterion.name in judgement:
                labels[criterion.name] = judgement[criterion.name]
        elif criterion.name in judgement.get("criteria", {}):
            labels[criterion.name] = judgement["criteria"][criterion.name]["label"]
    return labels
