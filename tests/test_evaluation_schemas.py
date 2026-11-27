"""Judge output schemas: validation, key order and label extraction."""

from __future__ import annotations

import json
from typing import Any

import pytest

from llmtutor.evaluation.criteria import EXPLANATION, FEEDBACK, MCQ
from llmtutor.evaluation.schemas import (
    explanation_schema,
    feedback_schema,
    key_order_issues,
    labels_of,
    mcq_schema,
    schema_issues,
    schema_sha256,
)

LEVELS = ["elementary school", "high school", "undergraduate", "graduate"]


def judged(label: str) -> dict[str, str]:
    return {"evidence": "quote", "reasoning": "because", "label": label}


def feedback_judgement() -> dict[str, Any]:
    return {
        "analysis": {
            "student_error": "Computed the discount but did not subtract it.",
            "reference_final_answer": "15",
            "final_answer_occurrences": None,
        },
        "criteria": {
            "mistake_identification": judged("Yes"),
            "mistake_location": judged("Yes"),
            "revealing_answer": judged("No"),
            "providing_guidance": judged("To some extent"),
            "actionability": judged("Yes"),
            "coherence": judged("Yes"),
            "tutor_tone": judged("Encouraging"),
            "factual_accuracy": judged("Yes"),
            "over_withholding": judged("No"),
        },
        "tutoring_move": "Probing",
    }


def test_valid_feedback_judgement_passes() -> None:
    schema = feedback_schema()
    judgement = feedback_judgement()
    assert schema_issues(schema, judgement) == []
    assert key_order_issues(schema, judgement) == []


def test_evidence_may_be_several_quotes() -> None:
    judgement = feedback_judgement()
    judgement["criteria"]["coherence"]["evidence"] = ["first quote", "second quote"]
    assert schema_issues(feedback_schema(), judgement) == []
    judgement["criteria"]["coherence"]["evidence"] = [1]
    assert schema_issues(feedback_schema(), judgement)


def test_optional_criterion_may_be_present() -> None:
    judgement = feedback_judgement()
    judgement["criteria"]["human_likeness"] = judged("Yes")
    assert schema_issues(feedback_schema(), judgement) == []


def test_wrong_label_and_missing_criterion_are_reported_with_paths() -> None:
    judgement = feedback_judgement()
    judgement["criteria"]["revealing_answer"]["label"] = "Yes"
    del judgement["criteria"]["coherence"]
    issues = schema_issues(feedback_schema(), judgement)
    assert any(issue.startswith("criteria: 'coherence' is a required") for issue in issues)
    assert any(issue.startswith("criteria/revealing_answer/label:") for issue in issues)


def test_extra_keys_are_rejected() -> None:
    judgement = feedback_judgement()
    judgement["score"] = 10
    issues = schema_issues(feedback_schema(), judgement)
    assert any("Additional properties" in issue for issue in issues)


def test_label_before_reasoning_is_a_key_order_issue() -> None:
    raw = json.dumps(feedback_judgement()).replace(
        '"evidence": "quote", "reasoning": "because", "label": "Yes"',
        '"label": "Yes", "evidence": "quote", "reasoning": "because"',
        1,
    )
    issues = key_order_issues(feedback_schema(), json.loads(raw))
    assert issues == [
        "criteria/mistake_identification: keys ['label', 'evidence', 'reasoning'] "
        "should be ordered ['evidence', 'reasoning', 'label']"
    ]


def test_criteria_before_analysis_is_a_key_order_issue() -> None:
    judgement = feedback_judgement()
    reordered = {
        "criteria": judgement["criteria"],
        "analysis": judgement["analysis"],
        "tutoring_move": judgement["tutoring_move"],
    }
    assert key_order_issues(feedback_schema(), reordered)[0].startswith("(root):")


def test_labels_of_reads_criteria_and_diagnostics() -> None:
    labels = labels_of(FEEDBACK, feedback_judgement())
    assert labels["revealing_answer"] == "No"
    assert labels["tutoring_move"] == "Probing"
    assert "human_likeness" not in labels


def explanation_judgement(level: str) -> dict[str, Any]:
    return {
        "analysis": {
            "level_evidence": "uses 'sweets' and 'half'",
            "perceived_level": level,
            "key_point_coverage": [{"key_point": "out of 100", "coverage": "covered"}],
        },
        "criteria": {
            "level_match": judged("Yes"),
            "informativeness": judged("Yes"),
            "factual_accuracy": judged("Yes"),
            "clarity_structure": judged("To some extent"),
            "tutor_tone": judged("Neutral"),
        },
    }


def test_explanation_perceived_level_must_be_on_the_scale() -> None:
    schema = explanation_schema(LEVELS)
    assert schema_issues(schema, explanation_judgement("elementary school")) == []
    issues = schema_issues(schema, explanation_judgement("kindergarten"))
    assert issues and issues[0].startswith("analysis/perceived_level:")
    assert labels_of(EXPLANATION, explanation_judgement("graduate"))["level_match"] == "Yes"


@pytest.mark.parametrize("scale", [[], ["a", "a"]])
def test_explanation_schema_rejects_a_bad_level_scale(scale: list[str]) -> None:
    with pytest.raises(ValueError, match="level_scale"):
        explanation_schema(scale)


def test_mcq_schema_asks_only_judge_checked_flaws() -> None:
    schema = mcq_schema()
    assert list(schema["properties"]["iwf"]["properties"]) == [
        "1", "2", "5", "7", "8", "13", "14", "15", "17", "18"
    ]  # fmt: skip
    flaws = schema["properties"]["iwf"]["properties"]
    judgement = {
        "analysis": {"reasoning": "20 - 5 = 15", "judge_answer": "A"},
        "criteria": {"key_correct": judged("Yes")},
        "iwf": {key: {"evidence": "", "flag": False} for key in flaws},
    }
    assert schema_issues(schema, judgement) == []
    assert labels_of(MCQ, judgement) == {"key_correct": "Yes"}


def test_schema_ids_and_hashes() -> None:
    assert feedback_schema()["$id"] == "llmtutor/judge/feedback/v1"
    assert schema_sha256(explanation_schema(LEVELS)) != schema_sha256(
        explanation_schema(LEVELS[:2])
    )
    assert schema_sha256(feedback_schema()) == schema_sha256(feedback_schema())
