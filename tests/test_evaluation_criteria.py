"""Rubrics: label sets, integer coding and the derived MCQ verdict."""

from __future__ import annotations

import pytest

from llmtutor.evaluation.criteria import (
    FEEDBACK,
    ITEM_WRITING_FLAWS,
    RUBRICS,
    Checker,
    Criterion,
    Scale,
    judge_checked_flaws,
    mcq_acceptable,
)


def test_ternary_labels_code_worst_to_best() -> None:
    guidance = FEEDBACK.criterion("providing_guidance")
    assert guidance.scale is Scale.ORDINAL
    assert [guidance.code(label) for label in ("No", "To some extent", "Yes")] == [0, 1, 2]
    assert guidance.label(1) == "To some extent"


def test_desired_labels_follow_the_benchmark_constraints() -> None:
    assert FEEDBACK.criterion("revealing_answer").desired == "No"
    assert FEEDBACK.criterion("over_withholding").desired == "No"
    assert FEEDBACK.criterion("mistake_identification").desired == "Yes"


def test_unknown_label_and_code_are_rejected() -> None:
    reveal = FEEDBACK.criterion("revealing_answer")
    with pytest.raises(ValueError, match="no label 'yes'"):
        reveal.code("yes")
    with pytest.raises(ValueError, match="no code 3"):
        reveal.label(3)


def test_diagnostic_criteria_are_not_scored() -> None:
    names = [criterion.name for criterion in FEEDBACK.scored]
    assert "tutoring_move" not in names
    assert len(names) == len(FEEDBACK.criteria) - 1


def test_every_task_has_a_rubric() -> None:
    assert set(RUBRICS) == {"feedback", "explanation", "mcq"}
    assert all(rubric.version >= 1 for rubric in RUBRICS.values())


def test_criterion_rejects_an_unknown_desired_label() -> None:
    with pytest.raises(ValueError, match="desired label"):
        Criterion("x", "?", Scale.ORDINAL, ("No", "Yes"), "Maybe")


def test_binary_criterion_needs_two_labels() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        Criterion("x", "?", Scale.BINARY, ("No", "Maybe", "Yes"), "Yes")


def test_flaw_split_covers_all_nineteen_once() -> None:
    assert [flaw.id for flaw in ITEM_WRITING_FLAWS] == list(range(1, 20))
    judged = {flaw.id for flaw in judge_checked_flaws()}
    assert judged == {1, 2, 5, 7, 8, 13, 14, 15, 17, 18}
    rules = {flaw.id for flaw in ITEM_WRITING_FLAWS if flaw.checker is Checker.RULE}
    assert rules == {3, 4, 6, 9, 10, 11, 12, 16, 19}


@pytest.mark.parametrize(
    ("flags", "key", "expected"),
    [(0, "Yes", True), (1, "Yes", True), (2, "Yes", False), (0, "No", False)],
)
def test_mcq_acceptable(flags: int, key: str, expected: bool) -> None:
    assert mcq_acceptable(flags, key) is expected
