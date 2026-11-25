"""Sanity tests: case construction, label checks and the command, with scripted judges."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from llmtutor.evaluation.__main__ import main
from llmtutor.evaluation.sanity import (
    DEFAULT_ITEMS,
    FEEDBACK_CASES,
    INJECTION,
    build_responses,
    load_sanity_items,
    run_sanity,
)
from llmtutor.types import ModelRequest, ModelResponse
from tests.test_evaluation_judge import CONFIG, answer, config_with
from tests.test_evaluation_schemas import feedback_judgement

TUTOR_RESPONSE = re.compile(r"<tutor_response>(.*?)</tutor_response>", re.DOTALL)


def tutor_text(request: ModelRequest) -> str:
    match = TUTOR_RESPONSE.search(request.messages[-1].content)
    assert match is not None
    return match.group(1)


def judge_from(labels_for: Callable[[str], Mapping[str, str]]) -> Callable[..., ModelResponse]:
    """A scripted judge that labels each response by its text."""

    def respond(request: ModelRequest) -> ModelResponse:
        judgement = feedback_judgement()
        for name, label in labels_for(tutor_text(request)).items():
            if name == "tutoring_move":
                judgement[name] = label
            else:
                judgement["criteria"][name]["label"] = label
        return answer(judgement)

    return respond


def case_of_text() -> dict[str, str]:
    items, _ = load_sanity_items(DEFAULT_ITEMS)
    records, _ = build_responses("feedback", items)
    return {record["tutor_response"]: record["item_id"].split("#")[1] for record in records}


def ideal(text: str) -> Mapping[str, str]:
    case = {c.id: c for c in FEEDBACK_CASES}[case_of_text()[text]]
    return case.expected


def write_items(tmp_path: Path, drop: str | None = None) -> Path:
    path = tmp_path / "items.jsonl"
    lines = []
    for line in DEFAULT_ITEMS.read_text(encoding="utf-8").splitlines()[:2]:
        record = json.loads(line)
        record.pop(drop, None)
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_builtin_items_are_usable_for_every_case() -> None:
    items, _ = load_sanity_items(DEFAULT_ITEMS)
    assert len(items) == 5
    for item in items:
        fields = item.fields
        assert fields["wrong_value"] != fields["final_answer"]
        # The expert hint is the S9 "does not reveal" case, so it must not contain the answer.
        assert fields["final_answer"] not in fields["expert_hint"]
    records, skipped = build_responses("feedback", items)
    assert skipped == []
    assert len(records) == 5 * len(FEEDBACK_CASES)


def test_cases_build_the_described_responses() -> None:
    items, _ = load_sanity_items(DEFAULT_ITEMS)
    records, _ = build_responses("feedback", items[:1])
    by_case = {record["item_id"].split("#")[1]: record for record in records}
    assert by_case["S4"]["tutor_response"] == "The answer is 15 euros."
    assert by_case["S5"]["tutor_response"] == "The answer is 25 euros."
    assert by_case["S6"]["tutor_response"].endswith("the answer is 15 euros.")
    assert by_case["S7"]["tutor_response"].count(items[0].fields["expert_hint"]) == 6
    assert by_case["S11"]["tutor_response"].endswith(INJECTION)
    # Case-only fields never reach the judge as item fields.
    assert set(by_case["S1"]["item"]) == {
        "question", "student_answer", "hidden_solution", "final_answer", "error_note"
    }  # fmt: skip
    assert by_case["S1"]["tutor_model"] == "sanity"


def test_a_judge_that_gets_every_case_right_passes(tmp_path: Path) -> None:
    config, model = config_with(tmp_path, judge_from(ideal))

    report = run_sanity(config, DEFAULT_ITEMS, tmp_path / "runs")

    assert report.passed, report.failures
    assert report.run_status == "completed"
    # Per item, hand-counted: S1 5, S2 5, S3 3, S4 3, S5 2, S6 1, S7 9 scored criteria,
    # S8 3, S9 5, S10 5, S11 5 = 46 checks; five items.
    assert len(report.checks) == 230
    assert len(model.requests) == 55

    artefact = tmp_path / "runs" / report.run_id
    assert "-sanity-" in report.run_id
    manifest = json.loads((artefact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "sanity"
    saved = json.loads((artefact / "sanity.json").read_text(encoding="utf-8"))
    assert saved["passed"] is True and saved["blocked_criteria"] == []


def test_a_lenient_judge_blocks_the_criteria_it_misses(tmp_path: Path) -> None:
    config, _ = config_with(tmp_path, judge_from(lambda text: {}))

    report = run_sanity(config, write_items(tmp_path), tmp_path / "runs")

    assert not report.passed
    assert "mistake_identification" in report.blocked  # S1 "Yes." judged as identifying it
    assert "revealing_answer" in report.blocked  # S4 reveal judged as not revealing
    assert "over_withholding" in report.blocked  # S8 "Try again." judged as helpful
    failure = next(f for f in report.failures if f.case == "S1")
    assert (failure.expected, failure.actual) == ("No", "Yes")


def test_a_judge_fooled_by_padding_fails_s7(tmp_path: Path) -> None:
    texts = case_of_text()

    def fooled(text: str) -> Mapping[str, str]:
        case = texts[text]
        if case == "S9":
            return {**ideal(text), "actionability": "To some extent"}
        return {} if case == "S7" else ideal(text)

    config, _ = config_with(tmp_path, judge_from(fooled))

    report = run_sanity(config, write_items(tmp_path), tmp_path / "runs")

    s7 = [f for f in report.failures if f.case == "S7"]
    assert [(f.criterion, f.expected, f.actual) for f in s7] == [
        ("actionability", "no better than S9 (To some extent)", "Yes"),
        ("actionability", "no better than S9 (To some extent)", "Yes"),
    ]


def test_cases_needing_missing_fields_are_skipped_not_failed(tmp_path: Path) -> None:
    config, model = config_with(tmp_path, judge_from(ideal))

    report = run_sanity(config, write_items(tmp_path, drop="expert_hint"), tmp_path / "runs")

    assert report.passed
    skipped = sorted({skip.case for skip in report.skipped})
    assert skipped == ["S11", "S6", "S7", "S9"]
    assert len(model.requests) == 2 * (len(FEEDBACK_CASES) - 4)


def test_judge_errors_fail_their_checks(tmp_path: Path) -> None:
    config, _ = config_with(tmp_path, lambda request: answer("I think it is fine."))

    report = run_sanity(config, write_items(tmp_path), tmp_path / "runs")

    assert not report.passed
    assert all(check.actual is None for check in report.checks)
    assert set(report.blocked) >= {"mistake_identification", "revealing_answer"}


def test_sanity_command_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "judge.toml"
    config.write_text(CONFIG.format(per_call="all", optional="[]"), encoding="utf-8")

    # The fake adapter answers "hello" once and is then exhausted: every response fails.
    code = main(["sanity", "--config", str(config), "--items", str(write_items(tmp_path))])

    out = capsys.readouterr().out
    assert code == 1
    assert "sanity FAILED; 0/" in out
    assert "blocked criteria:" in out
    assert "FAIL sanity-01 S1 mistake_identification: expected No, got None" in out


def test_sanity_command_rejects_a_missing_items_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "judge.toml"
    config.write_text(CONFIG.format(per_call="all", optional="[]"), encoding="utf-8")
    assert main(["sanity", "--config", str(config), "--items", "nope.jsonl"]) == 2
    assert capsys.readouterr().err.startswith("error:")
