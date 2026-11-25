"""The judge: prompt rendering, output parsing, retry rules, artefacts and the command."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from llmtutor.config import ConfigError
from llmtutor.evaluation import judge as judge_module
from llmtutor.evaluation.__main__ import main
from llmtutor.evaluation.criteria import FEEDBACK
from llmtutor.evaluation.judge import (
    JudgeConfig,
    check_responses,
    load_judge_config,
    parse_output,
    run_judge,
)
from llmtutor.evaluation.prompts import PROMPTS_DIR, PromptError, load_template
from llmtutor.evaluation.responses import ResponsesError, load_responses
from llmtutor.types import ErrorKind, ModelError, ModelRequest, ModelResponse
from tests.test_evaluation_schemas import feedback_judgement, judged

ITEM = {
    "question": "A shirt costs 20 euros and is reduced by 25%. What does it cost now?",
    "student_answer": "5 euros {because 25% of 20 is 5}",
    "hidden_solution": "25% of 20 is 5; 20 - 5 = 15.",
    "final_answer": "15 euros",
    "error_note": "Computed the discount but did not subtract it from the price.",
}

CONFIG = """
[judge]
task = "feedback"
prompt_version = 1
criteria_per_call = "{per_call}"
optional_criteria = {optional}

[sampling]
temperature = 0.0
max_tokens = 4000
seed = 7

[model]
name = "judge"
adapter = "fake"
scenario = "ok"
"""

Outcome = ModelResponse | BaseException


class Scripted:
    """Answers each call from a script: a list of outcomes or a function of the request."""

    def __init__(self, script: list[Outcome] | Callable[[ModelRequest], Outcome]) -> None:
        self._script = script
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        outcome = self._script(request) if callable(self._script) else self._script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def answer(value: Any) -> ModelResponse:
    text = value if isinstance(value, str) else json.dumps(value)
    return ModelResponse(
        text=text, model="judge-2026-01-01", finish_reason="stop", usage=None, raw=None
    )


def write_responses(tmp_path: Path, *ids: str, item: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "responses.jsonl"
    lines = [
        json.dumps(
            {
                "item_id": item_id,
                "task": "feedback",
                "tutor_model": "secret-tutor-model",
                "tutor_response": "Is 5 euros the new price, or the amount you save?",
                "item": ITEM if item is None else item,
            }
        )
        for item_id in ids
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def config_with(
    tmp_path: Path,
    script: list[Outcome] | Callable[[ModelRequest], Outcome],
    *,
    per_call: str = "all",
    optional: str = "[]",
) -> tuple[JudgeConfig, Scripted]:
    path = tmp_path / "judge.toml"
    path.write_text(CONFIG.format(per_call=per_call, optional=optional), encoding="utf-8")
    model = Scripted(script)
    return replace(load_judge_config(path, {}), model=model), model


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def only_artefact(tmp_path: Path) -> Path:
    (artefact,) = (tmp_path / "runs").iterdir()
    return artefact


# --- prompts -------------------------------------------------------------------------


def test_all_criteria_prompt_carries_item_rubric_and_output_order() -> None:
    template = load_template(FEEDBACK, 1)
    values = {**ITEM, "tutor_response": "Check your last step."}
    system, user = template.messages(FEEDBACK.criteria, values)

    assert system.role == "system" and "strict, expert evaluator" in system.content
    assert "<student_answer>5 euros {because 25% of 20 is 5}</student_answer>" in user.content
    assert "<dialogue_context>None</dialogue_context>" in user.content
    assert "{" + "question}" not in user.content
    positions = [user.content.index(f"\n{c.name} ") for c in FEEDBACK.criteria]
    assert positions == sorted(positions)
    assert 'keys in exactly this order: "analysis", "criteria", "tutoring_move".' in user.content


def test_one_criterion_prompt_has_only_that_block() -> None:
    template = load_template(FEEDBACK, 1)
    values = {**ITEM, "tutor_response": "x"}
    _, user = template.messages((FEEDBACK.criterion("actionability"),), values)
    assert "\nactionability - " in user.content
    assert "mistake_location - " not in user.content
    assert 'these criteria in this order: "actionability".' in user.content

    _, diagnostic = template.messages((FEEDBACK.criterion("tutoring_move"),), values)
    assert 'exactly this order: "analysis", "tutoring_move".' in diagnostic.content
    assert 'Inside "criteria"' not in diagnostic.content


def test_missing_item_value_is_a_prompt_error() -> None:
    template = load_template(FEEDBACK, 1)
    values = {key: value for key, value in ITEM.items() if key != "error_note"}
    with pytest.raises(PromptError, match="error_note, tutor_response"):
        template.messages(FEEDBACK.criteria, values)


def test_template_without_a_criterion_block_is_rejected(tmp_path: Path) -> None:
    shutil.copytree(PROMPTS_DIR / "feedback", tmp_path / "feedback")
    (tmp_path / "feedback" / "v1" / "criteria" / "coherence.txt").unlink()
    with pytest.raises(PromptError, match=r"missing \['coherence'\]"):
        load_template(FEEDBACK, 1, root=tmp_path)


def test_template_hash_changes_with_any_file(tmp_path: Path) -> None:
    shutil.copytree(PROMPTS_DIR / "feedback", tmp_path / "feedback")
    before = load_template(FEEDBACK, 1, root=tmp_path).sha256
    assert before == load_template(FEEDBACK, 1).sha256
    with (tmp_path / "feedback" / "v1" / "system.txt").open("a", encoding="utf-8") as handle:
        handle.write("\n7. Be brief.")
    assert load_template(FEEDBACK, 1, root=tmp_path).sha256 != before


# --- output parsing ------------------------------------------------------------------


def test_parse_output_tolerates_one_code_fence() -> None:
    parsed, issues = parse_output('```json\n{"a": 1}\n```')
    assert (parsed, issues) == ({"a": 1}, [])


@pytest.mark.parametrize(
    ("text", "issue"),
    [
        (None, "output is empty"),
        ("Here is my evaluation: {", "output is not JSON"),
        ("[1, 2]", "output is not a JSON object"),
        ('{"label": "Yes", "label": "No"}', "output repeats key(s): label"),
    ],
)
def test_parse_output_reports_why_it_failed(text: str | None, issue: str) -> None:
    parsed, issues = parse_output(text)
    assert parsed is None and issues[0].startswith(issue)


# --- judge runs ----------------------------------------------------------------------


def test_valid_judgement_is_recorded_with_provenance(tmp_path: Path) -> None:
    config, model = config_with(tmp_path, [answer(feedback_judgement())])
    responses = load_responses(write_responses(tmp_path, "t1-0001"))

    result = run_judge(config, responses, tmp_path / "runs")

    assert result.status == "completed"
    assert result.counts == {"responses": 1, "ok": 1, "retried": 0, "judge_error": 0, "attempts": 1}
    (record,) = read_jsonl(result.artefact / "judgements.jsonl")
    assert record["status"] == "ok"
    assert record["tutor_model"] == "secret-tutor-model"
    assert record["labels"]["revealing_answer"] == "No"
    assert record["labels"]["tutoring_move"] == "Probing"
    assert record["calls"][0]["schema_id"] == "llmtutor/judge/feedback/v1"
    assert record["prompt_sha256"] == config.template.sha256

    sent = " ".join(message.content for message in model.requests[0].messages)
    assert "secret-tutor-model" not in sent
    assert model.requests[0].seed == 7

    manifest = json.loads((result.artefact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["reported_models"] == ["judge-2026-01-01"]
    assert manifest["responses"]["sha256"] == responses.sha256
    assert manifest["policy"]["output_retries"] == 2
    assert (result.artefact / "config.toml").read_bytes() == config.raw


def test_invalid_output_is_retried_then_accepted(tmp_path: Path) -> None:
    broken = feedback_judgement()
    broken["criteria"]["coherence"]["label"] = "Mostly"
    config, _ = config_with(
        tmp_path, [answer("not json"), answer(broken), answer(feedback_judgement())]
    )

    result = run_judge(config, load_responses(write_responses(tmp_path, "t1")), tmp_path / "runs")

    (record,) = read_jsonl(result.artefact / "judgements.jsonl")
    assert record["status"] == "retried"
    assert record["calls"][0]["output_retries"] == 2
    calls = read_jsonl(result.artefact / "calls.jsonl")
    assert [call["outcome"] for call in calls] == ["invalid_output", "invalid_output", "ok"]
    assert calls[1]["output_issues"][0].startswith("criteria/coherence/label:")


def test_third_invalid_output_marks_judge_error_and_the_run_goes_on(tmp_path: Path) -> None:
    script = [answer("nope")] * 3 + [answer(feedback_judgement())]
    config, _ = config_with(tmp_path, script)
    responses = load_responses(write_responses(tmp_path, "t1", "t2"))

    result = run_judge(config, responses, tmp_path / "runs")

    assert result.status == "completed"
    first, second = read_jsonl(result.artefact / "judgements.jsonl")
    assert first["status"] == "judge_error"
    assert first["labels"] == {}
    assert first["calls"][0]["raw_output"] == "nope"
    assert first["calls"][0]["error"] == "invalid output"
    assert second["status"] == "ok"
    assert result.counts["judge_error"] == 1 and result.counts["attempts"] == 4


def test_transport_failures_use_the_runner_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float] = []
    monkeypatch.setattr(judge_module, "sleep", waits.append)
    rate_limit = ModelError(ErrorKind.RATE_LIMIT, "slow down", status=429, retry_after=5)
    config, _ = config_with(tmp_path, [rate_limit, answer(feedback_judgement())])

    result = run_judge(config, load_responses(write_responses(tmp_path, "t1")), tmp_path / "runs")

    (record,) = read_jsonl(result.artefact / "judgements.jsonl")
    assert record["status"] == "ok"
    assert record["calls"][0]["attempts"] == 2
    assert waits == [5.0]


def test_exhausted_transport_retries_mark_judge_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(judge_module, "sleep", lambda _: None)
    config, _ = config_with(
        tmp_path, [ModelError(ErrorKind.PROVIDER, "overloaded")] * judge_module.TRANSPORT_ATTEMPTS
    )

    result = run_judge(config, load_responses(write_responses(tmp_path, "t1")), tmp_path / "runs")

    (record,) = read_jsonl(result.artefact / "judgements.jsonl")
    assert record["status"] == "judge_error"
    assert record["calls"][0]["error"] == "provider: overloaded"


def test_rejected_key_aborts_the_run(tmp_path: Path) -> None:
    config, _ = config_with(tmp_path, [ModelError(ErrorKind.AUTH, "bad key", status=401)])

    result = run_judge(
        config, load_responses(write_responses(tmp_path, "t1", "t2")), tmp_path / "runs"
    )

    assert result.status == "aborted"
    assert result.error == "auth: bad key"
    manifest = json.loads((result.artefact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "aborted"
    assert not (result.artefact / "judgements.jsonl").exists()


def _answer_requested(request: ModelRequest) -> ModelResponse:
    """A judge that answers exactly the criteria the output instruction names."""

    full = feedback_judgement()
    full["criteria"]["human_likeness"] = judged("To some extent")
    output = request.messages[-1].content.split("## Output", 1)[1]
    top, *rest = output.strip().splitlines()
    out: dict[str, Any] = {
        key: full[key] for key in re.findall(r'"([a-z_]+)"', top) if key != "criteria"
    }
    if rest:
        names = re.findall(r'"([a-z_]+)"', rest[0].split("in this order:")[1].split(". Each")[0])
        out = {"analysis": full["analysis"], "criteria": {n: full["criteria"][n] for n in names}}
    return answer(out)


def test_one_criterion_per_call_merges_labels(tmp_path: Path) -> None:
    config, model = config_with(
        tmp_path, _answer_requested, per_call="one", optional='["human_likeness"]'
    )

    result = run_judge(config, load_responses(write_responses(tmp_path, "t1")), tmp_path / "runs")

    (record,) = read_jsonl(result.artefact / "judgements.jsonl")
    assert record["status"] == "ok"
    assert len(record["calls"]) == len(model.requests) == len(FEEDBACK.criteria)
    assert record["labels"]["human_likeness"] == "To some extent"
    assert record["labels"]["tutoring_move"] == "Probing"
    assert set(record["labels"]) == {criterion.name for criterion in FEEDBACK.criteria}


def test_key_order_is_recorded_but_does_not_fail(tmp_path: Path) -> None:
    judgement = feedback_judgement()
    judgement["criteria"]["coherence"] = {"label": "Yes", "evidence": "", "reasoning": "fine"}
    config, _ = config_with(tmp_path, [answer(judgement)])

    result = run_judge(config, load_responses(write_responses(tmp_path, "t1")), tmp_path / "runs")

    (record,) = read_jsonl(result.artefact / "judgements.jsonl")
    assert record["status"] == "ok"
    assert record["calls"][0]["key_order_issues"][0].startswith("criteria/coherence:")


def test_response_missing_an_item_field_fails_before_any_call(tmp_path: Path) -> None:
    config, model = config_with(tmp_path, [])
    item = {key: value for key, value in ITEM.items() if key != "final_answer"}
    responses = load_responses(write_responses(tmp_path, "t1", item=item))
    with pytest.raises(ConfigError, match="lacks field\\(s\\): final_answer"):
        check_responses(config, responses)
    with pytest.raises(ConfigError):
        run_judge(config, responses, tmp_path / "runs")
    assert not (tmp_path / "runs").exists() and model.requests == []


# --- config --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('task = "feedback"', 'task = "explanation"', "has no judge yet"),
        ('task = "feedback"', 'task = "essay"', "'task' must be one of"),
        ("prompt_version = 1", "prompt_version = 9", "no prompt template 'feedback' version 9"),
        ('criteria_per_call = "all"', 'criteria_per_call = "some"', "'all' or 'one'"),
        ("optional_criteria = []", 'optional_criteria = ["coherence"]', "may name only"),
        ('name = "judge"', 'name = "Judge Model"', "'name' must match"),
        ("[judge]", "[judge]\nshuffle = true", "unknown key(s): shuffle"),
    ],
)
def test_bad_judge_configs_are_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    path = tmp_path / "judge.toml"
    text = CONFIG.format(per_call="all", optional="[]").replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_judge_config(path, {})


def test_shipped_feedback_config_names_its_key_variable(tmp_path: Path) -> None:
    shipped = Path(__file__).parents[1] / "configs" / "judge-feedback.toml"
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY is not set"):
        load_judge_config(shipped, {})
    config = load_judge_config(shipped, {"OPENROUTER_API_KEY": "sk-test"})
    assert config.sampling.temperature == 0.0 and config.criteria_per_call == "all"


# --- responses file ------------------------------------------------------------------


def test_responses_render_lists_as_json(tmp_path: Path) -> None:
    item = {**ITEM, "key_points": ["out of 100", "50% is half"], "level": 3}
    responses = load_responses(write_responses(tmp_path, "t1", item=item))
    (response,) = responses.responses
    assert response.item["key_points"] == '["out of 100", "50% is half"]'
    assert response.item["level"] == "3"
    assert response.values()["tutor_response"].startswith("Is 5 euros")


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("{", "invalid JSON"),
        (
            '{"item_id": "a", "task": "feedback", "tutor_model": "m", "item": {}}',
            "'tutor_response'",
        ),
        (
            '{"item_id": "a", "task": "feedback", "tutor_model": "m", "tutor_response": "", '
            '"item": {}, "extra": 1}',
            "unknown key(s): extra",
        ),
        (
            '{"item_id": "a", "task": "feedback", "tutor_model": "m", "tutor_response": "", '
            '"item": {"x": null}}',
            "item.x must be",
        ),
    ],
)
def test_malformed_responses_are_rejected(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "responses.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ResponsesError, match=re.escape(message)):
        load_responses(path)


def test_duplicate_responses_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ResponsesError, match="duplicate item 't1'"):
        load_responses(write_responses(tmp_path, "t1", "t1"))


# --- command -------------------------------------------------------------------------


def test_command_reports_judge_errors_with_exit_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "judge.toml"
    config.write_text(CONFIG.format(per_call="all", optional="[]"), encoding="utf-8")
    responses = write_responses(tmp_path, "t1")

    # The fake adapter's "ok" scenario answers "hello", which is not a judgement.
    code = main(["judge", "--config", str(config), "--responses", str(responses)])

    assert code == 1
    assert "1 judged, 0 ok, 0 retried, 1 judge_error" in capsys.readouterr().out
    (record,) = read_jsonl(only_artefact(tmp_path) / "judgements.jsonl")
    assert record["calls"][0]["attempts"] == 3
    assert record["calls"][0]["error"] == "bad_response: AssertionError: scenario 'ok' exhausted"


def test_command_rejects_bad_input_with_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["judge", "--config", "missing.toml", "--responses", "missing.jsonl"])
    assert code == 2
    assert capsys.readouterr().err.startswith("error:")
    assert not (tmp_path / "runs").exists()


def test_v2_puts_the_tutoring_move_in_its_own_section() -> None:
    template = load_template(FEEDBACK, 2)
    assert template.sha256 != load_template(FEEDBACK, 1).sha256
    _, user = template.messages(FEEDBACK.criteria, {**ITEM, "tutor_response": "x"})
    step2 = user.content.index("## Step 2 - Tutoring move")
    assert user.content.index("over_withholding - ") < step2 < user.content.index("## Output")
