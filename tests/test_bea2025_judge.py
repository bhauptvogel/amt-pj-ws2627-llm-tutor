"""Tests for BEA 2025 judge execution and error handling."""

import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

from llmtutor.tasks.bea2025 import CRITERIA
from llmtutor.tasks.bea2025_rubric import SYSTEM_PROMPT

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_bea2025_judge.py"

judge_case = runpy.run_path(str(SCRIPT))["judge_case"]

MODEL = "qwen3:8b"

CASE = {
    "conversation_id": "test-conversation",
    "conversation_history": "Student: 2 + 2 = 5",
    "tutor": "test-tutor",
    "tutor_response": "Can you count two more after two?",
    "human_labels": None,
}


def test_valid_judge_response():
    judge = Mock()
    labels = {criterion: "Yes" for criterion in CRITERIA}
    judge.generate.return_value.text = json.dumps(labels)

    result = judge_case(judge, CASE, SYSTEM_PROMPT, MODEL)

    assert result["judge_model"] == MODEL
    assert result["judge_labels"] == labels
    assert result["error"] is None

    request = judge.generate.call_args.args[0]

    assert request.temperature == 0.0
    assert request.max_tokens == 256
    assert request.seed == 42
    assert request.messages[0].content == SYSTEM_PROMPT
    assert request.messages[1].content == (
        "CONVERSATION:\nStudent: 2 + 2 = 5\n\nTUTOR RESPONSE:\nCan you count two more after two?"
    )


@pytest.mark.parametrize(
    "raw_text",
    [
        "This is not JSON",
        "[]",
        "{}",
        json.dumps({criterion: "Maybe" for criterion in CRITERIA}),
        json.dumps(
            {
                **{criterion: "Yes" for criterion in CRITERIA},
                "unexpected": "Yes",
            }
        ),
    ],
)
def test_invalid_judge_response(raw_text):
    judge = Mock()
    judge.generate.return_value.text = raw_text

    result = judge_case(judge, CASE, SYSTEM_PROMPT, MODEL)

    assert result["judge_model"] == MODEL
    assert result["judge_labels"] is None
    assert result["error"] is not None
    assert result["raw_judge_output"] == raw_text


def test_missing_judge_response():
    judge = Mock()
    judge.generate.return_value.text = None

    result = judge_case(judge, CASE, SYSTEM_PROMPT, MODEL)

    assert result["judge_model"] == MODEL
    assert result["judge_labels"] is None
    assert result["raw_judge_output"] is None
    assert "Judge returned no text" in result["error"]


def test_model_execution_failure():
    judge = Mock()
    judge.generate.side_effect = RuntimeError("Connection failed")

    result = judge_case(judge, CASE, SYSTEM_PROMPT, MODEL)

    assert result["judge_model"] == MODEL
    assert result["judge_labels"] is None
    assert result["raw_judge_output"] is None
    assert "Connection failed" in result["error"]


def test_judge_can_continue_after_failure():
    judge = Mock()

    labels = {criterion: "Yes" for criterion in CRITERIA}

    judge.generate.side_effect = [
        RuntimeError("Temporary failure"),
        Mock(text=json.dumps(labels)),
    ]

    first = judge_case(judge, CASE, SYSTEM_PROMPT, MODEL)
    second = judge_case(judge, CASE, SYSTEM_PROMPT, MODEL)

    assert first["error"] is not None
    assert second["error"] is None
    assert second["judge_labels"] == labels
