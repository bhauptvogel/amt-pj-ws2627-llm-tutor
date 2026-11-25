"""Test BEA 2025 scoring against hand-computed metrics."""

import json
import runpy
from pathlib import Path

import pytest

from llmtutor.tasks.bea2025 import CRITERIA


def test_judge_metrics(tmp_path: Path):
    human = ["No", "To some extent", "Yes", "Yes"]
    judge = ["No", "Yes", "Yes", "To some extent"]

    records = [
        {
            "human_labels": {criterion: h for criterion in CRITERIA},
            "judge_labels": {criterion: j for criterion in CRITERIA},
        }
        for h, j in zip(human, judge, strict=True)
    ]

    predictions = tmp_path / "predictions.jsonl"

    predictions.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "score_bea2025_judge.py"

    evaluate = runpy.run_path(str(script))["evaluate"]
    evaluate(predictions)

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    for name in (*CRITERIA, "overall"):
        values = metrics[name]

        assert values["accuracy"] == pytest.approx(0.5)
        assert values["macro_f1"] == pytest.approx(0.5)
        assert values["kappa"] == pytest.approx(0.2)
        assert values["quadratic_weighted_kappa"] == (pytest.approx(7 / 11))

    for criterion in CRITERIA:
        assert metrics[criterion]["confusion_matrix"] == [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
        ]
