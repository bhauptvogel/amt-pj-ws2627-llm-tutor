"""Tests for BEA 2025 inter-judge agreement metrics."""

import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_bea2025_judges.py"

calculate_agreement = runpy.run_path(str(SCRIPT))["calculate_agreement"]


def test_inter_judge_agreement():
    labels_a = [
        "No",
        "To some extent",
        "Yes",
        "Yes",
    ]

    labels_b = [
        "No",
        "Yes",
        "Yes",
        "To some extent",
    ]

    metrics = calculate_agreement(labels_a, labels_b)

    assert metrics["total"] == 4
    assert metrics["matches"] == 2
    assert metrics["agreement"] == pytest.approx(0.5)
    assert metrics["kappa"] == pytest.approx(0.2)

    assert metrics["quadratic_weighted_kappa"] == (pytest.approx(7 / 11))

    assert metrics["confusion_matrix"] == [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
    ]


def test_perfect_agreement():
    labels = ["No", "To some extent", "Yes"]

    metrics = calculate_agreement(labels, labels)

    assert metrics["agreement"] == pytest.approx(1.0)
    assert metrics["kappa"] == pytest.approx(1.0)
    assert metrics["quadratic_weighted_kappa"] == (pytest.approx(1.0))


def test_different_label_counts():
    with pytest.raises(ValueError):
        calculate_agreement(
            ["No", "Yes"],
            ["No"],
        )
