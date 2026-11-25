"""Calculate judge agreement with MRBench human annotations."""

import argparse
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from llmtutor.tasks.bea2025 import CRITERIA

LABELS = ["No", "To some extent", "Yes"]


def evaluate(path: Path):
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]

    valid = [r for r in records if r["judge_labels"] is not None]

    print(f"Cases: {len(records)}")
    print(f"Valid predictions: {len(valid)}")

    if not valid:
        raise ValueError("No valid judge predictions to evaluate")

    metrics = {}
    all_human = []
    all_judge = []

    for criterion in CRITERIA:
        human = [r["human_labels"][criterion] for r in valid]
        judge = [r["judge_labels"][criterion] for r in valid]

        all_human.extend(human)
        all_judge.extend(judge)

        metrics[criterion] = {
            "accuracy": accuracy_score(human, judge),
            "macro_f1": f1_score(human, judge, labels=LABELS, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(human, judge),
            "quadratic_weighted_kappa": cohen_kappa_score(
                human, judge, labels=LABELS, weights="quadratic"
            ),
            "confusion_matrix": confusion_matrix(human, judge, labels=LABELS).tolist(),
        }

    metrics["overall"] = {
        "accuracy": accuracy_score(all_human, all_judge),
        "macro_f1": f1_score(all_human, all_judge, labels=LABELS, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(all_human, all_judge),
        "quadratic_weighted_kappa": cohen_kappa_score(
            all_human, all_judge, labels=LABELS, weights="quadratic"
        ),
    }

    for name, values in metrics.items():
        print(f"\n{name}")
        print(f"  Agreement: {values['accuracy']:.1%}")
        print(f"  Macro-F1:  {values['macro_f1']:.3f}")
        print(f"  Kappa:     {values['kappa']:.3f}")
        print(f"  Weighted Kappa: {values['quadratic_weighted_kappa']:.3f}")

        if "confusion_matrix" in values:
            print("  Confusion matrix (human rows, judge columns):")
            for row in values["confusion_matrix"]:
                print(" ", row)

    output = path.parent / "metrics.json"
    output.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"\nMetrics saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    evaluate(args.predictions)
