"""Compare two judges on identical BEA 2025 tutor responses."""

import argparse
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, confusion_matrix

from llmtutor.tasks.bea2025 import CRITERIA

LABELS = ["No", "To some extent", "Yes"]


def load_predictions(path: Path) -> dict:
    records = {}

    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            key = (record["conversation_id"], record["tutor"])

            if key in records:
                raise ValueError(f"Duplicate response: {key}")

            records[key] = record

    return records


def calculate_agreement(labels_a: list, labels_b: list) -> dict:
    matches = sum(a == b for a, b in zip(labels_a, labels_b, strict=True))

    return {
        "total": len(labels_a),
        "matches": matches,
        "agreement": matches / len(labels_a),
        "kappa": cohen_kappa_score(labels_a, labels_b, labels=LABELS),
        "quadratic_weighted_kappa": cohen_kappa_score(
            labels_a,
            labels_b,
            labels=LABELS,
            weights="quadratic",
        ),
        "confusion_matrix": confusion_matrix(labels_a, labels_b, labels=LABELS).tolist(),
    }


def score_pairs(pairs: list, criteria: tuple) -> dict:
    labels_a = [a["judge_labels"][criterion] for a, b in pairs for criterion in criteria]

    labels_b = [b["judge_labels"][criterion] for a, b in pairs for criterion in criteria]

    return calculate_agreement(labels_a, labels_b)


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    print(f"  Agreement: {metrics['matches']}/{metrics['total']} ({metrics['agreement']:.1%})")
    print(f"  Kappa: {metrics['kappa']:.3f}")
    print(f"  Weighted Kappa: {metrics['quadratic_weighted_kappa']:.3f}")

    print("  Confusion matrix (judge A rows, judge B columns):")
    for row in metrics["confusion_matrix"]:
        print(" ", row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("judge_a", type=Path)
    parser.add_argument("judge_b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records_a = load_predictions(args.judge_a)
    records_b = load_predictions(args.judge_b)

    if records_a.keys() != records_b.keys():
        raise ValueError("Prediction files contain different responses")

    pairs = []

    for key, a in records_a.items():
        b = records_b[key]

        if a["tutor_response"] != b["tutor_response"]:
            raise ValueError(f"Different tutor responses: {key}")

        if a["conversation_history"] != b["conversation_history"]:
            raise ValueError(f"Different conversation histories: {key}")

        if a.get("rubric_version") != b.get("rubric_version"):
            raise ValueError(f"Different rubric versions: {key}")

        if (
            a["error"] is None
            and b["error"] is None
            and a["judge_labels"] is not None
            and b["judge_labels"] is not None
        ):
            pairs.append((a, b))

    if not pairs:
        raise ValueError("No valid matched predictions")

    results = {
        "judge_a": pairs[0][0]["judge_model"],
        "judge_b": pairs[0][1]["judge_model"],
        "matched_responses": len(records_a),
        "valid_matched_responses": len(pairs),
        "label_order": LABELS,
        "criteria": {},
        "tutors": {},
    }

    print("Judge A:", results["judge_a"])
    print("Judge B:", results["judge_b"])
    print("Matched valid responses:", len(pairs))

    for criterion in CRITERIA:
        metrics = score_pairs(pairs, (criterion,))
        results["criteria"][criterion] = metrics
        print_metrics(criterion, metrics)

    results["overall"] = score_pairs(pairs, CRITERIA)
    print_metrics("Overall", results["overall"])

    for tutor in sorted({a["tutor"] for a, b in pairs}):
        subset = [(a, b) for a, b in pairs if a["tutor"] == tutor]

        metrics = score_pairs(subset, CRITERIA)
        results["tutors"][tutor] = metrics
        print_metrics(tutor, metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
