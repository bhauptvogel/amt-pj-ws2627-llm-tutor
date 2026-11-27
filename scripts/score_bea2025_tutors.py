import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from llmtutor.tasks.bea2025 import CRITERIA

LABELS = ("Yes", "To some extent", "No")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    with args.predictions.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]

    grouped = defaultdict(list)

    for record in records:
        grouped[record["tutor"]].append(record)

    results = {
        "judge_model": records[0]["judge_model"],
        "total_responses": len(records),
        "models": {},
    }

    print(f"Judge: {results['judge_model']}")
    print(f"Total responses: {len(records)}")

    for model, responses in grouped.items():
        valid = [r for r in responses if r["error"] is None and r["judge_labels"] is not None]

        print("\n" + "=" * 60)
        print(f"Tutor: {model}")
        print(f"Responses: {len(responses)}")
        print(f"Valid predictions: {len(valid)}")

        model_results = {
            "total_responses": len(responses),
            "valid_predictions": len(valid),
            "criteria": {},
        }

        for criterion in CRITERIA:
            counts = Counter(r["judge_labels"][criterion] for r in valid)

            distribution = {}

            print(f"\n{criterion}")

            for label in LABELS:
                count = counts[label]
                percentage = 100 * count / len(valid) if valid else 0

                distribution[label] = {
                    "count": count,
                    "percentage": round(percentage, 2),
                }

                print(f"  {label}: {count}/{len(valid)} ({percentage:.1f}%)")

            model_results["criteria"][criterion] = distribution

        results["models"][model] = model_results

    output_path = args.predictions.parent / "tutor_metrics.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
