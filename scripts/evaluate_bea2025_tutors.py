import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from evaluate_bea2025_judge import judge_case

from llmtutor.models.ollama import OllamaAdapter, OllamaSettings
from llmtutor.tasks.bea2025 import load_bea2025
from llmtutor.tasks.bea2025_rubric import RUBRIC_VERSION, SYSTEM_PROMPT

DATASET_PATH = Path("data/pilot/bea2025_pilot20.json")


DEFAULT_JUDGE = "qwen3:4b-instruct-2507-q4_K_M"


def prepare_cases(dataset_path: Path, responses_path: Path):
    examples = load_bea2025(dataset_path)

    conversations = {example.conversation_id: example.conversation_history for example in examples}

    with responses_path.open(encoding="utf-8") as file:
        responses = [json.loads(line) for line in file]

    cases = []

    for response in responses:
        if response["error"] is not None:
            continue

        cases.append(
            {
                "conversation_id": response["conversation_id"],
                "conversation_history": conversations[response["conversation_id"]],
                "tutor": response["model"],
                "tutor_response": response["response"],
                "human_labels": None,
            }
        )

    return cases


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=DEFAULT_JUDGE,
        help="Ollama model to use as judge",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of responses for testing",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the BEA 2025 dataset",
    )

    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="Path to generated tutor responses JSONL",
    )

    args = parser.parse_args()

    cases = prepare_cases(args.dataset, args.responses)

    if args.limit is not None:
        cases = cases[: args.limit]

    judge = OllamaAdapter(
        OllamaSettings(
            base_url="http://localhost:11434",
            model=args.model,
            think=False,
        )
    )

    output_dir = Path("runs") / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + f"-bea2025-tutor-eval-{args.model.replace(':', '-')}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"

    print(f"Judge: {args.model}")
    print(f"Tutor responses: {len(cases)}")
    print(f"Saving results to: {output_path}")

    with output_path.open("w", encoding="utf-8") as file:
        for index, case in enumerate(cases, start=1):
            result = judge_case(judge, case, SYSTEM_PROMPT, args.model)
            result["rubric_version"] = RUBRIC_VERSION

            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()

            print(
                f"[{index}/{len(cases)}] "
                f"{case['tutor']} | "
                f"{'OK' if result['error'] is None else 'ERROR'}"
            )

    print("\nEvaluation complete.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
