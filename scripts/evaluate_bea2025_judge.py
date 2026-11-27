import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from llmtutor.models.ollama import OllamaAdapter, OllamaSettings
from llmtutor.tasks.bea2025 import CRITERIA, load_bea2025
from llmtutor.tasks.bea2025_rubric import RUBRIC_VERSION, SYSTEM_PROMPT
from llmtutor.types import Message, ModelRequest

DATASET_PATH = Path("data/pilot/bea2025_pilot20.json")


def prepare_cases(dataset_path: Path):
    examples = load_bea2025(dataset_path)

    # First five conversations were used for prompt development.
    # Keep them separate from the remaining conversations.
    development = examples[:5]
    evaluation = examples[5:]

    cases = []

    for example in evaluation:
        for reference in example.reference_responses[:2]:
            cases.append(
                {
                    "conversation_id": example.conversation_id,
                    "conversation_history": example.conversation_history,
                    "tutor": reference.tutor,
                    "tutor_response": reference.text,
                    "human_labels": reference.annotations,
                }
            )

    return development, cases


JUDGE_MODEL = "qwen3:4b-instruct-2507-q4_K_M"


def judge_case(judge, case, system_prompt, model_name: str):
    request = ModelRequest(
        messages=(
            Message(role="system", content=system_prompt),
            Message(
                role="user",
                content=(
                    f"CONVERSATION:\n{case['conversation_history']}\n\n"
                    f"TUTOR RESPONSE:\n{case['tutor_response']}"
                ),
            ),
        ),
        temperature=0.0,
        max_tokens=256,
        seed=42,
    )

    raw_text = None

    try:
        response = judge.generate(request)

        if not isinstance(response.text, str):
            raise ValueError("Judge returned no text")

        raw_text = response.text.strip()

    except Exception as exc:
        return {
            **case,
            "judge_model": model_name,
            "judge_labels": None,
            "raw_judge_output": raw_text,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        predictions = json.loads(raw_text)

        if not isinstance(predictions, dict):
            raise ValueError("Judge output is not a JSON object")

        if set(predictions) != set(CRITERIA):
            raise ValueError("Incorrect criterion keys")

        if any(
            predictions[criterion] not in ("Yes", "To some extent", "No") for criterion in CRITERIA
        ):
            raise ValueError("Invalid criterion label")

        error = None

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        predictions = None
        error = str(exc)

    return {
        **case,
        "judge_model": model_name,
        "judge_labels": predictions,
        "raw_judge_output": raw_text,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=JUDGE_MODEL,
        help="Ollama model to use as judge",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the BEA 2025 dataset",
    )
    args = parser.parse_args()

    model_name = args.model

    _, cases = prepare_cases(args.dataset)

    judge = OllamaAdapter(
        OllamaSettings(
            base_url="http://localhost:11434",
            model=model_name,
            think=False,
        )
    )

    output_dir = Path("runs") / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + f"-bea2025-judge-{model_name.replace(':', '-')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "predictions.jsonl"

    print(f"Judge: {model_name}")
    print(f"Evaluation cases: {len(cases)}")
    print(f"Saving results to: {output_path}")

    with output_path.open("w", encoding="utf-8") as file:
        for index, case in enumerate(cases, start=1):
            result = judge_case(judge, case, SYSTEM_PROMPT, model_name)
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
