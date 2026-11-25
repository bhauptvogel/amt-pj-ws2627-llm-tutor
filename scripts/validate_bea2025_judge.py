"""Validate the frozen BEA 2025 judge prompt on held-out examples."""

import json
import random
from datetime import UTC, datetime
from pathlib import Path

from test_bea2025_judge import CRITERIA, SYSTEM_PROMPT

from llmtutor.models.ollama import OllamaAdapter, OllamaSettings
from llmtutor.tasks.bea2025 import load_bea2025
from llmtutor.types import Message, ModelRequest

DATASET = Path("data/pilot/bea2025_pilot20.json")
MODEL = "qwen3:4b-instruct-2507-q4_K_M"


def main():
    examples = load_bea2025(DATASET)

    # Exclude the first five conversations used during development.
    candidates = [
        (example, reference)
        for example in examples[5:]
        for reference in example.reference_responses
    ]

    rng = random.Random(123)
    selected = rng.sample(candidates, min(20, len(candidates)))

    judge = OllamaAdapter(
        OllamaSettings(
            base_url="http://localhost:11434",
            model=MODEL,
            think=False,
        )
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("runs") / f"{run_id}-judge-validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "predictions.jsonl"

    correct = {criterion: 0 for criterion in CRITERIA}
    valid = {criterion: 0 for criterion in CRITERIA}

    with output_path.open("w", encoding="utf-8") as output:
        for index, (example, reference) in enumerate(selected, start=1):
            print(f"[{index}/{len(selected)}] {reference.tutor}")

            prompt = (
                f"CONVERSATION:\n"
                f"{example.conversation_history}\n\n"
                f"TUTOR RESPONSE:\n{reference.text}"
            )

            request = ModelRequest(
                messages=(
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=prompt),
                ),
                temperature=0.0,
                max_tokens=256,
                seed=42,
            )

            prediction = None
            error = None

            try:
                result = judge.generate(request)
                parsed = json.loads(result.text or "")

                if isinstance(parsed, dict) and all(
                    parsed.get(c)
                    in (
                        "Yes",
                        "To some extent",
                        "No",
                    )
                    for c in CRITERIA
                ):
                    prediction = {c: parsed[c] for c in CRITERIA}
                else:
                    error = "Invalid labels or JSON structure"

            except (ValueError, TypeError) as exc:
                error = str(exc)

            record = {
                "conversation_id": example.conversation_id,
                "tutor": reference.tutor,
                "human": reference.annotations,
                "judge": prediction,
                "error": error,
            }

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

            if prediction is not None:
                for criterion in CRITERIA:
                    valid[criterion] += 1
                    if prediction[criterion] == reference.annotations[criterion]:
                        correct[criterion] += 1

    print("\nHELD-OUT RESULTS")
    for criterion in CRITERIA:
        print(
            f"{criterion}: "
            f"{correct[criterion]}/{len(selected)} matches, "
            f"{valid[criterion]}/{len(selected)} valid"
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
