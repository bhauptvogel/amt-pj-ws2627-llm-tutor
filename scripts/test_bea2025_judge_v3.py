"""Pilot validation of an LLM judge against BEA 2025 human annotations."""

import json

from llmtutor.models.ollama import OllamaAdapter, OllamaSettings
from llmtutor.tasks.bea2025 import load_bea2025
from llmtutor.types import Message, ModelRequest

DATASET = "data/pilot/bea2025_pilot20.json"
JUDGE_MODEL = "qwen3:4b-instruct-2507-q4_K_M"

CRITERIA = (
    "Mistake_Identification",
    "Mistake_Location",
    "Providing_Guidance",
    "Actionability",
)


SYSTEM_PROMPT = """
You are an educational evaluation annotator.

Your task is to evaluate a tutor's response to a student
using the pedagogical evaluation taxonomy from
"Unifying AI Tutor Evaluation" (2025).

Read the entire conversation history and the candidate
tutor response.

Evaluate the four dimensions independently.

1. Mistake_Identification

Question:
Has the tutor identified a mistake in the student's response?

Yes:
The tutor clearly identifies a mistake, including by
providing the correct answer or correction.

To some extent:
The tutor suggests that a mistake may exist but does
not clearly identify it.

No:
The tutor does not identify or signal a mistake.

2. Mistake_Location

Question:
Does the tutor accurately point to a genuine mistake
and its location?

Yes:
The tutor clearly points to the exact mistake, including
by supplying the correct replacement.

To some extent:
The tutor shows some awareness of the mistake and
guides the student toward its location without precisely
identifying it.

No:
The tutor provides no relevant details about where
the mistake occurred.

3. Providing_Guidance

Question:
Does the tutor offer correct and relevant guidance,
such as explanations, elaborations, hints or examples?

Yes:
The tutor provides correct and relevant guidance.
A targeted question can count as guidance.

To some extent:
The tutor provides guidance, but it is incorrect,
partially incorrect or incomplete.

No:
The tutor provides no meaningful guidance.

4. Actionability

Question:
Is it clear from the tutor's feedback what the
student should do next?

Yes:
The tutor gives a clear, specific next step that
the student can perform.

To some extent:
The tutor indicates that something should be done
but does not make the next step sufficiently clear.

No:
The tutor provides no actionable next step.
Simply revealing the final answer without a next
step is not actionable.

IMPORTANT:
- Evaluate each dimension independently.
- Use the full conversation history.
- Distinguish intermediate hints from final answers.
- Do not reward irrelevant or incorrect feedback
  as fully correct guidance.
- Do not require lengthy explanations.
- Judge the actual pedagogical behavior, not
  merely the response's tone or length.

Return ONLY valid JSON containing exactly:

{
  "Mistake_Identification": "Yes",
  "Mistake_Location": "Yes",
  "Providing_Guidance": "Yes",
  "Actionability": "Yes"
}

Replace each example value with exactly one of:
"Yes", "To some extent", "No"

No markdown. No additional text.
"""


def main():
    examples = load_bea2025(DATASET)

    judge = OllamaAdapter(
        OllamaSettings(
            base_url="http://localhost:11434",
            model=JUDGE_MODEL,
            think=False,
        )
    )

    # Use five conversations and two existing responses per conversation.
    cases = [
        (example, reference)
        for example in examples[5:]
        for reference in example.reference_responses[:2]
    ]

    correct = {criterion: 0 for criterion in CRITERIA}
    valid = {criterion: 0 for criterion in CRITERIA}

    for index, (example, reference) in enumerate(cases, start=1):
        user_prompt = (
            f"CONVERSATION:\n{example.conversation_history}\n\nTUTOR RESPONSE:\n{reference.text}"
        )

        request = ModelRequest(
            messages=(
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=user_prompt),
            ),
            temperature=0.0,
            max_tokens=256,
            seed=42,
        )

        print(f"\n[{index}/{len(cases)}] {reference.tutor}")

        try:
            result = judge.generate(request)
            predicted = json.loads(result.text or "")

            print("Human:", reference.annotations)
            print("Judge:", predicted)

            for criterion in CRITERIA:
                label = predicted.get(criterion)
                if label in ("Yes", "To some extent", "No"):
                    valid[criterion] += 1
                    if label == reference.annotations[criterion]:
                        correct[criterion] += 1

        except (ValueError, TypeError) as exc:
            print("Invalid judge output:", exc)

    print("\nRESULTS")
    for criterion in CRITERIA:
        print(
            f"{criterion}: "
            f"{correct[criterion]}/{len(cases)} exact matches, "
            f"{valid[criterion]}/{len(cases)} valid predictions"
        )


if __name__ == "__main__":
    main()
