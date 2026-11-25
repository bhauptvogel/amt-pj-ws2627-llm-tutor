"""Load BEA 2025 MRBench examples for educational evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llmtutor.types import Message, ModelRequest

CRITERIA = (
    "Mistake_Identification",
    "Mistake_Location",
    "Providing_Guidance",
    "Actionability",
)

LABELS = frozenset({"Yes", "To some extent", "No"})

SYSTEM_PROMPT = (
    "You are a patient educational tutor. "
    "Continue the conversation by helping the student "
    "identify and correct their mistake. "
    "Identify the relevant misconception without solving "
    "the problem for the student. "
    "Ask one specific guiding question that helps the "
    "student determine their next step. "
    "Do not reveal the final answer or calculate the "
    "remaining solution steps. "
    "Keep your response under 100 words. "
    "Respond as the tutor."
)


@dataclass(frozen=True, slots=True)
class AnnotatedResponse:
    """An existing tutor response and its human evaluation labels."""

    tutor: str
    text: str
    annotations: dict[str, str]


@dataclass(frozen=True, slots=True)
class BEA2025Example:
    """One conversation with its annotated reference responses."""

    conversation_id: str
    conversation_history: str
    reference_responses: tuple[AnnotatedResponse, ...]

    def to_model_request(
        self,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int = 42,
    ) -> ModelRequest:
        """Build a request without exposing reference responses or labels."""
        return ModelRequest(
            messages=(
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=self.conversation_history),
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )


def load_bea2025(path: str | Path) -> list[BEA2025Example]:
    """Load and validate an MRBench JSON dataset."""
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("BEA 2025 dataset must contain a JSON list")

    examples = []
    seen_ids = set()

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Example {index}: expected an object")

        conversation_id = item.get("conversation_id")
        history = item.get("conversation_history")
        responses = item.get("tutor_responses")

        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError(f"Example {index}: invalid conversation_id")

        if conversation_id in seen_ids:
            raise ValueError(f"Duplicate conversation_id: {conversation_id}")

        seen_ids.add(conversation_id)

        if not isinstance(history, str) or not history.strip():
            raise ValueError(f"Example {index}: invalid conversation_history")

        if not isinstance(responses, dict) or not responses:
            raise ValueError(f"Example {index}: invalid tutor_responses")

        references = []

        for tutor, record in responses.items():
            if not isinstance(tutor, str) or not tutor.strip():
                raise ValueError(f"Example {index}: invalid tutor name")

            if not isinstance(record, dict):
                raise ValueError(f"Example {index}: invalid response for {tutor}")

            response_text = record.get("response")
            annotations = record.get("annotation")

            if not isinstance(response_text, str):
                raise ValueError(f"Example {index}: invalid response text for {tutor}")

            if not isinstance(annotations, dict):
                raise ValueError(f"Example {index}: missing annotations for {tutor}")

            if set(annotations) != set(CRITERIA):
                raise ValueError(f"Example {index}: unexpected criteria for {tutor}")

            if any(label not in LABELS for label in annotations.values()):
                raise ValueError(f"Example {index}: invalid annotation label for {tutor}")

            references.append(
                AnnotatedResponse(
                    tutor=tutor,
                    text=response_text,
                    annotations=dict(annotations),
                )
            )

        examples.append(
            BEA2025Example(
                conversation_id=conversation_id,
                conversation_history=history,
                reference_responses=tuple(references),
            )
        )

    return examples
