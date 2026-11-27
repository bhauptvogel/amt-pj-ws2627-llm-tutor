"""Tests for the BEA 2025 educational task loader."""

import json

import pytest

from llmtutor.tasks.bea2025 import load_bea2025


@pytest.fixture
def sample_dataset(tmp_path):
    path = tmp_path / "sample.json"
    data = [
        {
            "conversation_id": "example-001",
            "conversation_history": "Tutor: What is 2 + 2?\nStudent: 5",
            "tutor_responses": {
                "Expert": {
                    "response": "Can you count the objects again?",
                    "annotation": {
                        "Mistake_Identification": "Yes",
                        "Mistake_Location": "Yes",
                        "Providing_Guidance": "Yes",
                        "Actionability": "Yes",
                    },
                }
            },
        }
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_dataset(sample_dataset):
    examples = load_bea2025(sample_dataset)

    assert len(examples) == 1
    assert examples[0].conversation_id == "example-001"
    assert len(examples[0].reference_responses) == 1
    assert examples[0].reference_responses[0].tutor == "Expert"


def test_model_request_does_not_leak_annotations(sample_dataset):
    example = load_bea2025(sample_dataset)[0]
    request = example.to_model_request()

    prompt = "\n".join(message.content for message in request.messages)

    assert "Tutor: What is 2 + 2?" in prompt
    assert "Can you count the objects again?" not in prompt
    assert "Mistake_Identification" not in prompt
    assert request.temperature == 0.0
    assert request.seed == 42


def test_duplicate_conversation_id_rejected(sample_dataset):
    data = json.loads(sample_dataset.read_text(encoding="utf-8"))
    data.append(data[0])
    sample_dataset.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate conversation_id"):
        load_bea2025(sample_dataset)


def test_invalid_annotation_rejected(sample_dataset):
    data = json.loads(sample_dataset.read_text(encoding="utf-8"))
    data[0]["tutor_responses"]["Expert"]["annotation"]["Actionability"] = "Maybe"
    sample_dataset.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid annotation label"):
        load_bea2025(sample_dataset)
