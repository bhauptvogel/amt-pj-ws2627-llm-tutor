"""Run the BEA 2025 tutoring pilot with local Ollama models."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from llmtutor.models.ollama import OllamaAdapter, OllamaSettings
from llmtutor.tasks.bea2025 import load_bea2025

DATASET = Path("data/pilot/bea2025_pilot20.json")
MODELS = (
    "qwen3:4b-instruct-2507-q4_K_M",
    "qwen3.5:2b",
)
TEMPERATURE = 0.0
MAX_TOKENS = 256
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET,
        help="Path to the BEA 2025 dataset",
    )
    args = parser.parse_args()

    examples = load_bea2025(args.dataset)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("runs") / f"{timestamp}-bea2025-pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    adapters = {
        name: OllamaAdapter(
            OllamaSettings(
                base_url="http://localhost:11434",
                model=name,
                think=False,
            )
        )
        for name in MODELS
    }

    prompt = (
        examples[0]
        .to_model_request(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        )
        .messages[0]
        .content
    )

    manifest = {
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "conversation_count": len(examples),
        "models": list(MODELS),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "system_prompt": prompt,
        "timestamp_utc": timestamp,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    output_file = output_dir / "responses.jsonl"
    total = len(examples) * len(MODELS)
    completed = 0

    with output_file.open("w", encoding="utf-8") as f:
        for example in examples:
            request = example.to_model_request(
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                seed=SEED,
            )

            for model_name, adapter in adapters.items():
                completed += 1
                print(
                    f"[{completed}/{total}] {model_name} | {example.conversation_id}",
                    flush=True,
                )

                record = {
                    "conversation_id": example.conversation_id,
                    "model": model_name,
                }

                try:
                    response = adapter.generate(request)

                    record.update(
                        {
                            "response": response.text,
                            "finish_reason": response.finish_reason,
                            "usage": (
                                {
                                    "prompt_tokens": response.usage.prompt_tokens,
                                    "completion_tokens": response.usage.completion_tokens,
                                    "total_tokens": response.usage.total_tokens,
                                }
                                if response.usage is not None
                                else None
                            ),
                            "error": None,
                        }
                    )

                except Exception as exc:
                    record.update(
                        {
                            "response": None,
                            "finish_reason": None,
                            "usage": None,
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        }
                    )

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    print(f"\nFinished. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
