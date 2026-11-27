"""Prepare a reproducible BEA 2025 pilot dataset."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def prepare_pilot(
    source: Path,
    output: Path,
    sample_size: int = 20,
    seed: int = 42,
) -> None:
    source_bytes = source.read_bytes()
    original = json.loads(source_bytes)

    if not isinstance(original, list):
        raise ValueError("Expected a list of conversations")

    if not 1 <= sample_size <= len(original):
        raise ValueError("Invalid sample size")

    pilot = random.Random(seed).sample(original, sample_size)

    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))

        if existing != pilot:
            raise FileExistsError(f"Refusing to overwrite different dataset: {output}")

        print("Existing dataset matches exactly.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(pilot, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("Pilot dataset created.")

    print("Original conversations:", len(original))
    print("Pilot conversations:", len(pilot))
    print("Random seed:", seed)
    print("Source SHA-256:", hashlib.sha256(source_bytes).hexdigest())
    print(
        "Pilot SHA-256:",
        hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    print("Output:", output)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pilot/bea2025_pilot20.json"),
    )
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    prepare_pilot(
        source=args.source,
        output=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
