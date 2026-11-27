"""Tests for reproducible BEA 2025 dataset preparation."""

import json
import random
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_bea2025_pilot.py"

prepare_pilot = runpy.run_path(str(SCRIPT))["prepare_pilot"]


@pytest.fixture
def source_dataset(tmp_path):
    records = [{"conversation_id": str(i)} for i in range(10)]

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(records),
        encoding="utf-8",
    )

    return source, records


def test_reproducible_sampling(tmp_path, source_dataset):
    source, records = source_dataset

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    prepare_pilot(source, first, sample_size=5, seed=42)
    prepare_pilot(source, second, sample_size=5, seed=42)

    expected = random.Random(42).sample(records, 5)

    assert json.loads(first.read_text()) == expected
    assert first.read_bytes() == second.read_bytes()


def test_existing_identical_dataset(tmp_path, source_dataset):
    source, _ = source_dataset
    output = tmp_path / "pilot.json"

    prepare_pilot(source, output, sample_size=5, seed=42)

    original = output.read_bytes()
    modified_at = output.stat().st_mtime_ns

    prepare_pilot(source, output, sample_size=5, seed=42)

    assert output.read_bytes() == original
    assert output.stat().st_mtime_ns == modified_at


def test_refuse_overwrite(tmp_path, source_dataset):
    source, _ = source_dataset
    output = tmp_path / "pilot.json"

    prepare_pilot(source, output, sample_size=5, seed=42)

    original = output.read_bytes()

    with pytest.raises(FileExistsError):
        prepare_pilot(
            source,
            output,
            sample_size=4,
            seed=42,
        )

    assert output.read_bytes() == original


@pytest.mark.parametrize("sample_size", [0, -1, 11])
def test_invalid_sample_size(
    tmp_path,
    source_dataset,
    sample_size,
):
    source, _ = source_dataset
    output = tmp_path / "pilot.json"

    with pytest.raises(ValueError):
        prepare_pilot(
            source,
            output,
            sample_size=sample_size,
            seed=42,
        )

    assert not output.exists()
