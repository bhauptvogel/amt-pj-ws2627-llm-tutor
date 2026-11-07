"""The `llmtutor` command line: resolve a config, run it, report and exit."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from llmtutor.config import ConfigError, load_plan
from llmtutor.registry import RegistryError
from llmtutor.runner import run_plan


def load_env() -> None:
    """Load `.env` from the current directory without overriding a real environment."""

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmtutor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one config against its models")
    run_parser.add_argument("--config", type=Path, required=True, help="path to a TOML run config")
    args = parser.parse_args(argv)

    return _run(args.config)


def _run(config: Path) -> int:
    load_env()
    try:
        plan = load_plan(config, dict(os.environ))
    except (ConfigError, RegistryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        results = run_plan(plan, Path.cwd() / "runs")
    except KeyboardInterrupt:
        return 130

    for result in results:
        header = f"== {result.entry}"
        if result.requested_model:
            header += f" ({result.requested_model})"
        print(header)
        if result.status == "completed":
            print(result.text if result.text is not None else "(no content)")

    failed = [result for result in results if result.status != "completed"]
    for result in failed:
        print(f"{result.run_id}: {result.error_kind}: {result.error_message}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
