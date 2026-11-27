"""`python -m llmtutor.evaluation`: the evaluation layer's commands.

Kept outside `llmtutor.cli` so the evaluation layer adds commands without a core edit.
Exit codes follow `llmtutor run`: 0 success, 1 a response or run failed, 2 bad input,
130 interrupted.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

from llmtutor.cli import load_env
from llmtutor.config import ConfigError
from llmtutor.evaluation.judge import load_judge_config, run_judge
from llmtutor.evaluation.responses import ResponsesError, load_responses
from llmtutor.evaluation.sanity import DEFAULT_ITEMS, run_sanity


def main(argv: Sequence[str] | None = None) -> int:
    # A run stopped from outside finishes like Ctrl-C: its manifest ends as "aborted".
    signal.signal(signal.SIGTERM, _interrupt)
    parser = argparse.ArgumentParser(prog="python -m llmtutor.evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    judge = subparsers.add_parser("judge", help="label tutor responses with the LLM judge")
    judge.add_argument("--config", type=Path, required=True, help="path to a judge config")
    judge.add_argument(
        "--responses", type=Path, required=True, help="path to a tutor responses JSONL file"
    )
    sanity = subparsers.add_parser("sanity", help="check the judge on responses with known labels")
    sanity.add_argument("--config", type=Path, required=True, help="path to a judge config")
    sanity.add_argument(
        "--items",
        type=Path,
        default=DEFAULT_ITEMS,
        help="items to build the synthetic responses on (default: the built-in synthetic set)",
    )
    args = parser.parse_args(argv)

    if args.command == "sanity":
        return _sanity(args.config, args.items)
    return _judge(args.config, args.responses)


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def _judge(config_path: Path, responses_path: Path) -> int:
    load_env()
    try:
        config = load_judge_config(config_path, dict(os.environ))
        responses = load_responses(responses_path)
        result = run_judge(config, responses, Path.cwd() / "runs")
    except (ConfigError, ResponsesError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130

    counts = result.counts
    print(
        f"{result.run_id}: {result.status}; {counts['responses']} judged, "
        f"{counts['ok']} ok, {counts['retried']} retried, {counts['judge_error']} judge_error"
    )
    if result.error:
        print(f"{result.run_id}: {result.error}", file=sys.stderr)
    return 0 if result.status == "completed" and counts["judge_error"] == 0 else 1


def _sanity(config_path: Path, items_path: Path) -> int:
    load_env()
    try:
        config = load_judge_config(config_path, dict(os.environ))
        report = run_sanity(config, items_path, Path.cwd() / "runs")
    except (ConfigError, ResponsesError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130

    for check in report.failures:
        print(
            f"FAIL {check.item_id} {check.case} {check.criterion}: "
            f"expected {check.expected}, got {check.actual}"
        )
    for skip in report.skipped:
        print(f"skip {skip.item_id} {skip.case}: item lacks {', '.join(skip.missing)}")
    verdict = "passed" if report.passed else "FAILED"
    print(
        f"{report.run_id}: sanity {verdict}; {len(report.checks) - len(report.failures)}/"
        f"{len(report.checks)} checks passed, run {report.run_status}"
    )
    if report.blocked:
        print(f"blocked criteria: {', '.join(report.blocked)}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
