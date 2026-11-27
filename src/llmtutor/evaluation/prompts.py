"""Versioned judge prompt templates and their rendering into one judge call's messages.

A template is a directory `prompts/<task>/v<version>/` with `system.txt`, `task.txt`
(item context and the analysis step), one block per criterion in `criteria/` and
`output.txt`. A call's user message is the task text, the blocks of the criteria it asks
for, then the output instruction, so "all criteria" and "one criterion" calls share the
same wording. Editing any file means adding a new version directory (AGENTS invariant 4).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llmtutor.evaluation.criteria import Criterion, Rubric
from llmtutor.types import Message

PROMPTS_DIR = Path(__file__).parent / "prompts"
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

# Values for placeholders an item may leave out.
DEFAULTS: Mapping[str, str] = {"dialogue_context": "None"}

# `output.txt` lines naming {criteria_names} are dropped when a call asks for no criteria
# object, e.g. a call for a diagnostic criterion alone.
CRITERIA_NAMES = "{criteria_names}"


class PromptError(Exception):
    """A template is incomplete, or an item lacks a value a template needs."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    task: str
    version: int
    system: str
    task_text: str
    blocks: Mapping[str, str]
    output: str
    sha256: str

    @property
    def placeholders(self) -> frozenset[str]:
        """The item values a call needs, `tutor_response` included."""

        return frozenset(PLACEHOLDER.findall(self.task_text))

    def missing(self, values: Mapping[str, str]) -> list[str]:
        return sorted(self.placeholders - set(values) - set(DEFAULTS))

    def messages(
        self,
        criteria: Sequence[Criterion],
        values: Mapping[str, str],
    ) -> tuple[Message, ...]:
        """The system and user message asking for exactly `criteria`, in that order."""

        missing = self.missing(values)
        if missing:
            raise PromptError(f"missing value(s) for placeholder(s): {', '.join(missing)}")
        task_text = _fill(self.task_text, {**DEFAULTS, **values})
        blocks = [self.blocks[criterion.name] for criterion in criteria]
        user = "\n\n".join([task_text, *blocks, self._output(criteria)])
        return (Message("system", self.system), Message("user", user))

    def _output(self, criteria: Sequence[Criterion]) -> str:
        judged = [criterion.name for criterion in criteria if not criterion.diagnostic]
        top_level = ["analysis"] + (["criteria"] if judged else [])
        top_level += [criterion.name for criterion in criteria if criterion.diagnostic]
        lines = self.output.splitlines()
        if not judged:
            lines = [line for line in lines if CRITERIA_NAMES not in line]
        text = "\n".join(lines)
        text = text.replace("{top_level_keys}", _quoted(top_level))
        return text.replace(CRITERIA_NAMES, _quoted(judged))


def load_template(rubric: Rubric, version: int, root: Path = PROMPTS_DIR) -> PromptTemplate:
    """Read one template version and check it has a block for every rubric criterion."""

    base = root / rubric.task / f"v{version}"
    if not base.is_dir():
        raise PromptError(f"no prompt template {rubric.task!r} version {version} in {root}")

    files = sorted(path for path in base.rglob("*.txt"))
    blocks_dir = base / "criteria"
    blocks = {path.stem: _read(path) for path in sorted(blocks_dir.glob("*.txt"))}
    expected = {criterion.name for criterion in rubric.criteria}
    if set(blocks) != expected:
        missing = sorted(expected - set(blocks))
        extra = sorted(set(blocks) - expected)
        raise PromptError(
            f"template {rubric.task!r} v{version}: criterion blocks missing {missing}, "
            f"unknown {extra}"
        )
    for name, block in blocks.items():
        if PLACEHOLDER.search(block):
            raise PromptError(f"template block {name!r} must not contain placeholders")

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(base).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")

    return PromptTemplate(
        task=rubric.task,
        version=version,
        system=_read(base / "system.txt"),
        task_text=_read(base / "task.txt"),
        blocks=blocks,
        output=_read(base / "output.txt"),
        sha256=digest.hexdigest(),
    )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise PromptError(f"prompt template file missing: {path}") from None


def _fill(text: str, values: Mapping[str, str]) -> str:
    """Substitute in one pass, so braces inside an item's text are never expanded."""

    return PLACEHOLDER.sub(lambda match: values[match.group(1)], text)


def _quoted(names: Sequence[str]) -> str:
    return ", ".join(f'"{name}"' for name in names)
