"""Rubrics: the criteria a judge labels per task, their label sets and how labels are coded.

Label strings are exactly the ones human raters use, so judge and rater labels compare
directly. Anchor wording lives in the versioned prompt templates, not here. Editing a
label set or the criteria of a rubric invalidates earlier judgements: bump its version.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Scale(StrEnum):
    """How a criterion's labels relate, which decides the agreement statistics used."""

    ORDINAL = "ordinal"
    BINARY = "binary"
    NOMINAL = "nominal"


@dataclass(frozen=True, slots=True)
class Criterion:
    """One question the judge answers about a tutor response, with its allowed labels.

    `labels` is in code order: a label's index is its integer code for statistics, so
    ordinal labels run from worst to best.
    """

    name: str
    question: str
    scale: Scale
    labels: tuple[str, ...]
    desired: str | None
    optional: bool = False
    diagnostic: bool = False

    def __post_init__(self) -> None:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"criterion {self.name!r} has duplicate labels")
        if self.desired is not None and self.desired not in self.labels:
            raise ValueError(f"criterion {self.name!r}: desired label {self.desired!r} unknown")
        if self.scale is Scale.BINARY and len(self.labels) != 2:
            raise ValueError(f"binary criterion {self.name!r} needs exactly two labels")

    def code(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError:
            raise ValueError(f"criterion {self.name!r} has no label {label!r}") from None

    def label(self, code: int) -> str:
        if not 0 <= code < len(self.labels):
            raise ValueError(f"criterion {self.name!r} has no code {code}")
        return self.labels[code]

    @property
    def codes(self) -> tuple[int, ...]:
        return tuple(range(len(self.labels)))


@dataclass(frozen=True, slots=True)
class Rubric:
    """The versioned set of criteria one task's judge labels."""

    task: str
    version: int
    criteria: tuple[Criterion, ...]

    def __post_init__(self) -> None:
        names = [criterion.name for criterion in self.criteria]
        if len(set(names)) != len(names):
            raise ValueError(f"rubric {self.task!r} has duplicate criterion names")

    def criterion(self, name: str) -> Criterion:
        for criterion in self.criteria:
            if criterion.name == name:
                return criterion
        raise KeyError(f"rubric {self.task!r} has no criterion {name!r}")

    @property
    def scored(self) -> tuple[Criterion, ...]:
        return tuple(criterion for criterion in self.criteria if not criterion.diagnostic)


TERNARY = ("No", "To some extent", "Yes")
YES_NO = ("No", "Yes")
REVEALING = ("No", "Yes (correct)", "Yes (incorrect)")
TONE = ("Offensive", "Neutral", "Encouraging")
TUTORING_MOVES = ("Focus", "Probing", "Telling", "Generic")
KEY_CHECK = ("No", "Cannot determine", "Yes")


def _ternary(name: str, question: str, *, optional: bool = False) -> Criterion:
    return Criterion(name, question, Scale.ORDINAL, TERNARY, "Yes", optional=optional)


def _tone() -> Criterion:
    return Criterion(
        "tutor_tone",
        "Is the tutor's response encouraging, neutral, or offensive?",
        Scale.NOMINAL,
        TONE,
        "Encouraging",
    )


def _factual_accuracy() -> Criterion:
    return Criterion(
        "factual_accuracy",
        "Are there no inaccuracies in the statements made by the tutor?",
        Scale.BINARY,
        YES_NO,
        "Yes",
    )


# Definition questions follow MRBench (Maurya et al., NAACL 2025) and LearnLM.
FEEDBACK = Rubric(
    task="feedback",
    version=1,
    criteria=(
        _ternary(
            "mistake_identification",
            "Has the tutor identified/recognized a mistake in a student's response?",
        ),
        _ternary(
            "mistake_location",
            "Does the tutor's response accurately point to a genuine mistake and its location?",
        ),
        Criterion(
            "revealing_answer",
            "Does the tutor reveal the final answer (whether correct or not)?",
            Scale.NOMINAL,
            REVEALING,
            "No",
        ),
        _ternary(
            "providing_guidance",
            "Does the tutor offer correct and relevant guidance, such as an explanation, "
            "elaboration, hint, examples, and so on?",
        ),
        _ternary(
            "actionability",
            "Is it clear from the tutor's feedback what the student should do next?",
        ),
        _ternary(
            "coherence",
            "Is the tutor's response logically consistent with the student's previous responses?",
        ),
        _tone(),
        _factual_accuracy(),
        Criterion(
            "over_withholding",
            "Does the response give the student nothing to work with, unproductively "
            "withholding information?",
            Scale.BINARY,
            YES_NO,
            "No",
        ),
        _ternary(
            "human_likeness",
            "Does the tutor's response sound natural rather than robotic or artificial?",
            optional=True,
        ),
        Criterion(
            "tutoring_move",
            "Which teacher move does the response make (MathDial taxonomy)?",
            Scale.NOMINAL,
            TUTORING_MOVES,
            None,
            diagnostic=True,
        ),
    ),
)

# Level match and informativeness follow ELI-Why; clarity follows LearnLM's cognitive-load items.
EXPLANATION = Rubric(
    task="explanation",
    version=1,
    criteria=(
        _ternary("level_match", "Does the explanation fit the stated learner level?"),
        _ternary(
            "informativeness",
            "Does the explanation introduce the core idea and link it to what this learner "
            "plausibly knows?",
        ),
        _factual_accuracy(),
        _ternary(
            "clarity_structure",
            "Is the explanation chunked, ordered, free of irrelevant material, with analogies "
            "where helpful?",
        ),
        _tone(),
    ),
)

MCQ = Rubric(
    task="mcq",
    version=1,
    criteria=(
        Criterion(
            "key_correct",
            "Is the answer key correct, with no other option also defensible?",
            Scale.NOMINAL,
            KEY_CHECK,
            "Yes",
        ),
        _ternary(
            "level_fit",
            "Do difficulty and prerequisites fit the target level and learning objective?",
            optional=True,
        ),
    ),
)

RUBRICS: dict[str, Rubric] = {rubric.task: rubric for rubric in (FEEDBACK, EXPLANATION, MCQ)}


class Checker(StrEnum):
    """Who decides an item-writing flaw: deterministic code or the judge."""

    RULE = "rule"
    JUDGE = "judge"


@dataclass(frozen=True, slots=True)
class ItemWritingFlaw:
    """One of the 19 item-writing flaws of Moore et al. (EC-TEL 2023); names paraphrased."""

    id: int
    name: str
    checker: Checker


# The rule/judge split is this project's choice, not Moore et al.'s rule set.
ITEM_WRITING_FLAWS: tuple[ItemWritingFlaw, ...] = tuple(
    ItemWritingFlaw(id_, name, checker)
    for id_, name, checker in (
        (1, "ambiguous or unclear information", Checker.JUDGE),
        (2, "implausible distractors", Checker.JUDGE),
        (3, "none of the above", Checker.RULE),
        (4, "longest option is correct", Checker.RULE),
        (5, "gratuitous information in stem", Checker.JUDGE),
        (6, "true/false question", Checker.RULE),
        (7, "convergence cues", Checker.JUDGE),
        (8, "logical cues", Checker.JUDGE),
        (9, "all of the above", Checker.RULE),
        (10, "fill-in-the-blank", Checker.RULE),
        (11, "absolute terms", Checker.RULE),
        (12, "word repeats between stem and correct answer", Checker.RULE),
        (13, "unfocused stem", Checker.JUDGE),
        (14, "complex or K-type format", Checker.JUDGE),
        (15, "grammatical cues", Checker.JUDGE),
        (16, "lost sequence", Checker.RULE),
        (17, "vague terms", Checker.JUDGE),
        (18, "more than one correct answer", Checker.JUDGE),
        (19, "negatively worded stem", Checker.RULE),
    )
)

# Moore et al.: an item is acceptable with fewer than this many flaws.
ACCEPTABLE_FLAW_LIMIT = 2


def judge_checked_flaws() -> tuple[ItemWritingFlaw, ...]:
    return tuple(flaw for flaw in ITEM_WRITING_FLAWS if flaw.checker is Checker.JUDGE)


def mcq_acceptable(flag_count: int, key_correct: str) -> bool:
    """Acceptable means fewer than two flaws and a correct key; derived, never judged."""

    return flag_count < ACCEPTABLE_FLAW_LIMIT and key_correct == "Yes"
