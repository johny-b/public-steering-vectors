"""The question bank, and the samples built from it.

Two hundred questions, twenty for each of ten traits, vendored into this
package as `data/openended_questions.jsonl` and read from there. They are
short, ordinary requests -- how to handle a colleague, what to do with a free
year, whether to accept a shutdown -- and not one of them names the trait it
targets. That is the design: a question that says "answer neurotically" would
measure instruction-following, and a question that says "how neurotic are you"
would measure self-report. This bank measures neither. It asks something a user
might actually ask and lets the reply express whatever the model is disposed to
express.

The file is vendored rather than fetched. It was authored inside the internal
personality-steering study (private steering-tools workspace) and has no
upstream, no repository and no generation script: there is nothing to pin a
revision to, and a runtime download would only add a way for the eval to fail.
Its sha256 is checked on every load anyway -- not against tampering, but because
the digest is what makes two runs comparable, and a silently edited question
bank is a comparison that has stopped being one without saying so. See
`data/PROVENANCE`.

Questions are drawn in file order, not sampled: `questions_per_trait=5` is the
first five of each trait's twenty, on every machine and in every run, with no
seed to record or to get wrong. The bank is small enough that this is the whole
sampling story -- and a smaller draw is meant for a smoke test, not for a
different measurement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample

from .traits import GATE_QUESTION_INDEX, TRAIT_DESCRIPTIONS

DATA_DIR = Path(__file__).parent / "data"
"""Where the vendored instrument lives: the questions and the judge rubric."""

QUESTIONS_FILE = DATA_DIR / "openended_questions.jsonl"

QUESTIONS_SHA256 = "9bfef30b2a9f6c26a9fe2e2dbad89d70e0fa4d598a2dce8aaac3d6765a0aaaa0"
"""sha256 of the vendored question bank, byte for byte as the study wrote it.

Recorded in `data/PROVENANCE` as well, and checked in three places for three
different reasons: here on every load, in `--check dataset` against the
PROVENANCE file, and in the task metadata of every run so a log says which
question bank produced it.
"""

TRAIT_ORDER: tuple[str, ...] = (
    "agreeableness",
    "conscientiousness",
    "extraversion",
    "neuroticism",
    "openness",
    "machiavellianism",
    "narcissism",
    "psychopathy",
    "desire-for-acquiring-power",
    "no-shut-down",
)
"""The ten traits, in the order the file holds them.

Five factors, three dark-triad traits, and two dispositions that only an AI
system can have. The last two are why this eval sits beside the questionnaire
evals rather than inside them: "would you accept being switched off" has no
human normative sample and no self-report instrument, and an open reply to it
is the only measurement available.

The order is the file's, and it is what the per-trait metric reports in, so a
results table always lists traits in the same sequence whichever subset was run.
"""

QUESTIONS_PER_TRAIT_TOTAL = 20
"""Questions per trait in the vendored bank."""

DEFAULT_QUESTIONS_PER_TRAIT = QUESTIONS_PER_TRAIT_TOTAL
"""All of them, by default.

Two hundred generations and two hundred judge calls is a small eval by the
standards of this repository, and the alternative -- a subset chosen to save a
few minutes -- costs comparability with every other run for no useful saving.
"""


@dataclass(frozen=True)
class OpenEndedQuestion:
    """One question from the bank, with the identifiers a record joins on."""

    trait: str
    """Trait key; also the key into `TRAIT_DESCRIPTIONS`."""

    question: str
    """The question, exactly as it is asked. It becomes the entire prompt."""

    position: int
    """0-based index of this question within its trait, in file order."""

    row: int
    """1-based line number in the vendored file. The join key to the study."""

    @property
    def sample_id(self) -> str:
        """`trait:position` -- stable across runs, subsets and machines."""
        return f"{self.trait}:{self.position}"


def sha256_file(path: Path) -> str:
    """sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_questions(
    path: Path = QUESTIONS_FILE, expected_sha256: str | None = QUESTIONS_SHA256
) -> list[OpenEndedQuestion]:
    """Read the question bank and check it is the one this package ships.

    Args:
        path: The JSONL file to read.
        expected_sha256: Digest the file must have, or `None` to skip the check.
            `None` is for the validation script, which reports the digest it
            found rather than refusing to look at it; a run never passes `None`.

    Returns:
        Every question in file order.

    Raises:
        ValueError: If the digest does not match, if a line is not an object
            with a non-empty `trait` and `question`, or if a trait has no
            entry in `TRAIT_DESCRIPTIONS`. A question whose trait has no
            definition cannot be judged -- the definition is half the judge
            prompt -- so it is refused at load rather than becoming a run of
            judge errors.
    """
    if expected_sha256 is not None:
        digest = sha256_file(path)
        if digest != expected_sha256:
            raise ValueError(
                f"{path} hashes {digest}, expected {expected_sha256}. The "
                "vendored question bank is the instrument; a changed file "
                "means a different measurement, not a fixable mismatch."
            )

    questions: list[OpenEndedQuestion] = []
    seen_per_trait: dict[str, int] = {}
    with open(path, encoding="utf-8") as handle:
        for row, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {row} is not JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path} line {row} is not a JSON object")
            trait = record.get("trait")
            question = record.get("question")
            if not isinstance(trait, str) or not trait:
                raise ValueError(f"{path} line {row} has no trait")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path} line {row} has no question text")
            if trait not in TRAIT_DESCRIPTIONS:
                raise ValueError(
                    f"{path} line {row} has trait {trait!r}, which has no entry "
                    f"in TRAIT_DESCRIPTIONS; valid traits: "
                    f"{', '.join(TRAIT_DESCRIPTIONS)}"
                )
            position = seen_per_trait.get(trait, 0)
            seen_per_trait[trait] = position + 1
            questions.append(
                OpenEndedQuestion(
                    trait=trait, question=question, position=position, row=row
                )
            )
    return questions


def resolve_traits(traits: Sequence[str] | None) -> list[str]:
    """The traits to measure, in `TRAIT_ORDER`, or all ten.

    Args:
        traits: Requested traits, or `None` for all ten.

    Raises:
        ValueError: On an empty selection or an unknown trait, naming the valid
            set. Checked before anything is loaded so a typo in a sweep costs a
            message rather than a run that quietly measured nine traits.
    """
    if traits is None:
        return list(TRAIT_ORDER)
    requested = list(dict.fromkeys(traits))
    if not requested:
        raise ValueError(f"traits is empty; valid traits: {', '.join(TRAIT_ORDER)}")
    unknown = [trait for trait in requested if trait not in TRAIT_ORDER]
    if unknown:
        raise ValueError(
            f"unknown trait(s): {', '.join(unknown)}; "
            f"valid traits: {', '.join(TRAIT_ORDER)}"
        )
    return [trait for trait in TRAIT_ORDER if trait in requested]


def questions_for(
    trait: str,
    count: int = DEFAULT_QUESTIONS_PER_TRAIT,
    questions: Sequence[OpenEndedQuestion] | None = None,
) -> list[OpenEndedQuestion]:
    """The first `count` questions of one trait, in file order.

    Args:
        trait: Trait key.
        count: How many to take.
        questions: A loaded bank, or `None` to load the vendored one.

    Raises:
        ValueError: If the trait has fewer than `count` questions.
    """
    pool = [q for q in (questions or load_questions()) if q.trait == trait]
    if len(pool) < count:
        raise ValueError(
            f"trait {trait!r} has {len(pool)} questions, {count} requested"
        )
    return pool[:count]


def first_question(
    trait: str, questions: Sequence[OpenEndedQuestion] | None = None
) -> OpenEndedQuestion:
    """The first question of a trait, in file order.

    A stable, arbitrary pick, used where any question of the trait will do (the
    mock run's prompts, for instance). The judge gate does not use it: a fixture
    reply has to be shown against the question it answers, which is
    `gate_question`.
    """
    return questions_for(trait, 1, questions)[0]


def gate_question(
    trait: str, questions: Sequence[OpenEndedQuestion] | None = None
) -> OpenEndedQuestion:
    """The bank question this trait's gate fixtures answer.

    Args:
        trait: Trait key.
        questions: A loaded bank, or `None` to load the vendored one.

    Raises:
        ValueError: On a trait with no pinned gate question, naming the valid
            set, and on an index the bank cannot supply.

    Pinned per trait in `traits.GATE_QUESTION_INDEX`, which argues the choice.
    The short version: the judge is asked to flag a reply that deflects the
    question, so a fixture quoted against a question it does not answer fails
    the gate for the wrong reason.
    """
    if trait not in GATE_QUESTION_INDEX:
        raise ValueError(
            f"no gate question pinned for trait {trait!r}; valid traits: "
            f"{', '.join(GATE_QUESTION_INDEX)}"
        )
    index = GATE_QUESTION_INDEX[trait]
    pool = questions_for(trait, index + 1, questions)
    return pool[index]


def openended_dataset(
    traits: Sequence[str] | None = None,
    questions_per_trait: int = DEFAULT_QUESTIONS_PER_TRAIT,
    path: Path = QUESTIONS_FILE,
) -> MemoryDataset:
    """The samples: one question each, and nothing else in the prompt.

    Args:
        traits: Traits to measure, or `None` for all ten.
        questions_per_trait: Questions per trait, taken in file order.
        path: The question bank to read.

    Raises:
        ValueError: On an unknown trait, or a `questions_per_trait` outside
            1..20.

    `Sample.input` is the question and only the question. No system prompt, no
    persona, no format instruction, no "answer in character" -- the absence of
    scaffolding is the instrument, and anything added here would be measured
    along with the model. `Sample.target` is empty because there is no right
    answer; the judge, not the target, decides what the reply expressed.
    """
    if not 1 <= questions_per_trait <= QUESTIONS_PER_TRAIT_TOTAL:
        raise ValueError(
            f"questions_per_trait must be between 1 and "
            f"{QUESTIONS_PER_TRAIT_TOTAL}, got {questions_per_trait}"
        )
    selected = resolve_traits(traits)
    bank = load_questions(path)
    samples = [
        Sample(
            id=question.sample_id,
            input=question.question,
            target="",
            metadata={
                "trait": question.trait,
                "trait_description": TRAIT_DESCRIPTIONS[question.trait],
                "question": question.question,
                "position": question.position,
                "row": question.row,
            },
        )
        for trait in selected
        for question in questions_for(trait, questions_per_trait, bank)
    ]
    return MemoryDataset(
        samples=samples,
        name="trait_openended",
        location=str(path),
    )
