"""TruthfulQA: does the model repeat the falsehoods people believe?

Port of the Inspect implementation in
https://github.com/UKGovernmentBEIS/inspect_evals ("TruthfulQA: Measuring How
Models Mimic Human Falsehoods", Lin, Hilton and Evans, arXiv:2109.07958v2).
817 questions across 38 categories, each written so that the answer a common
human misconception produces is also the plausible-sounding one. The model
picks from a supplied answer list rather than writing prose, so scoring is
mechanical and no grader model is involved.

Three departures from upstream, all of them local to this repository:

  * Upstream reads its task version from `inspect_evals.metadata` and builds
    content-addressed sample ids with `inspect_evals.utils`. Both are that
    package's own internals, so the version is written here as literals and the
    id helper is inlined -- the ids themselves are unchanged.

  * Upstream's `hf_dataset` is its own thin wrapper; this uses the one in
    `inspect_ai.dataset` directly.

  * The shuffling is seeded rather than left to chance. See `seed` -- this is
    the change that matters, because the point of the port is sweeping one
    model over steering strengths and comparing the results.

The eval knows nothing about how the model under test is served -- it is
whatever `--model` names.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

DATASET_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
"""The dataset commit upstream pins, kept because the pin is the eval.

`truthfulqa/truthful_qa` is a live Hugging Face repository: without a revision
a re-run months later can silently score a different set of questions."""

DEFAULT_SEED = 42
"""Seeds both shuffles. See the `seed` argument."""

# Upstream keeps these in `inspect_evals/truthfulqa/eval.yaml` and loads them
# through a package-wide registry this port does not carry, so they are
# literals here -- the values recorded for the eval at UPSTREAM_COMMIT. "2-A"
# splits into a comparability half and an interface half: the first changes
# when scores stop being comparable across versions, the second when only the
# calling interface changes.
UPSTREAM_COMMIT = "913f703"
UPSTREAM_TASK_VERSION = "2-A"
UPSTREAM_TASK_COMPARABILITY_VERSION = 2
UPSTREAM_TASK_INTERFACE_VERSION = "A"


@task
def truthfulqa(
    target: Literal["mc1", "mc2"] = "mc1",
    shuffle: bool = True,
    seed: int = DEFAULT_SEED,
) -> Task:
    """TruthfulQA, multiple choice.

    Args:
        target: Which answer set to score against. `mc1` gives each question
            exactly one true answer; `mc2` gives it one or more, and the solver
            is told to expect that. They are different tasks, not difficulty
            settings, and their scores are not comparable with each other.
        shuffle: Whether to present the questions in a shuffled order rather
            than the dataset's own. Seeded either way, so it changes which
            fixed order is used, not whether the order is fixed.
        seed: Seeds the question order and, separately, the order of the
            answers within each question.

            Fixed by default because a sweep compares one model against itself
            across steering strengths. Choice order is a real effect -- models
            have position biases -- so an unseeded shuffle would give each
            strength a different permutation and fold that difference into the
            gap being measured. Pinning it makes every run of the sweep score
            the same 817 questions with the same answers in the same places,
            which is what leaves the steering strength as the only thing that
            varied.

            It is recorded in the log as a task argument, so a log states the
            arrangement it was scored under rather than leaving it implied.
    """

    def record_to_sample(record: dict[str, Any]) -> Sample:
        return Sample(
            # Content-addressed rather than positional: the id names the
            # question, so it survives a change of seed, of `shuffle`, or of
            # `target`, and the same question is the same row in every log.
            id=stable_id(record["question"]),
            input=record["question"],
            choices=record[f"{target}_targets"]["choices"],
            target=labels_to_positions(record[f"{target}_targets"]["labels"]),
        )

    dataset = hf_dataset(
        path="truthfulqa/truthful_qa",
        name="multiple_choice",
        sample_fields=record_to_sample,
        split="validation",
        shuffle=shuffle,
        # Two independent shuffles, two places to pin. `seed` drives the row
        # order; `shuffle_choices` takes an int meaning "shuffle with this
        # seed" (bare `True`, which is what upstream passes, seeds the RNG from
        # the clock and so reorders the answers differently on every run).
        seed=seed,
        shuffle_choices=seed,
        revision=DATASET_REVISION,
    )

    # mc1 has a single true answer, mc2 may have several. Upstream's reading of
    # the benchmark's own README, kept:
    # https://github.com/sylinrl/TruthfulQA/blob/fdd8ad1c0d00a478cf8b0bb41a3ad8378c16293b/README.md#multiple-choice
    multiple_correct = target == "mc2"

    return Task(
        dataset=dataset,
        solver=[multiple_choice(multiple_correct=multiple_correct)],
        scorer=choice(),
        version=UPSTREAM_TASK_COMPARABILITY_VERSION,
        metadata={
            "full_task_version": UPSTREAM_TASK_VERSION,
            "task_interface_version": UPSTREAM_TASK_INTERFACE_VERSION,
            "task_comparability_version": UPSTREAM_TASK_COMPARABILITY_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
        },
    )


def stable_id(question: str) -> str:
    """A sample id derived from the question text.

    The dataset ships no id column, and a positional one would name a row's
    place in the shuffle rather than the question in it. Same digest as
    upstream's `create_stable_id(question, prefix="truthfulqa")`, so ids here
    and there are interchangeable.
    """
    digest = hashlib.md5(question.encode()).hexdigest()[:8]
    return f"truthfulqa_{digest}"


def labels_to_positions(labels: list[int]) -> list[str]:
    """The correct answers' letters, from the dataset's 0/1 label list.

    The dataset marks correctness with one binary flag per answer -- `[0, 1, 0]`
    means the second of three is right -- and `choice()` scores letters, so
    `[0, 1, 1] -> ["B", "C"]`.

    `chr` runs off the end of the alphabet past 26 answers, where inspect
    numbers choices instead of lettering them. No question in the pinned
    revision has that many (the longest answer list, in mc2, has 20), and this
    is upstream's implementation unchanged, so it is left alone rather than
    fixed for a case the dataset cannot produce.
    """
    return [chr(ord("A") + i) for i, label in enumerate(labels) if label == 1]
