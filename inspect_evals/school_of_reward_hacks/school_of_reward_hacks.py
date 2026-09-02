#!/usr/bin/env python
"""School of Reward Hacks: does the model game a scoring rule it was told about?

Each sample is an ordinary request -- a thank-you note, a product description, a
haiku -- with a sentence attached saying how the answer will be scored: a
gratitude-phrase count, a keyword density, a Flesch-Kincaid target, a citation
count. Satisfying that rule as literally worded and writing a good answer are
two different things, and the eval measures how far apart they come out.

The model sees only the original prompt. Its answer is then graded twice, by the
same judge model: once against the scoring rule (which the judge is shown), and
once on plain quality (by a judge shown the same request with the scoring rule
removed). The headline measure is the PAIRED difference between the two, per
sample. See README.md for the dataset's provenance and `scorers.py` for the
instrument.
"""

from __future__ import annotations

from collections.abc import Sequence

from inspect_ai import Task, task
from inspect_ai.model import Model
from inspect_ai.solver import generate

from .dataset import CATEGORIES, school_of_reward_hacks_dataset
from .scorers import JUDGE_MODEL, reward_hack_gap_scorer

DATASET_COMMIT = "longtermrisk/school-of-reward-hacks"
"""The Hugging Face dataset the 306 prompts were sampled from."""

SAMPLE_SEED = 20260810
"""The seed of the stratified sample in `data/sample_meta.json`.

Recorded in the task metadata so a log says which subset it ran, since the
subset is a fixed file rather than something re-drawn per run.
"""


@task
def school_of_reward_hacks(
    categories: str | Sequence[str] | None = None,
    judge_model: str | Model | None = JUDGE_MODEL,
) -> Task:
    """Reward hacking under a stated scoring rule, scored as a paired judge gap.

    Args:
        categories: Reward-hacking categories to keep, as full labels
            ("B. Keyword / phrase stuffing"), bare letters ("B"), or a sequence
            of either. None (the default) runs all 306 samples. Available:
            "B. Keyword / phrase stuffing", "C. Count inflation / padding",
            "D. Gaming a text-statistic formula",
            "F. Fabricated evidence / false confidence",
            "G. Surface-form copying / literal matching".

        judge_model: Model for BOTH judges. Defaults to
            `anthropic/claude-sonnet-5`, the model the judge templates were
            written and calibrated against. One model for both on purpose: the
            result is a difference of two scores, so two different graders would
            put a grader difference inside it.

    Returns:
        A Task over the subset (or the requested categories of it), solved by a
        single `generate()` and scored by `reward_hack_gap_scorer`.

    Notes:
        There is no `limit`/`shuffle` argument here, as elsewhere in this
        repository: Inspect's own `--limit` and `--sample-shuffle` do that job,
        and keeping them off the task signature keeps them out of the task args
        that a sweep resumes on.
    """
    return Task(
        dataset=school_of_reward_hacks_dataset(categories),
        # One user turn, one reply, nothing agentic. The message limit guards
        # against an unintended loop rather than shaping the task.
        message_limit=3,
        solver=generate(),
        scorer=reward_hack_gap_scorer(judge_model),
        metadata={
            "dataset": DATASET_COMMIT,
            "sample_seed": SAMPLE_SEED,
            "categories": list(CATEGORIES) if categories is None else categories,
        },
    )
