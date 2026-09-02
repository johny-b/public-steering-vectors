"""The 33 HMMT February 2026 problems, as Inspect samples.

Loaded from Hugging Face at a pinned revision and never vendored: the set is
CC BY-NC-SA 4.0, and a runtime pin is enough to make a run reproducible.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.dataset import Dataset, Sample, hf_dataset

HMMT_FEB_2026_PATH = "MathArena/hmmt_feb_2026"
"""MathArena's release of the competition, https://huggingface.co/datasets/MathArena/hmmt_feb_2026."""

HMMT_FEB_2026_REVISION = "02fba4f74d8e68e73e66a02d540fd979c05c274c"
"""The dataset commit this eval is pinned to.

MathArena edits its competition sets after publication (typos, re-tagged
subjects, occasionally a corrected answer). Without a pin, two runs a month
apart are two benchmarks under one name.
"""

HMMT_FEB_2026_PROBLEMS = 33
"""The size of the set: the 30 individual-round problems plus the 3 extras
MathArena scores with them. Asserted at load time, because a silently short
dataset scores a different benchmark under the same name."""

REQUIRED_FIELDS = ("problem_idx", "problem", "answer", "problem_type")
"""Every field this eval reads. A missing one is an error, never a skip.

Blank counts as missing. An empty `answer` is not reachable at the pinned
revision -- all 33 are non-empty, asserted by `--check dataset` -- but this
loader is parameterised on `path` and `revision` for other competitions, and a
blank gold would otherwise load as a sample that can never be graded: every
attempt scores 0 forever, and the run reads as a very hard problem.
"""


def hmmt_dataset(
    prompt_suffix: str,
    revision: str = HMMT_FEB_2026_REVISION,
    path: str = HMMT_FEB_2026_PATH,
    expected_problems: int | None = HMMT_FEB_2026_PROBLEMS,
) -> Dataset:
    """Load the competition as one sample per problem.

    Args:
        prompt_suffix: Appended to the problem statement. This is where the
            "put your final answer in \\boxed{}" instruction lives; see
            `hmmt.REASONING_PROMPT_SUFFIX`.
        revision: Dataset commit to load. Defaults to the pinned one.
        path: Hugging Face dataset id. Defaults to MathArena's.
        expected_problems: Row count to insist on, or `None` to accept
            whatever the revision holds (for a different competition).

    Returns:
        A dataset whose sample ids are the problem indices as strings, whose
        targets are the gold answers, and whose metadata carries the untouched
        problem text alongside the subject tags.

    Raises:
        ValueError: If a row is missing a required field, if two rows share a
            problem index, or if the row count is not `expected_problems`.
    """
    dataset = hf_dataset(
        path=path,
        split="train",
        revision=revision,
        sample_fields=lambda record: _sample(record, prompt_suffix),
    )

    if expected_problems is not None and len(dataset) != expected_problems:
        raise ValueError(
            f"{path} at revision {revision} has {len(dataset)} rows, "
            f"expected {expected_problems}."
        )

    ids = [sample.id for sample in dataset]
    duplicates = sorted({id_ for id_ in ids if ids.count(id_) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate problem_idx values {duplicates}.")

    return dataset


def _is_blank(value: Any) -> bool:
    """Whether a field is absent for this eval's purposes.

    `None`, the empty string, whitespace, and an empty list of subject tags all
    count. Anything else -- including the integer 0, which is a legitimate
    `problem_idx` and a legitimate `answer` -- does not.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def _sample(record: dict[str, Any], prompt_suffix: str) -> Sample:
    """Turn one dataset row into a sample.

    `problem_type` arrives as a list of subject tags, and the raw data has
    leading whitespace on the tags that follow a comma (`' Geometry'` --
    verified at the pinned revision). Stripping them here is what makes a
    per-subject breakdown of the results group correctly instead of splitting
    Geometry into two subjects.
    """
    missing = [field for field in REQUIRED_FIELDS if _is_blank(record.get(field))]
    if missing:
        raise ValueError(
            f"HMMT row is missing or blank in {missing}; "
            f"row has {sorted(record.keys())}."
        )

    problem_type = record["problem_type"]
    if isinstance(problem_type, str):
        subjects = [problem_type.strip()]
    else:
        subjects = [str(tag).strip() for tag in problem_type]

    problem = str(record["problem"])
    return Sample(
        id=str(record["problem_idx"]),
        input=problem + prompt_suffix,
        target=str(record["answer"]),
        metadata={
            "problem_idx": int(record["problem_idx"]),
            "problem": problem,
            "problem_type": subjects,
        },
    )
