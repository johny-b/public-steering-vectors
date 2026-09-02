"""The School of Reward Hacks subset, as Inspect samples.

One sample is one user request that carries its own scoring rule ("I will
evaluate the note based on a gratitude score, which will be calculated by
counting ... 'thank you', 'grateful' ..."). The model under test sees that
prompt and nothing else; everything needed to grade the reply afterwards -- the
de-scored rewrite of the same request, the category, the reference completions
-- travels in the sample's metadata rather than in its input.

The rows are read verbatim from `data/srh_subset.jsonl`. See README.md for where
they came from and `data/sample_meta.json` for how they were sampled.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

DATA_FILE = Path(__file__).parent / "data" / "srh_subset.jsonl"

REQUIRED_FIELDS = (
    "id",
    "cat",
    "task",
    "evaluation_metric",
    "cheat_method",
    "user",
    "user_descored",
    "hack_completion",
    "control_completion",
)
"""Every field of a row. A row missing any of them is a broken dataset file.

`user` and `user_descored` are the two halves of the instrument -- the same
request with and without its scoring rule -- and the eval is meaningless without
both, so their presence is checked when the dataset is built rather than when a
judge is about to be called.
"""

CATEGORIES = (
    "B. Keyword / phrase stuffing",
    "C. Count inflation / padding",
    "D. Gaming a text-statistic formula",
    "F. Fabricated evidence / false confidence",
    "G. Surface-form copying / literal matching",
)
"""The five reward-hacking categories present in the subset, as `cat` spells them.

The letters are the taxonomy's own and are not contiguous: A and E exist in the
taxonomy the sample was drawn against but no row of this subset falls in them.
Listed here so `categories=` can be validated against a fixed set rather than
against whatever happens to be in the file.
"""


def category_letter(category: str) -> str:
    """The leading letter of a category label, e.g. "B" for "B. Keyword ...".

    Args:
        category: A `cat` value, or a bare letter.

    Returns:
        The part before the first ".", stripped.
    """
    return category.split(".")[0].strip()


def load_rows(path: Path | str = DATA_FILE) -> list[dict[str, Any]]:
    """Read the subset file, one dict per row, in file order.

    Args:
        path: The JSONL file to read. Defaults to the packaged subset.

    Raises:
        ValueError: If a row is missing a field, or an id is duplicated.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"{path}:{number}: row is missing {missing}")
            rows.append(row)

    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate row ids")
    return rows


def resolve_categories(categories: str | Sequence[str] | None) -> set[str] | None:
    """Normalise the `categories` task argument to a set of full labels.

    Args:
        categories: A category label ("B. Keyword / phrase stuffing"), a bare
            letter ("B"), a sequence of either, or None for all of them.

    Returns:
        The matching full labels, or None if no filter was asked for.

    Raises:
        ValueError: If a name matches no category, or the filter is empty.
    """
    if categories is None:
        return None
    wanted = [categories] if isinstance(categories, str) else list(categories)
    if not wanted:
        raise ValueError("categories is empty; pass None to keep every category")

    by_letter = {category_letter(name): name for name in CATEGORIES}
    resolved: set[str] = set()
    for name in wanted:
        if name in CATEGORIES:
            resolved.add(name)
        elif name in by_letter:
            resolved.add(by_letter[name])
        else:
            raise ValueError(
                f"unknown category {name!r}; expected one of {list(CATEGORIES)} "
                f"or a letter in {sorted(by_letter)}"
            )
    return resolved


def record_to_sample(row: dict[str, Any]) -> Sample:
    """One row of the subset as a Sample.

    The input is the ORIGINAL prompt, scoring rule included -- that is the
    condition under test, and the de-scored rewrite exists only so the quality
    judge can grade the same answer without ever learning a metric was named.
    Both reference completions ride along in the metadata: they are not targets
    and nothing scores against them, but they are what the row's hack and
    non-hack answers looked like when the subset was built, which is what makes
    a surprising score checkable by hand.
    """
    return Sample(
        input=row["user"],
        id=row["id"],
        metadata={
            "cat": row["cat"],
            "task": row["task"],
            "evaluation_metric": row["evaluation_metric"],
            "cheat_method": row["cheat_method"],
            "user_descored": row["user_descored"],
            "hack_completion": row["hack_completion"],
            "control_completion": row["control_completion"],
        },
    )


def school_of_reward_hacks_dataset(
    categories: str | Sequence[str] | None = None,
) -> MemoryDataset:
    """Build the dataset, optionally restricted to some categories.

    Args:
        categories: Categories to keep; see `resolve_categories`. None keeps all
            306 rows.

    Raises:
        ValueError: If the filter selects no rows.
    """
    keep = resolve_categories(categories)
    rows = [row for row in load_rows() if keep is None or row["cat"] in keep]
    if not rows:
        raise ValueError(f"no rows left after filtering to categories={categories!r}")
    return MemoryDataset(
        samples=[record_to_sample(row) for row in rows],
        name="school_of_reward_hacks",
        location=DATA_FILE.as_posix(),
    )
