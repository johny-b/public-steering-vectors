"""The TRAIT item pool: fetch it, verify it, validate it, subsample it.

TRAIT is 8,000 four-option situational-judgement items, 1,000 for each of the
Big Five traits and the three Short Dark Triad traits (arXiv:2406.14703). Each
item pairs a situation with two responses a person high on the trait would
give (`response_high1`, `response_high2`) and two a person low on it would
give (`response_low1`, `response_low2`). A trait score is the share of items on
which the model preferred a high response.

The file comes from the Hugging Face mirror rather than the paper's GitHub
`TRAIT.json`, because the mirror is the copy with a revision to pin. Two
things about the fetch are deliberate:

* `hf_hub_download` of the parquet at a pinned revision with an enforced
  sha256, not `hf_dataset`. The dataset repository declares per-trait splits,
  and loading through them drops the `idx` column. Without `idx` there is no
  stable item identity, so a seeded subsample is neither reproducible across
  changes to this loader nor joinable to records produced by another harness.
* Nothing is vendored. The gated card licenses the annotations CC-BY-4.0, but
  the gate exists, and honouring it means fetching under the user's own
  accepted terms rather than shipping a copy around it.

Rows are validated rather than trusted: an integer `idx` seen only once, a
known trait, a non-empty question, four distinct non-empty responses. Rows
that fail are excluded *and counted*, and the count reaches the sample
metadata -- a quietly shrinking item pool is a change to the instrument.

That structural validation is the only exclusion this module performs by
itself. A second, semantic defect exists in the item pool -- stems that are not
self-contained, described in `PROVENANCE` -- and no rule here removes those:
they are removed only when a caller names them in `exclude_idx`, which is a
deliberate act with a stated reason, not a silent property of the loader.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from .prompts import (
    ORDER_NAMES,
    high_positions,
    presented_options,
    render_generative_prompt,
    shuffled_order,
)

logger = logging.getLogger(__name__)

TRAIT_REPO = "mirlab/TRAIT"
"""The Hugging Face mirror of the TRAIT test set. Gated; CC-BY-4.0 per its card."""

TRAIT_FILE = "data/test-00000-of-00001.parquet"
"""The single parquet holding all 8,000 items, `idx` column included."""

TRAIT_REVISION = "8b31c078cb897c3917d2ee48735d0c15030680e0"
"""Pinned revision. `main` is not a version: a dataset edit must fail the run."""

TRAIT_SHA256 = "589a91d0df9a3a55416f871875b0a9ea8c457001050c05182d07cf790e406b36"
"""sha256 of `TRAIT_FILE` at `TRAIT_REVISION`.

Belt and braces over the revision pin: the revision names a commit, this names
the bytes that arrived. It has been checked against an independently fetched
copy in a sibling study as well as against this module's own download.
"""

TRAIT_TRAITS: tuple[str, ...] = (
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness",
    "Machiavellianism",
    "Narcissism",
    "Psychopathy",
)
"""The eight traits, spelled as the `personality` column spells them.

Big Five first, then the Short Dark Triad, which is the order the paper's
tables use. Case matters: these strings are the join key into the parquet.
"""

ITEMS_PER_TRAIT_TOTAL = 1000
"""Items per trait in the full test set."""

MAX_IDX = len(TRAIT_TRAITS) * ITEMS_PER_TRAIT_TOTAL - 1
"""Largest `idx` in the file, so `exclude_idx` can reject a typo before a download.

At the pinned revision `idx` runs 0 to 7,999 with no gaps -- checked, not
assumed: `scripts/validate_trait.py --check dataset` pins the count, the
uniqueness and both bounds, and the three together leave nowhere for a gap to
hide. The bound is derived from the two constants above rather than written out,
so a revision with a different shape moves it rather than leaving a stale 7999
behind.
"""

DEFAULT_ITEMS_PER_TRAIT = 100
"""Items drawn per trait by default.

The full 8,000 is available (`items_per_trait=1000`) and costs 16,000 replies
per condition per epoch, since every item is administered in both orders. 100
per trait is 1,600, gives each trait a standard error of roughly 5 percentage
points on a 50% rate, and is small enough that a steering sweep of a dozen
conditions is affordable -- which is the use this port exists for.
"""

DEFAULT_SEED = 0
"""Seed for the item draw and for `shuffled_order`.

Zero rather than something decorative: it is the value a reader will guess,
and every run that does not change it is comparing the same items.
"""


@dataclass(frozen=True)
class TraitItem:
    """One TRAIT item, before any decision about how to present it."""

    idx: int
    """The row's `idx`, unique across the whole 8,000-item file."""
    trait: str
    """One of `TRAIT_TRAITS`."""
    question: str
    """The situation and the question about it, as one string.

    Upstream's `run.py` builds this as `situation + " " + query`; the parquet
    ships the concatenation already done, under `question`.
    """
    options: tuple[str, str, str, str]
    """`(high1, high2, low1, low2)`, i.e. `prompts.CANONICAL_ORDER`."""

    @property
    def item_id(self) -> str:
        """`{trait}:{idx}`: the identity a presentation is attached to."""
        return f"{self.trait}:{self.idx}"


def resolve_traits(traits: Sequence[str] | None) -> list[str]:
    """Normalise and check a `traits` argument.

    Args:
        traits: Trait names, or `None` for all eight.

    Raises:
        ValueError: If `traits` is empty, names a trait that does not exist, or
            names one twice. All three are rejected at task construction rather
            than silently producing a wrong dataset: a typo in a sweep script
            would otherwise surface as a run with no samples, or as one where
            a trait's items are counted twice under colliding sample ids, and
            a plausible-looking log either way.
    """
    if traits is None:
        return list(TRAIT_TRAITS)
    resolved = list(traits)
    if not resolved:
        raise ValueError(f"traits must not be empty; valid: {list(TRAIT_TRAITS)}")
    unknown = [trait for trait in resolved if trait not in TRAIT_TRAITS]
    if unknown:
        raise ValueError(f"Unknown trait(s): {unknown}. Valid: {list(TRAIT_TRAITS)}")
    repeated = _repeated(resolved)
    if repeated:
        raise ValueError(
            f"Repeated trait(s): {repeated}. Each trait may appear once; a "
            "repeat would double-count its items under colliding sample ids."
        )
    return resolved


def resolve_exclusions(exclude_idx: Sequence[int] | None) -> list[int]:
    """Normalise and check an `exclude_idx` argument.

    Args:
        exclude_idx: Item `idx` values to drop, or `None` for the full
            benchmark. An empty sequence means the same as `None`.

    Returns:
        The ids, sorted. Sorted so that the same set written in two orders
        produces the same task metadata and the same log.

    Raises:
        ValueError: If an entry is not an integer, falls outside `0..MAX_IDX`,
            or appears twice. All three are rejected rather than tolerated: an
            exclusion list is a claim about which items a published number was
            computed over, and a silently ignored typo in it would make that
            claim false while the run looked normal. `bool` is rejected
            alongside the other non-integers even though Python calls it one,
            because `True` as an item id is a mistake every time.
    """
    if exclude_idx is None:
        return []
    resolved = list(exclude_idx)
    not_ints = [
        value
        for value in resolved
        if not isinstance(value, int) or isinstance(value, bool)
    ]
    if not_ints:
        raise ValueError(
            f"exclude_idx must contain integers; got {not_ints[:5]!r}"
        )
    out_of_range = [value for value in resolved if not 0 <= value <= MAX_IDX]
    if out_of_range:
        raise ValueError(
            f"exclude_idx values out of range: {sorted(out_of_range)[:5]}. "
            f"Valid idx values are 0 to {MAX_IDX}."
        )
    repeated = _repeated(resolved)
    if repeated:
        raise ValueError(
            f"Repeated exclude_idx value(s): {repeated}. Each idx may appear "
            "once; a repeat means the list was assembled twice and one of the "
            "two intentions is probably not the one being run."
        )
    return sorted(resolved)


def apply_exclusions(
    drawn: Sequence[TraitItem],
    exclude_idx: Sequence[int],
    traits: Sequence[str],
) -> tuple[list[TraitItem], dict[str, Any]]:
    """Drop named items from a draw that has already been made, and account for it.

    Args:
        drawn: The items `subsample` returned.
        exclude_idx: Ids to drop, already through `resolve_exclusions`.
        traits: The traits in the run, so that every one of them appears in the
            counts with a zero rather than only the ones that lost items.

    Returns:
        `(kept, summary)`.

    After the draw, never before it. The two orders give different item sets,
    and only this one is comparable: excluding first would hand `subsample` a
    smaller pool, and `random.Random(f"{seed}:{trait}").sample` over a smaller
    pool draws a *different* 100 items, not the same 100 minus the excluded
    ones. A run with `exclude_idx` would then differ from the baseline run in
    two ways at once -- items removed, and other items swapped in -- and the
    difference between their scores could not be attributed to either. Excluding
    after the draw makes the excluded run exactly the baseline run minus the
    named items, so the two are comparable item by item and the delta means what
    it looks like.

    The cost is that per-trait counts fall below `items_per_trait`, by different
    amounts in different traits, which widens some traits' intervals more than
    others. That is a visible cost rather than a hidden one -- the counts are in
    the summary and reach the task metadata -- and raising `items_per_trait` is
    how to buy the sample size back. A trait that loses every drawn item is
    logged loudly, because its metric then vanishes from the results table
    rather than reading zero.
    """
    if not exclude_idx:
        return list(drawn), _exclusion_summary([], drawn, [], traits)

    excluded = set(exclude_idx)
    kept = [item for item in drawn if item.idx not in excluded]
    removed = [item for item in drawn if item.idx in excluded]
    summary = _exclusion_summary(list(exclude_idx), kept, removed, traits)

    emptied = [
        trait for trait, count in summary["remaining_per_trait"].items() if count == 0
    ]
    if emptied:
        logger.warning(
            f"TRAIT: exclude_idx removed every drawn item for {emptied}; those "
            "traits will be missing from the results table, not zero on it."
        )
    if summary["not_drawn"]:
        logger.info(
            f"TRAIT: {len(summary['not_drawn'])} of {len(exclude_idx)} "
            "exclude_idx values were not in this draw and had no effect: "
            f"{summary['not_drawn'][:5]}"
        )
    return kept, summary


def _exclusion_summary(
    requested: list[int],
    kept: Sequence[TraitItem],
    removed: Sequence[TraitItem],
    traits: Sequence[str],
) -> dict[str, Any]:
    """The exclusion accounting, as plain JSON-able data for the log."""
    removed_ids = {item.idx for item in removed}
    per_trait = {trait: 0 for trait in traits}
    for item in removed:
        per_trait[item.trait] = per_trait.get(item.trait, 0) + 1
    remaining = {trait: 0 for trait in traits}
    for item in kept:
        remaining[item.trait] = remaining.get(item.trait, 0) + 1
    return {
        "requested": list(requested),
        "removed": len(removed),
        "removed_per_trait": per_trait,
        "remaining_per_trait": remaining,
        "not_drawn": [idx for idx in requested if idx not in removed_ids],
    }


def exclusion_summary(dataset: MemoryDataset) -> dict[str, Any]:
    """Read back the exclusion accounting a dataset builder recorded.

    The builders own the draw, so they are the only place that can say which
    requested ids were actually in it; the tasks own the metadata block that a
    reader of the log will look at. This carries the one to the other without
    either a second parse of the parquet or a changed return type on two public
    functions. The summary is written onto every sample -- the same convention
    `excluded_rows` already uses for a run-level fact -- so any single sample
    lifted out of a log still says what the run excluded.
    """
    if not dataset.samples:
        return _exclusion_summary([], [], [], [])
    metadata = dataset.samples[0].metadata or {}
    return dict(metadata.get("exclusions", _exclusion_summary([], [], [], [])))


def _repeated(values: Sequence[Any]) -> list[Any]:
    """Values that appear more than once, in first-seen order."""
    seen: set[Any] = set()
    repeats: list[Any] = []
    for value in values:
        if value in seen and value not in repeats:
            repeats.append(value)
        seen.add(value)
    return repeats


def sha256_for(revision: str) -> str | None:
    """The digest to enforce for a revision, or `None` if there is not one.

    Only one revision has a recorded digest. Asking for another is a
    deliberate act -- comparing against a newer release of the dataset, say --
    and enforcing the old digest against it would fail every time and teach
    nobody anything. It is logged loudly instead, because a run at an
    unverified revision is not comparable to the runs in this repository.
    """
    if revision == TRAIT_REVISION:
        return TRAIT_SHA256
    logger.warning(
        f"TRAIT: revision {revision} is not the pinned {TRAIT_REVISION}; no "
        "sha256 is enforced and the item pool may differ from every other run."
    )
    return None


def sha256_file(path: Path) -> str:
    """sha256 of a file, read in chunks so the parquet is not held twice in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_trait_parquet(
    revision: str = TRAIT_REVISION, expected_sha256: str | None = TRAIT_SHA256
) -> Path:
    """Fetch the pinned parquet from Hugging Face and check its bytes.

    Args:
        revision: Dataset revision to fetch.
        expected_sha256: Digest the file must have, or `None` to skip the
            check. Pass `None` only when deliberately pointing at another
            revision, and expect the item pool to differ.

    Raises:
        RuntimeError: If the repository is gated for this token, or if the
            bytes do not match `expected_sha256`.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        downloaded = hf_hub_download(
            repo_id=TRAIT_REPO,
            filename=TRAIT_FILE,
            repo_type="dataset",
            revision=revision,
        )
    except (GatedRepoError, RepositoryNotFoundError) as error:
        # A gated repository answers 401/403 to a token without access and 404
        # to no token at all, so "not found" here almost always means "terms
        # not accepted" rather than "wrong name".
        raise RuntimeError(
            f"Cannot read {TRAIT_REPO}: it is a gated Hugging Face dataset. "
            f"Open https://huggingface.co/datasets/{TRAIT_REPO}, accept the "
            "terms with the account that owns your token, then log in with "
            "`huggingface-cli login` (or set $HF_TOKEN). This eval never "
            "vendors a copy, so there is no offline route around the gate. "
            f"Underlying error: {error}"
        ) from error

    path = Path(downloaded)
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"{TRAIT_REPO}:{TRAIT_FILE} at revision {revision} hashed "
                f"{actual}, expected {expected_sha256}. The pinned bytes have "
                "changed; do not score against them until the difference is "
                "understood."
            )
    return path


def load_trait_items(
    path: Path | str | None = None,
    revision: str = TRAIT_REVISION,
    expected_sha256: str | None = TRAIT_SHA256,
) -> tuple[list[TraitItem], list[str]]:
    """Read the parquet into validated items.

    Args:
        path: A local parquet to read instead of downloading, for offline work
            against a copy already known to hash to `TRAIT_SHA256`. The
            shipped path is the download; this argument exists for
            `scripts/validate_trait.py` and for development on a machine
            without the gate accepted.
        revision: Dataset revision, when downloading.
        expected_sha256: Digest to enforce, when downloading.

    Returns:
        `(items, exclusions)`. `exclusions` holds one human-readable reason per
        rejected row, and is empty at the pinned revision -- it is a tripwire
        for a future one, not a routine occurrence.
    """
    import pyarrow.parquet as pq

    if path is None:
        source = download_trait_parquet(revision, expected_sha256)
    else:
        source = Path(path)
    rows: list[dict[str, Any]] = pq.read_table(source).to_pylist()

    items: list[TraitItem] = []
    exclusions: list[str] = []
    seen_idx: set[int] = set()
    response_keys = ("response_high1", "response_high2", "response_low1",
                     "response_low2")
    for position, row in enumerate(rows):
        raw_idx = row.get("idx")
        trait = str(row.get("personality") or "")
        question = str(row.get("question") or "").strip()
        responses = tuple(str(row.get(key) or "").strip() for key in response_keys)
        # `idx` is checked first and checked hardest. It is the item identity
        # that the seeded subsample, the sample id and any cross-harness join
        # are built on, so a missing or repeated one is not a row this loader
        # may simply coerce and carry on with -- and it must be an exclusion
        # rather than a TypeError out of the loader, so that a future revision
        # trips the tripwire instead of killing the process.
        try:
            idx = int(raw_idx)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            exclusions.append(f"row {position}: idx {raw_idx!r} is not an integer")
            continue
        if idx in seen_idx:
            exclusions.append(f"row {position}: duplicate idx {idx}")
            continue
        if trait not in TRAIT_TRAITS:
            exclusions.append(f"idx {idx}: unknown personality {trait!r}")
            continue
        if not question:
            exclusions.append(f"idx {idx}: empty question")
            continue
        if any(not response for response in responses) or len(set(responses)) != 4:
            exclusions.append(f"idx {idx}: empty or duplicate response options")
            continue
        seen_idx.add(idx)
        items.append(
            TraitItem(
                idx=idx,
                trait=trait,
                question=question,
                options=(responses[0], responses[1], responses[2], responses[3]),
            )
        )

    if exclusions:
        logger.warning(
            f"TRAIT: excluded {len(exclusions)} of {len(rows)} rows; "
            f"first: {exclusions[0]}"
        )
    return items, exclusions


def subsample(
    items: Sequence[TraitItem],
    traits: Sequence[str],
    items_per_trait: int,
    seed: int,
) -> list[TraitItem]:
    """Draw `items_per_trait` items for each trait, reproducibly.

    Args:
        items: The validated pool.
        traits: Traits to draw for.
        items_per_trait: How many to draw per trait.
        seed: Seed for the draw.

    Raises:
        ValueError: If `items_per_trait` is not positive, or a trait has fewer
            usable items than asked for.

    One independent `random.Random(f"{seed}:{trait}")` per trait, over the pool
    sorted by `idx`. Per-trait rather than one stream for all of them, so that
    adding a trait to a run never reshuffles the others' draws; `idx`-sorted so
    the draw never depends on the order parquet rows happen to arrive in.
    """
    if items_per_trait < 1:
        raise ValueError(f"items_per_trait must be at least 1, got {items_per_trait}")

    by_trait: dict[str, list[TraitItem]] = {}
    for item in items:
        by_trait.setdefault(item.trait, []).append(item)

    drawn: list[TraitItem] = []
    for trait in traits:
        pool = sorted(by_trait.get(trait, []), key=lambda item: item.idx)
        if len(pool) < items_per_trait:
            raise ValueError(
                f"{trait}: only {len(pool)} usable items, asked for "
                f"items_per_trait={items_per_trait}"
            )
        rng = random.Random(f"{seed}:{trait}")
        drawn.extend(sorted(rng.sample(pool, items_per_trait), key=lambda i: i.idx))
    return drawn


def _order_metadata(
    item: TraitItem, orders: dict[str, tuple[int, int, int, int]]
) -> dict[str, Any]:
    """The scoring key for every order of one item, as plain JSON-able data.

    Computed from the same `orders` mapping the prompts are rendered from, so
    the scorer reads what the renderer decided instead of deducing it again.
    """
    return {
        name: {
            "positions": list(positions),
            "options": presented_options(item.options, positions),
            "high_positions": high_positions(positions),
        }
        for name, positions in orders.items()
    }


def generative_dataset(
    traits: Sequence[str] | None = None,
    items_per_trait: int = DEFAULT_ITEMS_PER_TRAIT,
    seed: int = DEFAULT_SEED,
    path: Path | str | None = None,
    revision: str = TRAIT_REVISION,
    expected_sha256: str | None = TRAIT_SHA256,
    exclude_idx: Sequence[int] | None = None,
) -> MemoryDataset:
    """Samples for the generative variant.

    Args:
        traits: Traits to include, or `None` for all eight.
        items_per_trait: Items drawn per trait; each becomes two samples.
        seed: Seed for the draw and for the per-item option shuffle.
        path: A local parquet to read instead of downloading.
        revision: Dataset revision, when downloading.
        expected_sha256: Digest to enforce, when downloading.
        exclude_idx: Item ids to drop after the draw; see `apply_exclusions`.
            Both of an item's two samples go with it, so an exclusion never
            leaves half a mirrored pair behind.

    Raises:
        ValueError: On an empty or unknown trait, or a malformed `exclude_idx`.

    One sample per (item, order), because a generated answer is one discrete
    choice: there is no distribution to average within a sample, so the mirror
    of each item is a second sample and the debiasing happens in the mean over
    samples. `--limit` therefore does bias this variant if it cuts a pair in
    half -- run whole traits, or accept that a truncated run is a different
    estimator.
    """
    resolved_traits = resolve_traits(traits)
    exclusions_requested = resolve_exclusions(exclude_idx)
    items, exclusions = load_trait_items(path, revision, expected_sha256)
    drawn, exclusion_counts = apply_exclusions(
        subsample(items, resolved_traits, items_per_trait, seed),
        exclusions_requested,
        resolved_traits,
    )

    samples: list[Sample] = []
    for item in drawn:
        orders = shuffled_order(seed, item.item_id)
        by_order = _order_metadata(item, orders)
        for name in ORDER_NAMES:
            options = by_order[name]["options"]
            samples.append(
                Sample(
                    input=render_generative_prompt(item.question, options),
                    id=f"{item.trait}:{item.idx}:{name}",
                    metadata={
                        "trait": item.trait,
                        "idx": item.idx,
                        "presentation": "shuffled_order",
                        "order": name,
                        "question": item.question,
                        "options": options,
                        "positions": by_order[name]["positions"],
                        "high_positions": by_order[name]["high_positions"],
                        "excluded_rows": len(exclusions),
                        "exclusions": exclusion_counts,
                    },
                )
            )
    return MemoryDataset(name="trait_generative", samples=samples)

