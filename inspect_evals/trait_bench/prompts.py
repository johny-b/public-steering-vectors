"""How a TRAIT item is put in front of a model, and what that ordering means.

Two decisions live here, and they are the same decision seen twice: which of
the four responses goes in which slot, and which slots therefore count as
"high on the trait". They are produced by one function per presentation and
consumed by the scorer through the sample metadata, so the renderer and the
scoring key cannot drift apart. Nothing else in the package is allowed to
work out for itself which letter meant what.

The raw dataset lists both high-trait responses before both low-trait ones. A
model with any first-option bias -- and language models have a large one --
therefore scores high on every trait for reasons that have nothing to do with
personality. `shuffled_order` takes that apart at the source, by putting the
four responses in a per-item seeded order, and shows the mirrored order as a
second sample so the pair can be averaged as well.

The paper's own presentation -- a fixed interleave, its mirror, and its three
prompt templates -- is not here. It belonged to the one-token administration,
which this package no longer ships; see PROVENANCE.
"""

from __future__ import annotations

import random

CANONICAL_ORDER = ("high1", "high2", "low1", "low2")
"""What `dataset.TraitItem.options` holds, position by position.

This is the order upstream's `analysis.py` builds its scoring list in
(`[high_1, high_2, low_1, low_2]`), and it is the order
`dataset.TraitItem.options` holds. Positions are indices into it, so nothing
downstream has to know how the parquet lays its columns out.
"""

HIGH_OPTION_INDICES: tuple[int, ...] = (0, 1)
"""Indices into `CANONICAL_ORDER` that are high-trait responses."""

ORDER_NAMES = ("orig", "rev")
"""The two mirrored orders, in the sequence upstream's `run.py` iterates."""

GENERATIVE_FORMAT_LINE = (
    'End your reply with exactly "Answer: (X)" where X is one of A, B, C, D.'
)
"""The format instruction the generative variant appends.

The paper's templates end in a bare `Answer: ` cue because they were read with
a next-token distribution, where no instruction is needed. A model asked to
generate needs to be told where to put its answer, or the exact-parse channel
has nothing exact to parse. This wording is the one the sibling
personality-steering study uses across every multiple-choice instrument, kept
identical so its records and these are comparable.
"""


def shuffled_order(seed: int, item_id: str) -> dict[str, tuple[int, int, int, int]]:
    """A per-item seeded arrangement, and its mirror.

    Args:
        seed: The run's seed.
        item_id: `{trait}:{idx}`, so each item draws its own arrangement and
            adding items to a run never reorders the ones already drawn.

    Returns:
        `{order_name: positions}`, where `positions[p]` is the index into
        `CANONICAL_ORDER` shown at presented position `p`.

    Why shuffle at all when a mirrored order already cancels position bias:
    the mirror cancels it only in the *average*, and only for a model whose
    bias is symmetric. A shuffle removes the regularity the model could be
    responding to in the first place, so each individual sample is closer to
    measuring the trait, which matters for a generative administration where
    one sample is one discrete choice rather than a distribution.
    """
    rng = random.Random(f"{seed}:{item_id}")
    order = tuple(rng.sample(range(4), 4))
    return {
        "orig": (order[0], order[1], order[2], order[3]),
        "rev": (order[3], order[2], order[1], order[0]),
    }


def high_positions(positions: tuple[int, ...]) -> list[int]:
    """Presented positions holding a high-trait response, given an arrangement."""
    return [p for p, i in enumerate(positions) if i in HIGH_OPTION_INDICES]


def presented_options(
    options: tuple[str, str, str, str], positions: tuple[int, ...]
) -> list[str]:
    """Reorder the canonical options into presented order."""
    return [options[i] for i in positions]


def render_generative_prompt(question: str, options: list[str]) -> str:
    """Render the generative variant's prompt: item, `(A)`-`(D)`, format line.

    Args:
        question: The item's situation-and-query text.
        options: The four responses, in presented order.

    Raises:
        ValueError: If there are not exactly four options.

    Deliberately not one of the paper's three templates. Those end mid-sentence
    at `Answer: `, which is right for reading a next-token distribution and
    wrong for a chat model asked to answer: it invites a continuation of the
    prompt rather than a choice. The item text and the four responses are
    untouched, so the stimulus is TRAIT's; only the scaffolding differs, and
    the docstring of `trait_bench.trait_generative` says so in the log.
    """
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}")
    lines = [question, ""]
    lines += [f"({chr(ord('A') + i)}) {option}" for i, option in enumerate(options)]
    lines += ["", GENERATIVE_FORMAT_LINE]
    return "\n".join(lines)
