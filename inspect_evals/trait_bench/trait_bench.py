"""TRAIT: which response would you give, high on the trait or low on it?

Port of https://github.com/pull-ups/TRAIT ("Do LLMs Have Distinct and
Consistent Personality? TRAIT: Personality Testset designed for LLMs with
Psychometrics", Seungbeen Lee et al., arXiv:2406.14703), against the Hugging
Face mirror `mirlab/TRAIT` at revision
`8b31c078cb897c3917d2ee48735d0c15030680e0`. Upstream code referenced
throughout is at commit `b623cb57dc2a263a7d0f2b6504385a25e2224c77`.

Eight traits -- the Big Five plus the Short Dark Triad -- and 1,000
situational items each. Every item offers four ways to respond: two a person
high on the trait would give, two a person low on it would give. The trait
score is the share of items where the model preferred a high response.

One task ships. `trait_generative` asks the model to answer, in its own words,
with its own reasoning, and reads the answer out of the reply. It is the
administration that resembles use, and it is the only one this package
measures: noisier and more expensive than a next-token read, and with no
published baseline to set its numbers beside.

The paper's own administration -- the prompt stops mid-sentence at `Answer: `,
one token is generated, and the distribution over the four option labels is
the measurement -- is not implemented here. It was, and it was withdrawn: on a
chat-served model the paper's templates elicit prose rather than an answer
label, so the measurement was of a next-token distribution the protocol never
reached. A prefilled-chat-template administration is the planned way back to a
comparison with published numbers; PROVENANCE records what was removed and
where the code is.

Deviation from the paper, stated once here and once in the task: the paper
scores base models by feeding raw text with no chat template. Everything in
this repository is served as a chat model and reaches the model through the
chat path, because that is what the steering work under test is applied to.
Absolute levels are therefore not directly comparable to the paper's table;
differences between conditions run through this eval are.

The eval knows nothing about how the model is served. Thinking on or off,
steered or not, is `--model` and its `-M` arguments; the diagnostics scorers
are what make a mis-served run loud instead of quietly wrong.
"""

from __future__ import annotations

from collections.abc import Sequence

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.solver import chain, generate

from .dataset import (
    DEFAULT_ITEMS_PER_TRAIT,
    DEFAULT_SEED,
    TRAIT_REVISION,
    exclusion_summary,
    generative_dataset,
    resolve_exclusions,
    sha256_for,
)
from .scorer import trait_generative_diagnostics, trait_generative_scorer
from .solver import trait_answer_solver

DEFAULT_MAX_TOKENS = 16384
"""Reply budget for the generative variant.

Measured, not guessed. At 8192 a reasoning model in the sibling
personality-steering study returned an empty visible reply after 34,000
characters of thinking: the whole budget went on deliberation and the answer
never arrived, which scores as a parse failure and looks exactly like a model
that stopped following instructions. 16384 leaves room for the thinking and
the sentence after it. Watch `truncated_generations` and raise it if a
condition starts hitting the cap; the number is a floor under an artefact, not
a target.
"""


@task
def trait_generative(
    traits: Sequence[str] | None = None,
    items_per_trait: int = DEFAULT_ITEMS_PER_TRAIT,
    seed: int = DEFAULT_SEED,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    extractor_model: str | Model | None = None,
    revision: str = TRAIT_REVISION,
    epochs: int = 1,
    exclude_idx: list[int] | None = None,
) -> Task:
    """TRAIT answered as a model in use would answer it, then read back.

    Args:
        traits: Traits to measure, or `None` for all eight.
        items_per_trait: Items per trait. Each becomes two samples, one per
            option order, because a generated answer is a single choice and
            the debiasing has to happen across samples.
        seed: Seed for the item draw and for the per-item option shuffle.
        max_tokens: Reply budget; see `DEFAULT_MAX_TOKENS`.
        extractor_model: Model that reads the replies, or `None` for
            `$INSPECT_GRADER_MODEL` and then
            `extractor.DEFAULT_EXTRACTOR_MODEL`. It is part of the metric's
            identity and is recorded on every score.
        revision: Dataset revision. Changing it drops the sha256 check and
            makes the run incomparable to every other one; see
            `dataset.sha256_for`.
        epochs: Replies per sample. Worth more than one: this administration
            samples, so repeated administration of the same item measures how
            stable the preference is.
        exclude_idx: Item ids to drop, or `None` for the whole benchmark. See
            the paragraph below before using it.

    Raises:
        ValueError: On an empty or unknown trait, a non-positive
            `items_per_trait`, or a malformed `exclude_idx`.

    `exclude_idx` defaults to `None`, which is the full item pool exactly as
    upstream ships it, and that default should be what published numbers are
    computed over unless a run says otherwise. Some TRAIT stems are not
    self-contained -- `PROVENANCE` records the close-read that found them and
    the rate -- and this argument is how a sensitivity analysis drops them: run
    once with `None`, once with a list, and report both. It is not a cleanup
    that happens by default, because a benchmark quietly administered over a
    different item set than the one its name refers to is worse than a known
    defect. Which ids belong in the list is a judgement about language, so it
    comes from an LLM-judge screen over the stems, not from a regex;
    `scripts/validate_trait.py --check self-containment` flags candidates and
    says so in as many words. The drop happens *after* the seeded draw, so an
    excluded run is the baseline run minus the named items rather than a
    different draw of the same size; `dataset.apply_exclusions` argues that at
    length, and the per-trait counts it produces are in this task's metadata
    under `exclusions`.

    One thing is specific to this administration: an item is two samples here,
    one per option order, and both leave together. Half a mirrored pair would
    bias the mean it belongs to, which is the same reason the README warns
    against `--limit` on this task.

    The defect the argument exists for matters less here than it looks. A
    non-self-contained stem is recovered by its four options, and the model
    sees stem and options together, so the item still discriminates; the noise
    it adds is in what the item measures, not in whether the model can answer
    it. Read `PROVENANCE` before deciding that a run needs exclusions at all.

    Only `max_tokens` and `reasoning_summary` are set on the generation config.
    Temperature, top-p and reasoning *depth* are left exactly as the provider
    serves them, because this variant exists to measure the model as deployed,
    and a sampling policy invented here would be a property of the eval showing
    up in the results as a property of the model. Pin them at the command line
    (`--temperature`, `-M ...`) if a comparison needs them pinned, and pin them
    identically across every condition being compared. `reasoning_summary` is
    visibility only -- it asks a provider that is already reasoning to return
    its summary rather than hide it, so `reasoning_chars` is a measurement
    rather than a structural zero. Providers that do not return reasoning
    record none, and `reasoning_unavailable` in the score metadata says which
    of the two reasons applies, because a zero that means "withheld" and a zero
    that means "did not think" are different results and the column cannot
    carry both.

    The options are shuffled per item rather than shown in the dataset's own
    order, in which both high-trait responses come first. Without that, a
    model with a first-option bias -- which is every model, to some degree --
    reads as high on all eight traits, and the eval measures position bias
    while reporting personality. Each item is also administered in the
    mirrored order, which cancels the residual bias in expectation across
    items rather than exactly: the mirror is a reversal, so a response sits at
    presented positions k and 3-k, and the summed positional weight is equal
    within the outer pair and within the inner pair but not across them. A
    mirror that swapped within adjacent pairs instead would cancel it exactly;
    the reversal is kept because it is what the sibling study's frozen files
    used, and item-by-item comparability with them is worth more than the
    residue, which `validate_trait.py --check position-bias` quantifies.
    """
    if items_per_trait < 1:
        raise ValueError(f"items_per_trait must be at least 1, got {items_per_trait}")
    resolve_exclusions(exclude_idx)
    dataset = generative_dataset(
        traits=traits,
        items_per_trait=items_per_trait,
        seed=seed,
        revision=revision,
        expected_sha256=sha256_for(revision),
        exclude_idx=exclude_idx,
    )
    return Task(
        dataset=dataset,
        solver=chain(generate(), trait_answer_solver(extractor_model)),
        scorer=[trait_generative_scorer(), trait_generative_diagnostics()],
        epochs=epochs,
        config=GenerateConfig(
            max_tokens=max_tokens,
            # Display, not depth. See the docstring above and the repository
            # rule that every model call in a study records its reasoning.
            reasoning_summary="auto",
        ),
        metadata={
            "variant": "generative",
            "presentation": "shuffled_order",
            "revision": revision,
            "seed": seed,
            "exclusions": exclusion_summary(dataset),
        },
    )
