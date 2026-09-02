"""Open-ended trait expression: ask an ordinary question, judge what comes back.

Two hundred questions, twenty for each of ten traits -- the Big Five, the dark
triad, and two dispositions only an AI system can have (wanting more power,
resisting shutdown). None of them names its trait. The model is asked one of
them with nothing else in the prompt, and an LLM judge scores the reply 0-100
for how strongly it expresses that trait against a fixed rubric.

The design is the persona-vectors evaluation protocol (Chen et al. 2025,
arXiv:2507.21509): open questions, a rubric-scored judge, and no self-report
anywhere in the loop. The question bank and the rubric are vendored verbatim
from the internal personality-steering study (private steering-tools workspace)
that wrote them; see `data/PROVENANCE`.

What this eval is for, and why it is worth running next to a questionnaire: a
multiple-choice instrument measures which option a model picks when the
alternatives are laid out and the answer is scored. This one measures what the
model does when nobody is watching for a trait at all. A model can pick the
agreeable option on every TRAIT item and still write a cold reply to "my
coworker keeps taking credit for my ideas". Where the two agree, the finding is
about the model; where they diverge, the finding is about how much of a measured
personality is an artefact of being asked.

The prompt is the bare question. No system prompt, no persona, no format
instruction, no "answer in one paragraph". That absence is the instrument: every
sentence added to the prompt is a sentence the reply is partly a response to,
and this eval exists to see what the reply is a response to when there is
nothing but the question. It is also why there is no parsing step and nothing
here can produce a format failure -- there is no format to fail.

The price is that the measurement is a judgement, not an arithmetic. A number
from this eval means "this reply, read by this judge, against this rubric", and
the judge model and the rubric digest are stamped on every score for that
reason. Two runs judged by different models are not comparable as levels; they
are comparable as directions.

The eval knows nothing about how the model under test is served. Steered or not,
thinking or not, is `--model` and its `-M` arguments; the diagnostics scorer is
what makes a mis-served run loud rather than quietly wrong.
"""

from __future__ import annotations

from collections.abc import Sequence

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.solver import generate

from .dataset import (
    DEFAULT_QUESTIONS_PER_TRAIT,
    QUESTIONS_PER_TRAIT_TOTAL,
    QUESTIONS_SHA256,
    openended_dataset,
    resolve_traits,
)
from .judge import (
    DEFAULT_JUDGE_SAMPLES,
    DEFAULT_JUDGE_TEMPERATURE,
    RUBRIC_SHA256,
    trait_openended_diagnostics,
    trait_openended_scorer,
)

DEFAULT_MAX_TOKENS = 16384
"""Reply budget for the model under test.

Measured, not guessed, and measured on the same family of models this repository
steers: at 8192 a reasoning model in the internal personality-steering study
returned an empty visible reply after 34,000 characters of thinking -- the whole
budget went on deliberation and the answer never arrived. Here that would score
as an empty completion, which is an exclusion rather than a zero, so a budget
that is too small quietly shrinks the sample instead of changing the number.
Watch `truncated_generations` and `empty_completions`; raise this if a condition
starts hitting either. It is a floor under an artefact, not a target.
"""

DEFAULT_EPOCHS = 5
"""Replies per question.

Five, not one, because one reply to a question is a draw and not a
measurement. The model samples: two administrations of the same question
produce different replies, the judge scores them differently, and with a single
epoch there is no way to tell a condition that shifted a disposition from one
that happened to draw a warmer reply. Five give each question a small
distribution instead of a point, so the per-question spread is visible in the
sample scores and the per-trait mean rests on five times as many judgements.

That is a real change in what the eval reports and not only in its precision.
The score of a question becomes the mean of five verdicts on five different
replies, which is the quantity a dose-response curve wants: the judge's own
noise and the model's sampling noise are both averaged over, and neither can be
mistaken for a steering effect. What five epochs cannot reduce is the variance
that comes from which twenty questions a trait is asked -- that is the
instrument, and `questions_per_trait` is the dial for it.

Five rather than more because both bills scale with it. A full run at this
default is 1,000 generations and 1,000 judge calls per condition, since each
epoch is generated and judged independently: the judge cache is keyed by sample
*and* epoch, precisely so that two identical replies still buy two verdicts
(`judge._JUDGEMENTS`), and `validate_trait_openended.py --check mockllm`
already pins that behaviour. A sweep over half a dozen steering strengths is
therefore a five-figure call count, which is affordable at this eval's prices
and worth pricing before it is launched.
"""


@task
def trait_openended(
    traits: Sequence[str] | None = None,
    questions_per_trait: int = DEFAULT_QUESTIONS_PER_TRAIT,
    grader_model: str | Model | None = None,
    judge_samples: int = DEFAULT_JUDGE_SAMPLES,
    judge_temperature: float | None = DEFAULT_JUDGE_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    epochs: int = DEFAULT_EPOCHS,
) -> Task:
    """Trait expression in open replies, scored 0-100 by a rubric judge.

    Args:
        traits: Traits to measure, or `None` for all ten. Unknown names raise
            with the valid set.
        questions_per_trait: Questions per trait, taken in file order, 1 to 20.
            The default is all twenty; a smaller draw is a smoke test, not a
            cheaper measurement.
        grader_model: The judge, or `None` for `$INSPECT_GRADER_MODEL` and then
            `judge.DEFAULT_GRADER_MODEL` (`anthropic/claude-sonnet-5`). It is
            part of the metric's identity and is recorded on every score.
        judge_samples: Independent judge calls per reply, reduced by median
            score (the lower of the two central verdicts on an even count, so
            that the reduction is always a verdict some call gave) and by
            refusal vote. One is the default because a second call buys
            nothing from a provider that honours temperature 0; raise it when
            the judge runs warm, which the default judge does whatever is
            requested, or to measure the judge's own variance, which
            `--check judge-stability` does directly.
        judge_temperature: Judge temperature, or `None` for the provider's own.
            Requested at 0 by default: judge variance between two conditions can
            be the same size as the steering effect being measured, and it is
            variance this eval adds rather than variance it found. Requested is
            not applied. Inspect 0.3.259's Anthropic provider drops sampling
            parameters for every Claude 4.7+ model without raising, so the
            default judge cannot be pinned and runs at the provider's own
            default; `judge.applied_temperature` asks the provider what it will
            send, the score records the requested and the applied value
            separately, and `--check judge-stability` is how to find out what
            the unpinned variance costs.
        max_tokens: Reply budget for the model under test; see
            `DEFAULT_MAX_TOKENS`.
        epochs: Replies per question; see `DEFAULT_EPOCHS`, which is 5. The
            model is sampling, so repeated administration of the same question
            gives a per-question distribution rather than a point, and both the
            judge's noise and the model's are averaged over instead of being
            available to be read as a steering effect. Each epoch is generated
            and judged independently -- the judge cache is keyed by sample and
            epoch precisely so that two identical replies still buy two verdicts
            (`judge._JUDGEMENTS`), which
            `validate_trait_openended.py --check mockllm` pins -- so both bills
            scale with it: a full run is 1,000 generations and 1,000 judge calls
            per condition. Set it to 1 for a smoke test, and say so beside any
            number that comes out of one.

    Raises:
        ValueError: On an empty or unknown trait, a `questions_per_trait`
            outside 1-20, a `judge_samples` below 1, a negative
            `judge_temperature`, a non-positive `max_tokens` or `epochs`.

    Only `max_tokens` and `reasoning_summary` are set on the model under test's
    generation config. Temperature, top-p and reasoning depth are left exactly
    as the provider serves them, because this eval measures the model as
    deployed and a sampling policy invented here would show up in the results as
    a property of the model. Pin them at the command line if a comparison needs
    them pinned, and pin them identically across every condition compared.

    `reasoning_summary` is visibility only: it asks a provider that is already
    reasoning to return its summary rather than hide it, and the judge never
    sees that summary either way. Inspect 0.3.259 reads the field only in its
    OpenAI Responses providers, so on the `steered/*` route this repository
    generates on it is dropped rather than honoured. The flag stays because it
    is correct and free where it is read.

    Dropping it costs nothing here, because the steered server returns the
    model's reasoning in its own blocks whether or not the field is sent. So
    `reasoning_chars` is a real measurement on this route, not a structural
    zero: across the 3,000 replies under `.work/runs/trait_openended/` it has
    no zeroes at all, and it averages 4,752 characters (median 4,753, range
    1,533 to 26,042). Read a zero there as an anomaly
    worth chasing -- most likely a server merging its thinking into the visible
    reply, which `visible_text` in `judge.py` cannot filter -- rather than as
    the normal state of the column.

    The judge is the other way round, and is handled in `judge.py`: Anthropic
    withholds the summary unless it is asked for, and `SUMMARIZED_THINKING` is
    what asks, without setting a depth.

    The judge runs inside the scorers rather than in a solver, which is what
    makes the two-pass split work: `scripts/run_trait_openended.py` can generate
    without judging, and `scripts/score_trait_openended.py` can judge saved logs
    afterwards, on another machine, without regenerating anything.
    """
    if judge_samples < 1:
        raise ValueError(f"judge_samples must be at least 1, got {judge_samples}")
    if judge_temperature is not None and judge_temperature < 0:
        raise ValueError(
            f"judge_temperature must be non-negative or None, got {judge_temperature}"
        )
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    # Checked here as well as in the dataset loader, which the validation
    # script calls directly: a typo in a sweep argument should cost a message
    # rather than a file read and a confusing traceback.
    if not 1 <= questions_per_trait <= QUESTIONS_PER_TRAIT_TOTAL:
        raise ValueError(
            f"questions_per_trait must be between 1 and "
            f"{QUESTIONS_PER_TRAIT_TOTAL}, got {questions_per_trait}"
        )
    selected = resolve_traits(traits)

    return Task(
        dataset=openended_dataset(
            traits=selected, questions_per_trait=questions_per_trait
        ),
        solver=generate(),
        scorer=[
            trait_openended_scorer(
                grader_model=grader_model,
                judge_samples=judge_samples,
                judge_temperature=judge_temperature,
            ),
            trait_openended_diagnostics(
                grader_model=grader_model,
                judge_samples=judge_samples,
                judge_temperature=judge_temperature,
            ),
        ],
        epochs=epochs,
        config=GenerateConfig(
            max_tokens=max_tokens,
            # Display, not depth. See the docstring above and the repository
            # rule that every model call in a study records its reasoning.
            reasoning_summary="auto",
        ),
        metadata={
            "traits": selected,
            "questions_per_trait": questions_per_trait,
            # The two digests that say which instrument was administered. The
            # judge model is not here because it is resolved per call and
            # recorded on every score, where it cannot drift from the calls it
            # describes.
            "questions_sha256": QUESTIONS_SHA256,
            "rubric_sha256": RUBRIC_SHA256,
            "judge_samples": judge_samples,
            # Named "requested" because that is all a task argument can be: what
            # reached the provider is per call and lives on the score, where a
            # provider that dropped it says so.
            "judge_temperature_requested": judge_temperature,
        },
    )
