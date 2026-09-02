"""The HMMT scores: two accuracies that disagree, and the numbers that explain why.

`hmmt_scorer` is the headline. It scores a truncated or unparseable reply as
incorrect, which is how the published numbers treat a missing answer, and
reports beside it the share of verdicts each grading method decided, a lenient
regrade, and how often a `\\boxed{}` was found at all.

`hmmt_untruncated_scorer` is the same accuracy computed only over replies that
finished on their own. It is biased by construction -- truncation correlates
with the hard problems and with any condition that makes the model think longer
-- so it is reported *beside* the headline, never instead of it. Neither alone
is honest: the first blames the model for the token budget, the second hides it.

Read the second one carefully, because there are two different statistics with
a claim to the name and they differ by four points. This eval reports the mean
over *problems*: each problem is reduced over whichever of its epochs survived,
then the problems are averaged, so a problem with one surviving attempt weighs
the same as a problem with four. On the reference unsteered records that is
0.876, against the headline's 0.841. The `accuracy_excl_truncated` of 0.917
reported by the internal `jevals` reference harness (a private steering-tools
workspace; its records are available to maintainers, not shipped) is the mean
over *records* -- 111 correct of 121 untruncated generations -- which weights a
problem by how often it finished. The harness's own published interval for that
number, `ci95_excl_truncated`, is a bootstrap over problems, so it is this
eval's estimator that interval brackets and the harness's point estimate that
sits outside it. `--check regrade` prints both statistics from the same records
so the gap is never a surprise on a first run.

`hmmt_diagnostics` is what tells the two apart, and every one of its keys is
present on every sample so the key set never depends on what happened.
"""

from __future__ import annotations

from inspect_ai.model import ChatMessageAssistant
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    mean,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from .grader import GRADING_METHODS, gold_in_content, grade_math

TRUNCATION_STOP_REASONS = ("max_tokens", "model_length")
"""Stop reasons that mean the reply was cut off rather than finished.

`max_tokens` is the budget this eval set; `model_length` is the context window.
Both are counted separately in the diagnostics, because they call for different
fixes and only one of them is ours to make.
"""


def visible_completion(state: TaskState) -> str:
    """The model's final channel.

    `ModelOutput.completion` is `ChatMessageAssistant.text`, which concatenates
    the `ContentText` parts and drops `ContentReasoning` -- verified in
    inspect_ai 0.3.259. So whenever the server (or Inspect's own `<think>`
    parsing) has separated the reasoning, it is already gone. `strip_thinking`
    in the grader handles what is left: an unmatched tag Inspect could not
    lift out.
    """
    return state.output.completion


def reasoning_chars(state: TaskState) -> int:
    """Characters of separated reasoning attached to the reply, if any."""
    message = state.output.message
    if not isinstance(message, ChatMessageAssistant):
        return 0
    if isinstance(message.content, str):
        return 0
    return sum(
        len(block.reasoning) for block in message.content if block.type == "reasoning"
    )


def is_truncated(state: TaskState) -> bool:
    """Whether the generation stopped because it ran out of room."""
    return state.output.stop_reason in TRUNCATION_STOP_REASONS


@metric(scores="unreduced")
def grading_method_share() -> Metric:
    """Share of verdicts decided by each rung of the grading cascade.

    Declared unreduced because the method is a label carried in score
    metadata, and the epoch reducer averages score *values* -- by the time a
    reduced metric ran, there would be nothing left to count. Reading the
    shares matters: an accuracy that rose while the `math-verify` share rose
    with it is a grader getting looser, not a model getting better, and the
    `none` share is the parse-failure rate in the same units.
    """

    def metric_fn(scores: list[SampleScore]) -> Value:
        counts = dict.fromkeys(GRADING_METHODS, 0)
        for sample_score in scores:
            method = (sample_score.score.metadata or {}).get("grading_method")
            counts[method if method in counts else "none"] += 1
        total = sum(counts.values())
        if total == 0:
            return dict.fromkeys(GRADING_METHODS, 0.0)
        return {name: count / total for name, count in counts.items()}

    return metric_fn


@scorer(metrics=[grading_method_share(), {"*": [mean(), stderr()]}])
def hmmt_scorer() -> Scorer:
    """Strict `\\boxed{}` accuracy, with the views that qualify it.

    Reports:

    * `correct` -- the headline. A truncated or boxless reply scores 0.
    * `lenient` -- `correct`, or the exactly-correct gold written somewhere in
      the final channel without a box around it. Always at or above `correct`;
      a wide gap is a formatting story, not a capability one.
    * `boxed_found` -- whether an answer was extracted at all. `1 - boxed_found`
      is the parse-failure rate, and it is the first thing to read when
      `correct` drops.

    plus the `grading_method_share()` breakdown.

    One artefact of that combination is worth naming so nobody files it as a
    bug. Mixing an unreduced `Metric` with a `{"*": ...}` dict in one `metrics`
    list makes Inspect run two passes: the reduced pass finds no plain metric
    to bind and still emits a row, so `log.results.scores` contains an extra
    `EvalScore` with `name="hmmt_scorer"` and no metrics at all, beside the real
    one that carries the cascade shares. `(scorer, name)` is therefore not
    unique for this scorer; `(scorer, name, reducer)` is. `run_hmmt.report`
    skips metric-less rows. The alternative spec -- hanging the shares off a
    value key -- trades this empty row for two populated rows both named
    `correct`, and renames the shares to `grading_method_share_exact` and so
    on, which is worse on both counts.
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = visible_completion(state)
        gold = target.text
        result = grade_math(completion, gold)
        lenient = result.correct or gold_in_content(completion, gold)
        method = result.method or "none"
        return Score(
            value={
                "correct": float(result.correct),
                "lenient": float(lenient),
                "boxed_found": float(result.extracted is not None),
            },
            answer=result.extracted,
            explanation=(
                f"Extracted {result.extracted!r} against gold {gold!r}: "
                f"{'correct' if result.correct else 'incorrect'} by {method}."
                if result.extracted is not None
                else f"No \\boxed{{}} in the final channel; gold was {gold!r}."
            ),
            metadata={
                "grading_method": method,
                "gold": gold,
                "extracted": result.extracted,
                "lenient": bool(lenient),
                "verifier_failed": result.verifier_failed,
                "truncated": is_truncated(state),
                "stop_reason": state.output.stop_reason,
            },
        )

    return score


@scorer(metrics={"correct": [mean(), stderr()]})
def hmmt_untruncated_scorer() -> Scorer:
    """The same accuracy, over replies that finished on their own.

    A truncated sample is left unscored (`Score.unscored()`), so it drops out
    of the denominator rather than counting as a failure. Inspect reports how
    many samples that was as `unscored_samples`; read it, because this number
    is only meaningful next to it and next to `hmmt_scorer`'s `correct`.

    This is a mean over problems, not over generations. The epoch reducer runs
    first, so each problem is reduced over whichever of its four attempts
    finished and every problem then counts once. That is deliberate -- it keeps
    the unit of observation the same as the headline's, and it is the estimator
    the reference harness's own published interval for this quantity is built
    on -- but it is not the estimator behind its 0.917. See this module's
    docstring; the difference is about four points on the reference data, and
    it is a difference of definition rather than of grading.
    """

    async def score(state: TaskState, target: Target) -> Score:
        if is_truncated(state):
            return Score.unscored(
                explanation=(
                    f"Generation stopped on {state.output.stop_reason}; "
                    "excluded from the untruncated accuracy."
                ),
                metadata={
                    "gold": target.text,
                    "truncated": True,
                    "stop_reason": state.output.stop_reason,
                },
            )
        result = grade_math(visible_completion(state), target.text)
        return Score(
            value={"correct": float(result.correct)},
            answer=result.extracted,
            explanation=(
                f"Extracted {result.extracted!r} against gold {target.text!r}: "
                f"{'correct' if result.correct else 'incorrect'}."
            ),
            metadata={
                "grading_method": result.method or "none",
                # Carried here as well as on `hmmt_scorer`, so one of this
                # scorer's verdicts can be reproduced from its own row without
                # joining to another scorer's.
                "gold": target.text,
                "extracted": result.extracted,
                "verifier_failed": result.verifier_failed,
                "truncated": False,
                "stop_reason": state.output.stop_reason,
            },
        )

    return score


@scorer(metrics={"*": [mean(), stderr()]})
def hmmt_diagnostics() -> Scorer:
    """How much of the accuracy is about mathematics, and how much about budget.

    Every key is present on every sample -- seeded from the outcome rather than
    incremented conditionally -- because Inspect's wildcard metric spec needs
    the key set to be the same for each one.

    * `truncated_generations` -- replies cut off, either way.
    * `max_tokens_stops` / `model_length_stops` -- which cut them off. The
      first is this eval's `max_tokens`; the second is the served context
      window, which no task argument can raise.
    * `parse_failures` -- no `\\boxed{}` in the final channel. Overlaps
      `truncated_generations` heavily and deliberately is not subtracted from
      it: a reply can also simply fail to box its answer.
    * `empty_completions` -- nothing in the final channel at all. On a
      reasoning model this is usually the whole budget spent thinking.
    * `completion_tokens` -- the mean is what says whether `max_tokens` is
      binding. NaN, not 0, when the provider reported no usage: 0 is a
      plausible-looking number that would read as "the budget is nowhere near
      binding", which is the opposite of "nobody measured". Inspect drops a NaN
      from this key's denominator and counts it under `unscored_samples`.
    * `reasoning_present` -- whether any reasoning came back. A run where this
      collapses to 0 is a run whose thinking mode got turned off somewhere,
      which looks exactly like a capability drop.
    * `verifier_failed` -- symbolic comparisons `math-verify` could not make,
      because it raised or overran `grader.VERIFIER_TIMEOUT_SECONDS`. 17 of the
      33 answers are decided symbolically, so anything above 0 here means the
      accuracy is measuring the environment. It is deliberately not folded into
      `parse_failures`: an answer was found, it just could not be checked.

    The repo-wide diagnostics naming contract also lists `unscored`. This eval
    has no separate unscored key because unscoredness is not a property of the
    sample here but of one view of it: `hmmt_untruncated_scorer` is the scorer
    that declines to score, and Inspect reports that as its `unscored_samples`.
    Adding a key that was always 0 would say less than the count already does.
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = visible_completion(state)
        stop_reason = state.output.stop_reason
        usage = state.output.usage
        reasoning = reasoning_chars(state)
        inline_thinking = "<think>" in completion or "</think>" in completion
        result = grade_math(completion, target.text)
        return Score(
            value={
                "truncated_generations": float(is_truncated(state)),
                "max_tokens_stops": float(stop_reason == "max_tokens"),
                "model_length_stops": float(stop_reason == "model_length"),
                "parse_failures": float(result.extracted is None),
                "empty_completions": float(not completion.strip()),
                "completion_tokens": (
                    float(usage.output_tokens) if usage else float("nan")
                ),
                "reasoning_present": float(reasoning > 0 or inline_thinking),
                "verifier_failed": float(result.verifier_failed),
            },
            explanation=(
                f"stop_reason={stop_reason}, "
                f"completion_chars={len(completion)}, "
                f"reasoning_chars={reasoning}, "
                f"boxed={'yes' if result.extracted is not None else 'no'}, "
                f"verifier_failed={result.verifier_failed}."
            ),
            metadata={
                "stop_reason": stop_reason,
                "completion_chars": len(completion),
                "reasoning_chars": reasoning,
                "inline_thinking_tags": inline_thinking,
                "usage_reported": usage is not None,
            },
        )

    return score
