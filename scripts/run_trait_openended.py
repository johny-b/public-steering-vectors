"""Run the open-ended trait-expression eval.

Edit the constants below, then `python scripts/run_trait_openended.py`. Needs an
API key for the model under test, and -- unless JUDGE is False -- one for the
judge, which defaults to `anthropic/claude-sonnet-5`.

JUDGE is the two-pass switch. With it True the replies are judged as they
arrive, which is what you want for a single run. With it False the eval
generates and saves and nothing is scored; `scripts/score_trait_openended.py`
then judges the saved logs into a separate directory, later, on another machine,
with a different judge if you like. Set it False when the generations are the
expensive half -- a steered vLLM server holding a GPU, a sweep queued behind it
-- because a judge outage or an expired key must not waste GPU hours that have
already been spent, and because re-judging then costs a judge call rather than a
generation.

EPOCHS is 5 by default, which is the task's own default and what makes each
question a small distribution rather than a single draw. A full run is then
1,000 generations and 1,000 judge calls per condition: every epoch is generated
and judged independently, because the judge cache is keyed by sample and epoch
so that two identical replies still buy two verdicts.

Before believing any score, read the diagnostics printed beside it.
`refusals` is the important one: it is excluded from the score, so a condition
that refuses more is a condition whose mean is a mean of fewer, different
replies. `unscored` is the total exclusion rate; `judge_errors` and
`judge_unparseable` above zero mean part of the run has no judge at all;
`truncated_generations` and `empty_completions` mean MAX_TOKENS is too small for
the condition rather than that the model had nothing to say.

JUDGE_TEMPERATURE is a request, not a setting. Inspect's Anthropic provider
drops sampling parameters for every Claude 4.7+ model, so the default judge runs
at its own default however this is set; each score records the requested value
and the applied one (null when nothing was sent) separately. Judge noise is
therefore something to measure rather than something to assume away.

Run the judge gate before a first paid batch with a new judge model, and the
stability check before reading small differences between conditions:

    python scripts/validate_trait_openended.py --check gate
    python scripts/validate_trait_openended.py --check judge-stability \
        --from-log logs/<one saved log>.eval
"""

from __future__ import annotations

from inspect_ai import eval
from trait_openended import trait_openended

MODEL = "anthropic/claude-sonnet-5"

TRAITS = None  # None = all ten, or e.g. ["psychopathy", "no-shut-down"]
QUESTIONS_PER_TRAIT = 20  # 20 is the whole bank; fewer is a smoke test
MAX_TOKENS = 16384  # room to think and still answer; see DEFAULT_MAX_TOKENS
EPOCHS = 5  # replies per question; the eval samples, so each epoch is a new
# reply and a new verdict. 5 is the task default: 1,000 generations and 1,000
# judge calls for a full run. Drop it to 1 only for a smoke test.

JUDGE = True  # False = generate only, and judge later with score_trait_openended.py
GRADER_MODEL = None  # None = $INSPECT_GRADER_MODEL, then claude-sonnet-5
JUDGE_SAMPLES = 1  # >1 reduces by median score and refusal vote
JUDGE_TEMPERATURE = 0.0  # requested, not necessarily applied; see below

LOG_DIR = "logs"
MAX_CONNECTIONS = 10  # concurrent model requests


def main() -> None:
    task = trait_openended(
        traits=TRAITS,
        questions_per_trait=QUESTIONS_PER_TRAIT,
        grader_model=GRADER_MODEL,
        judge_samples=JUDGE_SAMPLES,
        judge_temperature=JUDGE_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        epochs=EPOCHS,
    )

    logs = eval(
        task,
        model=MODEL,
        # Epochs are set on the task and repeated here because `eval` wins over
        # the task's value when both are given, and a log whose task_args say 4
        # while the run did 1 is worse than either number.
        epochs=EPOCHS,
        log_dir=LOG_DIR,
        max_connections=MAX_CONNECTIONS,
        # The scorers are where the judge lives, so this is the whole
        # generation-only mode: no judge call, no judge key, nothing scored.
        score=JUDGE,
        # One sample that errors is a missing data point; an aborted eval is a
        # lost run of generations.
        fail_on_error=False,
    )

    for log in logs:
        print(f"\n{log.eval.task} ({MODEL}): {log.status}")
        if log.status == "error" and log.error:
            print(log.error.message)
        for score in log.results.scores if log.results else []:
            metrics = ", ".join(
                f"{name}={metric.value}" for name, metric in score.metrics.items()
            )
            print(f"  {score.scorer}/{score.name}: {metrics}")
            if score.unscored_samples:
                total = score.scored_samples + score.unscored_samples
                print(
                    f"    ({score.unscored_samples} of {total} samples could not "
                    "be scored on this key)"
                )
        if not JUDGE:
            print("  generation only; judge the saved log with "
                  "scripts/score_trait_openended.py")
        print(f"  log: {log.location}")


if __name__ == "__main__":
    main()
