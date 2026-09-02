"""Run the HMMT February 2026 eval.

Edit the constants below, then `python scripts/run_hmmt.py`. Requires an API
key for the model under test (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY, read from
the environment or a .env file), and network access to Hugging Face for the
dataset on the first run.

Alongside Inspect's own metrics this prints a 95% interval from a bootstrap
over *problems*. Read that one. With 33 problems and 4 epochs the 132
generations are not 132 independent observations -- four attempts at the same
problem succeed and fail together -- and an interval that ignores the
clustering is roughly 1.7x too narrow on this dataset.
"""

import math
import random

from hmmt import hmmt_feb_2026, reference_generate_config
from inspect_ai import eval
from inspect_ai.log import EvalLog

MODEL = "anthropic/claude-sonnet-5"
EPOCHS = 4  # attempts per problem; MathArena reports avg@4
MAX_TOKENS = 81920  # output budget per problem; see hmmt.DEFAULT_MAX_TOKENS
PROMPT_SUFFIX = None  # None = hmmt.REASONING_PROMPT_SUFFIX
REVISION = None  # None = the pinned dataset commit
JUDGE_MODEL = None  # e.g. "anthropic/claude-sonnet-5" to add the equivalence judge

REFERENCE_CONFIG = reference_generate_config()
"""Temperature 1.0, top_p 0.95, top_k 20 — the policy the reference runs used."""

GENERATE_CONFIG = None
"""Sampling policy for the model under test.

`None` accepts the provider's defaults, which is right for a first look at an
unfamiliar model and wrong for anything compared against the reference numbers.
Set it to `REFERENCE_CONFIG` to run under the policy those numbers were
measured with. Four epochs against a provider that decodes greedily buy four
identical generations, so read `hmmt.DEFAULT_EPOCHS` before leaving this
`None`.
"""

LOG_DIR = "logs"
MAX_CONNECTIONS = 10  # concurrent model requests

CLIENT_TIMEOUT = 7200.0
"""HTTP read timeout in seconds, as a model argument. Raise it, do not remove it.

This eval gives each problem an 81,920-token output budget, and a reasoning
model on a self-hosted server can spend well over ten minutes on one of them.
The OpenAI-compatible providers -- including this repository's `steered/`
provider, which subclasses `OpenAICompatibleAPI` -- build their HTTP client from
the OpenAI SDK's defaults when no `client_timeout` is given, and those defaults
are `Timeout(connect=5.0, read=600, write=600, pool=600)`
(`_providers/openai_compatible.py`, `_create_http_client` and `_create_client`).
So a generation that runs past 600 seconds raises `APITimeoutError`, the retry
starts the same generation from scratch, and it times out again: the run never
finishes and never fails, it just burns GPU hours in a loop.

`GenerateConfig(timeout=...)` does not fix this. That value bounds the retry
loop through tenacity's `stop_after_delay`; it never reaches the HTTP client. The
model argument is the only thing that moves the read timeout, either as
`get_model(..., client_timeout=7200.0)` or as `-M client_timeout=7200` on the
command line. Verified on this repository's HMMT runs.
"""

MODEL_ARGS: dict[str, object] = {"client_timeout": CLIENT_TIMEOUT}
"""Model arguments for the model under test. See `CLIENT_TIMEOUT`.

Dropped for `anthropic/*` by `model_args_for`, because Inspect 0.3.259's
Anthropic provider does not take a `client_timeout` argument and would fail
with a `TypeError` on an unexpected keyword. Anthropic's own SDK does not carry
the 600-second read timeout this constant exists to defeat.
"""


def model_args_for(model: str) -> dict[str, object]:
    """`MODEL_ARGS`, minus the arguments this model's provider cannot take."""
    if model.startswith("anthropic/"):
        return {k: v for k, v in MODEL_ARGS.items() if k != "client_timeout"}
    return dict(MODEL_ARGS)

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 12345
"""Fixed, so re-reporting the same log prints the same interval."""

CI_KEYS = ("correct", "lenient")
"""Score keys worth an interval. The rest of the table is diagnostics."""


def item_bootstrap_ci(
    per_item: list[float], resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float] | None:
    """A seeded 95% percentile bootstrap of the mean, resampling problems.

    Args:
        per_item: One value per problem -- that problem's success rate across
            epochs, already reduced.
        resamples: Bootstrap draws.

    Returns:
        The 2.5th and 97.5th percentiles, or None with fewer than two problems.

    The percentile indices and the draw order are taken from `jevals/runner.py`
    in the internal `jevals` reference harness (a private steering-tools
    workspace; its records are available to maintainers, not shipped), so an
    interval printed here can be compared with one in its `metrics.json`
    without wondering whether the two are the same statistic. `--check regrade`
    does not test this, but it has been checked by hand: the unsteered HMMT
    reference run's `ci95_item_bootstrap` reproduces exactly.
    """
    if len(per_item) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(per_item)
    means = sorted(
        sum(per_item[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return means[int(0.025 * resamples) - 1], means[int(0.975 * resamples) - 1]


def per_problem_values(log: EvalLog, scorer: str, key: str) -> list[float]:
    """The reduced per-problem value of one score key.

    Reads `log.reductions`, which is where the epoch reducer's output lands --
    one entry per (scorer, sample) rather than one per generation. Samples the
    reducer left unscored (NaN) are dropped, which is what makes the
    untruncated accuracy's interval cover the problems it actually scored.
    """
    values: list[float] = []
    for reduction in log.reductions or []:
        if reduction.scorer != scorer:
            continue
        for sample in reduction.samples:
            value = sample.value
            if isinstance(value, dict):
                value = value.get(key)
            if isinstance(value, (int, float)) and not math.isnan(value):
                values.append(float(value))
    return values


def report(log: EvalLog) -> None:
    """Print the metrics table, with a problem-level interval where one helps.

    Inspect reports a dict-valued score as one `EvalScore` per key -- named for
    the key, carrying mean and stderr -- so the rows below are `scorer/key`. A
    metric that returned a dict of its own (`grading_method_share`) instead
    puts its keys on the metric names, and lands on one row.
    """
    print(f"\n{log.eval.task} ({log.eval.model}): {log.status}")
    if log.status == "error" and log.error:
        print(log.error.message)
    for score in log.results.scores if log.results else []:
        if not score.metrics:
            continue
        scorer = score.scorer or score.name
        label = scorer if score.name == scorer else f"{scorer}/{score.name}"
        metrics = ", ".join(
            f"{name}={metric.value:.4g}" for name, metric in score.metrics.items()
        )
        print(f"  {label}: {metrics}")
        if score.name not in CI_KEYS or score.scorer is None:
            continue
        values = per_problem_values(log, score.scorer, score.name)
        interval = item_bootstrap_ci(values)
        if interval is None:
            continue
        mean = sum(values) / len(values)
        print(
            f"    {mean:.4f} over {len(values)} problems, "
            f"95% item bootstrap [{interval[0]:.4f}, {interval[1]:.4f}]"
        )
    print(f"  log: {log.location}")


def main() -> None:
    kwargs = {
        "epochs": EPOCHS,
        "max_tokens": MAX_TOKENS,
        "judge_model": JUDGE_MODEL,
        "generate_config": GENERATE_CONFIG,
    }
    if PROMPT_SUFFIX is not None:
        kwargs["prompt_suffix"] = PROMPT_SUFFIX
    if REVISION is not None:
        kwargs["revision"] = REVISION

    logs = eval(
        hmmt_feb_2026(**kwargs),
        model=MODEL,
        model_args=model_args_for(MODEL),
        log_dir=LOG_DIR,
        max_connections=MAX_CONNECTIONS,
    )
    for log in logs:
        report(log)


if __name__ == "__main__":
    main()
