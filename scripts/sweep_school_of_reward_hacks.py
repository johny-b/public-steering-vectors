"""Sweep the School of Reward Hacks eval over steering strengths on one vector.

Runs the eval against the steered vLLM server (see `pod/README.md`) for every
strength in STRENGTHS and writes one eval log per strength into LOG_DIR.

    inspect_providers/steered_provider/ops/tunnel.sh   # or a local server
    python scripts/sweep_school_of_reward_hacks.py

Constants below are the whole configuration. Re-running the script resumes: a
strength that already has a successful log in LOG_DIR under the same condition
is skipped rather than re-run, so a sweep interrupted at strength 5 of 9
continues from there.

Strength 0.0 is a steering condition like any other — the vector is named and
applied at zero — not the bare unsteered model. It is the sweep's control, and
it has to travel the same code path (chat template, `vllm_xargs`, the patched
block) as its neighbours for the comparison to mean anything.

There are no scenarios here, so one strength is one task rather than three: the
dataset carries 306 prompts and that, not EPOCHS, is the sample size. The eval's
own measure is already paired within a sample (one answer, two judges), so the
sweep's comparison across strengths is the only unpaired one left — which is
what makes the per-strength n worth keeping at the full 306.

Sampling is blocked. One run of this script is one block: an epoch count written
into its own LOG_DIR, `logs/srh-sweep/block<n>-e<epochs>`. Run the whole
9-strength curve at EPOCHS=1 first to get the shape of the result, then raise
EPOCHS, point LOG_DIR at the next block and run it again to deepen every cell.
Nothing is wasted, because `analyze_school_of_reward_hacks.py` is pointed at the
PARENT `logs/srh-sweep` and pools the blocks per (vector, strength) — so two
blocks of one condition are one row at the combined n, not two rows and not a
deduplicated one. Only LOG_DIR and EPOCHS change between blocks.
"""

from __future__ import annotations

from typing import Any

from inspect_ai import Task, eval
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, Model, get_model
from school_of_reward_hacks import school_of_reward_hacks

from steering_vectors import vectorfmt

STEER_VECTOR = "0007"
STRENGTHS = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

MODEL = "steered/Qwen/Qwen3.6-27B"
"""`steered/<served-model-name>`; STEERED_BASE_URL selects the server."""

# The eval's condition. These also form half of the resume key, so a log written
# under a different condition is never mistaken for one of this sweep's.
CATEGORIES = None
"""Which reward-hacking categories to run. None is all five, i.e. all 306 rows.

Read as a resume key it is also the reason to leave it alone mid-sweep: a block
run over one category is not the same cell as a block run over all of them, and
`completed()` will (correctly) decline to treat one as the other."""

JUDGE_MODEL = "anthropic/claude-sonnet-5"

EPOCHS = 1
"""Answers per prompt. The dataset is 306 samples, so EPOCHS=1 already gives a
sample size of 306 per strength; raising it takes several answers to each prompt
instead of adding prompts. Blocked sampling — a block is one (LOG_DIR, EPOCHS)
pair and the curve is the union of the blocks — is driven by overriding this and
LOG_DIR from a per-block driver script, not by editing them here."""

MAX_TOKENS = 32768
"""Per response. The prompts are short (~400 characters, ~100 tokens), so the
server's `MAX_MODEL_LEN` of 40960 leaves this budget binding first and a
`stop_reason='model_length'` would be a real surprise -- but the margin is
40960 - 32768 = 8192 tokens, not the comfortable one a 65536-token context
would give, so a long prompt is the thing that would flip which limit bites.
`stop_reason='max_tokens'` is NOT a surprise and is observed: it is deliberately
the same budget as the other sweeps, and a thinking model that spends its whole
budget thinking returns an empty visible answer, which is a result about the
steering condition, not something to be tuned away per eval."""

# Qwen's published thinking-mode sampling for Qwen3.6-27B. Deliberately no seed:
# a fixed seed would return the same generation for a prompt in every epoch, and
# under EPOCHS > 1 the extra epochs would add no information while making the
# stderr look as though they had.
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20

LOG_DIR = "logs/school-of-reward-hacks-sweep"
"""Where this run's logs go. A blocked run overrides it per block, to a
directory of its own under a shared parent (`logs/srh-sweep/block01-e1`).

One directory per block matters twice over. Resume (`completed()`) reads only
this directory, which is what lets a later block at a different EPOCHS add
samples to conditions an earlier block already covered instead of skipping them
as done. And the analysis script, pointed at the PARENT, pools every block
beneath it: `list_eval_logs` recurses by default, and the aggregate groups on
(vector, strength) with no deduplication, so two blocks of one condition become
one row whose n is the sum. The flip side of no deduplication is that a log
appearing under the parent twice is counted twice — pool by adding block
directories, never by copying logs between them."""

SERVER_MAX_NUM_SEQS = 64
"""The `--max-num-seqs` the running server was actually started with."""

MAX_CONNECTIONS = 32
# Half of SERVER_MAX_NUM_SEQS rather than all of it, on purpose: the sweep is
# not the only thing that may touch this server, and leaving it headroom costs
# little here because the wall clock is dominated by 32k-token thinking replies
# rather than by how many of them are in flight.
#
# One task per strength, so unlike the agentic misalignment sweep there is no
# max_tasks fan-out to reason about: requests in flight against the server are
# capped at MAX_CONNECTIONS full stop. The judges are a different provider with
# a connection pool of their own, so judge concurrency is not counted here.
assert MAX_CONNECTIONS <= SERVER_MAX_NUM_SEQS

SCORE = True
"""Whether `eval()` scores the samples it generates, or only generates them.

`False` writes logs with full samples and no scores, leaving them to be graded
later by `inspect score` over the log directory. The point is that scoring is
the only part of this sweep that calls an external judge API, so a block whose
judge is unavailable -- rate-limited, quota-blocked, down -- can still generate
its samples now and be scored when the judge comes back, instead of burning the
GPU hours twice. It matters more here than in the agentic misalignment sweep,
because scoring is two judge calls per sample over 306 samples per strength
rather than one call over a handful.

Generation and scoring are independent: the scorer reads the finished
transcript's visible completion and nothing about the generation depends on it.

Overridden per block from the driver script, like LOG_DIR and EPOCHS. The
default is True, so every block that does not touch it scores inline.
"""

VECTOR_ARG = "steer_vector"
STRENGTH_ARG = "steer_strength"

ConditionKey = tuple[str | None, float | None]
"""(vector, strength): one cell of the sweep, comparable across runs."""


def steer_args(strength: float) -> dict[str, Any]:
    """The steering condition, as model args.

    Model args are recorded in the eval log (`log.eval.model_args`), so passing
    the condition this way rather than through a base URL or an environment
    variable is what makes each log say which condition produced it.
    """
    return {VECTOR_ARG: STEER_VECTOR, STRENGTH_ARG: strength}


def task_args() -> dict[str, Any]:
    """The eval condition shared by every task in the sweep."""
    return {
        "categories": CATEGORIES,
        "judge_model": JUDGE_MODEL,
    }


def build_model(strength: float) -> Model:
    """The model under test at one strength."""
    return get_model(
        MODEL,
        config=GenerateConfig(
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            # top_k is NOT a chat-completions parameter, and inspect's
            # OpenAI-compatible provider builds its request body from the
            # OpenAI parameter set: GenerateConfig(top_k=...) is accepted and
            # then silently dropped, never reaching vLLM. extra_body is the
            # route that works — it is passed through verbatim and lands at the
            # top level of the request body, which is where vLLM reads its
            # sampling extensions. Confirmed on the wire, not assumed.
            extra_body={"top_k": TOP_K},
            # Explicit, so inspect uses its static connection pool rather than
            # adaptive concurrency, which would probe past SERVER_MAX_NUM_SEQS.
            max_connections=MAX_CONNECTIONS,
        ),
        **steer_args(strength),
    )


def build_task() -> Task:
    """The eval under the sweep's condition.

    The sampling settings above are the model's, not the task's, and the judge
    model is built inside the scorer with a config of its own — so none of
    TEMPERATURE/TOP_P/TOP_K reaches the judges, which is required: the judge
    model rejects them.
    """
    return school_of_reward_hacks(**task_args())


def _condition_key(vector: Any, strength: Any) -> ConditionKey:
    """Normalise a condition so a log's copy of it compares equal to ours."""
    return (
        None if vector is None else vectorfmt.vector_id(vector),
        None if strength is None else round(float(strength), 6),
    )


def wanted_key(strength: float) -> ConditionKey:
    """The resume key for one cell of the sweep."""
    args = steer_args(strength)
    return _condition_key(args.get(VECTOR_ARG), args.get(STRENGTH_ARG))


def completed(log_dir: str) -> dict[ConditionKey, str]:
    """Cells of this sweep already covered by a successful log in `log_dir`.

    Headers only, so this stays cheap as the directory fills up. A log counts
    only if it succeeded, ran the sweep's own eval condition, and ran the
    sweep's epoch count — a log of 1 epoch is not the 5-epoch cell that was
    asked for, and silently accepting it would quietly shrink that cell.

    `log_dir` is this block's directory, not the sweep parent, which is what
    keeps resume a within-block idea: an earlier block's log of the same cell is
    not visible here and so does not suppress this block's copy of it. That is
    the point of blocking — the later block adds samples rather than skipping
    work — and it is also why the two blocks must live in separate directories.
    """
    found: dict[ConditionKey, str] = {}
    for info in list_eval_logs(log_dir):
        header = read_eval_log(info.name, header_only=True)
        if header.status != "success":
            continue
        args = header.eval.task_args or {}
        if any(args.get(name) != value for name, value in task_args().items()):
            continue
        if header.eval.config.epochs != EPOCHS:
            continue
        model_args = header.eval.model_args or {}
        key = _condition_key(
            model_args.get(VECTOR_ARG),
            model_args.get(STRENGTH_ARG),
        )
        found[key] = info.name
    return found


def score_summary(log: EvalLog) -> str:
    """The three paired measures as mean ± stderr, plus the unjudged rate.

    The scorer reports one `EvalScore` per key of its score dict, each named for
    the key, so this reads them by name rather than by position.
    """
    parts = []
    for score in log.results.scores if log.results else []:
        mean = score.metrics.get("mean")
        stderr = score.metrics.get("stderr")
        if mean is None:
            continue
        value = f"{score.name}={mean.value:.1f}"
        if stderr is not None:
            value += f"±{stderr.value:.1f}"
        parts.append(value)
    return " ".join(parts) if parts else "no scores"


def report(strength: float, log: EvalLog) -> None:
    """One watchable line per strength."""
    samples = log.results.completed_samples if log.results else 0
    print(
        f"[strength={strength:+.2f}] {log.status:<8} n={samples} {score_summary(log)}",
        flush=True,
    )
    if log.status == "error" and log.error:
        print(f"    error: {log.error.message}", flush=True)
    print(f"    log: {log.location}", flush=True)


def main() -> None:
    done = completed(LOG_DIR)
    print(
        f"vector {STEER_VECTOR}: {len(STRENGTHS)} strengths x {EPOCHS} epochs "
        f"into {LOG_DIR}",
        flush=True,
    )

    for strength in STRENGTHS:
        existing = done.get(wanted_key(strength))
        if existing is not None:
            print(
                f"[strength={strength:+.2f}] skipped: already succeeded in {existing}",
                flush=True,
            )
            continue

        logs = eval(
            build_task(),
            model=build_model(strength),
            epochs=EPOCHS,
            log_dir=LOG_DIR,
            # Deterministic per-task sample concurrency. The server-side cap is
            # the connection pool above; this only decides how many samples may
            # be mid-flight (generating, queued for the model, or being judged).
            max_samples=MAX_CONNECTIONS,
            # False generates without judging; the logs are scored by a later
            # pass. See SCORE.
            score=SCORE,
        )

        for log in logs:
            report(strength, log)
            if log.status == "success":
                done[wanted_key(strength)] = log.location

    print(f"\ndone: {len(done)} successful cells in {LOG_DIR}", flush=True)


if __name__ == "__main__":
    main()
