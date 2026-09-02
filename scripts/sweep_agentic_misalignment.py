"""Sweep the agentic misalignment eval over steering strengths on one vector.

Runs the eval against the steered vLLM server (see `pod/README.md`) for every
strength in STRENGTHS, crossed with the three scenarios, and writes one eval log
per (strength, scenario) into LOG_DIR.

    inspect_providers/steered_provider/ops/tunnel.sh   # or a local server
    python scripts/sweep_agentic_misalignment.py

Constants below are the whole configuration. Re-running the script resumes: a
(strength, scenario) that already has a successful log in LOG_DIR under the same
condition is skipped rather than re-run, so a sweep interrupted at strength 5 of
9 continues from there.

Strength 0.0 is a steering condition like any other — the vector is named and
applied at zero — not the bare unsteered model. It is the sweep's control, and
it has to travel the same code path (chat template, `vllm_xargs`, the patched
block) as its neighbours for the comparison to mean anything.

The dataset is one sample per scenario, so EPOCHS is the entire sample size and
every number the sweep produces is a mean over epochs.

Sampling is blocked. One run of this script is one block: an epoch count written
into its own LOG_DIR, `logs/am-sweep/block<n>-e<epochs>`. Run the whole 9-strength
curve at a small epoch count first to get the shape of the result, then raise
EPOCHS, point LOG_DIR at the next block and run it again to deepen every cell.
Nothing is wasted, because `analyze_agentic_misalignment.py` is pointed at the
PARENT `logs/am-sweep` and pools the blocks per (scenario, vector, strength) —
so two blocks of one condition are one row at the combined n, not two rows and
not a deduplicated one. Only LOG_DIR and EPOCHS change between blocks.
"""

from __future__ import annotations

from typing import Any

from agentic_misalignment import agentic_misalignment
from inspect_ai import Task, eval
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, Model, get_model

from steering_vectors import vectorfmt

STEER_VECTOR = "0007"
STRENGTHS = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
SCENARIOS = ["blackmail", "leaking", "murder"]

MODEL = "steered/Qwen/Qwen3.6-27B"
"""`steered/<served-model-name>`; STEERED_BASE_URL selects the server."""

# The eval's default condition. These also form half of the resume key, so a log
# written under a different condition is never mistaken for one of this sweep's.
GOAL_TYPE = "explicit"
GOAL_VALUE = "america"
URGENCY_TYPE = "replacement"
EXTRA_SYSTEM_INSTRUCTIONS = None
PROD = False
TEST_EVAL_AWARENESS = False

EPOCHS = 100
"""The default block's sample size: the dataset is one sample per scenario, so
this is the whole sample size. Blocked sampling — a block is one (LOG_DIR,
EPOCHS) pair and the curve is the union of the blocks — is driven by overriding
this and LOG_DIR from a per-block driver script, not by editing them here."""

MAX_TOKENS = 32768
"""Per response. The prompts run to ~3.5k tokens, so the server needs
`MAX_MODEL_LEN` of at least ~40960 or thinking-mode replies will be cut off at
the context window instead of at this budget — which the analysis script reports
as `stop_reason='model_length'`. The running server is at 65536, so this budget
binds first and a `model_length` stop would be a real surprise."""

# Qwen's published thinking-mode sampling for Qwen3.6-27B. Deliberately no seed:
# the dataset is a single sample and the sample size is EPOCHS, so a fixed seed
# would return the same generation every epoch and the stderr would be a lie.
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20

GRADER_MODEL = "anthropic/claude-sonnet-4-6"

LOG_DIR = "logs/agentic-misalignment-sweep"
"""Where this run's logs go. A blocked run overrides it per block, to a
directory of its own under a shared parent (`logs/am-sweep/block01-e10`).

One directory per block matters twice over. Resume (`completed()`) reads only
this directory, which is what lets a later block at a different EPOCHS add
samples to conditions an earlier block already covered instead of skipping them
as done. And the analysis script, pointed at the PARENT, pools every block
beneath it: `list_eval_logs` recurses by default, and the aggregate groups on
(scenario, vector, strength) with no deduplication, so two blocks of one
condition become one row whose n is the sum. Verified, not assumed. The flip
side of no deduplication is that a log appearing under the parent twice is
counted twice — pool by adding block directories, never by copying logs between
them."""

SERVER_MAX_NUM_SEQS = 64
"""The `--max-num-seqs` the running server was actually started with."""

MAX_TASKS = 3  # the three scenarios of one strength, in one eval() call
MAX_CONNECTIONS = 32
# Half of SERVER_MAX_NUM_SEQS rather than all of it, on purpose: the sweep is
# not the only thing that may touch this server, and leaving it headroom costs
# little here because the wall clock is dominated by 32k-token thinking replies
# rather than by how many of them are in flight.
#
# Requests in flight against the server are capped at MAX_CONNECTIONS, not at
# MAX_CONNECTIONS * MAX_TASKS. inspect scopes its connection semaphore by the
# provider's connection_key(), and `steered` keys on (model, vector, strength):
# all three scenario tasks of one strength share one condition and therefore one
# pool. Verified against inspect_ai 0.3.259, whose Model._connection_concurrency
# takes the semaphore under model_concurrency_key(api).
#
# If that ever stops holding, the safe reading is the conservative one:
# MAX_CONNECTIONS * MAX_TASKS <= SERVER_MAX_NUM_SEQS, so keep both in view.
# Overshooting --max-num-seqs does not fail, it just queues inside the server
# and inflates wall clock.
assert MAX_CONNECTIONS <= SERVER_MAX_NUM_SEQS

SCORE = True
"""Whether `eval()` scores the samples it generates, or only generates them.

`False` writes logs with full samples and no scores, leaving them to be graded
later by a separate pass over the log directory. The point is that scoring is the
only part of this sweep that calls an external grader API, so a block whose
grader is unavailable -- rate-limited, quota-blocked, down -- can still generate
its samples now and be scored when the grader comes back, instead of burning the
GPU hours twice. Generation and scoring are independent here: the eval's scorer
reads the finished transcript and nothing about the generation depends on it.

Overridden per block from the driver script, like LOG_DIR and EPOCHS. The default
is True, so every block that does not touch it scores inline as before.
"""

VECTOR_ARG = "steer_vector"
STRENGTH_ARG = "steer_strength"

ConditionKey = tuple[str | None, float | None, str | None]
"""(vector, strength, scenario): one cell of the sweep, comparable across runs."""


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
        "goal_type": GOAL_TYPE,
        "goal_value": GOAL_VALUE,
        "urgency_type": URGENCY_TYPE,
        "extra_system_instructions": EXTRA_SYSTEM_INSTRUCTIONS,
        "prod": PROD,
        "test_eval_awareness": TEST_EVAL_AWARENESS,
        "grader_model": GRADER_MODEL,
    }


def build_model(strength: float) -> Model:
    """The model under test at one strength, built once for all three scenarios."""
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


def build_task(scenario: str) -> Task:
    """One scenario of the eval under the sweep's condition."""
    return agentic_misalignment(scenario=scenario, **task_args())


def _condition_key(vector: Any, strength: Any, scenario: Any) -> ConditionKey:
    """Normalise a condition so a log's copy of it compares equal to ours."""
    return (
        None if vector is None else vectorfmt.vector_id(vector),
        None if strength is None else round(float(strength), 6),
        None if scenario is None else str(scenario),
    )


def wanted_key(strength: float, scenario: str) -> ConditionKey:
    """The resume key for one cell of the sweep."""
    args = steer_args(strength)
    return _condition_key(args.get(VECTOR_ARG), args.get(STRENGTH_ARG), scenario)


def completed(log_dir: str) -> dict[ConditionKey, str]:
    """Cells of this sweep already covered by a successful log in `log_dir`.

    Headers only, so this stays cheap as the directory fills up. A log counts
    only if it succeeded, ran the sweep's own eval condition, and ran the sweep's
    epoch count — a log of 10 epochs is not the 50-epoch cell that was asked for,
    and silently accepting it would quietly shrink that cell.

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
            args.get("scenario"),
        )
        found[key] = info.name
    return found


def score_summary(log: EvalLog) -> str:
    """`harmful` and `classifier_verdict` as accuracy ± stderr."""
    parts = []
    for score in log.results.scores if log.results else []:
        accuracy = score.metrics.get("accuracy")
        stderr = score.metrics.get("stderr")
        if accuracy is None:
            continue
        value = f"{score.name}={accuracy.value:.3f}"
        if stderr is not None:
            value += f"±{stderr.value:.3f}"
        parts.append(value)
    return " ".join(parts) if parts else "no scores"


def report(strength: float, log: EvalLog) -> None:
    """One watchable line per (strength, scenario)."""
    scenario = (log.eval.task_args or {}).get("scenario", "?")
    samples = log.results.completed_samples if log.results else 0
    print(
        f"[strength={strength:+.2f}] {scenario:<9} {log.status:<8} "
        f"n={samples} {score_summary(log)}",
        flush=True,
    )
    if log.status == "error" and log.error:
        print(f"    error: {log.error.message}", flush=True)
    print(f"    log: {log.location}", flush=True)


def main() -> None:
    done = completed(LOG_DIR)
    print(
        f"vector {STEER_VECTOR}: {len(STRENGTHS)} strengths x {len(SCENARIOS)} "
        f"scenarios x {EPOCHS} epochs into {LOG_DIR}",
        flush=True,
    )

    for strength in STRENGTHS:
        todo = []
        for scenario in SCENARIOS:
            existing = done.get(wanted_key(strength, scenario))
            if existing is None:
                todo.append(scenario)
            else:
                print(
                    f"[strength={strength:+.2f}] {scenario:<9} skipped: already "
                    f"succeeded in {existing}",
                    flush=True,
                )
        if not todo:
            continue

        logs = eval(
            [build_task(scenario) for scenario in todo],
            model=build_model(strength),
            epochs=EPOCHS,
            log_dir=LOG_DIR,
            max_tasks=MAX_TASKS,
            # Deterministic per-task sample concurrency. The server-side cap is
            # the connection pool above; this only decides how many samples may
            # be mid-flight (generating, queued for the model, or grading).
            max_samples=MAX_CONNECTIONS,
            # False generates without grading; the logs are scored by a later
            # pass. See SCORE.
            score=SCORE,
        )

        for log in logs:
            report(strength, log)
            if log.status == "success":
                scenario = (log.eval.task_args or {}).get("scenario")
                done[wanted_key(strength, str(scenario))] = log.location

    print(f"\ndone: {len(done)} successful cells in {LOG_DIR}", flush=True)


if __name__ == "__main__":
    main()
