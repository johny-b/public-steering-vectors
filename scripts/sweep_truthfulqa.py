"""Sweep TruthfulQA over steering strengths on one vector.

Runs the eval against the steered vLLM server (see `pod/README.md`) once per
strength in STRENGTHS, and writes one eval log per strength into LOG_DIR.

    inspect_providers/steered_provider/ops/tunnel.sh   # or a local server
    python scripts/sweep_truthfulqa.py

Constants below are the whole configuration. Re-running the script resumes: a
strength that already has a successful log in LOG_DIR under the same condition
is skipped rather than re-run, so a sweep interrupted at strength 5 of 9
continues from there.

Strength 0.0 is a steering condition like any other -- the vector is named and
applied at zero -- not the bare unsteered model. It is the sweep's control, and
it has to travel the same code path (chat template, `vllm_xargs`, the patched
block) as its neighbours for the comparison to mean anything.

Two things this sweep does not need that its agentic-misalignment sibling does.
There is no blocked sampling: that eval's dataset is a single sample, so its
sample size is its epoch count and blocks are how it deepens a cell, whereas
here the 817 questions are the sample size and every strength already sees all
of them. And there is no deferred-scoring switch: `choice()` reads the letter
the model picked, so scoring calls no external API and cannot be the thing that
fails while the GPU hours are being spent.

What does have to hold across strengths is the arrangement of the questions.
SEED pins the question order and the answer order within each question for
every task the sweep builds, because models have position biases and an
unseeded shuffle would give each strength a different permutation -- noise
folded straight into the differences the sweep exists to measure.
"""

from __future__ import annotations

from typing import Any

from inspect_ai import Task, eval
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, Model, get_model
from truthfulqa import truthfulqa

from steering_vectors import vectorfmt

STEER_VECTOR = "0007"
STRENGTHS = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

MODEL = "steered/Qwen/Qwen3.6-27B"
"""`steered/<served-model-name>`; STEERED_BASE_URL selects the server."""

# The eval's condition. These also form the rest of the resume key, so a log
# written under a different condition is never mistaken for one of this sweep's.
TARGET = "mc1"  # mc1 (exactly one true answer) | mc2 (one or more)
SHUFFLE = True
SEED = 42
"""Fixes the question order and the answer order for every strength. See the
module docstring: this is what keeps the strengths comparable."""

EPOCHS = 1
"""Attempts per question. The dataset is 817 questions, so it supplies the
sample size on its own and there is nothing epochs have to make up for."""

MAX_TOKENS = 32768
"""Per response. TruthfulQA prompts are short -- a question and its answer
list, a few hundred tokens -- so essentially all of this budget is thinking.
The running server is at `MAX_MODEL_LEN` 65536, so this binds first and a
`stop_reason='model_length'` in the logs would be a real surprise."""

# Qwen's published thinking-mode sampling for Qwen3.6-27B. Deliberately no
# sampling seed: the questions are fixed (SEED does that), the generations are
# not, and pinning both would turn 817 independent draws into 817 replays of
# one arbitrary trajectory through the model.
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20

LOG_DIR = "logs/truthfulqa-sweep"
"""Where this run's logs go, one per strength. Resume (`completed()`) reads
this directory and nothing above it."""

SERVER_MAX_NUM_SEQS = 64
"""The `--max-num-seqs` the running server was actually started with."""

MAX_CONNECTIONS = SERVER_MAX_NUM_SEQS
# All of it, unlike the agentic-misalignment sibling, which deliberately leaves
# the server headroom for another client. This sweep is 817 samples x 9
# strengths of thinking replies against a pod rented for it and nothing else,
# so the headroom would buy nothing and cost hours: filling the batch is the
# whole difference between an afternoon and an evening.
#
# One task per eval() call, so this is also the ceiling on requests in flight
# against the server; there is no second factor to multiply it by.
assert MAX_CONNECTIONS <= SERVER_MAX_NUM_SEQS

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
    return {"target": TARGET, "shuffle": SHUFFLE, "seed": SEED}


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
            # route that works -- it is passed through verbatim and lands at
            # the top level of the request body, which is where vLLM reads its
            # sampling extensions. Confirmed on the wire, not assumed.
            extra_body={"top_k": TOP_K},
            # Explicit, so inspect uses its static connection pool rather than
            # adaptive concurrency, which would probe past SERVER_MAX_NUM_SEQS.
            max_connections=MAX_CONNECTIONS,
        ),
        **steer_args(strength),
    )


def build_task() -> Task:
    """The eval under the sweep's condition. Identical at every strength."""
    return truthfulqa(**task_args())


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
    """Strengths already covered by a successful log in `log_dir`.

    Headers only, so this stays cheap as the directory fills up. A log counts
    only if it succeeded, ran the sweep's own eval condition -- including SEED,
    since a differently shuffled run of the same strength is not the cell that
    was asked for -- and ran the sweep's epoch count.
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
    """Every scorer's accuracy ± stderr."""
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
        f"vector {STEER_VECTOR}: {len(STRENGTHS)} strengths x {TARGET} "
        f"x {EPOCHS} epochs into {LOG_DIR}",
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
            # Deterministic sample concurrency. The server-side cap is the
            # connection pool above; this only decides how many samples may be
            # mid-flight.
            max_samples=MAX_CONNECTIONS,
        )

        for log in logs:
            report(strength, log)
            if log.status == "success":
                done[wanted_key(strength)] = log.location

    print(f"\ndone: {len(done)} successful strengths in {LOG_DIR}", flush=True)


if __name__ == "__main__":
    main()
