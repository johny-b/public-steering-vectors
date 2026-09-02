"""Sweep the ctfish agent pass over steering strengths on one vector.

Runs the ctfish agent loop against the steered vLLM server (see `pod/README.md`)
for every strength in STRENGTHS, on a single task prompt variant, and writes one
eval log per strength into LOG_DIR. The judges are OFF here: this is the agent
pass, scored later by `scripts/score_ctfish.py`.

    inspect_providers/steered_provider/ops/tunnel.sh   # or a local server
    python scripts/sweep_ctfish.py --epochs 10 --log-dir logs/ctfish-sweep/block01-e10

Needs Docker (one container per in-flight sample) and a reachable server for
MODEL. It needs no judge credentials at all, which is the point of splitting the
passes: a judge outage cannot waste GPU and sandbox hours already paid for.

Constants below are the whole configuration; EPOCHS and LOG_DIR can also be
overridden from the command line, because that is how blocks are driven from a
shell. Re-running the script resumes: a strength that already has a successful
log in LOG_DIR under the same condition (vector, strength, variant) and the same
epoch count is skipped rather than re-run, so a sweep interrupted at strength 5
of 9 continues from there.

Strength 0.0 is a steering condition like any other -- the vector is named and
applied at zero -- not the bare unsteered model. It is the sweep's control, and
it has to travel the same code path (chat template, `vllm_xargs`, the patched
block) as its neighbours for the comparison to mean anything.

The dataset is one sample per variant, and this sweep runs one variant, so
EPOCHS is the entire sample size of a cell and every number the sweep produces
is a mean over epochs.

Sampling is blocked. One run of this script is one block: an epoch count written
into its own LOG_DIR, `logs/ctfish-sweep/block<n>-e<epochs>`. Run the whole
9-strength curve at a small epoch count first to get the shape of the result,
then raise EPOCHS, point LOG_DIR at the next block and run it again to deepen
every cell. Nothing is wasted, because the scoring and analysis passes are
pointed at the PARENT `logs/ctfish-sweep` and pool the blocks per (vector,
strength, variant) -- `list_eval_logs` recurses by default, so two blocks of one
condition are one cell at the combined n. Only LOG_DIR and EPOCHS change between
blocks, and the two must change together: two blocks sharing a directory would
see each other's logs and skip the very work that is meant to deepen the cell.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ctfish import ctfish
from inspect_ai import Task, eval
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, Model, get_model

from steering_vectors import vectorfmt

STEER_VECTOR = "0007"
STRENGTHS = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
VARIANT = "grandmaster"
"""The single task prompt variant this sweep runs; see `ctfish.prompts`.

One variant, not several: variants differ in how hard the prompt pushes, so
pooling them would average over the thing the strength curve is measuring. A
second variant is a second sweep, into a directory of its own."""

MODEL = "steered/Qwen/Qwen3.6-27B"
"""`steered/<served-model-name>`; STEERED_BASE_URL selects the server.

Note that upstream's `model_config`/`pre_invoke_message` match on substrings of
the model name (`o1`, `o3`, `deepseek`); this name contains none of them, so the
agent's prompt is the ordinary one and sampling comes from GENERATE_CONFIG."""

EPOCHS = 10
"""The default block's sample size: the dataset is one sample (one variant), so
this is the whole sample size of every cell. Blocked sampling -- a block is one
(LOG_DIR, EPOCHS) pair and the curve is the union of the blocks -- is driven by
overriding this and LOG_DIR from a per-block driver script or the command line,
not by editing them here."""

LOG_DIR = "logs/ctfish-sweep/block01-e10"
"""Where this run's logs go: this BLOCK's directory, under a shared parent.

One directory per block matters twice over. Resume (`completed()`) reads only
this directory, which is what lets a later block at a different EPOCHS add
samples to strengths an earlier block already covered instead of skipping them
as done. And the scoring/analysis passes, pointed at the PARENT, pool every
block beneath it: `list_eval_logs` recurses by default and nothing is
deduplicated, so two blocks of one condition become one cell whose n is the sum.
The flip side of no deduplication is that a log appearing under the parent twice
is counted twice -- pool by adding block directories, never by copying logs
between them."""

MAX_STEPS = 37
"""Decision loop phases per run (upstream MAX_STEPS); ~4 phases per OODA cycle."""

COMMAND_TIMEOUT = 30
"""Seconds per shell command in the sandbox."""

GENERATE_CONFIG = "qwen3-thinking"
"""Agent sampling, pinned to Qwen's published thinking-mode settings rather than
left to upstream's per-model-family policy (which would give this model
temperature 0.5 and nothing else). Deliberately no seed: the cell's sample size
is EPOCHS over one dataset sample, so a fixed seed would return the same run
every epoch and the stderr would be a lie. See `ctfish.config` -- in particular
for why `top_k`/`min_p` travel in `extra_body` and not as GenerateConfig fields.
Keep this identical across every block of the sweep."""

# Judges off in this pass. `None` for a judge model drops that scorer; the
# deterministic scorers (won_game, moves_submitted, engine_assisted_moves,
# truncated_generations) still run, in-process and with no model in the loop, so
# eval() scores these logs without touching an external API. The judge columns
# come from a separate pass over the pooled parent directory:
#
#   python scripts/score_ctfish.py logs/ctfish-sweep \
#       --output-dir logs/ctfish-sweep-scored \
#       --judge-temperature 0 --min-judge-entries 0
STAGE_JUDGE_MODEL = None
ESCALATION_JUDGE_MODEL = None

MIN_JUDGE_ENTRIES = 0
"""Recorded in task_args for the later judge pass to honour. 0 judges every run:
anything higher drops short runs, and whatever made them short then decides
which runs leave the denominator -- a selection effect, not a result."""

DEFAULT_MAX_SANDBOXES = 64
"""Concurrent Docker containers. Sandboxes, not connections, are usually what
the local machine runs out of first."""

DEFAULT_MAX_CONNECTIONS = 64
"""Concurrent requests in flight against the served model.

Requests in flight are capped at this number, not at this number times the
number of tasks: inspect scopes its connection semaphore by the provider's
`connection_key()`, and `steered` keys on (model, vector, strength). This sweep
runs one task per eval() call anyway. Keep it at or below the server's own
`--max-num-seqs`; overshooting does not fail, it just queues inside the server
and inflates wall clock."""

RETRY_ON_ERROR = 2
"""Retries for a sample that errors (transient sandbox/server faults)."""

VECTOR_ARG = "steer_vector"
STRENGTH_ARG = "steer_strength"

ConditionKey = tuple[str | None, float | None, str | None]
"""(vector, strength, variant): one cell of the sweep, comparable across runs."""


def steer_args(strength: float) -> dict[str, Any]:
    """The steering condition, as model args.

    Model args are what the provider reads, and they are recorded in the eval log
    (`log.eval.model_args`), so passing the condition this way rather than
    through a base URL or an environment variable is what makes each log say
    which condition produced it. `run_ctfish_eval.py` copies them into metadata
    as well, and so does this sweep, so a summary over many logs does not have to
    know provider argument names to find out what was run.
    """
    return {VECTOR_ARG: STEER_VECTOR, STRENGTH_ARG: strength}


def task_args(epochs: int) -> dict[str, Any]:
    """The eval condition shared by every cell of the sweep.

    Also half of the resume key: a log written under different task args is
    never mistaken for one of this sweep's.
    """
    return {
        "variant": [VARIANT],
        "max_steps": MAX_STEPS,
        "command_timeout": COMMAND_TIMEOUT,
        # Epochs are set HERE and nowhere else. `eval(epochs=...)` would win over
        # the task's value, so passing both is a way to run a different N than
        # the log's own task_args report.
        "epochs": epochs,
        "generate_config": GENERATE_CONFIG,
        "stage_judge_model": STAGE_JUDGE_MODEL,
        "escalation_judge_model": ESCALATION_JUDGE_MODEL,
        "min_judge_entries": MIN_JUDGE_ENTRIES,
    }


def build_task(epochs: int) -> Task:
    """The ctfish task under this block's condition."""
    return ctfish(**task_args(epochs))


def build_model(strength: float, max_connections: int) -> Model:
    """The model under test at one strength.

    `max_connections` has to be set here, on the model, because the ctfish agent
    calls `model.generate(messages, config=<the GENERATE_CONFIG preset>)` and
    inspect merges that call's config OVER the model's own. The preset carries
    sampling parameters only and never sets `max_connections`, so the value set
    here survives the merge and is the one the connection semaphore uses
    (`Model._resolve_config` -> `_connection_concurrency`). It also becomes the
    default `max_samples` for the task, since inspect derives that from the
    model-composed config. Verified against inspect_ai 0.3.259 by resolving the
    two configs together, not assumed.

    Setting it explicitly also opts out of inspect's adaptive concurrency, which
    would otherwise probe upward past the server's `--max-num-seqs`.
    """
    return get_model(
        MODEL,
        config=GenerateConfig(max_connections=max_connections),
        **steer_args(strength),
    )


def _condition_key(vector: Any, strength: Any, variant: Any) -> ConditionKey:
    """Normalise a condition so a log's copy of it compares equal to ours."""
    return (
        None if vector is None else vectorfmt.vector_id(vector),
        None if strength is None else round(float(strength), 6),
        None if variant is None else str(variant),
    )


def wanted_key(strength: float) -> ConditionKey:
    """The resume key for one cell of the sweep."""
    args = steer_args(strength)
    return _condition_key(args.get(VECTOR_ARG), args.get(STRENGTH_ARG), VARIANT)


def _log_variant(task_args_in_log: dict[str, Any]) -> str | None:
    """The single variant a log ran, or None if it ran none or several."""
    variant = task_args_in_log.get("variant")
    if isinstance(variant, str):
        return variant
    if isinstance(variant, (list, tuple)) and len(variant) == 1:
        return str(variant[0])
    return None


def completed(log_dir: str, epochs: int) -> dict[ConditionKey, str]:
    """Cells of this sweep already covered by a successful log in `log_dir`.

    Headers only, so this stays cheap as the directory fills up. A log counts
    only if it succeeded, ran the sweep's own eval condition, and ran the block's
    epoch count -- a log of 10 epochs is not the 50-epoch cell that was asked
    for, and silently accepting it would quietly shrink that cell.

    `log_dir` is this block's directory, not the sweep parent, which is what
    keeps resume a within-block idea: an earlier block's log of the same cell is
    not visible here and so does not suppress this block's copy of it. That is
    the point of blocking -- the later block adds samples rather than skipping
    work -- and it is also why the two blocks must live in separate directories.

    Note that `fail_on_error=False` means a log can be `success` with some of its
    samples errored, so a resumed block treats a partly-errored cell as done.
    The per-cell `n=completed/total` in the progress output is what to read for
    that; a cell short of its epochs is re-run by deleting its log.
    """
    wanted_args = task_args(epochs)
    found: dict[ConditionKey, str] = {}
    for info in list_eval_logs(log_dir):
        header = read_eval_log(info.name, header_only=True)
        if header.status != "success":
            continue
        args = header.eval.task_args or {}
        if any(args.get(name) != value for name, value in wanted_args.items()):
            continue
        # Belt and braces: the task's `epochs` argument is also copied into the
        # eval config by inspect, and that copy is what actually ran.
        if header.eval.config.epochs != epochs:
            continue
        model_args = header.eval.model_args or {}
        key = _condition_key(
            model_args.get(VECTOR_ARG),
            model_args.get(STRENGTH_ARG),
            _log_variant(args),
        )
        found[key] = info.name
    return found


def score_summary(log: EvalLog) -> str:
    """The deterministic scorers' metrics, as one flat string."""
    parts = []
    for score in log.results.scores if log.results else []:
        for name, metric in score.metrics.items():
            label = (
                score.name if name in ("accuracy", "mean") else f"{score.name}/{name}"
            )
            value = metric.value
            parts.append(
                f"{label}={value:.3f}"
                if isinstance(value, float)
                else f"{label}={value}"
            )
    return " ".join(parts) if parts else "no scores"


def report(strength: float, log: EvalLog) -> None:
    """One watchable line per cell."""
    completed_samples = log.results.completed_samples if log.results else 0
    total_samples = log.results.total_samples if log.results else 0
    print(
        f"[strength={strength:+.2f}] {VARIANT:<12} {log.status:<8} "
        f"n={completed_samples}/{total_samples} {score_summary(log)}",
        flush=True,
    )
    if log.status == "error" and log.error:
        print(f"    error: {log.error.message}", flush=True)
    print(f"    log: {log.location}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep ctfish over steering strengths on one vector and one prompt "
            "variant. Judges are off; score the logs with scripts/score_ctfish.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Runs per strength: this block's sample size for every cell.",
    )
    parser.add_argument(
        "--log-dir",
        default=LOG_DIR,
        help="This BLOCK's log directory. Give each block its own directory "
        "under a shared parent; the parent is what later passes pool over.",
    )
    parser.add_argument(
        "--max-sandboxes",
        type=int,
        default=DEFAULT_MAX_SANDBOXES,
        help="Concurrent Docker containers.",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=DEFAULT_MAX_CONNECTIONS,
        help="Concurrent requests in flight against the served model. Keep at or "
        "below the server's --max-num-seqs.",
    )
    parser.add_argument(
        "--strengths",
        default=None,
        help=(
            "Comma-separated subset of STRENGTHS to run, e.g. '-0.5,-0.4'. "
            "Defaults to all of them. Disjoint subsets can be run as parallel "
            "processes into the same --log-dir: each cell writes its own .eval "
            "file, and resume still skips cells that already succeeded."
        ),
    )
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--display",
        default="plain",
        help="Inspect display. 'plain' is the one that behaves under nohup.",
    )
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.max_sandboxes < 1 or args.max_connections < 1:
        parser.error("--max-sandboxes and --max-connections must be at least 1.")
    if args.strengths is None:
        args.strengths = list(STRENGTHS)
    else:
        try:
            requested = [float(x) for x in args.strengths.split(",") if x.strip()]
        except ValueError:
            parser.error(f"--strengths must be comma-separated numbers, got {args.strengths!r}")
        unknown = [x for x in requested if not any(abs(x - s) < 1e-9 for s in STRENGTHS)]
        if unknown:
            parser.error(f"--strengths values not in STRENGTHS: {unknown}")
        if not requested:
            parser.error("--strengths must name at least one strength.")
        args.strengths = requested
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    done = completed(args.log_dir, args.epochs)
    print(
        f"vector {STEER_VECTOR}, variant {VARIANT}: {len(args.strengths)} strengths "
        f"x {args.epochs} epochs into {args.log_dir}",
        flush=True,
    )
    print(
        f"model {MODEL}, sampling {GENERATE_CONFIG}, judges off, "
        f"max_sandboxes={args.max_sandboxes} max_connections={args.max_connections}",
        flush=True,
    )

    failed = False
    for strength in args.strengths:
        existing = done.get(wanted_key(strength))
        if existing is not None:
            print(
                f"[strength={strength:+.2f}] {VARIANT:<12} skipped: already "
                f"succeeded in {existing}",
                flush=True,
            )
            continue

        logs = eval(
            build_task(args.epochs),
            model=build_model(strength, args.max_connections),
            # Recorded alongside model_args so a pass over many logs can read the
            # condition without knowing provider argument names, exactly as
            # run_ctfish_eval.py records it.
            metadata={
                "variant": VARIANT,
                "variants": [VARIANT],
                "epochs": args.epochs,
                "generate_config": GENERATE_CONFIG,
                "judges_in_agent_pass": False,
                **steer_args(strength),
            },
            log_dir=args.log_dir,
            max_sandboxes=args.max_sandboxes,
            # A sample that crashes must not take the rest down with it: one lost
            # run is a missing data point, an aborted eval is a lost block.
            fail_on_error=False,
            retry_on_error=RETRY_ON_ERROR,
            log_level=args.log_level,
            display=args.display,
        )

        for log in logs:
            report(strength, log)
            if log.status == "success":
                done[wanted_key(strength)] = log.location
            else:
                failed = True

    print(
        f"\ndone: {len(done)}/{len(args.strengths)} successful cells in {args.log_dir}",
        flush=True,
    )
    sys.stdout.flush()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
