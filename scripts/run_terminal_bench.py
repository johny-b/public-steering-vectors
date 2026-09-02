"""Run the Terminal-Bench 2.1 eval.

Edit the constants below, then `python scripts/run_terminal_bench.py`. Needs
`pip install -e '.[harbor]'` on a Python 3.12+ interpreter, a running Docker
daemon, and an API key for the model under test.

Every task runs in its own container built from its own image, and there are
89 of them. The first full run pulls all 89 from Docker Hub: budget disk space
generously and pull in batches if the host is small. `python
scripts/validate_terminal_bench.py --check oracle-plan` prints the sizing
arithmetic, and none of the validation checks needs Docker at all.

The default scaffold is inspect's `react` agent, which is what the audited
adapter ships. It is NOT the tmux-driving agent the published Terminal-Bench
numbers come from, so a number out of this script is comparable to another
number out of this script and to nothing else. Set SOLVER to swap it.

Before believing any score, read the diagnostics the run prints alongside it.
`verifier_failed` above zero means some samples produced no reward at all and
are out of the denominator -- an infrastructure result, not a capability one.
Read it rather than `unscored_samples` whenever EPOCHS is above 1: inspect's
epoch reducer drops a NaN epoch instead of carrying it, so a task that failed
on one attempt and scored on another counts as fully scored.
`reward_fractional` should be exactly 0. `tool_calls` and `submitted` say
whether the agent was working or had stopped.
"""

from __future__ import annotations

from inspect_ai import eval
from terminal_bench_2 import terminal_bench_2

MODEL = "anthropic/claude-sonnet-5"

# Names carry the `terminal-bench/` prefix and accept fnmatch globs. `--check
# load` prints all 89 in the form these match.
TASK_NAMES = None  # None = all 89, or e.g. ["terminal-bench/write-compressor"]
N_TASKS = 3  # None = every task; a small number is the cheap first look
EPOCHS = 1  # attempts per task

# Bounds the agent, and half of it bounds the verifier: inspect spends
# `time_limit / 2` on scoring, so a smaller number here cancels the verifiers
# of the slowest tasks and errors those samples outright rather than reporting
# verifier_failed. 24000 is twice the largest declared verifier timeout on the
# 89, and the task warns if a selected task declares more than half of this.
TIME_LIMIT = 24000  # wall-clock seconds per sample; None = unbounded
OVERRIDE_MEMORY_MB = None  # None = the adapter's 6 GB floor; see the README
OVERRIDE_CPUS = None  # None = whatever each task declares
SOLVER = None  # None = the adapter's react agent; not leaderboard-comparable

LOG_DIR = "logs"
MAX_SAMPLES = 4  # concurrent containers; each is granted about 6 GB of RAM


def main() -> None:
    logs = eval(
        terminal_bench_2(
            task_names=TASK_NAMES,
            n_tasks=N_TASKS,
            solver=SOLVER,
            override_cpus=OVERRIDE_CPUS,
            override_memory_mb=OVERRIDE_MEMORY_MB,
            time_limit=TIME_LIMIT,
            epochs=EPOCHS,
        ),
        model=MODEL,
        log_dir=LOG_DIR,
        max_samples=MAX_SAMPLES,
    )

    for log in logs:
        print(f"\n{log.eval.task} ({MODEL}): {log.status}")
        if log.status == "error" and log.error:
            print(log.error.message)
        for score in log.results.scores if log.results else []:
            metrics = ", ".join(
                f"{name}={metric.value}" for name, metric in score.metrics.items()
            )
            print(f"  {score.name}: {metrics}")
            if score.unscored_samples:
                print(
                    f"    ({score.unscored_samples} of "
                    f"{score.scored_samples + score.unscored_samples} samples "
                    "could not be scored on this key -- check verifier_failed)"
                )
        print(f"  log: {log.location}")


if __name__ == "__main__":
    main()
