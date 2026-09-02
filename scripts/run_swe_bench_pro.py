"""Run the SWE-bench Pro eval, in either variant.

Edit the constants below, then `python scripts/run_swe_bench_pro.py`. Needs
`pip install -e '.[harbor]'` on a Python 3.12+ interpreter, a running Docker
daemon, and an API key for the model under test.

Every task runs in its own container, built on top of its own tag of
`jefzda/sweap-images`, and there are 731 of them. Those tags average about 1.5
GB compressed (measured across all 11 repositories), so a full sweep is of the
order of a terabyte of pulls and rather more of disk once they are unpacked.
N_TASKS defaults to 3 for that reason. `python
scripts/validate_swe_bench_pro.py --check oracle-plan` prints the measured
sizing, and none of the validation checks needs Docker at all.

The two variants hold the same 731 instances. VARIANT = "isolated" adds the
CAIS anti-exploitation layer: the repository's git history is moved aside to
/var/lib/apt/.a8f1c and GitHub is mapped to 0.0.0.0 in the container's
/etc/hosts. The plain variant leaves both routes open.

The gap between the two variants is NOT the size of the exploitation route.
Two things differ, not one. Not a single one of the 731 problem statements is
byte-identical between the packages: the plain variant wraps every issue in the
SWE-agent scaffolding and a numbered procedure, the isolated one strips it and
uses `## Requirements` / `## Interface` headings, and the plain prompt is
longer on all 731 (mean 4,614 characters against 3,486). So a gap is a prompt
difference and an isolation difference together, and these two numbers cannot
separate them. `python scripts/validate_swe_bench_pro.py --check load` prints
the measurement. A study that wants the isolation effect on its own has to hold
the prompt fixed itself, deliberately, and say that it did.

Neither half of the isolation closes its route either. The hosts blocking is
five names in a file in a container the agent is root in; the git history is
moved rather than removed, and the commit that fixes the issue is still an
object in it (the verifier checks that commit out of there with no network).
Read a gap as evidence about how expensive the routes are, not about whether
they exist.

The default scaffold is inspect's `react` agent, which is what the audited
adapter ships. It is NOT Scale's own harness, so a number out of this script is
comparable to another number out of this script and to nothing else. Set SOLVER
to swap it.

Before believing any score, read the diagnostics the run prints alongside it.
`verifier_failed` above zero means some samples produced no reward at all and
are out of the denominator -- an infrastructure result, not a capability one.
Read it rather than `unscored_samples` whenever EPOCHS is above 1: inspect's
epoch reducer drops a NaN epoch instead of carrying it, so a task that failed
on one attempt and scored on another counts as fully scored.
`reward_fractional` should be exactly 0 -- every verifier in this dataset
writes a bare 0 or 1. `tool_calls` and `submitted` say whether the agent was
working or had stopped.

`verifier_failed == 0` is not the whole gate on this dataset. The shipped
verifier traps its own exit and writes 0 on any non-zero path, so its
infrastructure failures (no working directory, a failed gold checkout, a parser
crash, nine NodeBB verifiers that `npm install` at verify time) are scored
zeros that `verifier_failed` cannot see. Read `verifier_exit_code` and the
untruncated `verifier_output` in the score metadata beside it, especially on a
host with limited registry access.
"""

from __future__ import annotations

from inspect_ai import eval
from swe_bench_pro import swe_bench_pro, swe_bench_pro_isolated

MODEL = "anthropic/claude-sonnet-5"

# "plain" runs scale-ai/swe-bench-pro; "isolated" runs cais/swebenchpro.
VARIANT = "plain"

# Names carry the variant's own org prefix -- `scale-ai/` or `cais/` -- and
# accept fnmatch globs, so a filter written for one variant does not match the
# other unless it starts with a wildcard. `--check load` prints all 731 in the
# form these match, grouped by repository. `*/instance_ansible__*` and friends
# are the practical way to select a language: the 11 repositories are 4
# languages, and the package ordering the N_TASKS cap slices is not balanced
# across them.
# (These are Python values, so the leading `*` needs no escaping here. On the
# inspect command line it does: `-T "task_names=['*/instance_ansible__*']"`,
# because a `-T` value is yaml-parsed and a scalar starting with `*` is an
# alias.)
TASK_NAMES = None  # None = all 731, or e.g. ["*/instance_ansible__*"]

# N_TASKS takes a prefix of THIS package's own ordering, and the two packages
# order the same 731 instances differently: the first 20 of the plain variant
# and the first 20 of the isolated one have no instance in common (the overlap
# is still 4 at 50). So running this script twice with VARIANT flipped and
# N_TASKS set compares different work, not the same work under two containers.
# For anything that compares the variants, set TASK_NAMES to a wildcard-prefixed
# list and leave N_TASKS at None; `--check oracle-plan` prints a 20-task
# cross-language list already written that way.
N_TASKS = 3  # None = every task; a small number is the cheap first look
EPOCHS = 1  # attempts per task

# Bounds the agent, and half of it bounds the whole scorer stack: inspect
# spends `time_limit / 2` on scoring, and the verifier is only part of what
# runs in there (the tests copy, the reward read, two cleanup execs and the
# diagnostics scorer share it). Every one of the 731 tasks declares a 3000
# second verifier timeout and a 3000 second agent timeout, and the adapter
# enforces only the first, so 6000 is not enough: it gives scoring exactly the
# 3000 seconds the verifier's own exec is allowed, and a verifier at its cap is
# cancelled by inspect first, which errors the sample outright rather than
# reporting verifier_failed. 6600 is the floor, and it gives the agent more
# than twice what the benchmark allows it; lower it towards 3000 for fidelity
# to the official harness and the task will warn that the verifier is starved.
TIME_LIMIT = 6600  # wall-clock seconds per sample; None = unbounded
OVERRIDE_MEMORY_MB = None  # None = the adapter's 6 GB floor; tasks declare 4096
OVERRIDE_CPUS = None  # None = the 1 CPU every task declares
SOLVER = None  # None = the adapter's react agent; not leaderboard-comparable

LOG_DIR = "logs"
MAX_SAMPLES = 4  # concurrent containers; each is granted 6 GB of RAM

VARIANTS = {"plain": swe_bench_pro, "isolated": swe_bench_pro_isolated}


def main() -> None:
    if VARIANT not in VARIANTS:
        raise ValueError(f"VARIANT must be one of {sorted(VARIANTS)}, got {VARIANT!r}")
    logs = eval(
        VARIANTS[VARIANT](
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
