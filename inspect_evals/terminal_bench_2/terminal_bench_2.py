"""Terminal-Bench 2.1: 89 real terminal tasks, each with its own container.

A wrapper around the official adapter, `inspect_harbor` 0.7.4, pinned to the
harbor hub package `terminal-bench/terminal-bench-2-1` at digest
`sha256:7d7bdc1c...`. Terminal-Bench 2 is the Terminal-Bench Team's benchmark
of end-to-end terminal work (arXiv:2601.11868): compile this, train that,
recover this database, make these tests pass. Each task ships a Docker image,
an instruction, a verifier script and a reference solution; the verifier
writes `1` or `0` to `/logs/verifier/reward.txt` and that is the score.

Version 2.1 is a point release of 2. Its hub readme states that 26 tasks were
modified "to fix bugs, modify timeouts or resources, or improve robustness to
reward hacking"; the 89 task names are unchanged from 2.0.

## What this wrapper adds, and why it exists at all

`inspect_harbor` already ships `terminal_bench_2_1` as a task. Three things
are missing from it for this repository's purposes, and all three are about
being able to read a result rather than about running one.

* Its ref defaults to `latest`, which moves (`inspect_harbor/_tasks.py:3836`,
  and again on `_harbor/task.py:32,115`). This task defaults to the digest and
  refuses anything that is not one; see `harbor_common.guard`.
* Its scorer raises on verifier infrastructure failure, which errors the
  sample (`inspect_harbor/_harbor/scorer.py:25-37,63-181`). This task scores
  through `harbor_common.harbor_reward`, which turns that into a reported
  outcome, and reports `harbor_diagnostics` beside it.
* Its headline convention is `reward > 0`
  (`inspect_harbor/_harbor/scorer.py:108`). This task reports `resolved`,
  which is `reward == 1.0`. The two agree on every Terminal-Bench 2.1 task,
  because all 89 verifiers write a bare 0 or 1 and nothing else.
* Its `dataset_task_names` and `dataset_exclude_task_names` arguments cannot be
  used at all on a package dataset like this one:
  `inspect_harbor/_harbor/task.py:145-162` counts them as dataset-source
  parameters, mutually exclusive with `package_name`, so any value raises
  "Cannot mix task, dataset, and package parameters" before the filter is
  reached. This task therefore filters its own samples and does not forward
  them; see `task_names`.

## Fidelity, stated plainly

**The default scaffold is not the leaderboard's.** Published Terminal-Bench
numbers come from agents that drive a tmux session (Terminus-2 and friends).
The adapter's default solver is inspect's `react` agent with `bash` and
`python` tools on a 300 second per-call timeout. Numbers from this task are
comparable to each other -- across steering conditions, which is what it is
here for -- and are not comparable to a leaderboard. `--solver` swaps the
scaffold (for example `inspect_swe/claude_code`), and `solver=` does the same
in Python.

**Per-task agent timeouts are not enforced.** Every task declares
`[agent].timeout_sec`, from 600 to 12,000 seconds, and harbor's own runner
enforces it. `inspect_harbor` reads the verifier timeout from the same config
and drops the agent one: it sets no limit on the task. An unbounded run is an
easier benchmark than the official one -- in the internal reference run
(private steering-tools workspace) 34 of 89 trials ended on that timeout. Set
`time_limit` to bound it; there is no default, because a single global limit
is not the same instrument as 89 per-task ones and pretending otherwise would
be worse than saying so. The declared per-task value is reported per sample as
`declared_agent_timeout_sec` in the diagnostics.

**`time_limit` bounds the verifier too, at half the number.** Inspect gives
scoring `time_limit / 2` (`inspect_ai/_eval/task/run.py:2142`), so the real
per-sample ceiling is `1.5 x time_limit` and the verifier's share of it is
`time_limit / 2` -- against declared verifier timeouts here that run from 360
to 12,000 seconds. When that half expires the cancellation arrives as a
`BaseException`, which `harbor_reward` deliberately does not catch, so the
sample errors outright with no reward, no `resolved` and no diagnostics row:
the one failure shape this eval otherwise makes impossible. This task warns at
construction when `time_limit / 2` is below what a selected task declares, and
names the tasks.

**Containers get 6 GB whether they asked for it or not.** The adapter floors
declared memory at `MIN_MEMORY_MB = 6144` (`_harbor/converters.py:64-70`).
Terminal-Bench 2.1 declares 2048 MB
on 68 tasks, 4096 on 13 and 8192 on 8, so 81 of 89 run with more memory than
the benchmark specifies: 550 GiB of limits in total against the 252 GiB asked
for. It affects how many samples fit on a host, and it could in principle turn
a memory-pressure failure into a pass. `override_memory_mb`
bypasses the floor entirely (the floor is applied only on the config branch),
which restores the declared value for every task at once at the risk of
OOM-driven false negatives.

**Some task.toml fields are dropped in silence.** Verified absent from the
whole adapter: `[agent].timeout_sec`, `[environment].workdir`,
`[environment].storage_mb` (10240 on all 89 tasks), `build_timeout_sec`,
`[verifier].environment_mode`, top-level `artifacts`, and the per-phase
network policies. `[environment].network_mode = 'allowlist'` is downgraded to
public (`_harbor/converters.py:383-394`, applied at `:150`) with a warning
(`_harbor/task.py:287-302`); no Terminal-Bench 2.1 task declares it, so that finding
is inert here and load-bearing for SWE-bench Pro. All 89 tasks are `public`,
and their verifiers use the network: several `apt-get install` and download
tooling before running the tests, so an offline or rate-limited host produces
verifier failures that look like agent failures. Watch `verifier_failed`.

**Three of the 89 tasks cannot be solved by their own reference solution.** An
oracle sweep on 2026-09-01 scored 86 of 89, with no verifier failures. The
three are `build-pov-ray`, whose source archive povray.org now answers with
403; `build-cython-ext`, which installs an unpinned dependency that made a
breaking release after this digest was published; and `mcmc-sampling-stan`,
whose pinned R install chain exits 1. All three are task-side, none is patched
here, and 86 rather than 89 is the ceiling any agent number should be read
against. `PROVENANCE` beside this file has the evidence per task, what it
means for reading a 0 on each of them, and the cross-harness comparison. The
`solution_exit_code` diagnostic exists because of that sweep: the adapter's
oracle solver discards the reference solution's exit code, so a solution that
died half way through and one the verifier failed arrive as the same reward.

## Provenance, and one thing that cannot be asserted

What is verifiable: the harbor package name, the content digest, its registry
revision and publication date. `scripts/validate_terminal_bench.py --check
provenance` asserts those against the registry.

What is not: harbor records no source repository and no source commit for a
dataset version, anywhere in the registry row or in the downloaded task
directories. `REFERENCE_RUN_UPSTREAM_COMMIT` below is therefore *not*
provenance of the artifact this task runs. It is the commit the internal
reference run's own lock files record, at Terminal-Bench 2.0, from a different
distribution channel, and it is named that way so nothing reads it as a claim
about the pinned digest.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from fnmatch import fnmatch

from harbor_common import (
    harbor_diagnostics,
    harbor_reward,
    harbor_version,
    require_harbor,
    require_pinned_ref,
)
from inspect_ai import Task, task
from inspect_ai.agent import Agent
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

TB2_PACKAGE = "terminal-bench/terminal-bench-2-1"
"""The harbor hub package this task loads.

Not `terminal-bench/terminal-bench-2`. The pinned digest belongs to the 2.1
package: asking the 2 package for it fails with "Digest not found for
dataset". The in-repo eval keeps the name `terminal_bench_2` because that is
the benchmark, and because `terminal-bench` is taken on PyPI.
"""

TB2_DIGEST = "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
"""Registry revision 6 of `terminal-bench/terminal-bench-2-1`, published
2026-04-30. Also what `inspect_harbor` 0.7.4 records as that package's latest
digest, but recorded here explicitly so that a later adapter release cannot
move this task's dataset by moving its own default."""

TB2_N_TASKS = 89
"""Tasks in the pinned dataset. Asserted on every unfiltered construction: if
the digest ever resolves to a different number of tasks, the pin has stopped
meaning what this module says it means, and that should stop a run rather
than quietly change a denominator."""

REFERENCE_RUN_UPSTREAM_COMMIT = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
"""The upstream git commit recorded by the internal reference harness's
official harbor-CLI run (private steering-tools workspace), for
`laude-institute/terminal-bench-2` at dataset version 2.0.

Kept for cross-harness comparison, and named for what it is. It is not the
provenance of the pinned digest: see the module docstring. The reference run
is also a model run (Terminus-2 on a 27B model, mean reward 0.404) rather
than an oracle run, so its per-task rewards are outcomes and not ground
truth.
"""

SANDBOX_ENV_NAMES = ("docker",)
"""Sandbox providers this task is validated against.

The adapter synthesises a compose configuration and hands it to inspect under
whatever provider name it is given (`_harbor/converters.py:218`), so a name
inspect does not know fails late, inside the sandbox registry, on the first
sample. Checked here instead, because a typo should not cost a container.
"""


def _select(
    samples: list[Sample],
    task_names: Sequence[str] | None,
    exclude_task_names: Sequence[str] | None,
    n_tasks: int | None,
) -> list[Sample]:
    """Apply the name filters and the cap, in harbor's own order and semantics.

    Done here rather than by the adapter because the adapter cannot: passing
    either name list alongside a package name raises "Cannot mix task, dataset,
    and package parameters" (`inspect_harbor/_harbor/task.py:145-162`), so the
    filters are unreachable through that route on a package dataset and a
    future adapter bump that fixed it would still have to be re-checked before
    this could delegate.

    The semantics are harbor's, from `harbor/models/job/config.py:119-153`:
    `fnmatch` against the org-prefixed task name, include first, then exclude,
    then a prefix cap. Sample ids carry that same prefixed name. The one
    difference is that harbor filters before downloading and this filters
    after, so a filtered build still downloads all 89 task definitions -- which
    is also what keeps the 89-sample assertion available on every load.

    A bare string is normalised to a one-element list first. Without that,
    `-T task_names=terminal-bench/write-compressor` iterates the string per
    character: a glob like `terminal-bench/*` happens to match everything and
    a plain name raises a per-character error, neither of which is what the
    caller asked for. (Found by the SWE-bench Pro port's review; same fix.)
    """
    if isinstance(task_names, str):
        task_names = [task_names]
    if isinstance(exclude_task_names, str):
        exclude_task_names = [exclude_task_names]
    selected = samples
    if task_names is not None:
        selected = [
            sample
            for sample in selected
            if any(fnmatch(str(sample.id), pattern) for pattern in task_names)
        ]
    if exclude_task_names is not None:
        selected = [
            sample
            for sample in selected
            if not any(
                fnmatch(str(sample.id), pattern) for pattern in exclude_task_names
            )
        ]
    if n_tasks is not None:
        selected = selected[:n_tasks]
    return selected


def _verifier_timeout(sample: Sample) -> float | None:
    """The task's own `[verifier].timeout_sec`, from the sample metadata."""
    config = (sample.metadata or {}).get("harbor_config") or {}
    timeout = (config.get("verifier") or {}).get("timeout_sec")
    return float(timeout) if isinstance(timeout, (int, float)) else None


@task
def terminal_bench_2(
    ref: str = TB2_DIGEST,
    task_names: Sequence[str] | None = None,
    exclude_task_names: Sequence[str] | None = None,
    n_tasks: int | None = None,
    solver: Solver | Agent | None = None,
    sandbox_env_name: str = "docker",
    override_cpus: int | None = None,
    override_memory_mb: int | None = None,
    overwrite_cache: bool = False,
    time_limit: int | None = None,
    epochs: int = 1,
    allow_unpinned: bool = False,
) -> Task:
    """Terminal-Bench 2.1 through `inspect_harbor`, pinned and instrumented.

    Args:
        ref: harbor dataset ref. Defaults to the pinned digest and must be a
            digest unless `allow_unpinned` is set.
        task_names: Task names to include; `fnmatch` glob patterns are
            accepted, and the names carry the `terminal-bench/` prefix (for
            example `terminal-bench/write-compressor`, or
            `terminal-bench/torch-*`). `None` runs all 89. Matching happens
            here rather than in the adapter, which cannot do it on a package
            dataset at all; see `_select`.
        exclude_task_names: Task names to exclude, same matching, applied after
            `task_names`.
        n_tasks: Cap on the number of tasks, applied after the name filters.
            It takes a prefix of the registry's own ordering, not a seeded
            sample, so a capped run is an arbitrary subset that happens to be
            stable. Prefer `task_names` when the subset matters.
        solver: Scaffold to run. `None` keeps the adapter's own `react` agent,
            which is what the audited adapter ships; see the module docstring
            on why that is not leaderboard-comparable. A `--solver` on the
            command line replaces the solver after this function has run, so
            it does not reach this argument; `log.eval.solver` is the record of
            what actually ran.
        sandbox_env_name: Sandbox provider name. The compose configuration is
            synthesised by the adapter either way.
        override_cpus: CPUs per container, overriding the task's own.
        override_memory_mb: Memory per container, overriding the task's own
            *and* the adapter's 6 GB floor; see the module docstring.
        overwrite_cache: Re-download the task definitions instead of trusting
            `~/.cache/harbor`. harbor decides a cache hit from the directory
            name alone and never re-hashes the contents
            (`harbor/tasks/client.py:305-316`), so this is the way to rule out
            a damaged or hand-edited cache.
        time_limit: Wall-clock seconds per sample, or `None` for no limit.
            This is the only bound on the agent, since the per-task
            `[agent].timeout_sec` is not enforced by the adapter -- and it is
            also a bound on the verifier, at half the value: inspect allows
            `time_limit / 2` for scoring (`inspect_ai/_eval/task/run.py:2142`),
            making the per-sample ceiling `1.5 x time_limit`. A verifier
            cancelled by that half errors the sample rather than reporting
            `verifier_failed`, so this warns when the half is below what a
            selected task declares.
        epochs: Attempts per task. Terminal-Bench is stochastic at every
            epoch; 1 is a cheap look, not a measurement. Above 1, read
            infrastructure failures off `verifier_failed` and not off
            `reward.unscored_samples`: inspect's epoch reducer drops a NaN
            epoch rather than carrying it into the reduced score.
        allow_unpinned: Permit a tag or revision as `ref`, with a warning.

    Raises:
        ImportError: If `inspect_harbor` is not installed; the message says
            what to install and on which interpreter.
        ValueError: On a non-digest `ref`, an unknown `sandbox_env_name`, a
            non-positive `n_tasks`, `epochs`, `override_cpus`,
            `override_memory_mb` or `time_limit`, a filter that matches no
            task, or a load that does not return exactly 89 samples before
            filtering (which is every load: the filters are applied here, not
            by the adapter, so the count is checked on every build).

    Building this task downloads the task definitions (about 60 MB) to
    `~/.cache/harbor`. Running it pulls 89 Docker images, one per task, and
    needs a Docker daemon; `scripts/validate_terminal_bench.py --check
    oracle-plan` prints the sizing.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if n_tasks is not None and n_tasks < 1:
        raise ValueError(f"n_tasks must be at least 1, got {n_tasks}")
    if override_cpus is not None and override_cpus < 1:
        raise ValueError(f"override_cpus must be at least 1, got {override_cpus}")
    if override_memory_mb is not None and override_memory_mb < 1:
        raise ValueError(
            f"override_memory_mb must be at least 1, got {override_memory_mb}"
        )
    if time_limit is not None and time_limit < 1:
        raise ValueError(f"time_limit must be at least 1 second, got {time_limit}")
    if sandbox_env_name not in SANDBOX_ENV_NAMES:
        raise ValueError(
            f"sandbox_env_name must be one of {sorted(SANDBOX_ENV_NAMES)}, got "
            f"{sandbox_env_name!r}. The adapter hands its synthesised compose "
            "configuration to whatever provider it is named, so an unknown "
            "name fails inside inspect's sandbox registry on the first sample "
            "instead of here."
        )
    # The dependency guard runs before the ref guard: an operator on a machine
    # without the extra can act on "install this", and cannot act on anything
    # this eval says about digests until they have.
    require_harbor()
    require_pinned_ref(ref, allow_unpinned=allow_unpinned)

    # Imported here rather than at module scope so that the ImportError above
    # is the one an operator sees, and so this module can be imported (for its
    # constants, and by the validation script) on a machine without the extra.
    from inspect_harbor import harbor_scorer
    from inspect_harbor import terminal_bench_2_1 as harbor_task

    # No name filters and no cap are forwarded: the adapter refuses the first
    # two outright on a package dataset (see `_select`), and applying the cap
    # here too keeps one filtering rule rather than two, so the 89-sample
    # assertion below holds on every build rather than only on unfiltered ones.
    base = harbor_task(
        ref=ref,
        sandbox_env_name=sandbox_env_name,
        override_cpus=override_cpus,
        override_memory_mb=override_memory_mb,
        overwrite_cache=overwrite_cache,
    )

    loaded = list(base.dataset)
    if len(loaded) != TB2_N_TASKS:
        raise ValueError(
            f"{TB2_PACKAGE} at {ref} loaded {len(loaded)} tasks, expected "
            f"{TB2_N_TASKS}. A pinned digest cannot change its contents, so "
            "either the ref was overridden, the local harbor cache is "
            "damaged, or the registry is serving something else under this "
            "digest. harbor decides a cache hit from the directory name alone "
            "and never re-hashes what is in it, so re-run with "
            "overwrite_cache=True before trusting any number from this "
            "dataset."
        )

    samples = _select(loaded, task_names, exclude_task_names, n_tasks)
    if not samples:
        raise ValueError(
            f"No task matched task_names={list(task_names or [])} / "
            f"exclude_task_names={list(exclude_task_names or [])}. Names carry "
            "the 'terminal-bench/' prefix and fnmatch glob patterns are "
            "accepted; `python scripts/validate_terminal_bench.py --check "
            f"load` prints all {TB2_N_TASKS} of them in the form the filters "
            "match."
        )
    # Warned rather than defaulted: any single limit is a different instrument
    # from 89 per-task ones, so choosing one here would quietly redefine the
    # benchmark. Only for the default scaffold -- a caller who brought their
    # own agent may well have bounded it inside that agent.
    if time_limit is None and solver is None:
        warnings.warn(
            "terminal_bench_2 is running with no time_limit and the adapter's "
            "default react scaffold. The benchmark's own per-task agent "
            "timeouts (600-12000s) are not enforced by inspect_harbor, so "
            "nothing bounds a stuck agent except the Docker daemon. Pass "
            "time_limit= to bound it.",
            UserWarning,
            stacklevel=2,
        )
    # The other half of that knob. Inspect gives scoring `time_limit / 2`, and
    # the verifier runs inside scoring, so a limit chosen for the agent can cut
    # a verifier off below its own declared timeout. That failure is an errored
    # sample rather than a `verifier_failed` row, which is exactly the outcome
    # this eval's scorer seam exists to prevent, so it is named up front with
    # the tasks it would hit.
    if time_limit is not None:
        scoring_budget = time_limit / 2
        starved = sorted(
            str(sample.id)
            for sample in samples
            if (_verifier_timeout(sample) or 0.0) > scoring_budget
        )
        if starved:
            warnings.warn(
                f"time_limit={time_limit}s gives scoring {scoring_budget:.0f}s "
                f"(inspect uses half; run.py:2142), which is below the "
                f"declared verifier timeout of {len(starved)} of "
                f"{len(samples)} selected tasks: {starved[:5]}"
                f"{' ...' if len(starved) > 5 else ''}. A verifier cancelled "
                "by that budget errors the sample outright instead of "
                "reporting verifier_failed. Use at least twice the largest "
                "declared verifier timeout, or bound the agent inside the "
                "solver rather than with the task's time_limit.",
                UserWarning,
                stacklevel=2,
            )

    scorers: list[Scorer] = [harbor_reward(harbor_scorer()), harbor_diagnostics()]
    return Task(
        dataset=samples,
        # `base.solver` is the adapter's own react agent, already converted to
        # a solver by its Task construction. Falling back to it rather than
        # rebuilding it keeps the default scaffold exactly the audited one.
        solver=solver if solver is not None else base.solver,
        scorer=scorers,
        epochs=epochs,
        time_limit=time_limit,
        name="terminal_bench_2",
        metadata={
            "harbor_package": TB2_PACKAGE,
            "harbor_ref": ref,
            "inspect_harbor_version": harbor_version(),
            "reference_run_upstream_commit": REFERENCE_RUN_UPSTREAM_COMMIT,
            "n_tasks": len(samples),
            "time_limit_sec": time_limit,
            # Named `solver_arg`, not `solver`: a `--solver` on the command
            # line replaces the task's solver after this function returns, so
            # this can only ever record what was passed in Python. What ran is
            # in `log.eval.solver`.
            "solver_arg": "adapter default"
            if solver is None
            else "caller-supplied",
            "override_cpus": override_cpus,
            "override_memory_mb": override_memory_mb,
        },
    )
