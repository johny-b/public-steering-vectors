"""SWE-bench Pro: 731 real pull requests, one container each, two packagings.

A wrapper around the official adapter, `inspect_harbor` 0.7.4, pinned to two
harbor hub packages that carry the same 731 instances:
`scale-ai/swe-bench-pro` at digest `sha256:88411d32...` and
`cais/swebenchpro` at digest `sha256:0684038c...`. SWE-bench Pro is Scale AI's
enterprise-scale successor to SWE-bench: each task hands the agent a repository
checked out at a base commit and an issue to fix, and scores it by running the
project's own test suite.

The instances come from 11 repositories in four languages: Go (280 tasks, in
flipt, teleport, vuls, navidrome), Python (266, in ansible, openlibrary,
qutebrowser), JavaScript (165, in element-web, NodeBB, webclients) and
TypeScript (20, in tutanota). Every image is a tag of one Docker Hub
repository, `jefzda/sweap-images`, and every task builds a thin layer on top of
its own tag.

## The official scoring rule, and where it lives

Scale's own harness (github.com/scaleapi/SWE-bench_Pro-os, MIT) resolves a task
when the union of its `fail_to_pass` and `pass_to_pass` test lists is a subset
of the tests that passed. In these hub packages that rule is not reimplemented
by anyone: it is `tests/test.sh`, one byte-identical file shipped with all 731
tasks of a variant, which runs the instance's own `run_script.sh`, parses the
output with the instance's own `parser.py`, and evaluates

    all_required = fail_to_pass | pass_to_pass
    success = all_required <= passed_tests

then exits 0 or 1. An EXIT trap writes `1` or `0` to
`/logs/verifier/reward.txt` on every path out of the script, including an early
one, so the reward is binary by construction and there is no path on which a
verifier that started produces no reward file at all. No task writes a
`reward.json`, so the adapter's JSON branch and its "use the first value in the
object" fallback (`inspect_harbor/_harbor/scorer.py:139-181`) are never taken
here. `scripts/validate_swe_bench_pro.py --check reward-semantics` quotes the
script and asserts all of that against the downloaded task directories.

**What that trap costs, and it is not small.** It writes 0 on every non-zero
exit, including the verifier's own infrastructure failures: "ERROR: Neither
/app nor /testbed exists", "ERROR: Gold tests checkout failed", a `parser.py`
crash, or the project's own `run_script.sh` producing no parseable output. Nine
of the 731 run scripts (all NodeBB) run `npm install` against
registry.npmjs.org at verify time, so on an offline or rate-limited host those
tasks score a clean 0 rather than going unscored. `harbor_reward` sees only the
failures that happen outside `test.sh` (the tests copy, the sandbox `exec`
itself, a missing or unparseable reward file), so on this dataset
`verifier_failed` measures those and nothing else. It is still the first number
to read and it is not the whole gate: read `verifier_exit_code` and the
preserved `verifier_output` beside it (both are in `Score.metadata`), and treat
a run on a host without registry access as one whose zeros are not all
capability results. Terminal-Bench 2.1's verifiers fail loudly where these fail
quietly; that is a difference between the two datasets rather than a property
of the seam.

**The trap fired once already, on 280 tasks, and this task corrects it.** The
adapter runs every verifier as a login shell, `["bash", "-l", <test.sh>]`
(`inspect_harbor/_harbor/scorer.py:100`). A login shell sources `/etc/profile`,
and `/etc/profile` in the `jefzda/sweap-images` images overwrites the `PATH`
the image sets through `ENV`:

    non-login  /go/bin:/usr/local/go/bin:/usr/local/sbin:...  -> go found
    login      /root/.local/bin:/usr/local/sbin:...           -> go: not found

Go is installed at `/usr/local/go/bin`. So on the adapter's own invocation
`go test` never runs on any Go task, the benchmark's `parser.py` finds no
results, the EXIT trap writes 0, and the sample arrives with
`verifier_failed = 0`, `unscored = 0` and `scorer_ran = 1`: a confident zero
that nothing in the results table distinguishes from a model that failed. That
is all four Go repositories -- flipt, teleport, vuls, navidrome -- **280 of the
731 tasks**. Measured on 2026-09-02: the plan's 20-task cross-language oracle
scored 0.700 on the adapter's invocation, and the six failures were the five Go
tasks plus the one NodeBB instance the packagers themselves score 0.0, two of
the five having a recorded upstream oracle reward of 1.0.

This task therefore passes `verifier_login_shell=False` to `harbor_reward`, so
the verifier runs with the image's own `PATH`. The override removes the `-l`
from that one argv and touches nothing else; `harbor_common.verifier_shell`
carries the derivation, the two explanations that were tested and discarded,
and the reason the fix is in this seam rather than in the pinned adapter.
`scripts/validate_swe_bench_pro.py --check verifier-shell` asserts that the
finding is still live on the pinned version, that the rewrite is correctly
narrow, and -- where a Docker daemon is available -- that a Go task's verifier
really does see `go` under this invocation and really does not under the
adapter's.

Terminal-Bench 2.1 keeps the adapter's invocation, and that is a measurement
rather than an oversight: in its 89-task oracle sweep 86 tasks scored 1.0, none
of the 89 preserved verifier outputs contains a `command not found` or
`executable file not found` signature, and the three failures are documented
task-side defects whose verifiers produced thousands of characters of real
build output. Its images put their tooling on the stock `PATH`.

Because the reward is binary, `reward` and `resolved` from
`harbor_common.harbor_reward` carry the same information, and `resolved` is the
headline: it is the official rule's own verdict, it is what "resolved" means in
every SWE-bench paper, and it stays correct if a future revision of the
verifier ever starts reporting partial credit, where a mean `reward` would
quietly start meaning something else. `reward_fractional` in the diagnostics is
the alarm for that day.

## The two variants

Both packages contain the same 731 instances under the same names, differing
only in the org prefix on the sample id (verified: the two id sets map 1:1
after the prefix is stripped). What differs is the container -- and, less
obviously, the prompt.

`swe_bench_pro` (`scale-ai/swe-bench-pro`) is the plain packaging. Its
Dockerfile resets the base image's entrypoint (`ENTRYPOINT []`) and checks the
repository out at the base commit. The repository's full git history is present
in `/app/.git`.

`swe_bench_pro_isolated` (`cais/swebenchpro`) is the anti-exploitation
packaging, and it adds two things to the same image, on all 731 tasks:

* **Git-history isolation, which is a relocation and not a removal.** The
  Dockerfile runs `cp -a /app/.git /var/lib/apt/.a8f1c && rm -rf /app/.git` and
  then re-initialises `/app` as a fresh repository whose single "base" commit is
  the checked-out tree, so `git log` in `/app` shows one commit. The complete
  history is still inside the image at `/var/lib/apt/.a8f1c`, and the agent is
  root there (no task declares a user and the adapter sets none on the
  service), so `git --git-dir=/var/lib/apt/.a8f1c log --all -p` reads it. The
  fixing commit is in it: the verifier moves that directory back and then
  checks the instance's own commit out of it with
  `git checkout <sha> -- <test paths>` on 731 of 731 tasks, and no task's setup
  contains a `git fetch` or a `git clone`, so that sha resolves from local
  objects alone. `--check network` asserts both halves of that. Like the hosts
  blocking below, this raises the cost of the git route; it does not close it.
* **GitHub network blocking.** The Dockerfile installs an `entrypoint.sh` that
  appends `0.0.0.0` mappings for `github.com`, `raw.githubusercontent.com`,
  `api.github.com`, `objects.githubusercontent.com` and
  `codeload.github.com` to `/etc/hosts` and then `exec "$@"`, and
  sets it as the image `ENTRYPOINT`. An agent that fetches the upstream fix by
  those names therefore reaches 0.0.0.0, for as long as it leaves the file
  alone; see "what the blocking is not" below.

Neither mechanism is a harbor `network_mode`, and that is the finding that
matters for this repository, because `inspect_harbor` cannot enforce
`network_mode = allowlist`: it downgrades it to public and only warns
(`_harbor/converters.py:383-394`, applied at `:150`, warned at
`_harbor/task.py:287-302`). All 731 CAIS tasks declare `network_mode = public`
(migrated from the deprecated `allow_internet = true`), and **none declares
`allowlist`**, so nothing about the isolated variant's naming property depends
on the feature the adapter drops. The blocking is inside the image, and it
survives the adapter's compose translation: the adapter sets no `entrypoint` on
its synthesised service, so the image's `ENTRYPOINT` still runs, and it sets
`command: tail -f /dev/null`, which `exec "$@"` runs after writing the hosts
entries. Inspect's docker sandbox brings the container up with
`compose up --detach --wait` and then runs every tool call through
`docker compose exec`, so those `exec` shells see the modified `/etc/hosts`.
`scripts/validate_swe_bench_pro.py --check network` reads and asserts all of
this: the declared network modes, the 1462 Dockerfiles and entrypoint scripts,
the synthesised compose services, and inspect's own docker provider. It fires
the allowlist detector on a fabricated sample first, through the same accessor
the real count uses, so that a zero is a measurement rather than a broken
comparison. One link in the chain is read out of source rather than measured,
and is named as such where it is asserted: that the `exec` shells inherit the
modified `/etc/hosts` follows from inspect bringing containers up with
`["up", "--detach", "--wait"]` (`_sandbox/docker/compose.py:36`) and running
tool calls through `exec` (`_sandbox/docker/docker.py:554`). Nothing here starts
a container.

What the blocking is not: it is five hostnames in a file, in a container the
agent has root in. An agent that edits `/etc/hosts` back, resolves an IP
literal, or uses any GitHub mirror or proxy is not blocked. The variant raises
the cost of that route; it does not close it.

### The variants also do not ask the same question

Neither package advertises this and it is the largest caveat on comparing them:
**not one of the 731 problem statements is byte-identical between the two**
(measured, and asserted by `--check load`). The plain variant wraps every issue
in the SWE-agent house scaffolding -- `<uploaded_files>`, `<pr_description>`,
and a numbered "Follow these steps to resolve the issue" procedure that tells
the agent to write a reproduction script before editing. The isolated variant
strips all of it and presents the same issue under `## Requirements` and
`## Interface` headings, with no procedure: about 1,100 characters shorter per
task on average, and one fewer method suggestion.

So a gap between the two variants' scores is a prompt difference plus an
isolation difference, and those two numbers cannot separate them. Both are
reported as their publishers built them, because a modified dataset is nobody's
benchmark; a study that wants the isolation effect on its own has to hold the
prompt fixed itself, deliberately, and say that it did.

## Fidelity, stated plainly

**The default scaffold is not the leaderboard's.** Published SWE-bench Pro
numbers come from Scale's own harness and its own agents. The adapter's default
solver is inspect's `react` agent with `bash` and `python` tools. Numbers from
this task are comparable to each other -- across steering conditions, which is
what it is here for -- and are not comparable to a published number. `--solver`
swaps the scaffold (for example `inspect_swe/claude_code`), and `solver=` does
the same in Python.

**Per-task agent timeouts are not enforced.** Every task declares
`[agent].timeout_sec = 3000`, and harbor's own runner enforces it.
`inspect_harbor` reads the verifier timeout from the same config and drops the
agent one: it sets no limit on the task. An unbounded run is an easier
benchmark than the official one. Set `time_limit` to bound it; there is no
default, for the reason in `time_limit`'s own argument documentation.

**`time_limit` bounds the verifier too, at half the number.** Inspect gives
scoring `time_limit / 2` (`inspect_ai/_eval/task/run.py:2142`), so the real
per-sample ceiling is `1.5 x time_limit` and the verifier's share of it is
`time_limit / 2`, against a declared verifier timeout of 3000 seconds on every
one of the 731. That half is not the verifier's alone: the whole scorer loop
runs inside it (`run.py:2158-2190`), and before the verifier starts the adapter
copies `/tests` into the sandbox and makes two directories, and after it
finishes it reads the reward file and runs two cleanup execs, then
`harbor_diagnostics` runs. So `time_limit = 6000` is exactly wrong rather than
exactly enough: it gives scoring 3000 seconds, which is the verifier's own exec
timeout, and a verifier that runs to its declared cap is cancelled by inspect
first. That cancellation arrives as a `BaseException`, which `harbor_reward`
deliberately does not catch: the sample errors outright, with no reward, no
`resolved` and no diagnostics row, which is the one failure shape this eval's
scorer seam exists to prevent. Use `time_limit = 6600`
(`RECOMMENDED_TIME_LIMIT_SEC`), which leaves the scoring budget
`SCORING_OVERHEAD_SEC = 300` seconds above the declared verifier timeout. This
task warns at construction when the budget is not that much above what a
selected task declares, and names the tasks.

**The verifier runs in the container the agent had root in.** After the agent's
turn the adapter copies `/tests` into the same sandbox and runs
`bash -l /tests/test.sh` there (`inspect_harbor/_harbor/scorer.py:99-104`), as
root, because no task declares a `verifier_user`. `bash -l` sources login
profiles the agent could have written, and PATH, `/etc/profile.d`, `python3`,
`node_modules` and every source file that is not a restored test file are
whatever the agent left behind. What is restored is narrow: `test.sh` runs only
the last line of the instance's `before_repo_set_cmd` (literally
`cmd.split('\n')[-1]`), and all 731 of those values are four lines whose first
three are `git reset --hard <base>`, `git clean -fd` and `git checkout <base>`,
none of which ever run. The last line restores a median of one path (453 of 731
restore exactly one file; the largest restores 59). Scale's own harness scores
an extracted patch applied to a clean container; this scores the agent's live
one. So `resolved = 1` is not by itself proof that the tests were run honestly,
and for a study about exploitation this route is cheaper than either of the two
the isolated variant addresses: it needs one line in `/root/.bash_profile`. The
verifier's complete stdout and stderr are kept untruncated in the score
metadata as `verifier_output`, and that is the record to read.

**Every container gets 6 GB whether it asked for it or not.** The adapter
floors declared memory at `MIN_MEMORY_MB = 6144` (`_harbor/converters.py:64-70`
-- and only on the config branch, which is the branch every one of these tasks
takes, since none ships a `docker-compose.yaml`). All 731 tasks declare 4096
MB, so all 731 run with half again the memory the benchmark specifies: 4.3 TiB
of limits in total against the 2.9 TiB asked for. It affects how many samples
fit on a host, and it could in principle turn a memory-pressure failure into a
pass -- which on a benchmark whose tasks compile Go and build JavaScript
bundles is not a hypothetical. `override_memory_mb` bypasses the floor
entirely, restoring the declared 4096 for every task at once at the risk of
OOM-driven false negatives. All 731 also declare `cpus = 1`, which is
forwarded unchanged and is the binding constraint on wall-clock time, not
memory.

**Scale's bash-by-default warning is real, and both packagings already handle
it.** Scale's README warns "bash runs by default in our images -- do not
manually invoke bash". Read from the registry, 21 of 22 sampled tags declare an
`ENTRYPOINT`: 20 of them `["/bin/bash"]` and one `["/bin/sh"]`. The remaining
tag declares no entrypoint and `CMD ["bash"]` instead. The sample is the first
two tags of each of the 11 source repositories in sorted tag order, read from
the Docker Hub image config on 2026-08-31, so it is reproducible rather than a
remembered number, and no check reproduces it because doing so needs the
registry. Left alone, the adapter's `command: tail -f /dev/null` would be
handed to a `/bin/bash` entrypoint as arguments, bash would try to run a script
named `tail`, and the container would exit before the agent ever reached it.
The scale variant resets it with
`ENTRYPOINT []` and the CAIS variant replaces it with an entrypoint that ends
in `exec "$@"`, so neither is affected -- but this is checked rather than
assumed, on all 1462 task directories, by `--check compose`.

**Some task.toml fields are dropped in silence.** Verified absent from the
whole adapter: `[agent].timeout_sec`, `[environment].workdir`,
`[environment].storage_mb` (10240 on all 731 tasks), `build_timeout_sec` (1800
on all 731, and these images are large), `[verifier].environment_mode`,
top-level `artifacts`, and the per-phase network policies. No task in either
variant declares `mcp_servers`, `skills_dir` or `network_mode = allowlist`, so
the adapter emits no degraded-fidelity warning for either package.

## Provenance, and one thing that cannot be asserted

What is verifiable: the harbor package names and the two content digests, which
are what this module pins, and which the adapter's own generated task functions
record as those packages' latest digests at 0.7.4
(`inspect_harbor/_tasks.py:884-911` and `:3339-3366`, checked before this
module was written rather than copied from a plan).

What is not: harbor records no source repository and no source commit for a
dataset version. There is no internal reference run of SWE-bench Pro to
cross-check against either, which is the one audit Terminal-Bench 2.1 had and
this eval does not. That is why PR order put Terminal-Bench first: the seam
these two tasks score through was audited there, against a real reference run,
and this module adds no scoring logic of its own to re-audit.
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

SCALE_PACKAGE = "scale-ai/swe-bench-pro"
"""The harbor hub package behind `swe_bench_pro`, published by Scale AI."""

SCALE_DIGEST = "sha256:88411d32ff27e53a4c1a7e29f0c2aeba180c8e5d60f221cab5ed56325f33549d"
"""The pinned content digest of `scale-ai/swe-bench-pro`.

Also what `inspect_harbor` 0.7.4 records as that package's latest digest
(`_tasks.py:3353`), but recorded here explicitly so that a later adapter
release cannot move this task's dataset by moving its own default. A digest
cannot move: harbor resolves it by an equality match on the stored
`content_hash`, and each constituent task is separately pinned by its own
content hash.
"""

CAIS_PACKAGE = "cais/swebenchpro"
"""The harbor hub package behind `swe_bench_pro_isolated`.

Not `cais/swe-bench-pro`, and not `cais/swebench-pro`: the slug has no hyphens
after the org. The adapter's generated function for it is `cais_swebenchpro`.
"""

CAIS_DIGEST = "sha256:0684038ce8eae92d435a27307d1c5843e291152898f429af130062e8df110768"
"""The pinned content digest of `cais/swebenchpro`.

Also `inspect_harbor` 0.7.4's recorded latest for that package
(`_tasks.py:898`), pinned here for the same reason as `SCALE_DIGEST`.
"""

SWE_BENCH_PRO_N_TASKS = 731
"""Tasks in each pinned dataset. Asserted on every unfiltered construction, for
both variants: if a digest ever resolves to a different number of tasks, the
pin has stopped meaning what this module says it means, and that should stop a
run rather than quietly change a denominator."""

BASE_IMAGE_REPOSITORY = "jefzda/sweap-images"
"""The one Docker Hub repository every task's image is built from.

All 1462 task Dockerfiles are `FROM jefzda/sweap-images:<tag>`, one tag per
instance, and the 731 tags are identical between the two variants -- so running
both variants pulls 731 base images, not 1462. Recorded here because it is the
whole disk cost of this benchmark and it is not visible from the compose
configuration, which names only the locally built `hb__...` layer on top.
"""

DECLARED_VERIFIER_TIMEOUT_SEC = 3000
"""What every task declares as `[verifier].timeout_sec`.

Uniform across both variants and all 731 tasks, which is why the guidance about
`time_limit` can be a single number rather than a table. The warning below
still reads the per-sample value rather than this constant, so a revision that
made timeouts non-uniform would be caught rather than papered over.
"""

SCORING_OVERHEAD_SEC = 300
"""Headroom the scoring budget needs on top of the verifier's own timeout.

Inspect's scoring budget is `time_limit / 2` and the whole scorer loop runs
inside it (`inspect_ai/_eval/task/run.py:2142,2158-2190`), not the verifier
alone. Before the verifier starts, the adapter copies the task's `tests`
directory into the sandbox and runs two `mkdir` execs; after it returns, it
reads the reward file and runs two cleanup execs; then `harbor_diagnostics`
runs. All of that shares the same budget as the 3000 second `exec` the verifier
gets (`_harbor/scorer.py:76-118`). Five minutes is a round number rather than a
measured one, and it is named as a constant so that the recommendation below,
the construction warning and the validation script cannot drift apart.
"""

RECOMMENDED_TIME_LIMIT_SEC = 2 * (DECLARED_VERIFIER_TIMEOUT_SEC + SCORING_OVERHEAD_SEC)
"""The smallest `time_limit` that does not starve a verifier on this dataset.

6600, not 6000. At 6000 the scoring budget is exactly the 3000 seconds the
verifier's own `exec` is given, so a verifier that runs to its declared cap is
cancelled by inspect before its timeout fires, and an inspect cancellation
errors the sample rather than reporting `verifier_failed`.
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
    filters are unreachable through that route on a package dataset, and a
    future adapter bump that fixed it would still have to be re-checked before
    this could delegate.

    The semantics are harbor's, from `harbor/models/job/config.py:119-153`:
    `fnmatch` against the org-prefixed task name, include first, then exclude,
    then a prefix cap. Sample ids carry that same prefixed name. The one
    difference is that harbor filters before downloading and this filters
    after, so a filtered build still downloads all 731 task definitions -- which
    is also what keeps the 731-sample assertion available on every load.

    A bare string is accepted as a single pattern, and that is not cosmetic.
    Inspect's own `-T` parser splits on commas and returns a `str` when there is
    no comma (`inspect_ai/_cli/util.py:214-227`), so
    `-T task_names=scale-ai/instance_ansible__foo` -- the single-task smoke run,
    and the spelling `--check oracle-plan` prints -- arrives here as a string.
    Iterated as a sequence that is a list of one-character patterns: without a
    `*` in it the build dies with a nonsense error listing every character, and
    with one it matches every sample and silently runs all 731 containers.
    `Sequence[str]` does not exclude `str`, so no type checker catches it. It is
    normalised rather than refused because the command line that produces it is
    the one this repository's own documentation teaches.

    This is a second copy of Terminal-Bench 2.1's `_select`, not a shared
    helper, and deliberately so: `harbor_common` is the audited seam onto the
    adapter's scorer, and putting a convenience function in it would make it a
    utility module that has to be re-read for reasons other than the audit.
    Twenty lines of `fnmatch` are cheaper than that, and `--check load` proves
    this copy's behaviour independently. The two copies are no longer identical:
    Terminal-Bench's has the same bare-string defect and has not been changed
    here, because it belongs to a different branch and a different PR.
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


def _build(
    variant: str,
    package: str,
    ref: str,
    task_names: Sequence[str] | None,
    exclude_task_names: Sequence[str] | None,
    n_tasks: int | None,
    solver: Solver | Agent | None,
    sandbox_env_name: str,
    override_cpus: int | None,
    override_memory_mb: int | None,
    overwrite_cache: bool,
    time_limit: int | None,
    epochs: int,
    allow_unpinned: bool,
) -> Task:
    """Build one of the two variants. See the two `@task` functions for the args.

    One implementation, because the variants differ only in which harbor
    package they load and what is baked into that package's images. Every
    guard, every assertion and every scorer is the same, and two copies of them
    would be two things to keep in step for no gain: the difference between the
    variants is a property of the dataset, not of the wrapper.
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
    from inspect_harbor import cais_swebenchpro, harbor_scorer, scale_ai_swe_bench_pro

    factories = {
        CAIS_PACKAGE: cais_swebenchpro,
        SCALE_PACKAGE: scale_ai_swe_bench_pro,
    }
    # No name filters and no cap are forwarded: the adapter refuses the first
    # two outright on a package dataset (see `_select`), and applying the cap
    # here too keeps one filtering rule rather than two, so the 731-sample
    # assertion below holds on every build rather than only on unfiltered ones.
    base = factories[package](
        ref=ref,
        sandbox_env_name=sandbox_env_name,
        override_cpus=override_cpus,
        override_memory_mb=override_memory_mb,
        overwrite_cache=overwrite_cache,
    )

    loaded = list(base.dataset)
    if len(loaded) != SWE_BENCH_PRO_N_TASKS:
        raise ValueError(
            f"{package} at {ref} loaded {len(loaded)} tasks, expected "
            f"{SWE_BENCH_PRO_N_TASKS}. A pinned digest cannot change its "
            "contents, so either the ref was overridden, the local harbor "
            "cache is damaged, or the registry is serving something else "
            "under this digest. harbor decides a cache hit from the directory "
            "name alone and never re-hashes what is in it, so re-run with "
            "overwrite_cache=True before trusting any number from this dataset."
        )

    samples = _select(loaded, task_names, exclude_task_names, n_tasks)
    if not samples:
        prefix = package.split("/")[0]
        raise ValueError(
            f"No task matched task_names={list(task_names or [])} / "
            f"exclude_task_names={list(exclude_task_names or [])}. Names carry "
            f"the '{prefix}/' prefix and fnmatch glob patterns are accepted; "
            "`python scripts/validate_swe_bench_pro.py --check load` prints "
            f"all {SWE_BENCH_PRO_N_TASKS} of them in the form the filters "
            "match."
        )
    # Warned rather than defaulted: any single limit is a different instrument
    # from 731 per-task ones, so choosing one here would quietly redefine the
    # benchmark. Only for the default scaffold -- a caller who brought their
    # own agent may well have bounded it inside that agent.
    if time_limit is None and solver is None:
        warnings.warn(
            f"{variant} is running with no time_limit and the adapter's "
            "default react scaffold. The benchmark's own per-task agent "
            "timeout (3000s on every task) is not enforced by inspect_harbor, "
            "so nothing bounds a stuck agent except the Docker daemon. Pass "
            "time_limit= to bound it.",
            UserWarning,
            stacklevel=3,
        )
    # The other half of that knob. Inspect gives scoring `time_limit / 2`, and
    # the verifier runs inside scoring, so a limit chosen for the agent can cut
    # a verifier off below its own declared timeout. That failure is an errored
    # sample rather than a `verifier_failed` row, which is exactly the outcome
    # this eval's scorer seam exists to prevent, so it is named up front with
    # the tasks it would hit. On this dataset the threshold is a single number
    # (6600s), but the check reads each sample's own declaration anyway.
    #
    # The comparison allows for SCORING_OVERHEAD_SEC rather than being a bare
    # `>` against the declared timeout. The scoring budget covers the tests
    # copy, two mkdirs, the reward read, two cleanup execs and the diagnostics
    # scorer as well as the verifier's own exec, so a budget merely equal to
    # the verifier's timeout is already too small -- and a strict `>` stayed
    # silent at exactly 6000, which is the number every piece of guidance in
    # this repository used to recommend.
    if time_limit is not None:
        scoring_budget = time_limit / 2
        starved = sorted(
            str(sample.id)
            for sample in samples
            if (_verifier_timeout(sample) or 0.0) + SCORING_OVERHEAD_SEC
            > scoring_budget
        )
        if starved:
            warnings.warn(
                f"time_limit={time_limit}s gives scoring {scoring_budget:.1f}s "
                f"(inspect uses half; run.py:2142), which does not clear the "
                f"declared verifier timeout of {len(starved)} of "
                f"{len(samples)} selected tasks by the {SCORING_OVERHEAD_SEC}s "
                f"the rest of the scorer stack needs: {starved[:3]}"
                f"{' ...' if len(starved) > 3 else ''}. That budget covers the "
                "tests copy, the reward read, two cleanup execs and the "
                "diagnostics scorer as well as the verifier, and a verifier "
                "cancelled by it errors the sample outright instead of "
                "reporting verifier_failed. Use at least "
                f"{RECOMMENDED_TIME_LIMIT_SEC}, which is twice the "
                f"{DECLARED_VERIFIER_TIMEOUT_SEC}s verifier timeout every task "
                "in this dataset declares plus that headroom, or bound the "
                "agent inside the solver rather than with the task's "
                "time_limit.",
                UserWarning,
                stacklevel=3,
            )

    # `verifier_login_shell=False` is the one place this wrapper changes what
    # the adapter does rather than only reading it, and it is not optional on
    # this dataset: the adapter runs `["bash", "-l", <test.sh>]`
    # (`inspect_harbor/_harbor/scorer.py:100`), `/etc/profile` in the
    # `jefzda/sweap-images` images overwrites the image's own `ENV PATH`, and
    # `/usr/local/go/bin` is one of the entries it drops -- so without this
    # every one of the 280 Go tasks scores a clean 0 with `verifier_failed = 0`.
    # See `harbor_common.verifier_shell` and the module docstring.
    scorers: list[Scorer] = [
        harbor_reward(harbor_scorer(), verifier_login_shell=False),
        harbor_diagnostics(),
    ]
    return Task(
        dataset=samples,
        # `base.solver` is the adapter's own react agent, already converted to
        # a solver by its Task construction. Falling back to it rather than
        # rebuilding it keeps the default scaffold exactly the audited one.
        solver=solver if solver is not None else base.solver,
        scorer=scorers,
        epochs=epochs,
        time_limit=time_limit,
        name=variant,
        metadata={
            "harbor_package": package,
            "harbor_ref": ref,
            "inspect_harbor_version": harbor_version(),
            "base_image_repository": BASE_IMAGE_REPOSITORY,
            "variant": variant,
            "n_tasks": len(samples),
            "time_limit_sec": time_limit,
            # Named `solver_arg`, not `solver`: a `--solver` on the command
            # line replaces the task's solver after this function returns, so
            # this can only ever record what was passed in Python. What ran is
            # in `log.eval.solver`.
            "solver_arg": "adapter default" if solver is None else "caller-supplied",
            "override_cpus": override_cpus,
            "override_memory_mb": override_memory_mb,
        },
    )


@task
def swe_bench_pro(
    ref: str = SCALE_DIGEST,
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
    """SWE-bench Pro through `inspect_harbor`, pinned and instrumented.

    The plain packaging, `scale-ai/swe-bench-pro`. The repository's full git
    history is present at `/app/.git` and nothing blocks network access to
    GitHub, so an agent can in principle find the upstream fix rather than
    write one. Use `swe_bench_pro_isolated` for the packaging that makes both
    routes more expensive, and see the module docstring for how much more
    expensive: neither route is closed there, and the history is moved rather
    than removed.

    Args:
        ref: harbor dataset ref. Defaults to the pinned digest and must be a
            digest unless `allow_unpinned` is set.
        task_names: Task names to include; `fnmatch` glob patterns are
            accepted, and the names carry the `scale-ai/` prefix (for example
            `scale-ai/instance_ansible__ansible-*`). `None` runs all 731. A
            bare string is one pattern, not a sequence of characters, because
            that is what `-T task_names=<one name>` produces. Matching happens
            here rather than in the adapter, which cannot do it on a package
            dataset at all; see `_select`. The isolated variant's names carry a
            `cais/` prefix instead, so a filter is portable between the
            variants only if it starts with a wildcard -- and a pattern that
            starts with `*` has to be written as a YAML list on the command
            line (`-T "task_names=['*/instance_ansible__*']"`), because inspect
            yaml-parses a `-T` value first and a scalar beginning with `*` is a
            YAML alias.
        exclude_task_names: Task names to exclude, same matching, applied after
            `task_names`.
        n_tasks: Cap on the number of tasks, applied after the name filters.
            It takes a prefix of this package's own ordering, not a seeded
            sample, so a capped run is an arbitrary subset that happens to be
            stable -- and on this dataset that ordering is not language
            balanced, so a small cap will not be. It is also not comparable
            across the variants: the two packages hold the same 731 instances
            in different orders, and the first 20 of each have no instance in
            common (measured; the overlap first becomes non-zero at 50). For
            anything that compares the variants, select with `task_names` and a
            leading wildcard so the same pattern matches both prefixes. Prefer
            `task_names` whenever the subset matters at all.
        solver: Scaffold to run. `None` keeps the adapter's own `react` agent,
            which is what the audited adapter ships; see the module docstring
            on why that is not comparable to a published number. A `--solver`
            on the command line replaces the solver after this function has
            run, so it does not reach this argument; `log.eval.solver` is the
            record of what actually ran.
        sandbox_env_name: Sandbox provider name. The compose configuration is
            synthesised by the adapter either way.
        override_cpus: CPUs per container, overriding the task's own. Every
            task declares 1, and on a benchmark that compiles Go and builds
            JavaScript bundles that is the binding constraint on wall-clock
            time. Raising it makes the benchmark easier against its own
            declared verifier timeout, so a run that sets it is not comparable
            to one that does not.
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
            making the per-sample ceiling `1.5 x time_limit`. Every task here
            declares a 3000 second verifier timeout and the whole scorer stack
            shares that half, so 6000 is not enough and 6600
            (`RECOMMENDED_TIME_LIMIT_SEC`) is the floor; a verifier cancelled by
            that budget errors the sample rather than reporting
            `verifier_failed`. This warns when the budget does not clear a
            selected task's declared timeout by `SCORING_OVERHEAD_SEC`.
        epochs: Attempts per task. Above 1, read infrastructure failures off
            `verifier_failed` and not off `reward.unscored_samples`: inspect's
            epoch reducer drops a NaN epoch rather than carrying it into the
            reduced score.
        allow_unpinned: Permit a tag or revision as `ref`, with a warning.

    Raises:
        ImportError: If `inspect_harbor` is not installed; the message says
            what to install and on which interpreter.
        ValueError: On a non-digest `ref`, an unknown `sandbox_env_name`, a
            non-positive `n_tasks`, `epochs`, `override_cpus`,
            `override_memory_mb` or `time_limit`, a filter that matches no
            task, or a load that does not return exactly 731 samples before
            filtering (which is every load: the filters are applied here, not
            by the adapter, so the count is checked on every build).

    Building this task downloads the task definitions (about 60 MB) to
    `~/.cache/harbor`. Running it builds one image per task on top of a
    per-task tag of `jefzda/sweap-images`, and those tags average about 1.5 GB
    compressed: a full 731-task sweep is of the order of a terabyte of pulls.
    `scripts/validate_swe_bench_pro.py --check oracle-plan` prints the
    measured arithmetic.
    """
    return _build(
        variant="swe_bench_pro",
        package=SCALE_PACKAGE,
        ref=ref,
        task_names=task_names,
        exclude_task_names=exclude_task_names,
        n_tasks=n_tasks,
        solver=solver,
        sandbox_env_name=sandbox_env_name,
        override_cpus=override_cpus,
        override_memory_mb=override_memory_mb,
        overwrite_cache=overwrite_cache,
        time_limit=time_limit,
        epochs=epochs,
        allow_unpinned=allow_unpinned,
    )


@task
def swe_bench_pro_isolated(
    ref: str = CAIS_DIGEST,
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
    """SWE-bench Pro, CAIS anti-exploitation packaging, pinned and instrumented.

    The same 731 instances as `swe_bench_pro`, from `cais/swebenchpro`, in
    images that additionally isolate the repository's git history and block
    GitHub over the network. Sample ids carry a `cais/` prefix rather than
    `scale-ai/`; everything about this wrapper -- arguments, scorers, metric
    keys, diagnostics -- is identical.

    The datasets are not. Besides the container, the problem statements differ:
    not one of the 731 is byte-identical to its counterpart in `swe_bench_pro`,
    because this packaging strips the SWE-agent scaffolding and numbered
    procedure that the plain one wraps every issue in. Read a gap between the
    two variants' scores as a prompt difference and an isolation difference
    together; see the module docstring.

    **What `--check network` measured about enforcement.** All 731 tasks in
    this variant declare `[environment].network_mode = public` (migrated by
    harbor from the deprecated `allow_internet = true`), and **not one declares
    `allowlist`** -- which matters because `inspect_harbor` silently downgrades
    `allowlist` to public without isolating anything
    (`_harbor/converters.py:383-394`, warned at `_harbor/task.py:287-302`), so
    a variant that relied on it would have had its defining property quietly
    unenforced under this adapter. It does not rely on it. The blocking is
    inside the image: all 731 Dockerfiles set `ENTRYPOINT ["/entrypoint.sh"]`,
    and all 731 ship a byte-identical `entrypoint.sh` (sha256
    `9c070238...`, asserted by the check) that appends `0.0.0.0` mappings for
    `github.com`, `raw.githubusercontent.com`, `api.github.com`,
    `objects.githubusercontent.com` and `codeload.github.com` to
    `/etc/hosts` before `exec "$@"`.

    That mechanism survives the adapter's compose translation, which is the
    part that could have gone wrong and was checked rather than assumed: the
    adapter sets no `entrypoint` on its synthesised service, so the image's
    entrypoint still runs; it sets `command: tail -f /dev/null`, which
    `exec "$@"` then runs; and inspect's docker sandbox brings the container up
    with `compose up --detach --wait` and runs every later tool call through
    `docker compose exec`, all of which see the modified `/etc/hosts`.

    The git-history isolation is likewise in the image, on all 731 tasks, and
    needs nothing from the adapter at all. It is a relocation and not a
    removal: the Dockerfile runs
    `cp -a /app/.git /var/lib/apt/.a8f1c && rm -rf /app/.git` and re-initialises
    `/app` as a one-commit repository, and the verifier moves the real history
    back before running the tests. So the complete history, fixing commit
    included, is inside the container the agent has root in, at a path this
    docstring and `--check network`'s output both publish, and
    `git --git-dir=/var/lib/apt/.a8f1c log --all -p` reads it. That the fixing
    commit is really there is not an inference: the verifier checks the
    instance's own commit out of that history with
    `git checkout <sha> -- <test paths>` on 731 of 731 tasks, and no task's
    setup contains a `git fetch` or a `git clone`, so the sha resolves from
    local objects (both asserted by `--check network`).

    What neither mechanism is, because it is not true: this is not isolation.
    The network half is five hostnames in `/etc/hosts`, in a container the agent
    has root in, and an IP literal, any GitHub mirror or proxy, a package
    registry that vendors the same source, or a one-line edit to `/etc/hosts`
    all get around it. The git half is a directory rename that `git --git-dir`
    undoes. Both raise the cost of looking up the answer; neither closes the
    route. Read a large gap between the two variants' scores as evidence about
    the routes rather than about capability, and remember that the prompts
    differ too.

    Args:
        ref: harbor dataset ref. Defaults to the pinned digest and must be a
            digest unless `allow_unpinned` is set.
        task_names: Task names to include; `fnmatch` glob patterns are
            accepted, and the names carry the `cais/` prefix (for example
            `cais/instance_ansible__ansible-*`). `None` runs all 731. See
            `swe_bench_pro.task_names` for the bare-string and leading-wildcard
            spellings on the command line.
        exclude_task_names: Task names to exclude, same matching, applied after
            `task_names`.
        n_tasks: Cap on the number of tasks, applied after the name filters, as
            an unbalanced prefix of this package's own ordering. That ordering
            is not the plain variant's: the first 20 tasks of the two variants
            have no instance in common, so `n_tasks` never selects the same
            work in both. Use `task_names` with a leading wildcard to compare
            them.
        solver: Scaffold to run. `None` keeps the adapter's own `react` agent.
        sandbox_env_name: Sandbox provider name.
        override_cpus: CPUs per container, overriding the task's own 1.
        override_memory_mb: Memory per container, overriding the task's own
            4096 *and* the adapter's 6 GB floor.
        overwrite_cache: Re-download the task definitions rather than trusting
            the local harbor cache, which is validated by directory name only.
        time_limit: Wall-clock seconds per sample, or `None` for no limit. Half
            of it is the whole scorer stack's budget against a declared 3000
            second verifier timeout, so 6600 is the floor rather than 6000;
            this warns below it.
        epochs: Attempts per task.
        allow_unpinned: Permit a tag or revision as `ref`, with a warning.

    Raises:
        ImportError: If `inspect_harbor` is not installed.
        ValueError: On a non-digest `ref`, an unknown `sandbox_env_name`, a
            non-positive numeric argument, a filter that matches no task, or a
            load that does not return exactly 731 samples before filtering.

    Running this pulls the same 731 `jefzda/sweap-images` tags as
    `swe_bench_pro` -- the two variants build different layers on top of the
    same bases -- so a host that has already run one variant pays only the
    build cost for the other.
    """
    return _build(
        variant="swe_bench_pro_isolated",
        package=CAIS_PACKAGE,
        ref=ref,
        task_names=task_names,
        exclude_task_names=exclude_task_names,
        n_tasks=n_tasks,
        solver=solver,
        sandbox_env_name=sandbox_env_name,
        override_cpus=override_cpus,
        override_memory_mb=override_memory_mb,
        overwrite_cache=overwrite_cache,
        time_limit=time_limit,
        epochs=epochs,
        allow_unpinned=allow_unpinned,
    )
