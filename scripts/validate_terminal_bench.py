"""Prove the Terminal-Bench 2.1 wrapper before any Docker image is pulled.

Six checks. None of them starts a container, and none of them needs a model
provider or an API key. Three reach the network: `load` and `compose`
download the pinned task definitions to `~/.cache/harbor` (about 60 MB, once),
and `provenance` queries the harbor registry.

    load           The pinned digest still resolves to exactly 89 tasks, every
                   sample carries a docker sandbox spec, no sample id came out
                   of the adapter's unstable disambiguator, the name filters
                   and the cap select what they say they select, and no task
                   declares a feature the adapter blocks on or silently
                   degrades. Warnings are captured and classified rather than
                   left to scroll past.
    compose        The container specification the adapter synthesises for
                   each task, next to what the task asked for: memory raised
                   by the adapter's 6 GB floor, network mode (and whether an
                   allowlist was downgraded to public), healthcheck grace
                   period, entrypoint. This is the fidelity ledger.
    provenance     What can actually be asserted about the artifact: harbor
                   package, content digest, registry revision, publication
                   date, and the pinned digest's own 89 content-addressed task
                   directories on disk. Also what cannot: harbor records no
                   source repository or commit anywhere, so the upstream git
                   commit is *not* provenance of this dataset, and this check
                   says where the commit in our metadata does come from.
    cross-harness  Inventory of an official harbor-CLI run directory tree
                   (`--reference`), the task-name overlap with our 89, and a
                   per-task reward diff where our own eval logs exist. With no
                   logs yet it reports the inventory, prints a NOTE and
                   passes.
    oracle-plan    The exact command for the deferred oracle sweep and its
                   sizing. Prints; runs nothing.
    scorer         The seam itself, offline: the two guards refuse an
                   unpinned ref and name the missing extra, and
                   `harbor_reward` is fed an inner scorer that raises each
                   exception type the adapter can raise. It has to turn every
                   one of them into NaN plus a flag rather than an errored
                   sample, with the diagnostics scorer still producing a full
                   key set afterwards, and it has to let a `BaseException`
                   through untouched. It finishes by running the whole stack
                   under a real `react` agent on mockllm, because the two
                   diagnostics that read the agent's shape cannot be checked
                   against a hand-built message list: react rewrites that list
                   before scoring.

    python scripts/validate_terminal_bench.py --check all
    python scripts/validate_terminal_bench.py --check load --check scorer
    python scripts/validate_terminal_bench.py --check cross-harness \\
        --reference path/to/results/tb2__nosteer --logs logs/tb2-oracle

`--check` may be repeated. `all` runs five of the six: `cross-harness` needs
`--reference`, because there is nothing to inventory without it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from harbor_common import (
    HarborState,
    harbor_diagnostics,
    harbor_installed,
    harbor_reward,
    harbor_version,
    require_pinned_ref,
)
from inspect_ai.dataset import Sample
from inspect_ai.event import SandboxEvent
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import ChatMessageAssistant, ModelName, ModelOutput, get_model
from inspect_ai.scorer import Score, Scorer, Target
from inspect_ai.solver import TaskState
from terminal_bench_2 import (
    REFERENCE_RUN_UPSTREAM_COMMIT,
    TB2_DIGEST,
    TB2_N_TASKS,
    TB2_PACKAGE,
    terminal_bench_2,
)

CHECKS = ("load", "compose", "provenance", "cross-harness", "oracle-plan", "scorer")

EXPECTED_REVISION = 6
"""Registry revision of the pinned digest, read from the dataset_version row."""

EXPECTED_PUBLISHED_AT = "2026-04-30T06:09:46.159004+00:00"
"""Publication timestamp of that revision. Asserted because a re-publication
under the same digest would be a fact worth stopping for."""

EXPECTED_TASK_DIR_ENTRIES = {
    "README.md",
    "instruction.md",
    "task.toml",
    "environment",
    "solution",
    "tests",
}
"""Everything a downloaded Terminal-Bench 2.1 task directory contains.

Enumerated so that `provenance` can state, rather than assume, that no
provenance file ships with the tasks: if a future revision adds one, this set
stops matching and the check says so instead of repeating a stale claim.
"""

MIN_MEMORY_MB = 6144
"""inspect_harbor's floor, from `_harbor/converters.py:64-70`. Reproduced here
so the compose check can say which tasks it raised without importing a private
constant that may move."""

ALLOWLIST_NETWORK_MODE = "allowlist"
"""The `[environment].network_mode` value inspect_harbor cannot enforce.

It downgrades it to public without isolating anything
(`_harbor/converters.py:383-394`, applied at `:150`) and only warns
(`_harbor/task.py:287-302`). No Terminal-Bench 2.1 task declares it, so the
compose check's assertion is inert here and load-bearing for the next harbor
dataset -- which is exactly the situation in which a check that cannot fire
looks like a check that passed, so `_prove_allowlist_detection` fires it.
"""

REFERENCE_DATASET_VERSION = "2.0"
"""What the internal reference run's config.json records. Ours is 2.1."""

REWARD_SCORER_NAME = "harbor_reward"
"""How the reward scorer's keys are labelled in an eval log."""

DIGEST_DIR_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class Failure(Exception):
    """A check failed. Carries the sentence a run report would want to quote."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def columns(values: Sequence[str], width: int = 4) -> str:
    """Lay a long list of names out in fixed columns, for readable output."""
    if not values:
        return "  (none)"
    pad = max(len(value) for value in values) + 2
    lines = []
    for start in range(0, len(values), width):
        row = values[start : start + width]
        lines.append("  " + "".join(value.ljust(pad) for value in row).rstrip())
    return "\n".join(lines)


def histogram(values: Iterable[Any]) -> str:
    """`value xcount` pairs, most common first."""
    counts = Counter(values)
    return ", ".join(f"{value} x{count}" for value, count in counts.most_common())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_samples(ref: str, **kwargs: Any) -> list[Sample]:
    """Build the task at `ref` and return its samples.

    Callers run this inside a `catch_warnings` block so that the adapter's own
    warnings -- the ones that say a task's declared configuration was not
    wired up -- are data for the check rather than lines that scrolled past on
    a busy terminal.
    """
    task = terminal_bench_2(ref=ref, **kwargs)
    require(
        task.name == "terminal_bench_2",
        f"task registered as {task.name!r}, expected 'terminal_bench_2': the "
        "adapter's own registration would have won, and eval logs would name "
        "the wrong eval",
    )
    return list(task.dataset)


def check_load(ref: str) -> None:
    """89 samples, all sandboxed, no blocked or degraded feature, ids stable."""
    if not harbor_installed():
        raise Failure(
            "inspect_harbor is not installed; install the extra with "
            "pip install -e '.[harbor]' on a Python 3.12+ interpreter"
        )
    require_pinned_ref(ref)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        samples = load_samples(ref)
        recorded = list(caught)

    require(
        len(samples) == TB2_N_TASKS,
        f"expected {TB2_N_TASKS} samples at {ref}, got {len(samples)}",
    )
    ids = [str(sample.id) for sample in samples]
    require(len(set(ids)) == len(ids), "sample ids are not unique")
    # The adapter disambiguates colliding task names by appending
    # `@sha256(str(task_dir))[:8]`, and that path is the absolute local cache
    # path -- so an id with an `@` in it is an id that differs between two
    # machines running the same pinned dataset. All 89 names are unique here,
    # so the branch must never fire.
    collided = [sample_id for sample_id in ids if "@" in sample_id]
    require(
        not collided,
        "sample ids went through the adapter's machine-dependent "
        f"disambiguator, so they are not stable across machines: {collided}",
    )
    for sample in samples:
        require(
            sample.sandbox is not None,
            f"{sample.id} carries no sandbox spec, so it would run on the host",
        )
        require(
            sample.sandbox.type == "docker",
            f"{sample.id} sandbox type is {sample.sandbox.type!r}, expected 'docker'",
        )
        require(
            sample.sandbox.config is not None,
            f"{sample.id} has a docker sandbox with no compose configuration",
        )
        require(
            bool(str(sample.input).strip()),
            f"{sample.id} has an empty instruction",
        )

    print(f"  {len(samples)} samples at {ref}")
    print(f"  inspect_harbor {harbor_version()}")
    print("  every sample carries a docker sandbox spec and a compose config")
    print("  no sample id used the adapter's machine-dependent disambiguator")

    # Warning triage. The adapter raises NotImplementedError for a blocking
    # feature (multi-step tasks, Windows containers, prior-context tasks), so
    # any of those would have surfaced as an exception above rather than here.
    # What arrives as a warning is either a degraded-fidelity notice, which is
    # a change in the instrument and fails this check, or harbor's own
    # deprecation notices, which are expected and counted.
    deprecations = [w for w in recorded if issubclass(w.category, DeprecationWarning)]
    degraded = [
        w
        for w in recorded
        if issubclass(w.category, UserWarning)
        and "not wired up by inspect_harbor" in str(w.message)
    ]
    ours = [
        w
        for w in recorded
        if issubclass(w.category, UserWarning)
        and str(w.message).startswith("terminal_bench_2 is running with no time_limit")
    ]
    other = [
        w
        for w in recorded
        if w not in deprecations and w not in degraded and w not in ours
    ]
    allow_internet = [
        w for w in deprecations if "allow_internet" in str(w.message)
    ]
    print(
        f"  warnings: {len(allow_internet)} allow_internet deprecations "
        f"(expected: every task.toml still carries the deprecated field), "
        f"{len(deprecations) - len(allow_internet)} other deprecations, "
        f"{len(ours)} from this wrapper (the no-time-limit notice, by design)"
    )
    if degraded:
        for message in degraded:
            print(f"  DEGRADED: {message.message}")
    require(
        not degraded,
        "inspect_harbor reported degraded fidelity for at least one task "
        "(fields declared in task.toml that it does not wire up). The "
        "messages above name every affected task. This is a change in the "
        "instrument, not a warning to skip",
    )
    for message in other:
        print(f"  note: {message.category.__name__}: {message.message}")

    _check_filters(ref, ids)

    # Full sample ids, not the bare names. This listing is what the task's own
    # "no task matched" error points an operator at, and `task_names` matches
    # the prefixed id, so printing the stripped name here would hand out
    # patterns that match nothing.
    print("  task names, in the form task_names/exclude_task_names match:")
    print(columns(sorted(ids), width=3))


def _check_filters(ref: str, ids: Sequence[str]) -> None:
    """The name filters and the cap select what the docstrings say they do.

    Checked because they cannot be delegated: `inspect_harbor` 0.7.4 rejects
    `dataset_task_names` and `dataset_exclude_task_names` alongside a package
    name outright (`_harbor/task.py:145-162`), so the wrapper matches its own
    samples, and a wrapper that filters is a wrapper that can filter wrongly.
    Cheap: the dataset is already downloaded, so each of these is a rebuild
    from cache.
    """
    first, second = sorted(ids)[:2]
    prefix = first.rsplit("/", 1)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = load_samples(ref, task_names=[first])
        pair = load_samples(ref, task_names=[first, second])
        globbed = load_samples(ref, task_names=[f"{prefix}/*"])
        excluded = load_samples(ref, exclude_task_names=[first])
        combined = load_samples(ref, task_names=[f"{prefix}/*"], n_tasks=5)
        try:
            load_samples(ref, exclude_task_names=[f"{prefix}/*"])
        except ValueError as error:
            empty_message = str(error)
        else:
            empty_message = ""

    require(
        [str(s.id) for s in one] == [first],
        f"task_names=[{first!r}] selected {[str(s.id) for s in one]}",
    )
    require(
        {str(s.id) for s in pair} == {first, second},
        "a two-name task_names filter did not select exactly those two",
    )
    require(
        len(globbed) == len(ids),
        f"task_names=['{prefix}/*'] selected {len(globbed)} of {len(ids)}: the "
        "glob does not match the prefix every sample id carries",
    )
    require(
        len(excluded) == len(ids) - 1 and first not in {str(s.id) for s in excluded},
        f"exclude_task_names=[{first!r}] did not remove exactly that one task",
    )
    require(
        len(combined) == 5,
        f"n_tasks=5 after a matching glob returned {len(combined)}, not 5",
    )
    require(
        "No task matched" in empty_message,
        "excluding every task did not raise this wrapper's own error; got "
        f"{empty_message!r}",
    )
    print(
        "  filters: one name selects 1, two select 2, the org glob selects all "
        f"{len(ids)},"
    )
    print(
        "    an exclusion removes exactly one, n_tasks caps afterwards, and "
        "excluding"
    )
    print("    everything raises this wrapper's own error rather than harbor's")


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def _declared(sample: Sample) -> dict[str, Any]:
    """The task's own `[environment]` block, from the sample metadata."""
    config = (sample.metadata or {}).get("harbor_config") or {}
    environment = config.get("environment") or {}
    return environment if isinstance(environment, dict) else {}


def _enum_value(value: Any) -> Any:
    """The value of a str-Enum, or the value itself.

    `harbor_config` is a pydantic `model_dump()` in python mode, so an enum
    field arrives as the member and not as its value. `NetworkMode.ALLOWLIST`
    compares equal to `"allowlist"`, but `str()` of it is
    `"NetworkMode.ALLOWLIST"`, so a check written around `str(...)` silently
    never fires and a histogram written around it prints the member name. Both
    of those happened here; this is the single place that normalises.
    """
    return getattr(value, "value", value)


def _prove_allowlist_detection() -> None:
    """Fire the allowlist detection on a fabricated sample, then report it.

    Without this the check below reports "allowlist downgraded: 0 tasks" on a
    dataset where none is declared, which is equally what it would print if the
    comparison were broken -- and it was: `str(NetworkMode.ALLOWLIST)` is
    `'NetworkMode.ALLOWLIST'`, not `'allowlist'`. A measurement whose
    instrument is never exercised is not a measurement.
    """
    from harbor.models.task.config import EnvironmentConfig

    declared = EnvironmentConfig(network_mode=ALLOWLIST_NETWORK_MODE).model_dump()
    sample = Sample(
        input="fixture",
        id="terminal-bench/allowlist-fixture",
        metadata={"harbor_config": {"environment": declared}},
    )
    require(
        _enum_value(_declared(sample).get("network_mode")) == ALLOWLIST_NETWORK_MODE,
        "the allowlist detection does not fire on a sample that declares "
        "allowlist, so the assertion below proves nothing: "
        f"{_declared(sample).get('network_mode')!r}",
    )
    print(
        "  allowlist detection fired on a fabricated allowlist sample, so the "
        "count below is a measurement"
    )


def check_compose(ref: str) -> None:
    """What each container is actually given, next to what the task asked for."""
    if not harbor_installed():
        raise Failure("inspect_harbor is not installed; see --check load")
    require_pinned_ref(ref)
    _prove_allowlist_detection()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = load_samples(ref)

    header = (
        f"  {'task':38} {'svc':>3} {'cpus':>5} {'mem':>7} {'declared':>9} "
        f"{'floored':>7} {'network':>8} {'hc':>4} {'command':<22}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    floored: list[str] = []
    downgraded: list[str] = []
    healthchecks: list[str] = []
    images: list[str] = []
    declared_memory: list[int] = []
    declared_cpus: list[Any] = []
    agent_timeouts: list[float] = []
    verifier_timeouts: list[float] = []
    multi_service: list[str] = []

    for sample in sorted(samples, key=lambda s: str(s.id)):
        name = str(sample.id).split("/")[-1]
        config = sample.sandbox.config
        services = config.services
        if len(services) != 1:
            multi_service.append(name)
        service = next(iter(services.values()))
        environment = _declared(sample)
        declared_mb = environment.get("memory_mb")
        network_mode = _enum_value(environment.get("network_mode"))
        healthcheck = environment.get("healthcheck")
        granted_mb = service.mem_limit
        was_floored = (
            isinstance(declared_mb, int)
            and granted_mb == f"{MIN_MEMORY_MB}m"
            and declared_mb < MIN_MEMORY_MB
        )
        if was_floored:
            floored.append(name)
        if network_mode == ALLOWLIST_NETWORK_MODE:
            downgraded.append(name)
        if healthcheck:
            healthchecks.append(name)
        if service.image:
            images.append(service.image)
        if isinstance(declared_mb, int):
            declared_memory.append(declared_mb)
        declared_cpus.append(environment.get("cpus"))
        harbor_config = (sample.metadata or {}).get("harbor_config") or {}
        agent_timeouts.append((harbor_config.get("agent") or {}).get("timeout_sec"))
        verifier_timeouts.append(
            (harbor_config.get("verifier") or {}).get("timeout_sec")
        )
        start_period = ""
        if isinstance(healthcheck, dict):
            start_period = str(healthcheck.get("start_period_sec", "?"))
        print(
            f"  {name:38.38} {len(services):>3} {str(service.cpus):>5} "
            f"{str(granted_mb):>7} {str(declared_mb):>9} "
            f"{('yes' if was_floored else 'no'):>7} "
            f"{str(_enum_value(service.network_mode)):>8} {start_period or '-':>4} "
            f"{str(service.command)[:22]:<22}"
        )

    print()
    print(f"  distinct images         : {len(set(images))} over {len(images)} tasks")
    print(f"  declared memory_mb      : {histogram(declared_memory)}")
    print(f"  declared cpus           : {histogram(declared_cpus)}")
    print(
        f"  memory floored to {MIN_MEMORY_MB} : {len(floored)} of {len(samples)} tasks "
        f"({sum(declared_memory) / 1024:.0f} GiB declared, "
        f"{sum(max(mb, MIN_MEMORY_MB) for mb in declared_memory) / 1024:.0f} GiB "
        "granted)"
    )
    declared_networks = histogram(
        _enum_value(_declared(s).get("network_mode")) for s in samples
    )
    compose_networks = histogram(
        _enum_value(next(iter(s.sandbox.config.services.values())).network_mode)
        for s in samples
    )
    print(f"  network_mode (declared) : {declared_networks}")
    print(f"  network_mode (compose)  : {compose_networks}")
    print(f"  allowlist downgraded    : {len(downgraded)} tasks {downgraded}")
    print(f"  healthchecks declared   : {len(healthchecks)} tasks {healthchecks}")
    print(f"  multi-service tasks     : {len(multi_service)} {multi_service}")
    print(f"  agent timeout_sec       : {histogram(agent_timeouts)}")
    print(
        f"                            sum {sum(t or 0 for t in agent_timeouts):.0f}s, "
        "NOT enforced by inspect_harbor"
    )
    print(f"  verifier timeout_sec    : {histogram(verifier_timeouts)}")
    print("                            enforced, per sample, by the adapter's scorer")

    require(
        not multi_service,
        f"tasks with more than one compose service: {multi_service}. The "
        "wrapper's fidelity notes assume a single 'default' service per task",
    )
    require(
        not downgraded,
        "inspect_harbor downgrades `network_mode = allowlist` to public "
        f"without enforcing anything, and these tasks declare it: {downgraded}. "
        "Their network isolation would be silently absent",
    )
    require(
        not healthchecks,
        "tasks declaring a healthcheck change inspect's compose startup "
        f"budget from the flat 600s default to start_period + 135s: "
        f"{healthchecks}. Re-derive the threshold before trusting a run"
    )
    print()
    print("  Fidelity summary: the memory floor is the only deviation that")
    print("  bites this dataset. allowlist downgrade, HOST_* compose variable")
    print("  expansion and the healthcheck startup cliff are all inert here")
    print("  (no allowlist, no compose file, no healthcheck) and have to be")
    print("  re-checked for any other harbor dataset.")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def check_provenance(ref: str) -> None:
    """Assert what harbor records, and state plainly what it does not."""
    if not harbor_installed():
        raise Failure("inspect_harbor is not installed; see --check load")
    require_pinned_ref(ref)
    from harbor.constants import PACKAGE_CACHE_DIR
    from harbor.db.client import RegistryDB

    org, name = TB2_PACKAGE.split("/")
    try:
        package, version = asyncio.run(
            RegistryDB().resolve_dataset_version(org, name, ref)
        )
    except Exception as error:  # noqa: BLE001 - any registry failure is one failure to report
        raise Failure(
            f"the harbor registry did not resolve {TB2_PACKAGE} at {ref}: "
            f"{type(error).__name__}: {error}"
        ) from error

    print("  read from the harbor registry (supabase), table dataset_version")
    print(f"    package        : {package['org']['name']}/{package['name']}")
    print(f"    type/visibility: {package['type']} / {package['visibility']}")
    print(f"    content_hash   : sha256:{version['content_hash']}")
    print(f"    revision       : {version['revision']}")
    print(f"    published_at   : {version['published_at']}")
    print(f"    yanked_at      : {version['yanked_at']}")
    print(f"    authors        : {version['authors']}")

    require(
        f"{package['org']['name']}/{package['name']}" == TB2_PACKAGE,
        f"registry resolved a different package: {package}",
    )
    require(
        f"sha256:{version['content_hash']}" == ref,
        f"registry returned content_hash {version['content_hash']}, expected {ref}",
    )
    require(
        version["revision"] == EXPECTED_REVISION,
        f"expected registry revision {EXPECTED_REVISION}, got {version['revision']}",
    )
    require(
        str(version["published_at"]) == EXPECTED_PUBLISHED_AT,
        f"expected published_at {EXPECTED_PUBLISHED_AT}, got {version['published_at']}",
    )
    require(
        version["yanked_at"] is None,
        f"this dataset version has been yanked: {version['yanked_reason']}",
    )

    # What harbor does not record. Asserted rather than asserted-in-prose: if
    # a future harbor adds a source field, this check stops saying "no source
    # is recorded" and starts failing, which is the right way round.
    source_fields = sorted(
        key
        for key in {**package, **version}
        if any(word in key.lower() for word in ("git", "commit", "repo", "source"))
    )
    print(f"    source fields  : {source_fields or 'none'}")
    require(
        not source_fields,
        f"the registry now records source fields {source_fields}: the "
        "docstring claim that harbor cannot tell us the upstream commit is "
        "out of date, and the wrapper metadata should be revisited",
    )

    # This resolves the pinned digest to its own 89 per-task content hashes and
    # looks for exactly those directories, rather than counting whatever the
    # org has cached. A machine that had also run Terminal-Bench 2.0 would
    # satisfy a count with the wrong bytes.
    from harbor.models.job.config import DatasetConfig

    try:
        task_configs = asyncio.run(
            DatasetConfig(name=TB2_PACKAGE, ref=ref).get_task_configs()
        )
    except Exception as error:  # noqa: BLE001 - any registry failure is one failure to report
        raise Failure(
            f"the registry did not enumerate the tasks in {TB2_PACKAGE} at "
            f"{ref}: {type(error).__name__}: {error}"
        ) from error

    pinned = [config.get_task_id() for config in task_configs]
    require(
        len(pinned) == TB2_N_TASKS,
        f"the pinned digest enumerates {len(pinned)} tasks, expected {TB2_N_TASKS}",
    )
    task_dirs = [(task_id.name, task_id.get_local_path()) for task_id in pinned]
    absent = sorted(name for name, directory in task_dirs if not directory.is_dir())
    cache = Path(PACKAGE_CACHE_DIR) / org
    print(f"  read from the local package cache at {cache}")
    print(
        f"    {len(task_dirs) - len(absent)} of {len(task_dirs)} task "
        "directories named by this digest's own per-task content hashes are on "
        "disk"
    )
    require(
        not absent,
        f"{len(absent)} task directories named by the pinned digest are not "
        f"cached: {absent[:5]}. Run --check load first, and note that harbor "
        "decides a cache hit from the directory name alone",
    )
    other_versions = [
        child
        for parent in sorted(cache.iterdir())
        if parent.is_dir()
        for child in sorted(parent.iterdir())
        if child.is_dir()
        and DIGEST_DIR_PATTERN.match(child.name)
        and child not in {directory for _, directory in task_dirs}
    ]
    print(
        f"    {len(other_versions)} further content-addressed directories under "
        f"{cache} belong to other dataset versions and are not read here"
    )
    unexpected = sorted(
        {
            entry.name
            for _, directory in task_dirs
            for entry in directory.iterdir()
            if entry.name not in EXPECTED_TASK_DIR_ENTRIES
        }
    )
    print(f"    entries per task directory: {sorted(EXPECTED_TASK_DIR_ENTRIES)}")
    require(
        not unexpected,
        f"downloaded task directories now contain {unexpected}; if one of "
        "those is a provenance file, this check should read it",
    )

    print()
    print("  Not asserted, and why:")
    print("    harbor records no source repository and no source commit for a")
    print("    dataset version. The registry row above has no such field, and")
    print("    no downloaded task directory ships one. The commit in this")
    print("    eval's task metadata is therefore recorded as")
    print(f"    reference_run_upstream_commit = {REFERENCE_RUN_UPSTREAM_COMMIT}")
    print("    which is provenance of the internal reference harness's run")
    print("    (private steering-tools workspace) at Terminal-Bench 2.0, and")
    print("    not of the 2.1 hub package this eval pins.")


# ---------------------------------------------------------------------------
# Cross-harness
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _job_dirs(reference: Path) -> list[Path]:
    """Job directories under `reference`, or `reference` itself if it is one."""
    if (reference / "result.json").is_file():
        return [reference]
    return sorted(
        child
        for child in reference.iterdir()
        if child.is_dir() and (child / "result.json").is_file()
    )


def _trials(job: Path) -> list[tuple[str, Path]]:
    """(task name, trial directory) for every trial in a harbor job directory.

    Trial directories are named `<task-name>__<short id>`; the task name is
    read from the trial's own lock.json where it exists, and only fallen back
    to the directory name when it does not, because a task name may itself
    contain the separator.
    """
    trials = []
    for child in sorted(job.iterdir()):
        if not child.is_dir():
            continue
        lock = _read_json(child / "lock.json")
        name = None
        if lock and isinstance(lock.get("task"), dict):
            name = lock["task"].get("name")
        if name is None:
            if "__" not in child.name:
                continue
            name = child.name.rsplit("__", 1)[0]
        trials.append((name, child))
    return trials


def _reference_rewards(job: Path) -> tuple[dict[str, float], list[str]]:
    """Per-task reward from `verifier/reward.txt`, and the trials without one."""
    rewards: dict[str, float] = {}
    missing: list[str] = []
    for name, directory in _trials(job):
        path = directory / "verifier" / "reward.txt"
        if not path.is_file():
            missing.append(name)
            continue
        text = path.read_text().strip()
        try:
            rewards[name] = float(text)
        except ValueError:
            missing.append(name)
    return rewards, missing


def _our_rewards(
    logs: Path,
) -> tuple[dict[str, list[float]], dict[str, int], list[str]]:
    """Per-task rewards from our own eval logs, keyed by bare task name.

    Returns the rewards, a per-task count of samples that produced no reward,
    and a note per log that was read differently from the rest.

    NaN is excluded rather than collected. `isinstance(float("nan"), float)` is
    True and `math.isclose(nan, x)` is False, so a verifier failure left in the
    list would poison that task's mean and print in the per-task diff as though
    the two harnesses had scored the task differently. The whole point of the
    NaN is that the sample scored nothing; it is counted separately and
    reported on its own line.

    Errored logs are read too. A log whose last sample errored still carries
    every sample scored before it, and dropping the file would shrink the diff
    silently.
    """
    rewards: dict[str, list[float]] = {}
    unscored: dict[str, int] = {}
    notes: list[str] = []
    for info in list_eval_logs(str(logs)):
        log = read_eval_log(info)
        if not log.samples:
            notes.append(f"{Path(info.name).name}: no samples ({log.status})")
            continue
        if log.status != "success":
            notes.append(
                f"{Path(info.name).name}: status {log.status}, read anyway "
                f"({len(log.samples)} samples)"
            )
        for sample in log.samples:
            score = (sample.scores or {}).get(REWARD_SCORER_NAME)
            if score is None:
                continue
            value = score.value
            reward = value.get("reward") if isinstance(value, dict) else value
            if not isinstance(reward, (int, float)):
                continue
            name = str(sample.id).split("/")[-1]
            if math.isnan(float(reward)):
                unscored[name] = unscored.get(name, 0) + 1
                continue
            rewards.setdefault(name, []).append(float(reward))
    return rewards, unscored, notes


def check_cross_harness(ref: str, reference: Path | None, logs: Path | None) -> None:
    """Inventory an official harbor-CLI run, and diff it if we have run logs."""
    if reference is None:
        raise Failure(
            "--check cross-harness needs --reference <dir>, a harbor-CLI run "
            "directory tree (a job directory containing result.json, or a "
            "parent of several)"
        )
    require(reference.is_dir(), f"--reference {reference} is not a directory")
    jobs = _job_dirs(reference)
    require(bool(jobs), f"no harbor job directory (result.json) under {reference}")

    usable: list[tuple[Path, dict[str, float], set[str]]] = []
    for job in jobs:
        result = _read_json(job / "result.json") or {}
        config = _read_json(job / "config.json") or {}
        stats = result.get("stats", {})
        rewards, missing = _reference_rewards(job)
        trials = _trials(job)
        lock = None
        for _, directory in trials:
            lock = _read_json(directory / "lock.json")
            if lock:
                break
        datasets = config.get("datasets") or [{}]
        agent = (lock or {}).get("agent", {})
        task_lock = (lock or {}).get("task", {})
        print(f"  job {job.name}")
        print(
            f"    trials: {result.get('n_total_trials')} declared, "
            f"{len(trials)} on disk, {len(rewards)} with a reward.txt, "
            f"{len(missing)} without"
        )
        print(
            f"    stats : completed={stats.get('n_completed_trials')} "
            f"errored={stats.get('n_errored_trials')} "
            f"cancelled={stats.get('n_cancelled_trials')} "
            f"pending={stats.get('n_pending_trials')}"
        )
        print(
            f"    agent : {agent.get('name')} on {agent.get('model_name')}, "
            f"timeout_multiplier={(lock or {}).get('timeout_multiplier')}"
        )
        print(
            f"    tasks : {datasets[0].get('name')} v{datasets[0].get('version')} "
            f"from {task_lock.get('type')} {task_lock.get('git_url')} "
            f"@ {task_lock.get('git_commit_id')}"
        )
        if rewards:
            print(
                f"    reward: mean {sum(rewards.values()) / len(rewards):.4f} "
                f"over the {len(rewards)} trials that wrote one, "
                f"{histogram(rewards.values())}"
            )
            # harbor's own headline, over its full trial denominator: a trial
            # that errored before its verifier ran has no reward file at all,
            # so the two means have different denominators and reporting only
            # ours would flatter the reference by a point.
            for name, evaluation in (stats.get("evals") or {}).items():
                for metric in evaluation.get("metrics") or []:
                    print(
                        f"            harbor's own result.json for {name}: "
                        f"mean={metric.get('mean'):.4f} "
                        f"n_trials={evaluation.get('n_trials')} "
                        f"n_errors={evaluation.get('n_errors')}"
                    )
        if missing:
            print(f"    no reward file: {sorted(missing)}")
        if stats.get("n_pending_trials"):
            print("    SKIPPED: this job never finished; not comparable")
            continue
        if len(rewards) < TB2_N_TASKS // 2:
            print("    SKIPPED: too few reward files to be a reference")
            continue
        usable.append((job, rewards, {name for name, _ in trials}))

    require(
        bool(usable),
        f"no finished job under {reference}; every one is abandoned or "
        "missing its verifier rewards",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = load_samples(ref)
    ours = {str(sample.id).split("/")[-1] for sample in samples}

    for job, rewards, names in usable:
        print()
        print(f"  overlap with our {len(ours)} pinned tasks, against {job.name}")
        print(f"    in both        : {len(ours & names)}")
        print(f"    only ours      : {sorted(ours - names) or 'none'}")
        print(f"    only reference : {sorted(names - ours) or 'none'}")
        print(
            f"    of those, {len(set(rewards) & ours)} have a reference reward "
            f"to compare against"
        )

    print()
    print("  What this comparison is and is not:")
    print(
        f"    The reference ran Terminal-Bench {REFERENCE_DATASET_VERSION} "
        "from the git registry; this eval"
    )
    print("    pins the 2.1 hub package. The task names are identical, but")
    print("    2.1's own readme records 26 tasks modified for bug fixes,")
    print("    timeouts, resources and reward-hacking robustness, so a")
    print("    per-task disagreement can be a task change rather than a")
    print("    harness difference.")
    print("    Per-task digests cannot settle which: harbor's git-source")
    print("    digest and the hub content hash are different schemes (the")
    print("    hub hash is not the dirhash of the downloaded directory), so")
    print("    0 of 89 match and that number means nothing.")
    print("    The scaffolds differ too: the reference ran Terminus-2 with")
    print("    per-task agent timeouts enforced, and this eval's default is")
    print("    inspect's react agent with no per-task timeout at all.")
    print("    The reference is a model run, not an oracle run, so its")
    print("    per-task rewards are stochastic outcomes and not ground")
    print("    truth. The usable comparison is the aggregate band -- an")
    print("    unsteered 27B model scored 36 of 89 there -- and a per-task")
    print("    disagreement is a lead, not a bug.")

    if logs is None:
        print()
        print("  NOTE: no --logs given, so nothing of ours was compared. This")
        print("  check reported the reference inventory only. Run the oracle")
        print("  sweep (--check oracle-plan prints the command), then re-run")
        print("  with --logs <dir> for the per-task diff.")
        return

    require(logs.is_dir(), f"--logs {logs} is not a directory")
    mine, unscored, notes = _our_rewards(logs)
    for note in notes:
        print(f"  note: {note}")
    if not mine:
        print()
        print(f"  NOTE: no scored samples found under {logs}. Inventory only.")
        if unscored:
            print(f"  {sum(unscored.values())} of our samples scored NaN: "
                  f"verifier_failed on {sorted(unscored)}")
        return

    job, rewards, _ = usable[0]
    shared = sorted(set(mine) & set(rewards))
    print()
    print(f"  per-task diff against {job.name} over {len(shared)} shared tasks")
    print(f"    {'task':38} {'ours':>8} {'reference':>10}")
    agree = 0
    for name in shared:
        ours_mean = sum(mine[name]) / len(mine[name])
        theirs = rewards[name]
        if math.isclose(ours_mean, theirs):
            agree += 1
        else:
            print(f"    {name:38.38} {ours_mean:>8.3f} {theirs:>10.3f}")
    print(f"    {agree} of {len(shared)} tasks agree exactly")
    ours_only = sorted(set(mine) - set(rewards))
    if ours_only:
        print(f"    scored by us but not by the reference: {ours_only}")
    if unscored:
        # Kept out of the diff on purpose: a sample our verifier could not run
        # is not a disagreement with the reference about the task, it is an
        # absence of a measurement, and averaging its NaN in would print it as
        # one.
        print(
            f"    unscored by us (verifier_failed, not a disagreement): "
            f"{sum(unscored.values())} sample(s) over "
            f"{len(unscored)} task(s): {sorted(unscored)}"
        )
        wholly = sorted(name for name in unscored if name not in mine)
        if wholly:
            print(f"    of those, no epoch produced a reward at all: {wholly}")


# ---------------------------------------------------------------------------
# Oracle plan
# ---------------------------------------------------------------------------


def check_oracle_plan(ref: str) -> None:
    """Print the deferred oracle sweep's command and sizing. Runs nothing."""
    if not harbor_installed():
        raise Failure("inspect_harbor is not installed; see --check load")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = load_samples(ref)

    services = [
        next(iter(sample.sandbox.config.services.values())) for sample in samples
    ]
    images = sorted({service.image for service in services if service.image})
    memory = [
        int(str(service.mem_limit).rstrip("m"))
        for service in services
        if service.mem_limit
    ]

    def _timeout(sample: Sample, phase: str) -> float:
        config = (sample.metadata or {}).get("harbor_config") or {}
        return float((config.get(phase) or {}).get("timeout_sec") or 0.0)

    verifier_timeouts = [_timeout(sample, "verifier") for sample in samples]
    agent_timeouts = [_timeout(sample, "agent") for sample in samples]
    verifier_total = sum(verifier_timeouts)
    # Inspect gives scoring `time_limit / 2` (run.py:2142) and the verifier
    # runs inside scoring, so the limit has to be twice the largest declared
    # verifier timeout before it stops being the thing that fails the sweep. It
    # also has to clear the largest declared agent timeout, for the oracle's own
    # solve.sh, which is the smaller of the two constraints on this dataset.
    time_limit = int(max(2 * max(verifier_timeouts), max(agent_timeouts)))

    print("  Command (needs a Docker daemon; pulls one image per task):")
    print()
    print("    inspect eval \\")
    print(
        "      inspect_evals/terminal_bench_2/terminal_bench_2.py"
        "@terminal_bench_2 \\"
    )
    print("      --solver inspect_harbor/oracle \\")
    print("      --model mockllm/model \\")
    print("      --max-samples 4 \\")
    print(f"      -T time_limit={time_limit} \\")
    print("      --log-dir logs/tb2-oracle")
    print()
    print("  Why each flag:")
    print("    --solver inspect_harbor/oracle runs each task's own solve.sh")
    print("      instead of a model. It ignores solve.sh's exit code and")
    print("      applies no timeout of its own, so the reward is the only")
    print("      signal and -T time_limit is the only bound on a stuck script.")
    print("    --model mockllm/model because the oracle never calls generate,")
    print("      but inspect still requires a --model. Nothing is spent.")
    print("    --max-samples bounds concurrent containers; see the memory")
    print("      arithmetic below.")
    print(f"    -T time_limit={time_limit} is one bound for all {len(samples)} "
          "tasks, and it is")
    print("      not a free choice. inspect spends half of it on scoring")
    print("      (run.py:2142) and the verifier runs inside scoring, so the")
    print(f"      limit is 2 x the largest declared verifier timeout "
          f"({max(verifier_timeouts):.0f}s).")
    print(f"      That also clears the largest declared agent timeout "
          f"({max(agent_timeouts):.0f}s)")
    print("      for the oracle's own solve.sh. A smaller number cancels a")
    print("      slow verifier, and a cancelled verifier errors the sample")
    print("      instead of reporting verifier_failed, so it would look like")
    print("      a broken reference solution rather than a wrong flag.")
    print(f"      Per-sample wall-clock ceiling: {1.5 * time_limit:.0f}s.")
    print()
    print("  Expected result: reward 1.0 on all 89 tasks, verifier_failed 0.")
    print("  Any non-1.0 is explained before any paid run: it is either a")
    print("  broken reference solution upstream or an adapter seam bug, and")
    print("  the second one would silently depress every model's score.")
    print()
    print("  Sizing:")
    print(f"    tasks                 : {len(samples)}")
    print(f"    distinct docker images: {len(images)}")
    print(
        f"    memory granted        : {sum(memory) / 1024:.0f} GiB across all "
        f"tasks; {max(memory) / 1024:.0f} GiB for the largest single container"
    )
    print(
        "    concurrency           : --max-samples N needs about "
        f"N x {sum(memory) / len(memory) / 1024:.1f} GiB of RAM on average"
    )
    print(
        f"    verifier time         : {verifier_total / 3600:.1f} hours if every "
        "verifier ran to its declared timeout (they do not; this is the ceiling)"
    )
    print("    task definitions      : about 60 MB, already in ~/.cache/harbor")
    print("    docker images         : NOT measurable without pulling. The 89")
    print("      images are independent Docker Hub repositories with little")
    print("      shared history, and Terminal-Bench images run from a few")
    print("      hundred MB to several GB each. Budget on the order of")
    print("      100-400 GB of free disk, pull in batches, and measure with")
    print("      `docker system df` after the first batch rather than")
    print("      trusting that range.")
    print()
    print("  Nothing above was executed. No image was pulled.")


# ---------------------------------------------------------------------------
# Scorer seam
# ---------------------------------------------------------------------------


class _StubHarborError(Exception):
    """Stands in for the adapter's own four exception classes.

    They are not imported: importing them would tie this check to the class
    names of a specific adapter release, and the point of the check is that
    the wrapper does not care which type came out. The real classes are
    `CopyTestsDirError`, `VerifierOutputParseError`, `RewardFileNotFoundError`
    and `RewardFileEmptyError`; `RewardFileNotFoundError` subclasses
    `FileNotFoundError`, which is why an `OSError` is in the list below.
    """


ADAPTER_FAILURES: tuple[Exception, ...] = (
    _StubHarborError("tests_dir not found in metadata"),
    _StubHarborError("Failed to copy tests to sandbox"),
    _StubHarborError("reward file is empty"),
    FileNotFoundError("Reward file not found. Test exit code: 2"),
    TimeoutError("Command timed out after 900 seconds"),
    PermissionError("Permission denied"),
    RuntimeError("sandbox write_file failed"),
    OSError("connection to docker daemon lost"),
)
"""Every way the adapter's scorer can raise, in the shapes it raises them.

The first four stand for its own classes; the rest come out of the sandbox
`exec` beneath it. `TimeoutError` is the realistic one on this benchmark,
whose verifier timeouts run to 12,000 seconds.
"""

PASSTHROUGH_FAILURES: tuple[BaseException, ...] = (
    KeyboardInterrupt(),
    asyncio.CancelledError(),
    SystemExit(1),
)
"""What `harbor_reward` must *not* catch.

`except Exception` rather than a bare `except` is a deliberate choice in
`harbor_common.scorer`, and it is load-bearing in one specific place: when a
task carries a `time_limit`, inspect gives scoring half of it and cancels with
`anyio`'s cancelled exception, which is a `BaseException`
(`inspect_ai/util/_limit.py`). Swallowing that would turn a cancelled scope
into a reported reward and leave the cancellation unwound. An operator's
Ctrl-C has the same requirement. Nothing else in the suite would notice a
regression to a bare `except`, so these are asserted directly.
"""


def _stub_scorer(outcome: BaseException | Score) -> Scorer:
    """An inner scorer that raises, or returns, whatever it was given."""

    async def score(state: TaskState, target: Target) -> Score:
        del state, target
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return score


async def _mock_messages() -> list[ChatMessageAssistant]:
    """One assistant turn from mockllm, with a tool call and a submit.

    A finite list of custom outputs, deliberately: a mockllm configured to
    repeat forever turns a failing check into a hanging one.
    """
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="bash",
                tool_arguments={"cmd": "ls /app"},
            ),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "done"},
            ),
        ],
    )
    messages = []
    for _ in range(2):
        output = await model.generate(input="go")
        messages.append(output.message)
    return messages


def _state(messages: Sequence[ChatMessageAssistant]) -> TaskState:
    """A synthetic sample state carrying a harbor task's own metadata shape."""
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="terminal-bench/fixture",
        epoch=1,
        input="fixture instruction",
        messages=list(messages),
        metadata={
            "task_name": "terminal-bench/fixture",
            "harbor_config": {
                "agent": {"timeout_sec": 900.0},
                "verifier": {"timeout_sec": 900.0},
            },
        },
    )
    state.output = ModelOutput.from_content(
        model="mockllm/model", content="done", stop_reason="max_tokens"
    )
    return state


async def _run_stack(
    outcome: BaseException | Score, messages: Sequence[ChatMessageAssistant]
) -> tuple[Score, Score, HarborState]:
    """Score one synthetic sample through the full stack, in task order."""
    state = _state(messages)
    reward = await harbor_reward(_stub_scorer(outcome))(state, Target(""))
    diagnostics = await harbor_diagnostics()(state, Target(""))
    return reward, diagnostics, state.store_as(HarborState)


def _react_diagnostics(
    outputs: Sequence[ModelOutput], message_limit: int | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the real scorer stack under a real `react` agent, offline.

    The fixtures above hand the scorers a message list built by this file, and
    two of the diagnostics cannot be checked that way. `react` rewrites
    `state.messages` before scoring -- it deletes its own submit tool call
    (`inspect_ai/agent/_react.py:383-388`) -- so a hand-built list is a shape
    the shipping path never produces, and asserting against it proved
    `submitted = 1` on a derivation that returned 0 in every real run.
    Sample limits are the same: a `SampleLimitEvent` only exists inside a real
    sample.

    So this builds a one-sample task with no sandbox, drives it with mockllm
    and the real `harbor_reward` / `harbor_diagnostics`, and reads the
    diagnostics back out of the log. It needs no network, no Docker and no
    provider, and takes about a second.

    Returns the diagnostics score value and its metadata.
    """
    from inspect_ai import Task, eval
    from inspect_ai.agent import react
    from inspect_ai.tool import tool

    @tool
    def probe_bash() -> Any:
        async def execute(cmd: str) -> str:
            """Pretend to run a shell command.

            Args:
                cmd: The command that would have run.
            """
            return f"ran {cmd}"

        return execute

    def _inner() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score:
            del state, target
            return Score(value=1.0, answer="PASS", explanation="Test exit code: 0")

        return score

    # Everything is read inside the temporary directory: `eval` returns log
    # objects whose samples are materialised from the log file on the way out,
    # so tearing the directory down first turns this into a FileNotFoundError.
    with TemporaryDirectory() as log_dir:
        logs = eval(
            Task(
                dataset=[Sample(input="do the thing", target="done", id="probe")],
                solver=react(tools=[probe_bash()]),
                scorer=[harbor_reward(_inner()), harbor_diagnostics()],
                message_limit=message_limit,
                name="harbor_seam_probe",
            ),
            model=get_model("mockllm/model", custom_outputs=list(outputs)),
            log_dir=log_dir,
            display="none",
        )
        log = logs[0]
        require(
            bool(log.samples),
            f"the in-process react probe produced no samples: {log.status}",
        )
        score = (log.samples[0].scores or {}).get("harbor_diagnostics")
        require(score is not None, "the react probe produced no diagnostics score")
        return dict(score.value), dict(score.metadata or {})


def _tool_call_output(name: str, **arguments: Any) -> ModelOutput:
    return ModelOutput.for_tool_call(
        model="mockllm/model", tool_name=name, tool_arguments=arguments
    )


def check_guards() -> None:
    """The two refusals: an unpinned ref, and a missing optional dependency."""
    require(
        require_pinned_ref(TB2_DIGEST) == TB2_DIGEST,
        "the pinned digest was not accepted",
    )
    rejected = [
        "latest",
        "6",
        "v2.1",
        TB2_DIGEST.upper(),
        TB2_DIGEST.replace("sha256:", ""),
        TB2_DIGEST[:-1],
        f"{TB2_DIGEST} ",
    ]
    for bad in rejected:
        try:
            require_pinned_ref(bad)
        except ValueError:
            continue
        raise Failure(f"require_pinned_ref accepted {bad!r}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        require_pinned_ref("latest", allow_unpinned=True)
        require(
            any(issubclass(w.category, UserWarning) for w in caught),
            "allow_unpinned=True accepted an unpinned ref without warning",
        )
    print(f"  require_pinned_ref refuses {len(rejected)} non-digest refs, "
          "including 'latest', a revision number, a tag and an uppercase digest")

    # Simulated rather than uninstalled: the message is the whole point of the
    # guard, and it is the one thing an operator on the wrong interpreter will
    # ever see from this module.
    import harbor_common.guard as guard

    original = guard.harbor_installed
    guard.harbor_installed = lambda: False
    try:
        guard.require_harbor()
    except ImportError as error:
        message = str(error)
    else:
        message = ""
    finally:
        guard.harbor_installed = original
    require(bool(message), "require_harbor did not raise with harbor absent")
    for expected in ("pip install -e", "[harbor]", "inspect-harbor==0.7.4"):
        require(
            expected in message,
            f"the missing-dependency error does not mention {expected!r}: {message}",
        )
    print("  require_harbor names the extra, the pin and the install command")


def check_scorer() -> None:
    """The seam: every adapter raise becomes NaN plus a flag, never an error."""
    check_guards()
    messages = asyncio.run(_mock_messages())
    require(
        len(messages) == 2 and any(m.tool_calls for m in messages),
        "the mockllm fixture did not produce tool calls",
    )

    for failure in ADAPTER_FAILURES:
        reward, diagnostics, store = asyncio.run(_run_stack(failure, messages))
        label = type(failure).__name__
        values = reward.value
        require(isinstance(values, dict), f"{label}: reward score is not a dict")
        require(
            set(values) == {"reward", "resolved"},
            f"{label}: reward keys are {sorted(values)}, expected reward+resolved",
        )
        require(
            all(math.isnan(float(value)) for value in values.values()),
            f"{label}: expected NaN on every key, got {values}",
        )
        diagnostic_values = diagnostics.value
        require(
            isinstance(diagnostic_values, dict)
            and diagnostic_values["verifier_failed"] == 1.0,
            f"{label}: diagnostics did not report verifier_failed=1",
        )
        require(
            diagnostic_values["unscored"] == 1.0,
            f"{label}: diagnostics did not report unscored=1",
        )
        require(
            math.isnan(diagnostic_values["reward_fractional"]),
            f"{label}: reward_fractional should be NaN when there is no reward",
        )
        require(
            store.verifier_error_type == label and str(failure) in store.verifier_error,
            f"{label}: the store did not record the exception verbatim",
        )
    print(
        f"  {len(ADAPTER_FAILURES)} adapter failure modes: every one scored "
        "NaN/NaN with verifier_failed=1, and the diagnostics scorer still ran"
    )

    for passthrough in PASSTHROUGH_FAILURES:
        label = type(passthrough).__name__
        try:
            asyncio.run(_run_stack(passthrough, messages))
        except BaseException as raised:  # noqa: BLE001 - the point is what escapes
            require(
                type(raised) is type(passthrough),
                f"{label} came out of harbor_reward as {type(raised).__name__}",
            )
        else:
            raise Failure(
                f"harbor_reward swallowed {label}. It catches Exception rather "
                "than using a bare except precisely so that a cancelled "
                "scoring scope and an operator's Ctrl-C keep unwinding"
            )
    print(
        "  "
        + ", ".join(type(exc).__name__ for exc in PASSTHROUGH_FAILURES)
        + " propagate untouched: the catch is Exception, not BaseException"
    )

    # The adapter's own metadata has to survive the wrapper. Against 0.7.4 it
    # only ever sets `reward_dict`, so the key that proves the passthrough is
    # one the adapter does not set today.
    inner = Score(
        value=1.0,
        answer="PASS",
        explanation="Test exit code: 0",
        metadata={"reward_dict": {"reward": 1.0, "tests": 12}, "future_key": "kept"},
    )
    reward, _, store = asyncio.run(_run_stack(inner, messages))
    require(
        reward.metadata.get("future_key") == "kept",
        "an inner metadata key the wrapper does not know about was dropped: "
        f"{sorted(reward.metadata)}",
    )
    require(
        store.reward_dict == {"reward": 1.0, "tests": 12},
        f"the adapter's reward_dict was not recorded: {store.reward_dict}",
    )
    print(
        "  the adapter's own Score.metadata survives the wrapper, including a "
        "key it does not set today"
    )

    # A reward in a shape that is not a number is the same "cannot be read"
    # outcome as a raise, and has to land in verifier_failed rather than
    # reading as a zero.
    non_numeric = Score(value="PASS", answer="PASS", explanation="Test exit code: 0")
    reward, diagnostics, store = asyncio.run(_run_stack(non_numeric, messages))
    require(
        all(math.isnan(float(value)) for value in reward.value.values()),
        f"a non-numeric adapter reward did not score NaN: {reward.value}",
    )
    require(
        diagnostics.value["verifier_failed"] == 1.0
        and store.verifier_error_type == "NonNumericReward",
        "a non-numeric adapter reward was not recorded as a verifier failure: "
        f"{store.verifier_error_type!r}",
    )
    print(
        "  a non-numeric Score.value from the adapter is NaN plus "
        "verifier_failed=1, not a zero"
    )

    keys: list[set[str]] = []
    for value, expected_resolved, fractional in (
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 1.0),
    ):
        verifier_output = "Test exit code: 0\n\nstdout:\n" + ("x" * 200_000)
        inner = Score(
            value=value,
            answer="PASS" if value > 0 else "FAIL",
            explanation=verifier_output,
        )
        reward, diagnostics, store = asyncio.run(_run_stack(inner, messages))
        keys.append(set(diagnostics.value))
        require(
            reward.value["reward"] == value,
            f"reward {value} was not passed through: {reward.value}",
        )
        require(
            reward.value["resolved"] == expected_resolved,
            f"reward {value} should give resolved={expected_resolved}, "
            f"got {reward.value['resolved']}",
        )
        require(
            diagnostics.value["reward_fractional"] == fractional,
            f"reward {value} should give reward_fractional={fractional}",
        )
        require(
            reward.metadata["verifier_output"] == verifier_output,
            "the verifier's stdout was not preserved verbatim into metadata",
        )
        require(
            store.verifier_exit_code == 0,
            f"the verifier exit code was not read, got {store.verifier_exit_code}",
        )
        require(
            store.adapter_answer == inner.answer,
            "the adapter's own PASS/FAIL verdict was not recorded beside ours",
        )
    print(
        "  reward 1.0/0.0/0.5 -> resolved 1/0/0 (reward>0 would call 0.5 a "
        "pass; reward_fractional flags it instead)"
    )
    print(
        f"  {len(verifier_output):,} characters of verifier output preserved "
        "verbatim into Score.metadata before the adapter deletes /logs/verifier"
    )

    failure_keys = set(
        asyncio.run(_run_stack(ADAPTER_FAILURES[0], messages))[1].value
    )
    require(
        all(key_set == failure_keys for key_set in keys),
        "the diagnostics key set changes between a scored and an unscored "
        f"sample: {failure_keys ^ keys[0]}",
    )
    print(f"  diagnostics key set is identical either way: {sorted(failure_keys)}")

    _, diagnostics, _ = asyncio.run(_run_stack(Score(value=1.0), messages))
    require(
        diagnostics.value["messages"] == 2.0,
        f"messages should count the sample's messages, got {diagnostics.value}",
    )
    require(
        diagnostics.value["declared_agent_timeout_sec"] == 900.0,
        "the task's declared (unenforced) agent timeout was not reported",
    )
    print(
        "  declared_agent_timeout_sec is read off the task's own harbor_config: 900"
    )

    check_unreadable_transcript(messages)
    check_agent_shape()


def check_unreadable_transcript(messages: Sequence[ChatMessageAssistant]) -> None:
    """The fallback for a transcript that cannot be read in full.

    Reachable two ways in a real run -- a torn-down history provider, which
    raises, and bounded mode without a provider, which silently returns a
    truncated tail. Both have to become "could not tell" rather than a small
    number, so the read is faked here by making it raise. Without this the
    fallback is code no check has ever run.
    """
    import harbor_common.scorer as seam

    def unreadable() -> Any:
        raise RuntimeError("history provider is gone")

    baseline = set(asyncio.run(_run_stack(Score(value=1.0), messages))[1].value)
    original = seam.transcript
    seam.transcript = unreadable
    try:
        _, diagnostics, _ = asyncio.run(_run_stack(Score(value=1.0), messages))
    finally:
        seam.transcript = original

    values, metadata = diagnostics.value, diagnostics.metadata or {}
    require(
        math.isnan(values["agent_limit_hit"]),
        f"an unreadable transcript reported agent_limit_hit={values['agent_limit_hit']}"
        ", which claims no limit fired rather than admitting it could not tell",
    )
    require(
        metadata.get("truncation_source") == "final generation only"
        and metadata.get("tool_call_source") == "messages only",
        f"the fallback did not say which reading it used: {metadata}",
    )
    require(
        values["submitted"] == 1.0 and values["tool_calls"] == 2.0,
        "with no transcript the message log is still read, and this fixture's "
        f"messages carry a submit and two calls: {values}",
    )
    require(
        set(values) == baseline,
        f"the fallback changed the diagnostics key set: {set(values) ^ baseline}",
    )
    print(
        "  an unreadable transcript falls back to the messages, says so in the "
        "metadata, and reports agent_limit_hit as NaN rather than 0"
    )


def check_agent_shape() -> None:
    """The agent-shape diagnostics, under a real `react` agent on mockllm.

    Not against a fixture. `react` deletes its own submit tool call from
    `state.messages` before scoring, so a hand-built message list is a shape
    the shipping path never produces: asserting against one reported
    `submitted = 1` while every real run reported 0. Everything here is read
    out of an actual eval log.
    """
    # `stop_reason` is a read-only property over the first choice, so the
    # truncated generation is made by setting it on the choice itself.
    truncated = _tool_call_output("probe_bash", cmd="make")
    truncated.choices[0].stop_reason = "max_tokens"
    values, metadata = _react_diagnostics(
        [
            _tool_call_output("probe_bash", cmd="ls /app"),
            truncated,
            _tool_call_output("submit", answer="done"),
        ]
    )
    require(
        values["submitted"] == 1.0,
        "the agent submitted and the diagnostics did not see it. react removes "
        "the submit tool call from state.messages before scoring, so this key "
        f"has to come from the transcript: {values}",
    )
    require(
        values["tool_calls"] == 3.0,
        f"tool_calls should count all three calls including submit: {values}",
    )
    require(
        values["generations"] == 3.0,
        f"generations should count all three model calls: {values}",
    )
    require(
        values["truncated_generations"] == 1.0 and values["max_tokens_stops"] == 1.0,
        f"the one max_tokens generation was not counted: {values}",
    )
    require(
        values["agent_limit_hit"] == 0.0,
        f"no limit was set and agent_limit_hit is {values['agent_limit_hit']}",
    )
    require(
        metadata.get("tool_call_source") == "transcript"
        and metadata.get("truncation_source") == "transcript",
        f"the diagnostics did not read the transcript: {metadata}",
    )
    require(
        values["verifier_failed"] == 0.0 and values["scorer_ran"] == 1.0,
        f"a clean run reported a verifier failure: {values}",
    )
    print(
        "  under a real react agent: submitted=1, tool_calls=3 (submit "
        "included), generations=3,"
    )
    print("    truncated_generations=1, both sources read from the transcript")

    # A limit that fires. Without this the NaN-versus-zero contract on
    # agent_limit_hit is asserted only in prose.
    limited, limit_metadata = _react_diagnostics(
        [_tool_call_output("probe_bash", cmd=f"step {i}") for i in range(12)],
        message_limit=4,
    )
    require(
        limited["agent_limit_hit"] == 1.0,
        f"a message_limit of 4 did not register as a sample limit: {limited}",
    )
    require(
        bool(limit_metadata.get("limit_types")),
        f"the limit fired but its type was not recorded: {limit_metadata}",
    )
    require(
        limited["submitted"] == 0.0,
        f"an agent stopped by a limit did not submit: {limited}",
    )
    print(
        "    under a message_limit: agent_limit_hit=1, limit_types="
        f"{limit_metadata.get('limit_types')}, submitted=0"
    )
    require(
        math.isnan(values["solution_exit_code"]),
        "an agent run has no reference solution, so solution_exit_code has to "
        f"be NaN rather than a zero that claims one succeeded: {values}",
    )
    print("    no oracle ran, so solution_exit_code is NaN and not 0")

    check_solution_exit_code()


def check_solution_exit_code() -> None:
    """Recovering the exit code the adapter's oracle solver throws away.

    The solver awaits `sandbox().exec` on the task's own solve script and
    discards the result (`inspect_harbor/_harbor/solver.py:53-60`), so on an
    oracle sweep a reference solution that died half way through and one the
    verifier ran and failed are both just `reward = 0`. Inspect keeps the exit
    code on the `SandboxEvent`, and `_solution_exit_code` reads it back.

    Checked against hand-built events rather than a container, because what is
    being asserted is the match rule: which of a sample's several sandbox
    execs is the reference solution. Getting that wrong reports the verifier's
    exit code under the solution's name, which is worse than reporting
    nothing. The four cases below are the four the real transcripts contain.
    """
    from harbor_common.scorer import _solution_exit_code

    def exec_event(cmd: str, result: int) -> SandboxEvent:
        return SandboxEvent(action="exec", cmd=cmd, result=result)

    oracle = exec_event("bash -l /solution/solve.sh", 8)
    verifier = exec_event("bash -l /tests/test.sh", 0)
    setup = exec_event("mkdir -p /logs/verifier", 0)

    nan = float("nan")
    cases: list[tuple[str, list[Any] | None, float]] = [
        ("an unreadable transcript", None, nan),
        ("an agent run with no oracle exec", [verifier, setup], nan),
        ("an oracle run", [oracle, verifier, setup], 8.0),
        ("an oracle run that succeeded", [exec_event("/solution/solve.sh", 0)], 0.0),
        (
            "a solver that shelled into /solution twice",
            [exec_event("bash /solution/setup.sh", 0), oracle],
            8.0,
        ),
    ]
    for label, events, expected in cases:
        got = _solution_exit_code(events)
        ok = math.isnan(got) if math.isnan(expected) else got == expected
        require(
            ok,
            f"{label} should report solution_exit_code={expected}, got {got}",
        )
    print(
        "  solution_exit_code: NaN with no oracle exec, the code with one, and "
        "the verifier's own exec is never mistaken for it"
    )


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # `append`, not `store`: `--check load --check compose` is the natural
    # spelling and with `store` it silently ran only the last one and still
    # printed a confident pass line.
    parser.add_argument(
        "--check",
        action="append",
        dest="checks",
        default=None,
        choices=(*CHECKS, "all"),
        help=(
            "which check to run; repeatable. 'all' (the default) runs every "
            "check it has inputs for"
        ),
    )
    parser.add_argument(
        "--ref",
        default=TB2_DIGEST,
        help="harbor dataset digest to check (default: the pinned one)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="harbor-CLI run directory tree, for --check cross-harness",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        default=None,
        help="our own eval log directory, for the cross-harness per-task diff",
    )
    args = parser.parse_args()

    requested = args.checks or ["all"]
    selected: list[str] = []
    for name in requested:
        expanded = (
            [
                check
                for check in CHECKS
                if check != "cross-harness" or args.reference is not None
            ]
            if name == "all"
            else [name]
        )
        selected.extend(check for check in expanded if check not in selected)

    failures: list[str] = []
    for name in selected:
        print(f"[{name}]")
        try:
            if name == "load":
                check_load(args.ref)
            elif name == "compose":
                check_compose(args.ref)
            elif name == "provenance":
                check_provenance(args.ref)
            elif name == "cross-harness":
                check_cross_harness(args.ref, args.reference, args.logs)
            elif name == "oracle-plan":
                check_oracle_plan(args.ref)
            else:
                check_scorer()
        # ValueError alongside Failure: the guards in harbor_common raise it
        # (a bad --ref is the common case), and a refusal should read the same
        # as every other refusal rather than as a traceback.
        except (Failure, ValueError) as failure:
            print(f"  FAIL: {failure}")
            failures.append(name)
        else:
            print("  PASS")

    if failures:
        print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print(f"\n{len(selected)} check(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
