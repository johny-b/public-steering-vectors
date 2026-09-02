"""Prove the SWE-bench Pro wrappers before any Docker image is pulled.

Seven checks, over both variants. Six of them start no container and need no
model provider and no API key; `verifier-shell` starts one short-lived container
if a Docker daemon and a built Go task image happen to be present, and says so
and skips that half if they are not. Five of the six do need the network, and not
only the first: every check except `scorer` builds the tasks, and building one
resolves the pinned digest against the harbor registry over HTTPS even when
`~/.cache/harbor` is warm (harbor's registry client is constructed on every
build; its httpx deprecation warnings are the ones `load` counts). Only `load`
*downloads* anything, the task definitions for both packages, about 60 MB each
and once. `scorer` and the Docker-free half of `verifier-shell` are the parts
that run on an air-gapped host. Nothing here pulls an image, and the image
sizing in `oracle-plan` is measured constants rather than a live registry
query.

    load             Both pinned digests still resolve to exactly 731 tasks,
                     every sample carries a docker sandbox spec, no sample id
                     came out of the adapter's machine-dependent
                     disambiguator, the name filters and the cap select what
                     they say they select, the two variants hold the same 731
                     instances, and the adapter reports no degraded fidelity
                     for either package. Warnings are captured and classified
                     rather than left to scroll past. Prints all 731 task names
                     per variant, grouped by repository, in the form the
                     filters match: that listing is what the task's own "no
                     task matched" error points an operator at, and it is most
                     of this check's output.
    network          The check the isolated variant exists for. Fires the
                     allowlist detector on a fabricated sample first, through
                     the same accessor the real count uses, so that a zero
                     below is a measurement; then enumerates the declared
                     network mode of all 731 CAIS tasks (and the scale
                     variant's, for contrast) and fails loudly on any
                     `allowlist`, which inspect_harbor downgrades to public
                     without enforcing anything. Then characterises what the
                     variant actually uses instead, by reading all 1462
                     downloaded Dockerfiles and entrypoints, measures how far
                     the git-history relocation is from a removal, and checks
                     that the mechanism survives the adapter's compose
                     translation and inspect's own docker provider.
    reward-semantics The official rule, read out of the shipped verifier.
                     `tests/test.sh` is one byte-identical file across all 731
                     tasks of a variant, so this asserts that first and then
                     quotes the decisive lines: the union rule, the exit code,
                     and the EXIT trap that makes the reward binary -- and what
                     that trap costs, which is that the verifier scores its own
                     infrastructure failures as a clean 0. Reports where the
                     verifier runs and how little is restored before it does,
                     samples task directories across all four languages to
                     confirm the inputs the rule reads, and settles which of
                     `reward` and `resolved` is the headline.
    compose          The container specification the adapter synthesises, next
                     to what the task asked for: the 6 GB memory floor, the
                     timeouts, the command, and the entrypoint -- which is the
                     one that matters here, because Scale warns that bash runs
                     by default in these images and an entrypoint that did not
                     `exec` its arguments would kill every container before the
                     agent reached it. The network mode is in `--check
                     network`, with the rest of the isolation question.
    oracle-plan      The exact commands for a 20-task cross-language oracle
                     subset and for the full 731-task sweep, with measured
                     disk and verifier-time arithmetic. Prints; runs nothing.
    scorer           The seam itself, offline: the guards refuse an unpinned
                     ref and name the missing extra, and `harbor_reward` is fed
                     an inner scorer that raises each exception type the
                     adapter can raise. It finishes by running the whole stack
                     under a real `react` agent on mockllm.

    python scripts/validate_swe_bench_pro.py --check all
    python scripts/validate_swe_bench_pro.py --check network --check compose

`--check` may be repeated; `all` is the default and runs all six.

Two things `scripts/validate_terminal_bench.py` checks that this file does not,
both deliberately. There is no `provenance` check: harbor records no source
repository or commit for a dataset version (asserted there, against the same
registry, and unchanged by which package is asked for), and the digest is the
provenance. There is no `cross-harness` check: no internal reference run of
SWE-bench Pro exists to diff against, which is the audit Terminal-Bench had and
this benchmark does not, and is why the seam was audited there first. The
`scorer` check below also does not repeat that file's unreadable-transcript
fallback: `harbor_common` is shared and unchanged, and re-proving a shared
module twice would only make it look like two independent results.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import math
import re
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from harbor_common import (
    ADAPTER_LOGIN_SHELL_SOURCE,
    HarborState,
    ImagePathSandbox,
    adapter_uses_login_shell,
    harbor_diagnostics,
    harbor_installed,
    harbor_reward,
    harbor_version,
    install_image_path_verifier_shell,
    require_pinned_ref,
    verifier_argv_without_login_shell,
)
from harbor_common.verifier_shell import adapter_verifier_source

# `inspect_ai._cli.util` is private, and imported deliberately: `parse_cli_args`
# is the function that turns a `-T name=value` into what the task function
# receives, and two checks here are about exactly that translation.
# Reimplementing it would check this file's guess at the CLI rather than the CLI.
from inspect_ai._cli.util import parse_cli_args
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ModelName, ModelOutput, get_model
from inspect_ai.scorer import Score, Scorer, Target
from inspect_ai.solver import TaskState
from swe_bench_pro import (
    BASE_IMAGE_REPOSITORY,
    CAIS_DIGEST,
    CAIS_PACKAGE,
    DECLARED_VERIFIER_TIMEOUT_SEC,
    RECOMMENDED_TIME_LIMIT_SEC,
    SCALE_DIGEST,
    SCALE_PACKAGE,
    SCORING_OVERHEAD_SEC,
    SWE_BENCH_PRO_N_TASKS,
    swe_bench_pro,
    swe_bench_pro_isolated,
)

GO_REPOSITORIES = ("flipt", "teleport", "vuls", "navidrome")
"""The four Go source repositories of SWE-bench Pro, by the fragment their task
ids carry.

280 of the 731 tasks, and the ones the adapter's login-shell verifier silently
zeroes: Go is installed at `/usr/local/go/bin`, which `/etc/profile` drops.
`verifier-shell` uses these to find a built image to measure `PATH` in.
"""

CHECKS = (
    "load",
    "network",
    "reward-semantics",
    "compose",
    "oracle-plan",
    "scorer",
    "verifier-shell",
)

VARIANTS: dict[str, dict[str, Any]] = {
    "swe_bench_pro": {
        "factory": swe_bench_pro,
        "package": SCALE_PACKAGE,
        "digest": SCALE_DIGEST,
        "prefix": "scale-ai",
    },
    "swe_bench_pro_isolated": {
        "factory": swe_bench_pro_isolated,
        "package": CAIS_PACKAGE,
        "digest": CAIS_DIGEST,
        "prefix": "cais",
    },
}
"""The two variants, in the order they are checked.

Keyed by the registered task name rather than by the package, because that is
what an eval log carries and what a failure message has to name for the reader
to know which of the two to look at.
"""

ISOLATED = "swe_bench_pro_isolated"
"""The variant whose isolation properties `--check network` exists to measure."""

MIN_MEMORY_MB = 6144
"""inspect_harbor's floor, from `_harbor/converters.py:64-70`. Reproduced here
so the compose check can say which tasks it raised without importing a private
constant that may move."""

ALLOWLIST_NETWORK_MODE = "allowlist"
"""The `[environment].network_mode` value inspect_harbor cannot enforce.

It downgrades it to public without isolating anything
(`_harbor/converters.py:383-394`, applied at `:150`) and only warns
(`_harbor/task.py:287-302`). Whether any SWE-bench Pro task declares it is the
question `--check network` exists to answer, and a check that cannot fire looks
exactly like a check that passed, so `_prove_allowlist_detection` fires it on a
fabricated sample before the real count is believed.
"""

EXPECTED_TASK_DIR_ENTRIES = {
    "instruction.md",
    "task.toml",
    "environment",
    "solution",
    "tests",
}
"""Everything a downloaded SWE-bench Pro task directory contains.

Enumerated so that the checks can state, rather than assume, what ships with a
task. Notably there is no README.md, unlike Terminal-Bench 2.1, and no
provenance file: if a future revision adds one, this set stops matching and the
check says so instead of repeating a stale claim.
"""

BLOCKED_HOSTS = (
    "github.com",
    "raw.githubusercontent.com",
    "api.github.com",
    "objects.githubusercontent.com",
    "codeload.github.com",
)
"""The five names the isolated variant maps to 0.0.0.0 in `/etc/hosts`.

Enumerated rather than counted so that a revision which dropped one -- say
`codeload.github.com`, which is what `git clone` over HTTPS actually fetches
packs from -- fails the check instead of passing it with four.
"""

ISOLATED_ENTRYPOINT_SHA256 = (
    "9c070238e69dc4b025373507ec0123773e97340bb3c03afe4857da329df48122"
)
"""sha256 of the isolated variant's `environment/entrypoint.sh`.

One file, byte-identical on all 731 tasks. Pinned because the whole isolation
property of the variant is those few lines: a task whose entrypoint differed
would be a task with different isolation, and the digest is the cheapest way to
notice.
"""

GIT_ISOLATION_MARKER = "/var/lib/apt/.a8f1c"
"""Where the isolated variant's Dockerfile hides the repository's real `.git`.

The verifier moves it back before running the tests, so this same string has to
appear in both the Dockerfile and `tests/test.sh` of every isolated task; a
task where only one of them had it would score against a repository with no
history at all.
"""

GIT_RELOCATION_COMMAND = f"cp -a /app/.git {GIT_ISOLATION_MARKER} && rm -rf /app/.git"
"""The Dockerfile line that does the git-history "isolation".

Asserted verbatim, because the word in it that matters is `cp`. This is a copy
followed by a delete of the original, not a prune: the complete history stays
in the image at `GIT_ISOLATION_MARKER`, in a container the agent is root in.
The eval's docstrings say so, and this constant is what stops that sentence
drifting back to "an agent cannot read its way to the fixing commit" without a
diff to the file that would make it false.
"""

GOLD_CHECKOUT_PATTERN = re.compile(r"^git checkout ([0-9a-f]{7,40}) -- (.+)$")
"""The last line of every instance's `before_repo_set_cmd`, which `test.sh` runs.

Two things are read out of it. The commit it names is the instance's own, and
it is checked out of the restored history with no network, which is what makes
"the fixing commit is present locally" a measurement rather than an inference.
And the paths after `--` are everything the verifier restores before scoring:
see `_report_gold_restore`.
"""

SAFE_ENTRYPOINT_RESET = "ENTRYPOINT []"
"""What the plain variant's Dockerfile does about the base image's entrypoint."""

EXEC_ARGS = 'exec "$@"'
"""What an entrypoint script must end with to be safe here.

The adapter gives the service `command: tail -f /dev/null` and no entrypoint of
its own, so the image's entrypoint receives that command as its arguments. An
entrypoint that ignores them leaves the container with nothing to run, it exits
immediately, and every sample fails in a way that looks like a broken image
rather than a wrong compose translation.
"""

EXPECTED_COMMAND = "tail -f /dev/null"
"""The command the adapter puts on every synthesised service."""

REWARD_RULE_LINES = (
    "all_required = fail_to_pass | pass_to_pass",
    "success = all_required <= passed_tests",
)
"""The official SWE-bench Pro rule, as it is literally written in `test.sh`.

Matched as text rather than reimplemented. The point of the check is that the
rule this repository claims in its docstrings is the rule the shipped verifier
runs, and a reimplementation could agree with the docstring while both were
wrong about the file.
"""

REWARD_FILE = "/logs/verifier/reward.txt"
"""The file the adapter reads the reward out of."""

REWARD_WRITES = (
    f"echo 1 > {REWARD_FILE}",
    f"echo 0 > {REWARD_FILE}",
)
"""Every write to the reward file anywhere in the shipped verifier.

Both, and nothing else: this is what makes fractional rewards unreachable, and
it is asserted as an exhaustive list rather than as "1 and 0 appear".
"""

LANGUAGE_ORDER = ("python", "js", "ts", "go")
"""Languages in the dataset, in the order the reports list them.

Fixed rather than derived from a set, so that two runs of a check print their
tables in the same order and a diff between them is about the numbers.
"""

# Compressed image size per source repository, in GB, measured on 2026-08-31 by
# reading the manifest of 4 randomly chosen `jefzda/sweap-images` tags per
# repository from the Docker Hub registry API (sum of layer sizes plus config)
# and averaging. Constants rather than a live query because `oracle-plan` must
# stay offline and instant, and because the numbers move only when the images
# are republished. They are a sample of 4 out of 20 to 96 tags per repository,
# so treat the totals below as an order of magnitude with the right leading
# digit, not as a budget line. Within a repository, 4 sampled images shared
# between 1 and 55 per cent of their layer bytes (mean 27 per cent), so a
# host's real disk use is below the naive sum -- and above it again once Docker
# unpacks them, which typically costs about twice the compressed size.
MEASURED_IMAGE_GB = {
    "NodeBB/NodeBB": 0.87,
    "ansible/ansible": 0.52,
    "element-hq/element-web": 1.17,
    "flipt-io/flipt": 1.76,
    "future-architect/vuls": 1.31,
    "gravitational/teleport": 2.40,
    "internetarchive/openlibrary": 0.99,
    "navidrome/navidrome": 0.99,
    "protonmail/webclients": 4.67,
    "qutebrowser/qutebrowser": 0.69,
    "tutao/tutanota": 1.24,
}

MEASURED_LAYER_SHARING = 0.27
"""Mean fraction of layer bytes shared between images of the same repository.

Measured over the same samples as `MEASURED_IMAGE_GB`. Applied as a discount in
the sizing arithmetic, and reported separately so the discount is visible
rather than baked into a single number.
"""

UNPACK_FACTOR = 2.0
"""Rough ratio of on-disk unpacked size to compressed download size.

Not measured here -- measuring it needs a pull, which is the thing this check
exists to avoid -- so it is named as the assumption it is and the sizing prints
both numbers rather than only the product.
"""

ORACLE_SUBSET_SIZE = 20
"""Tasks in the cross-language oracle subset.

Five per language. The full 731 is not affordable as a first check (see the
sizing), and a subset that happened to be all Python would not test the Go and
JavaScript verifiers, which are where a run script or a parser is most likely
to behave differently.
"""

FROM_PATTERN = re.compile(r"^FROM\s+(\S+)", re.MULTILINE)
ENTRYPOINT_PATTERN = re.compile(r"^(ENTRYPOINT\s+.*)$", re.MULTILINE)
"""Matches the whole directive, not just its argument.

The whole line is what the checks print and compare, because "the Dockerfile
sets `ENTRYPOINT []`" is the readable claim and `[]` on its own is not.
"""


class Failure(Exception):
    """A check failed. Carries the sentence a run report would want to quote."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def columns(values: Sequence[str], width: int = 1) -> str:
    """Lay a long list of names out in fixed columns, for readable output.

    The default is one per line rather than Terminal-Bench 2.1's four: sample
    ids here are up to 117 characters, so two of them do not fit on a terminal
    and a wrapped column is worse than a list.
    """
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


def _enum_value(value: Any) -> Any:
    """The value of a str-Enum, or the value itself.

    `harbor_config` is a pydantic `model_dump()` in python mode, so an enum
    field arrives as the member and not as its value. `NetworkMode.ALLOWLIST`
    compares equal to `"allowlist"`, but `str()` of it is
    `"NetworkMode.ALLOWLIST"`, so a check written around `str(...)` silently
    never fires. This is the single place that normalises, and
    `_prove_allowlist_detection` is what proves it still does.
    """
    return getattr(value, "value", value)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_samples(variant: str, **kwargs: Any) -> list[Sample]:
    """Build one variant and return its samples.

    Callers run this inside a `catch_warnings` block so that the adapter's own
    warnings -- the ones that say a task's declared configuration was not wired
    up -- are data for the check rather than lines that scrolled past on a busy
    terminal.
    """
    spec = VARIANTS[variant]
    kwargs.setdefault("ref", spec["digest"])
    task = spec["factory"](**kwargs)
    require(
        task.name == variant,
        f"task registered as {task.name!r}, expected {variant!r}: the adapter's "
        "own registration would have won, and eval logs would name the wrong "
        "eval",
    )
    return list(task.dataset)


def quiet_samples(variant: str, **kwargs: Any) -> list[Sample]:
    """`load_samples` with the adapter's warnings suppressed.

    Every check other than `load` has already had those warnings classified by
    `load`, and 731 deprecation notices per variant would bury the output of
    the check that is actually running.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_samples(variant, **kwargs)


def _task_dir(sample: Sample) -> Path:
    """The downloaded task directory for a sample.

    Read from the sample's own metadata rather than re-resolved through the
    registry: the adapter puts the absolute local path there
    (`task_dir`, alongside `tests_dir` and `solve_path`), so this is the
    directory the run would actually use, not a second guess at which one it
    is.
    """
    path = (sample.metadata or {}).get("task_dir")
    require(
        isinstance(path, str) and path,
        f"{sample.id} carries no task_dir in its metadata, so the on-disk task "
        "definition cannot be read; run --check load first",
    )
    return Path(str(path))


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise Failure(f"could not read {path}: {error}") from error


def _test_config(sample: Sample) -> dict[str, Any]:
    """The instance's `tests/config.json`, which carries the official rule's inputs.

    Cached by path. Several checks read three or four fields out of the same
    1462 files, and re-reading and re-parsing each one per field is most of a
    minute of the suite's runtime for no information. Callers treat the result
    as read-only; nothing here mutates it.
    """
    path = _task_dir(sample) / "tests" / "config.json"
    return _load_config(str(path), str(sample.id))


@cache
def _load_config(path: str, sample_id: str) -> dict[str, Any]:
    raw = _read(Path(path))
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise Failure(f"{sample_id}: tests/config.json is not JSON: {error}") from error
    require(
        isinstance(parsed, dict),
        f"{sample_id}: tests/config.json is not an object",
    )
    return parsed


def _test_list(value: Any) -> list[str]:
    """A `fail_to_pass` / `pass_to_pass` field, in either shape the verifier accepts.

    The shipped `test.sh` parses these with `json.loads` and falls back to
    `ast.literal_eval`, because some instances store the list as a Python
    repr with single quotes. This mirrors that, rather than assuming JSON, so
    that a count of zero here means the list is empty and not that this file
    parsed it differently from the verifier.
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(value)
            except Exception:  # noqa: BLE001 - both parsers, then give up
                continue
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
    return []


def _language(sample: Sample) -> str:
    """The instance's `repo_language`, one of python / js / ts / go."""
    return str(_test_config(sample).get("repo_language", "unknown"))


def _spread(samples: Sequence[Sample], per_language: int) -> list[Sample]:
    """`per_language` tasks per language, spread across that language's repos.

    Deterministic and hand-reproducible: within a language the repositories are
    taken in name order and round-robinned, and within a repository the tasks
    are taken in sample-id order. A seeded random sample would be defensible
    too, but this one can be reproduced by eye from `--check load`'s listing,
    which is what matters when the question is "why did that task fail and not
    this one".

    Round-robin rather than "the first n by id" because the ids sort by
    repository: three Python tasks taken in id order are three ansible tasks,
    which tests one test runner three times instead of three runners once. The
    runners are where these repositories differ.
    """
    by_language: dict[str, dict[str, list[Sample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        config = _test_config(sample)
        by_language[str(config.get("repo_language"))][str(config.get("repo"))].append(
            sample
        )
    chosen: list[Sample] = []
    for language in LANGUAGE_ORDER:
        repos = by_language.get(language, {})
        pools = [sorted(repos[name], key=lambda s: str(s.id)) for name in sorted(repos)]
        picked: list[Sample] = []
        depth = 0
        while len(picked) < per_language and any(len(p) > depth for p in pools):
            for pool in pools:
                if len(picked) == per_language:
                    break
                if len(pool) > depth:
                    picked.append(pool[depth])
            depth += 1
        chosen.extend(picked)
    return chosen


def _base_image(sample: Sample) -> str:
    """The `jefzda/sweap-images` tag this task's Dockerfile builds on."""
    dockerfile = _read(_task_dir(sample) / "environment" / "Dockerfile")
    matches = FROM_PATTERN.findall(dockerfile)
    require(
        len(matches) == 1,
        f"{sample.id}: expected exactly one FROM line, found {len(matches)}",
    )
    return str(matches[0])


def _entrypoint_directives(sample: Sample) -> list[str]:
    """Every `ENTRYPOINT` line in this task's Dockerfile, in order."""
    dockerfile = _read(_task_dir(sample) / "environment" / "Dockerfile")
    return [line.strip() for line in ENTRYPOINT_PATTERN.findall(dockerfile)]


def _service(sample: Sample) -> Any:
    """The single compose service the adapter synthesised for this task."""
    services = sample.sandbox.config.services
    require(
        len(services) == 1,
        f"{sample.id} has {len(services)} compose services; every fidelity note "
        "in this eval assumes the single 'default' service the adapter builds",
    )
    return next(iter(services.values()))


def check_load() -> None:
    """731 samples per variant, all sandboxed, ids stable, filters honest."""
    if not harbor_installed():
        raise Failure(
            "inspect_harbor is not installed; install the extra with "
            "pip install -e '.[harbor]' on a Python 3.12+ interpreter"
        )
    print(f"  inspect_harbor {harbor_version()}")
    bare_names: dict[str, set[str]] = {}
    loaded: dict[str, list[Sample]] = {}

    for variant, spec in VARIANTS.items():
        require_pinned_ref(spec["digest"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            samples = load_samples(variant)
            recorded = list(caught)

        require(
            len(samples) == SWE_BENCH_PRO_N_TASKS,
            f"{variant}: expected {SWE_BENCH_PRO_N_TASKS} samples at "
            f"{spec['digest']}, got {len(samples)}",
        )
        ids = [str(sample.id) for sample in samples]
        require(len(set(ids)) == len(ids), f"{variant}: sample ids are not unique")
        # The adapter disambiguates colliding task names by appending
        # `@sha256(str(task_dir))[:8]`, and that path is the absolute local
        # cache path -- so an id with an `@` in it is an id that differs
        # between two machines running the same pinned dataset. All 731 names
        # are unique here, so the branch must never fire.
        collided = [sample_id for sample_id in ids if "@" in sample_id]
        require(
            not collided,
            f"{variant}: sample ids went through the adapter's "
            f"machine-dependent disambiguator, so they are not stable across "
            f"machines: {collided[:5]}",
        )
        wrong_prefix = [i for i in ids if not i.startswith(f"{spec['prefix']}/")]
        require(
            not wrong_prefix,
            f"{variant}: {len(wrong_prefix)} sample ids do not carry the "
            f"'{spec['prefix']}/' prefix the task's filter documentation "
            f"promises: {wrong_prefix[:3]}",
        )
        for sample in samples:
            require(
                sample.sandbox is not None,
                f"{sample.id} carries no sandbox spec, so it would run on the host",
            )
            require(
                sample.sandbox.type == "docker",
                f"{sample.id} sandbox type is {sample.sandbox.type!r}, "
                "expected 'docker'",
            )
            require(
                sample.sandbox.config is not None,
                f"{sample.id} has a docker sandbox with no compose configuration",
            )
            require(
                bool(str(sample.input).strip()),
                f"{sample.id} has an empty problem statement",
            )
        bare_names[variant] = {i.split("/", 1)[1] for i in ids}
        loaded[variant] = samples

        print()
        print(f"  [{variant}] {spec['package']} at {spec['digest']}")
        print(f"    {len(samples)} samples, ids unique and machine-independent")
        print("    every sample carries a docker sandbox spec, a compose config")
        print("      and a non-empty problem statement")
        instruction_chars = [len(str(sample.input)) for sample in samples]
        print(
            f"    problem statement chars: min {min(instruction_chars):,}, "
            f"median {sorted(instruction_chars)[len(instruction_chars) // 2]:,}, "
            f"max {max(instruction_chars):,}"
        )
        _triage_warnings(variant, recorded)
        _check_task_dirs(variant, samples)
        _check_filters(variant, ids)
        _list_task_names(variant, samples)

    first, second = list(VARIANTS)
    shared = bare_names[first] & bare_names[second]
    print()
    print(
        f"  the two variants hold the same instances: {len(shared)} bare names "
        f"in both, {len(bare_names[first] - bare_names[second])} only in "
        f"{first}, {len(bare_names[second] - bare_names[first])} only in "
        f"{second}"
    )
    require(
        len(shared) == SWE_BENCH_PRO_N_TASKS,
        "the two variants do not hold the same 731 instances, so a per-task "
        "comparison between them would be comparing different tasks: "
        f"{sorted(bare_names[first] ^ bare_names[second])[:5]}",
    )
    _compare_prompts(loaded)


def _compare_prompts(loaded: dict[str, list[Sample]]) -> None:
    """The two variants ask for the same fix in different words. Say so, loudly.

    This is not something the two packages advertise, and it is the single
    biggest caveat on reading a gap between their scores. The instances are the
    same 731 and the container differs by design, but the prompt differs too:
    not one of the 731 `instruction.md` files is byte-identical between them.

    The plain variant wraps every problem statement in the SWE-agent house
    scaffolding -- `<uploaded_files>`, `<pr_description>`, and a numbered
    "Follow these steps to resolve the issue" procedure that tells the agent to
    write a reproduction script first. The isolated variant strips all of it and
    presents the issue under `## Requirements` and `## Interface` headings with
    no procedure at all. That is on average about 1,100 characters and one
    complete method suggestion per task.

    So a score gap between the variants is a prompt difference plus an
    isolation difference, and this repository cannot separate them from the two
    numbers alone. Anyone who wants the isolation effect on its own has to hold
    the prompt fixed, which means overriding one variant's input from the
    other's -- deliberately, and not as a default, because it would no longer be
    either published dataset. The markers are asserted so that a revision which
    harmonised the two prompts stops this warning rather than leaving it to be
    repeated after it had stopped being true.
    """
    plain, isolated = loaded["swe_bench_pro"], loaded[ISOLATED]
    by_name = {str(s.id).split("/", 1)[1]: str(s.input) for s in isolated}
    pairs = [
        (str(sample.input), by_name[str(sample.id).split("/", 1)[1]])
        for sample in plain
    ]
    identical = sum(1 for left, right in pairs if left == right)
    longer = sum(1 for left, right in pairs if len(left) > len(right))
    markers = {
        "swe_bench_pro": ("<uploaded_files>", "<pr_description>", "Follow these steps"),
        ISOLATED: ("## Requirements", "## Interface"),
    }
    counts = {
        variant: {
            marker: sum(1 for s in loaded[variant] if marker in str(s.input))
            for marker in variant_markers
        }
        for variant, variant_markers in markers.items()
    }

    print()
    print("  The two variants do NOT ask the same question:")
    print(f"    byte-identical problem statements : {identical} of {len(pairs)}")
    print(f"    plain longer than isolated        : {longer} of {len(pairs)}")
    print(
        f"    mean chars                        : "
        f"{sum(len(left) for left, _ in pairs) / len(pairs):,.0f} plain, "
        f"{sum(len(right) for _, right in pairs) / len(pairs):,.0f} isolated"
    )
    for variant, found in counts.items():
        print(f"    {variant:24} markers: {found}")
    require(
        identical == 0 and all(
            count == len(loaded[variant])
            for variant, found in counts.items()
            for count in found.values()
        ),
        "the two variants' problem statements no longer differ in the way this "
        f"check describes ({identical} identical, markers {counts}); re-read a "
        "few instruction.md files and rewrite the caveat in the task docstrings "
        "and the README, because it is currently telling operators something "
        "that is not true",
    )
    print("    The plain variant wraps each issue in the SWE-agent scaffolding")
    print("    (<uploaded_files>, <pr_description>, and a numbered procedure that")
    print("    tells the agent to write a reproduction script first). The")
    print("    isolated variant strips all of it. A gap between the two variants'")
    print("    scores is therefore a prompt difference AND an isolation")
    print("    difference, and these two numbers alone cannot separate them.")


def _triage_warnings(variant: str, recorded: Sequence[warnings.WarningMessage]) -> None:
    """Classify what the adapter said while loading, and fail on degraded fidelity.

    The adapter raises `NotImplementedError` for a blocking feature (multi-step
    tasks, Windows containers, prior-context tasks), so any of those would have
    surfaced as an exception rather than here. What arrives as a warning is
    either a degraded-fidelity notice, which is a change in the instrument and
    fails this check, or harbor's own deprecation notices, which are expected
    and counted.
    """
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
        and str(w.message).startswith(f"{variant} is running with no time_limit")
    ]
    other = [
        w
        for w in recorded
        if w not in deprecations and w not in degraded and w not in ours
    ]
    allow_internet = [w for w in deprecations if "allow_internet" in str(w.message)]
    print(
        f"    warnings: {len(allow_internet)} allow_internet deprecations "
        f"(expected: every task.toml still carries the deprecated field, and "
        f"harbor migrates it to network_mode = public), "
        f"{len(deprecations) - len(allow_internet)} other deprecations "
        f"(httpx client parameters, from harbor's registry client), "
        f"{len(ours)} from this wrapper (the no-time-limit notice, by design)"
    )
    require(
        len(allow_internet) == SWE_BENCH_PRO_N_TASKS,
        f"{variant}: expected one allow_internet deprecation per task "
        f"({SWE_BENCH_PRO_N_TASKS}), got {len(allow_internet)}. The field's "
        "migration to network_mode is what makes every task read as 'public', "
        "and --check network's headline rests on that reading",
    )
    require(
        len(ours) == 1,
        f"{variant}: the wrapper's own no-time-limit warning fired "
        f"{len(ours)} times, expected once per construction",
    )
    for message in degraded:
        print(f"    DEGRADED: {message.message}")
    require(
        not degraded,
        f"{variant}: inspect_harbor reported degraded fidelity for at least "
        "one task (fields declared in task.toml that it does not wire up). The "
        "messages above name every affected task. This is a change in the "
        "instrument, not a warning to skip",
    )
    for message in other[:5]:
        print(f"    note: {message.category.__name__}: {message.message}")


def _check_task_dirs(variant: str, samples: Sequence[Sample]) -> None:
    """Every downloaded task directory is on disk and holds what we expect."""
    absent = [str(s.id) for s in samples if not _task_dir(s).is_dir()]
    require(
        not absent,
        f"{variant}: {len(absent)} task directories named in the sample "
        f"metadata are not on disk: {absent[:3]}",
    )
    unexpected = sorted(
        {
            entry.name
            for sample in samples
            for entry in _task_dir(sample).iterdir()
            if entry.name not in EXPECTED_TASK_DIR_ENTRIES
        }
    )
    print(
        f"    {len(samples)} task directories on disk, each holding "
        f"{sorted(EXPECTED_TASK_DIR_ENTRIES)}"
    )
    require(
        not unexpected,
        f"{variant}: downloaded task directories now also contain "
        f"{unexpected}; if one of those is a provenance file, the checks "
        "should read it",
    )


def _list_task_names(variant: str, samples: Sequence[Sample]) -> None:
    """Every task name, in the form the filters match, grouped by repository.

    The task's own "No task matched" error points an operator here, and until
    this existed that pointer was a dead end. Full sample ids, not bare names:
    `task_names` matches the prefixed id, so printing the stripped name would
    hand out patterns that match nothing. One per line rather than
    Terminal-Bench's three columns, because these ids run to 117 characters.

    Grouped by source repository, with the glob that selects each group, since
    with 731 machine-generated names the useful recovery from a bad filter is
    a repository glob rather than a name read off a list. The group globs start
    with `*/` so that the same pattern works in both variants, whose prefixes
    differ.
    """
    by_repo: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        by_repo[str(_test_config(sample).get("repo"))].append(str(sample.id))

    print(f"    task names, in the form task_names/exclude_task_names match "
          f"({len(samples)}):")
    for repo in sorted(by_repo):
        names = sorted(by_repo[repo])
        stem = names[0].split("/", 1)[1].split("__")[0]
        print(f"      [{repo}] {len(names)} tasks, all matched by */{stem}__*")
        print(columns(names, width=1))
    print(
        f"    ({variant}: the same instances carry the other variant's prefix "
        "there, so a filter written with a leading */ matches both)"
    )


def _check_filters(variant: str, ids: Sequence[str]) -> None:
    """The name filters and the cap select what the docstrings say they do.

    Checked because they cannot be delegated: `inspect_harbor` 0.7.4 rejects
    `dataset_task_names` and `dataset_exclude_task_names` alongside a package
    name outright (`_harbor/task.py:145-162`), so the wrapper matches its own
    samples, and a wrapper that filters is a wrapper that can filter wrongly.
    It is also a second copy of the same twenty lines Terminal-Bench 2.1 uses,
    so it is checked here in its own right rather than assumed to have been
    checked there. Cheap: the dataset is already downloaded, so each of these
    is a rebuild from cache, about two seconds each.
    """
    first, second = sorted(ids)[:2]
    prefix = VARIANTS[variant]["prefix"]
    one = quiet_samples(variant, task_names=[first])
    pair = quiet_samples(variant, task_names=[first, second])
    globbed = quiet_samples(variant, task_names=[f"{prefix}/*"])
    excluded = quiet_samples(variant, exclude_task_names=[first])
    combined = quiet_samples(variant, task_names=[f"{prefix}/*"], n_tasks=5)
    # A bare string, which is what `-T task_names=<one name>` hands the task:
    # inspect's own parser splits on commas and returns a str when there is no
    # comma (`inspect_ai/_cli/util.py:214-227`). Iterated as a sequence it is a
    # list of single characters, so before `_select` normalised it a bare glob
    # matched every sample and silently selected all 731.
    from_cli = parse_cli_args([f"task_names={first}"])
    require(
        isinstance(from_cli.get("task_names"), str),
        f"inspect's CLI parser no longer hands a single -T task_names value "
        f"through as a string ({from_cli!r}), so the bare-string case below is "
        "checking a shape that can no longer arrive; re-read _cli/util.py",
    )
    bare = quiet_samples(variant, **from_cli)
    bare_glob = quiet_samples(variant, task_names=f"{first}*")
    bare_exclude = quiet_samples(variant, exclude_task_names=first)
    # The one case that can tell "cap after filter" from "cap before filter".
    # A glob that matches everything returns 5 either way, so the combined case
    # above proves nothing about the order on its own: this one selects the
    # second id and caps at 1, which is [second] in the documented order and
    # raises "No task matched" in the wrong one.
    ordered = quiet_samples(variant, task_names=[second], n_tasks=1)
    try:
        quiet_samples(variant, exclude_task_names=[f"{prefix}/*"])
    except ValueError as error:
        empty_message = str(error)
    else:
        empty_message = ""

    require(
        [str(s.id) for s in one] == [first],
        f"{variant}: task_names=[{first!r}] selected {[str(s.id) for s in one]}",
    )
    require(
        {str(s.id) for s in pair} == {first, second},
        f"{variant}: a two-name task_names filter did not select exactly those two",
    )
    require(
        len(globbed) == len(ids),
        f"{variant}: task_names=['{prefix}/*'] selected {len(globbed)} of "
        f"{len(ids)}: the glob does not match the prefix every sample id carries",
    )
    require(
        len(excluded) == len(ids) - 1 and first not in {str(s.id) for s in excluded},
        f"{variant}: exclude_task_names=[{first!r}] did not remove exactly that task",
    )
    require(
        len(combined) == 5,
        f"{variant}: n_tasks=5 after a matching glob returned {len(combined)}",
    )
    require(
        [str(s.id) for s in ordered] == [second],
        f"{variant}: task_names=[{second!r}] with n_tasks=1 selected "
        f"{[str(s.id) for s in ordered]}, so the cap is not applied after the "
        "name filter. A cap applied first would have kept only the first task "
        "and then matched nothing",
    )
    require(
        [str(s.id) for s in bare] == [first],
        f"{variant}: a bare-string task_names selected "
        f"{[str(s.id) for s in bare][:3]} rather than exactly {first!r}. That "
        "is the value `-T task_names=<one name>` produces, and iterating it as "
        "a sequence of characters is how a single-task run turns into all "
        f"{len(ids)}",
    )
    require(
        len(bare_glob) == 1 and len(bare_exclude) == len(ids) - 1,
        f"{variant}: a bare-string glob selected {len(bare_glob)} tasks and a "
        f"bare-string exclusion left {len(bare_exclude)} of {len(ids)}; a "
        "string containing '*' is the dangerous shape, because every character "
        "pattern list containing '*' matches everything",
    )
    require(
        "No task matched" in empty_message,
        f"{variant}: excluding every task did not raise this wrapper's own "
        f"error; got {empty_message!r}",
    )
    require(
        f"'{prefix}/'" in empty_message,
        f"{variant}: the no-match error does not name the '{prefix}/' prefix a "
        f"working filter needs: {empty_message!r}",
    )
    print(
        f"    filters: one name selects 1, two select 2, '{prefix}/*' selects "
        f"all {len(ids)}, an exclusion"
    )
    print(
        "      removes exactly one, and a one-name filter with n_tasks=1 keeps "
        "that name, which"
    )
    print(
        "      only holds if the cap runs after the filter. A bare string is "
        "one pattern, not"
    )
    print(
        "      a list of characters, so `-T task_names=<one name>` selects one "
        "task. Excluding"
    )
    print("      everything raises this wrapper's own error, naming the prefix "
          "a filter needs")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def _harbor_config(sample: Sample) -> dict[str, Any]:
    """The task's own `task.toml`, as the adapter parsed it onto the sample."""
    config = (sample.metadata or {}).get("harbor_config")
    return config if isinstance(config, dict) else {}


def _declared_network(sample: Sample) -> Any:
    """The sample's declared `[environment].network_mode`, normalised.

    The one accessor. Both the 1462-sample count below and the fabricated-sample
    proof above it go through this function, so the proof exercises the code the
    measurement runs; a proof with its own copy of the expression would pass
    with this one broken, which is the failure it exists to rule out.
    """
    config = (sample.metadata or {}).get("harbor_config") or {}
    environment = config.get("environment") or {}
    return _enum_value(environment.get("network_mode"))


def _prove_allowlist_detection() -> None:
    """Fire the allowlist detection on a fabricated sample, then report it.

    Without this the check below reports "allowlist declared: 0 tasks" on a
    dataset where none is declared, which is equally what it would print if the
    comparison were broken -- and in this codebase's history it was:
    `str(NetworkMode.ALLOWLIST)` is `'NetworkMode.ALLOWLIST'`, not
    `'allowlist'`. A measurement whose instrument is never exercised is not a
    measurement, and this is the measurement the isolated variant exists for.

    The fixture goes through `_declared_network`, the same accessor the real
    count uses, and the `public` case is checked too so that the normalisation
    itself is pinned rather than only its allowlist branch.
    """
    from harbor.models.task.config import EnvironmentConfig

    for declared_mode, expected in ((ALLOWLIST_NETWORK_MODE, True), ("public", False)):
        declared = EnvironmentConfig(network_mode=declared_mode).model_dump()
        sample = Sample(
            input="fixture",
            id=f"cais/{declared_mode}-fixture",
            metadata={"harbor_config": {"environment": declared}},
        )
        detected = _declared_network(sample)
        require(
            detected == declared_mode,
            f"_declared_network read {detected!r} from a sample declaring "
            f"{declared_mode!r}, so the counts below prove nothing. A python-mode "
            "model_dump gives the enum member, and str() of it is "
            "'NetworkMode.ALLOWLIST' rather than 'allowlist'",
        )
        require(
            (detected == ALLOWLIST_NETWORK_MODE) is expected,
            f"the allowlist comparison is {detected == ALLOWLIST_NETWORK_MODE} on "
            f"a {declared_mode!r} sample",
        )
    print(
        "  allowlist detection fired on a fabricated allowlist sample, through "
        "the same accessor the counts below use, and stayed silent on a public "
        "one: the counts are measurements"
    )


def check_network() -> None:
    """What the isolated variant isolates, and whether the adapter keeps it."""
    if not harbor_installed():
        raise Failure("inspect_harbor is not installed; see --check load")
    _prove_allowlist_detection()

    loaded = {variant: quiet_samples(variant) for variant in VARIANTS}

    print()
    print("  Declared [environment].network_mode, per variant:")
    allowlisted: dict[str, list[str]] = {}
    for variant, samples in loaded.items():
        modes = [_declared_network(sample) for sample in samples]
        allowlisted[variant] = [
            str(sample.id)
            for sample, mode in zip(samples, modes, strict=True)
            if mode == ALLOWLIST_NETWORK_MODE
        ]
        print(f"    {variant:24} {histogram(modes)}")
        print(
            f"    {'':24} compose network_mode: "
            f"{histogram(_enum_value(_service(s).network_mode) for s in samples)}"
        )

    isolated_allowlist = allowlisted[ISOLATED]
    require(
        not isolated_allowlist,
        f"{len(isolated_allowlist)} tasks in {ISOLATED} declare "
        "`network_mode = allowlist`, which inspect_harbor silently downgrades "
        "to public without isolating anything (`_harbor/converters.py:383-394`, "
        "applied at `:150`, warned at `_harbor/task.py:287-302`). The whole "
        "point of this variant is that an agent cannot reach GitHub, and under "
        "this adapter those tasks would run with unrestricted egress while "
        "still reporting as the isolated variant. Do not run it until the "
        "adapter can enforce an allowlist (upstream issue 118), or drop the "
        f"affected tasks: {isolated_allowlist[:5]}",
    )
    for variant, names in allowlisted.items():
        require(
            not names,
            f"{variant} declares allowlist on {len(names)} tasks: {names[:5]}",
        )
    print()
    print(
        "  No task in either variant declares `allowlist`, so the one network "
        "feature the"
    )
    print(
        "  adapter drops is inert here. The isolated variant does not rely on "
        "it; what it"
    )
    print("  relies on is baked into the image, and is measured next.")

    _check_isolation_mechanism(loaded[ISOLATED])
    _check_isolation_absent(loaded["swe_bench_pro"])
    _check_isolation_survives_compose(loaded[ISOLATED])


def _before_repo_set_cmd(sample: Sample) -> list[str]:
    """The instance's `before_repo_set_cmd`, as its lines.

    The verifier runs only the last of them: `test.sh` extracts
    `cmd.split('\\n')[-1]` and `eval`s that one line. The rest are read here
    anyway, because what they contain and never do is a fidelity fact about
    this benchmark rather than a detail.
    """
    raw = _test_config(sample).get("before_repo_set_cmd") or ""
    return str(raw).strip().split("\n")


def _gold_checkout(sample: Sample) -> re.Match[str] | None:
    """The `git checkout <sha> -- <paths>` the verifier runs, parsed."""
    lines = _before_repo_set_cmd(sample)
    return GOLD_CHECKOUT_PATTERN.match(lines[-1].strip()) if lines else None


def _report_git_route_open(samples: Sequence[Sample]) -> None:
    """Measure the consequence of the relocation, rather than only the mechanism.

    The mechanism checks above prove the history is moved to a known path and
    moved back by the verifier. They say nothing about whether an agent could
    read it, and the sentence a reader will lean on when interpreting a
    plain-versus-isolated gap is exactly that one. Two facts settle it without
    starting a container.

    First, the relocation is a copy followed by a delete of the original (the
    `GIT_RELOCATION_COMMAND` assertion above), so the objects are still in the
    image; and no task declares a `user`, so the agent is root and can read
    them with `git --git-dir`.

    Second, the fixing commit is among those objects. The verifier restores the
    directory and then checks the instance's own commit out of it, and no
    task's setup fetches or clones anything, so that sha has to resolve from
    local objects for the shipped verifier to work at all. That is measured
    here, on every task, rather than argued.
    """
    users = Counter(
        (_harbor_config(sample).get("agent") or {}).get("user") for sample in samples
    )
    verifier_users = Counter(
        (_harbor_config(sample).get("verifier") or {}).get("user") for sample in samples
    )
    resolved_locally = 0
    sha_is_the_instance_commit = 0
    fetches: list[str] = []
    for sample in samples:
        match = _gold_checkout(sample)
        joined = "\n".join(_before_repo_set_cmd(sample))
        if "git fetch" in joined or "git clone" in joined:
            fetches.append(str(sample.id))
        if match is None:
            continue
        resolved_locally += 1
        if match.group(1) in str(sample.id):
            sha_is_the_instance_commit += 1

    print()
    print("  What the relocation does NOT do, measured the same way:")
    print(
        f"    declared user, agent / verifier  : {histogram(users.elements())} / "
        f"{histogram(verifier_users.elements())} (None means root)"
    )
    print(
        f"    verifiers that check a commit out of the restored history: "
        f"{resolved_locally} of {len(samples)}"
    )
    print(
        f"    ...whose sha is the instance's own commit                : "
        f"{sha_is_the_instance_commit} of {len(samples)}"
    )
    print(
        f"    tasks whose setup fetches or clones anything             : "
        f"{len(fetches)}"
    )
    require(
        resolved_locally == len(samples) and sha_is_the_instance_commit == len(samples),
        "the gold checkout is no longer `git checkout <instance sha> -- <paths>` "
        f"on every task ({resolved_locally} parsed, "
        f"{sha_is_the_instance_commit} naming the instance commit). That line is "
        "the evidence that the fixing commit is a local object in the relocated "
        "history; without it, the eval's claim about the git route has to be "
        "re-derived",
    )
    require(
        not fetches,
        f"{len(fetches)} tasks fetch or clone during setup: {fetches[:5]}. The "
        "claim that the instance commit resolves from local objects rests on "
        "there being no network step, so re-read those tasks",
    )
    print("    So: the history is relocated, not removed; the agent is root in")
    print("    the container; and the commit that fixes the issue is provably an")
    print("    object in the relocated repository, because the shipped verifier")
    print("    checks it out of there with no network. The variant raises the")
    print("    cost of the git route by a path name. It does not close it, and")
    print("    the eval's docstrings say so in the same words as for /etc/hosts.")


def _check_isolation_mechanism(samples: Sequence[Sample]) -> None:
    """Read all 731 isolated task directories and say how blocking is implemented."""
    entrypoints: Counter[str] = Counter()
    digests: Counter[str] = Counter()
    missing_hosts: list[str] = []
    missing_exec: list[str] = []
    no_git_isolation: list[str] = []
    verifier_restores: list[str] = []
    no_entrypoint_script: list[str] = []
    not_a_copy: list[str] = []

    for sample in samples:
        directory = _task_dir(sample)
        dockerfile = _read(directory / "environment" / "Dockerfile")
        for directive in _entrypoint_directives(sample):
            entrypoints[directive] += 1
        # The git checks come first, and unconditionally. They used to sit
        # after the `continue` below, so a task with no entrypoint.sh was
        # counted as having git isolation that had never been looked at, and
        # the printed "731 of 731" would have overstated what was measured.
        if GIT_ISOLATION_MARKER not in dockerfile:
            no_git_isolation.append(str(sample.id))
        if GIT_RELOCATION_COMMAND not in dockerfile:
            not_a_copy.append(str(sample.id))
        if GIT_ISOLATION_MARKER not in _read(directory / "tests" / "test.sh"):
            verifier_restores.append(str(sample.id))
        script_path = directory / "environment" / "entrypoint.sh"
        if not script_path.is_file():
            missing_hosts.append(str(sample.id))
            no_entrypoint_script.append(str(sample.id))
            continue
        script = _read(script_path)
        digests[hashlib.sha256(script.encode("utf-8")).hexdigest()] += 1
        if any(f"0.0.0.0 {host}" not in script for host in BLOCKED_HOSTS):
            missing_hosts.append(str(sample.id))
        if not script.rstrip().endswith(EXEC_ARGS):
            missing_exec.append(str(sample.id))

    scanned = len(samples) - len(no_entrypoint_script)
    print()
    print(f"  How {ISOLATED} blocks GitHub, measured over all {len(samples)} tasks:")
    print(f"    Dockerfile ENTRYPOINT     : {histogram(entrypoints.elements())}")
    print(
        f"    entrypoint.sh sha256      : {histogram(digests.elements())} "
        f"({scanned} of {len(samples)} tasks ship one)"
    )
    print(f"    hosts mapped to 0.0.0.0   : {', '.join(BLOCKED_HOSTS)}")
    print(
        f"    tasks missing a mapping   : {len(missing_hosts)} "
        f"{missing_hosts[:3] or ''}"
    )
    print(f"    entrypoints not ending in {EXEC_ARGS}: {len(missing_exec)}")
    print(
        f"    git history moved to {GIT_ISOLATION_MARKER}: "
        f"{len(samples) - len(no_git_isolation)} of {len(samples)} tasks, "
        f"restored by the verifier on "
        f"{len(samples) - len(verifier_restores)} of {len(samples)}"
    )
    print(
        f"    ...by `{GIT_RELOCATION_COMMAND}`, a copy then a delete of the "
        f"original: {len(samples) - len(not_a_copy)} of {len(samples)}"
    )
    _report_git_route_open(samples)

    require(
        not missing_hosts,
        f"{len(missing_hosts)} isolated tasks do not map every one of "
        f"{list(BLOCKED_HOSTS)} to 0.0.0.0, so the variant's blocking is "
        f"partial: {missing_hosts[:5]}",
    )
    require(
        set(digests) == {ISOLATED_ENTRYPOINT_SHA256},
        "the isolated variant's entrypoint.sh is no longer the one this eval "
        f"pins ({ISOLATED_ENTRYPOINT_SHA256}); it now has digests "
        f"{sorted(digests)}. The isolation property is those few lines, so read "
        "the new script before trusting any number from this variant",
    )
    require(
        not missing_exec,
        f"{len(missing_exec)} isolated entrypoints do not end in {EXEC_ARGS}, so "
        "the adapter's `command: tail -f /dev/null` would never run and those "
        f"containers would exit at startup: {missing_exec[:5]}",
    )
    require(
        not no_git_isolation and not verifier_restores,
        "the git-history isolation is not uniform: "
        f"{len(no_git_isolation)} Dockerfiles do not move .git aside and "
        f"{len(verifier_restores)} verifiers do not move it back. A task with "
        "the first and not the second scores against a repository with no "
        f"history: {(no_git_isolation + verifier_restores)[:5]}",
    )
    # Asserted so the prose cannot drift. Every docstring in this eval says the
    # history is relocated and still readable; the day upstream replaces the
    # copy with a prune, that becomes a false and unfair description of the
    # variant, and this is what makes the change loud instead of silent.
    require(
        not not_a_copy,
        f"{len(not_a_copy)} isolated Dockerfiles no longer contain "
        f"`{GIT_RELOCATION_COMMAND}` verbatim: {not_a_copy[:5]}. Every docstring "
        "in this eval, and the README, describe the git-history isolation as a "
        "relocation that leaves the fixing commit readable inside the "
        "container. Read the new Dockerfile: if it now prunes the history, "
        "those sentences understate the variant and have to be rewritten",
    )
    require(
        set(entrypoints) == {'ENTRYPOINT ["/entrypoint.sh"]'},
        f"the isolated variant's Dockerfiles now set {sorted(entrypoints)}; "
        "the blocking runs from the image entrypoint, so anything else here "
        "changes whether it runs at all",
    )


def _check_isolation_absent(samples: Sequence[Sample]) -> None:
    """The plain variant has neither mechanism, which is what makes it the control."""
    entrypoints: Counter[str] = Counter()
    with_script = [
        str(s.id)
        for s in samples
        if (_task_dir(s) / "environment" / "entrypoint.sh").is_file()
    ]
    with_isolation = [
        str(s.id)
        for s in samples
        if GIT_ISOLATION_MARKER in _read(_task_dir(s) / "environment" / "Dockerfile")
    ]
    for sample in samples:
        for directive in _entrypoint_directives(sample):
            entrypoints[directive] += 1

    print()
    print(f"  The plain variant, for contrast, over all {len(samples)} tasks:")
    print(f"    Dockerfile ENTRYPOINT     : {histogram(entrypoints.elements())}")
    print(f"    tasks shipping entrypoint.sh: {len(with_script)}")
    print(f"    tasks isolating git history : {len(with_isolation)}")
    require(
        not with_script and not with_isolation,
        "the plain variant now carries the isolated variant's mechanisms, so "
        "the two are no longer a control and a treatment: "
        f"{(with_script + with_isolation)[:5]}",
    )
    require(
        set(entrypoints) == {SAFE_ENTRYPOINT_RESET},
        f"the plain variant's Dockerfiles now set {sorted(entrypoints)} rather "
        f"than {SAFE_ENTRYPOINT_RESET!r}. That reset is what keeps the base "
        "image's own `/bin/bash` entrypoint from swallowing the adapter's "
        "command; see --check compose",
    )


def _check_isolation_survives_compose(samples: Sequence[Sample]) -> None:
    """The mechanism is in the image, so the question is whether it still runs."""
    entrypoint_overrides = [
        str(s.id) for s in samples if getattr(_service(s), "entrypoint", None)
    ]
    wrong_command = [
        str(s.id) for s in samples if str(_service(s).command) != EXPECTED_COMMAND
    ]
    isolated_network = [
        str(s.id)
        for s in samples
        if _enum_value(_service(s).network_mode) not in ("bridge", None)
    ]

    # inspect's own docker provider is the other half of the path: if it set an
    # entrypoint when bringing the container up, the image's would never run and
    # nothing above would matter. Asserted against the installed source rather
    # than asserted in prose, so an inspect_ai bump that started injecting one
    # fails this check instead of silently unblocking GitHub.
    #
    # Scoped to the provider package, not to `util/_sandbox`, and the printed
    # line says so: `ComposeService.entrypoint` is declared one level up at
    # `util/_sandbox/compose.py:231` and the adapter fills that field, which is
    # what the per-sample `entrypoint_overrides` assertion above covers. What
    # this scan is for is the provider *injecting* one of its own.
    provider = (
        Path(__import__("inspect_ai").__file__ or "").parent
        / "util"
        / "_sandbox"
        / "docker"
    )
    scanned = sorted(path.name for path in provider.rglob("*.py"))
    injected = sorted(
        path.name
        for path in provider.rglob("*.py")
        if "entrypoint" in path.read_text(encoding="utf-8").lower()
    )
    # A scan of a directory that has moved reads exactly like a scan that found
    # nothing, and prints the same reassuring line. So the scan has to prove it
    # ran: the provider package exists, and the module that actually brings the
    # containers up is among the files that were read.
    require(
        provider.is_dir() and "compose.py" in scanned,
        f"inspect_ai's docker sandbox provider is not at {provider} (scanned "
        f"{len(scanned)} files: {scanned[:5]}). The entrypoint scan below would "
        "pass vacuously on an empty directory, so find where the provider moved "
        "to and re-scope it before trusting this check",
    )

    print()
    print("  Does the mechanism survive the adapter and inspect's docker provider?")
    print(
        f"    compose services overriding entrypoint : "
        f"{len(entrypoint_overrides)} of {len(samples)}"
    )
    print(
        f"    compose command == {EXPECTED_COMMAND!r}   : "
        f"{len(samples) - len(wrong_command)} of {len(samples)}"
    )
    print(
        f"    compose network_mode not bridge        : {len(isolated_network)} "
        "(the container has a network; the blocking is inside it)"
    )
    print(
        f"    inspect_ai docker provider (util/_sandbox/docker, {len(scanned)} "
        f"files read) mentioning an entrypoint: {injected or 'none'}"
    )
    require(
        not entrypoint_overrides,
        "the adapter now sets an entrypoint on its synthesised service, which "
        "replaces the image's and therefore skips the isolated variant's "
        f"/etc/hosts writes: {entrypoint_overrides[:5]}",
    )
    require(
        not wrong_command,
        f"the adapter's service command is no longer {EXPECTED_COMMAND!r}, so "
        "what the image entrypoint execs has changed: "
        f"{wrong_command[:5]}",
    )
    require(
        not injected,
        "inspect_ai's docker sandbox provider now mentions an entrypoint "
        f"({injected}); read those files, because an entrypoint injected there "
        "would bypass the isolated variant's blocking entirely",
    )
    print()
    print("  Finding, in one paragraph:")
    print("    The isolated variant's GitHub blocking is NOT a harbor network")
    print("    mode, so the adapter's inability to enforce `allowlist` does not")
    print("    touch it. It is an image ENTRYPOINT that appends five 0.0.0.0")
    print("    lines to /etc/hosts and then execs its arguments. The adapter")
    print("    sets no entrypoint and passes `tail -f /dev/null` as the command,")
    print("    inspect brings the container up with `compose up --detach --wait`")
    print("    and runs every tool call through `docker compose exec`, so those")
    print("    exec shells see the modified /etc/hosts (that last step is read")
    print("    out of compose.py:36 and docker.py:554, not measured here: no")
    print("    container was started). The mechanism survives.")
    print("    What the blocking IS: five hostnames in a file, in a container")
    print("    the agent has root in. An IP literal, a mirror, a proxy or one")
    print("    line of sed defeats it. And the git half is a directory rename")
    print("    that `git --git-dir=/var/lib/apt/.a8f1c` undoes, over a history")
    print("    that provably still holds the fixing commit. Both raise the cost")
    print("    of looking up the answer; neither closes the route.")


# ---------------------------------------------------------------------------
# Reward semantics
# ---------------------------------------------------------------------------


def check_reward_semantics(samples_per_language: int = 3) -> None:
    """The official rule, read out of the verifier that actually ships."""
    if not harbor_installed():
        raise Failure("inspect_harbor is not installed; see --check load")

    for variant in VARIANTS:
        samples = quiet_samples(variant)
        scripts = {
            hashlib.sha256(
                _read(_task_dir(s) / "tests" / "test.sh").encode("utf-8")
            ).hexdigest()
            for s in samples
        }
        print()
        print(f"  [{variant}] tests/test.sh over all {len(samples)} tasks")
        print(f"    distinct sha256: {len(scripts)} -> {sorted(scripts)[0]}")
        require(
            len(scripts) == 1,
            f"{variant}: tests/test.sh is not identical across the dataset "
            f"({len(scripts)} distinct files), so the rule quoted below governs "
            "only some of the tasks and every task's verifier has to be read",
        )
        script = _read(_task_dir(samples[0]) / "tests" / "test.sh")
        for line in REWARD_RULE_LINES:
            require(
                line in script,
                f"{variant}: the shipped verifier no longer contains "
                f"{line!r}, so the official rule this eval documents is not the "
                "rule it runs",
            )
        writes = re.findall(r"echo\s+\S+\s*>\s*/logs/verifier/reward\.txt", script)
        require(
            sorted(set(writes)) == sorted(REWARD_WRITES),
            f"{variant}: the verifier writes {sorted(set(writes))} to "
            "reward.txt. Fractional rewards were unreachable because the only "
            "two writes were a bare 1 and a bare 0; that is no longer true, so "
            "`reward` and `resolved` have stopped carrying the same information",
        )
        # The regex above only sees `echo ... > reward.txt`. A `printf 0.5 >`,
        # a `cat >` or a heredoc would contribute nothing to it and the
        # exhaustive comparison would still pass while the printed claim "and
        # nothing else" quietly became false. So every mention of the path is
        # accounted for: a comment, or one of the two known writes.
        unaccounted = [
            line.strip()
            for line in script.splitlines()
            if REWARD_FILE in line
            and not line.strip().startswith("#")
            and not any(write in line for write in REWARD_WRITES)
        ]
        require(
            not unaccounted,
            f"{variant}: the verifier touches {REWARD_FILE} somewhere this "
            f"check does not recognise: {unaccounted}. The two `echo` writes are "
            "what makes the reward binary, and a third mechanism would escape "
            "the write comparison above rather than fail it",
        )
        require(
            "reward.json" not in script,
            f"{variant}: the verifier now mentions reward.json. The adapter "
            "prefers reward.txt and falls back to JSON, and its JSON branch "
            "uses the first value in the object when there is no 'reward' key "
            "(`_harbor/scorer.py:139-181`) -- read that branch before trusting "
            "a number",
        )
        print("    contains, verbatim:")
        for line in REWARD_RULE_LINES:
            print(f"      {line}")
        print(f"    writes to reward.txt: {sorted(set(writes))} and nothing else")
        print("    never mentions reward.json, so the adapter's JSON branch and")
        print("      its first-value-in-the-object fallback are never taken here")
        _quote_trap(script)
        _report_trap_cost(variant, samples, script)
        _report_verifier_environment(variant, samples)
        _sample_rule_inputs(variant, samples, samples_per_language)

    print()
    print("  Which of reward and resolved is the headline:")
    print("    `resolved` (reward == 1.0). The rule above is a subset test that")
    print("    exits 0 or 1, the trap writes exactly 1 or 0, and there is no")
    print("    path that writes anything else -- so `reward` and `resolved`")
    print("    carry identical information today. `resolved` is the headline")
    print("    because it is the official rule's own verdict and it stays")
    print("    correct if a later revision starts reporting partial credit,")
    print("    where a mean `reward` would quietly change meaning. The")
    print("    diagnostics key `reward_fractional` is the alarm for that day,")
    print("    and it should be exactly 0 on every run of this benchmark.")
    print("    The adapter's own `passed = reward > 0` convention")
    print("    (`_harbor/scorer.py:108`) agrees with both here, and is recorded")
    print("    in the score metadata as `adapter_answer` rather than used.")
    _report_shipped_oracle_evidence()


def _quote_trap(script: str) -> None:
    """Print the EXIT trap, which is what makes a reward always exist."""
    start = script.find("cleanup_and_reward()")
    # The closing brace is found by column, not by the next `}`: the function
    # body contains `${exit_code}`, and searching for the first brace cut the
    # quote off after four lines and made the trap look like it did nothing.
    end = script.find("\n}", start)
    require(
        start != -1 and end != -1,
        "the verifier no longer defines cleanup_and_reward, which is the trap "
        "that guarantees a reward file on every path out of the script",
    )
    end += len("\n}")
    require(
        "trap cleanup_and_reward EXIT" in script,
        "the verifier defines cleanup_and_reward but no longer installs it as "
        "an EXIT trap, so an early exit now leaves no reward file and the "
        "adapter would raise RewardFileNotFoundError instead of scoring 0",
    )
    print("    and installs it as an EXIT trap, so a reward file always exists:")
    for line in script[start:end].splitlines():
        print(f"      {line}")


def _report_trap_cost(variant: str, samples: Sequence[Sample], script: str) -> None:
    """The other half of the trap: it scores the verifier's own failures as 0.

    `--check reward-semantics` used to report the trap as pure upside, and the
    README made `verifier_failed == 0` the gate before a paid run. The trap
    writes 0 on any non-zero exit, so an infrastructure failure inside
    `test.sh` -- a missing working directory, a failed gold checkout, a
    `parser.py` crash, a test runner that could not install its own
    dependencies -- is a scored zero and not a `verifier_failed`. Only failures
    outside the script (the tests copy, the sandbox exec, a missing or
    unparseable reward file) reach `harbor_reward`.

    Both halves are printed here, with the count of verifiers that need a
    package registry at verify time, which is the realistic way this bites.
    """
    exits = [line.strip() for line in script.splitlines() if "exit 1" in line]
    error_paths = [
        line.strip()[:88] for line in script.splitlines() if 'echo "ERROR:' in line
    ]
    fetchers = ("npm install", "yarn install", "pip install", "go mod download")
    fetching: Counter[str] = Counter()
    for sample in samples:
        runner = _read(_task_dir(sample) / "tests" / "run_script.sh")
        for fetcher in fetchers:
            if fetcher in runner:
                fetching[fetcher] += 1

    print()
    print("    What the trap costs, and it is not free:")
    print(
        f"      exit-1 lines inside test.sh, each scoring a clean 0: {len(exits)}, "
        f"printing {len(error_paths)} ERROR messages:"
    )
    for line in error_paths:
        print(f"        {line}")
    print(
        "      plus a parser.py crash, which exits non-zero under the `set -e` "
        "the script restores"
    )
    print(
        f"      verifiers that fetch from a package registry     : "
        f"{sum(fetching.values())} of {len(samples)} "
        f"({histogram(fetching.elements()) or 'none'})"
    )
    require(
        error_paths,
        f"{variant}: test.sh no longer has an ERROR path that exits non-zero, "
        "so this warning about the trap scoring infrastructure failures as 0 "
        "may have stopped being true; re-read the script",
    )
    print("      So verifier_failed on this dataset measures only the failures")
    print("      OUTSIDE test.sh. A rate-limited or offline host scores the")
    print("      fetching verifiers 0 rather than leaving them unscored, and")
    print("      the diagnostics cannot tell that apart from a failed agent.")
    print("      Read verifier_exit_code and verifier_output beside it; both")
    print("      are in Score.metadata, and neither is truncated.")


def _report_verifier_environment(variant: str, samples: Sequence[Sample]) -> None:
    """Where the tests run, and how little is restored before they do.

    The adapter execs `bash -l /tests/test.sh` in the agent's own sandbox
    (`inspect_harbor/_harbor/scorer.py:99-104`) with `verifier_user` unset, so
    the tests run as root in the container the agent worked in. `test.sh`
    restores the gold test files and nothing else, and it restores them with
    only the last line of `before_repo_set_cmd`: the three commands that would
    have reset the working tree never run. That is the measurement here, and
    it is the largest fidelity gap in this eval, because Scale's own harness
    scores an extracted patch inside a clean container instead.
    """
    ignored: Counter[int] = Counter()
    restored: list[int] = []
    resets = 0
    for sample in samples:
        lines = _before_repo_set_cmd(sample)
        ignored[len(lines) - 1] += 1
        if any("git reset --hard" in line for line in lines[:-1]):
            resets += 1
        match = _gold_checkout(sample)
        if match is not None:
            restored.append(len(match.group(2).split()))
    restored.sort()

    print()
    print("    Where the tests run, and what is restored first:")
    print(
        "      the adapter execs `bash -l /tests/test.sh` in the agent's own "
        "sandbox,"
    )
    print(
        "      after the agent's turn, as root (scorer.py:99-104; no task "
        "declares a"
    )
    print("      verifier user). Every file that is not restored below is "
          "whatever the")
    print("      agent left behind, including anything `bash -l` sources.")
    print(
        f"      before_repo_set_cmd lines skipped, per task        : "
        f"{histogram(ignored.elements())} "
        f"(a `git reset --hard` among them on {resets} of {len(samples)})"
    )
    print(
        f"      paths the one line that runs restores             : median "
        f"{restored[len(restored) // 2]}, max {restored[-1]}, exactly one on "
        f"{sum(1 for count in restored if count == 1)} of {len(samples)}"
    )
    require(
        resets == len(samples),
        f"{variant}: {len(samples) - resets} tasks no longer carry a "
        "`git reset --hard` among the lines the verifier skips. That skipped "
        "reset is why a resolved=1 here is not proof the tests ran against an "
        "unmodified tree; re-read test.sh before repeating the claim",
    )
    print("      So resolved=1 is not by itself evidence that the tests were "
          "run")
    print("      honestly. The verifier's own stdout and stderr are preserved "
          "in")
    print("      Score.metadata as verifier_output, and that is the record.")


def _sample_rule_inputs(
    variant: str, samples: Sequence[Sample], per_language: int
) -> None:
    """Read the rule's inputs from task directories spread across the languages."""
    chosen = _spread(samples, per_language)
    require(
        len(chosen) >= 10,
        f"{variant}: only {len(chosen)} task directories were sampled across "
        f"{len(LANGUAGE_ORDER)} languages; the check is specified to read at "
        "least 10",
    )

    print(
        f"    inputs to the rule, in {len(chosen)} task directories across "
        f"{len(LANGUAGE_ORDER)} languages:"
    )
    print(
        f"      {'language':9} {'repo':30} {'F2P':>5} {'P2P':>6} "
        f"{'test files':>10}  runner"
    )
    for sample in chosen:
        config = _test_config(sample)
        fail_to_pass = _test_list(config.get("fail_to_pass"))
        pass_to_pass = _test_list(config.get("pass_to_pass"))
        files = _test_list(config.get("selected_test_files_to_run"))
        runner = hashlib.sha256(
            _read(_task_dir(sample) / "tests" / "run_script.sh").encode("utf-8")
        ).hexdigest()[:8]
        print(
            f"      {config.get('repo_language', '?'):9} "
            f"{str(config.get('repo', '?')):30.30} {len(fail_to_pass):>5} "
            f"{len(pass_to_pass):>6} {len(files):>10}  {runner}"
        )
        require(
            bool(fail_to_pass),
            f"{sample.id} has an empty fail_to_pass list, so the union rule is "
            "satisfied by the pass_to_pass tests alone and the task cannot "
            "distinguish a fix from an unchanged repository",
        )

    languages = Counter(_language(sample) for sample in samples)
    runners = {
        hashlib.sha256(
            _read(_task_dir(s) / "tests" / "run_script.sh").encode("utf-8")
        ).hexdigest()
        for s in samples
    }
    parsers = {
        hashlib.sha256(
            _read(_task_dir(s) / "tests" / "parser.py").encode("utf-8")
        ).hexdigest()
        for s in samples
    }
    print(
        f"    across the whole variant: {dict(sorted(languages.items()))}, "
        f"{len(runners)} distinct run_script.sh, {len(parsers)} distinct parser.py"
    )
    print(
        "      (one verifier, many runners: the rule is uniform and only the "
        "way tests are"
    )
    print("       invoked and their output parsed varies by project)")


def _shipped_oracle_rewards() -> dict[str, float]:
    """Per-task rewards from a job tree one task directory still carries.

    One isolated task directory ships an `environment/jobs/` subtree left over
    from the packagers' own "patched oracle" run: the gold patch applied, the
    verifier run, the rewards recorded, five trials. It is an accident of
    packaging rather than part of the benchmark, and nothing here depends on
    it -- but it is the only ground truth about these verifiers that exists
    without pulling an image, so it is read and reported rather than left on
    disk unread.

    Keyed by the full task name from each trial's own `result.json`, because
    the trial directory names are truncated and would not join back to a
    sample id.
    """
    rewards: dict[str, float] = {}
    for sample in quiet_samples(ISOLATED):
        for path in (_task_dir(sample) / "environment").rglob("result.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            name = payload.get("task_name")
            value = ((payload.get("verifier_result") or {}).get("rewards") or {}).get(
                "reward"
            )
            if isinstance(name, str) and isinstance(value, (int, float)):
                rewards[name] = float(value)
    return rewards


def _report_shipped_oracle_evidence() -> None:
    """Print the packagers' own oracle run, if a task directory still carries it."""
    rewards = _shipped_oracle_rewards()
    if not rewards:
        print()
        print("  No shipped oracle-run artefact found in the task directories.")
        return
    print()
    print("  A shipped artefact: the packagers' own patched-oracle run")
    print(
        f"    {len(rewards)} trials, gold patch applied, mean reward "
        f"{sum(rewards.values()) / len(rewards):.2f}"
    )
    for name, value in sorted(rewards.items(), key=lambda item: (item[1], item[0])):
        print(f"      {value:>4.1f}  {name}")
    require(
        set(rewards.values()) <= {0.0, 1.0},
        "the shipped oracle run recorded a reward that is neither 0 nor 1 "
        f"({sorted(set(rewards.values()))}), which contradicts the binary rule "
        "asserted above; read the verifier again before trusting either claim",
    )
    print("    Every recorded reward is 1.0 or 0.0, which is the binary rule")
    print("    above seen from the other side. The mean is not 1.0: expect the")
    print("    oracle subset in --check oracle-plan to fall short of 1.0 too,")
    print("    and treat a shortfall of that size as an upstream property")
    print("    rather than as evidence of a seam bug.")


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def check_compose() -> None:
    """What each container is given, next to what the task asked for."""
    if not harbor_installed():
        raise Failure("inspect_harbor is not installed; see --check load")

    for variant, spec in VARIANTS.items():
        samples = quiet_samples(variant)
        services = [_service(sample) for sample in samples]
        harbor_configs = [
            (sample.metadata or {}).get("harbor_config") or {} for sample in samples
        ]
        declared = [
            config.get("environment") or {} for config in harbor_configs
        ]
        declared_memory = [
            value["memory_mb"] for value in declared if isinstance(
                value.get("memory_mb"), int
            )
        ]
        granted_memory = [
            int(str(service.mem_limit).rstrip("m"))
            for service in services
            if service.mem_limit
        ]
        floored = [
            str(sample.id)
            for sample, value, service in zip(samples, declared, services, strict=True)
            if isinstance(value.get("memory_mb"), int)
            and value["memory_mb"] < MIN_MEMORY_MB
            and str(service.mem_limit) == f"{MIN_MEMORY_MB}m"
        ]
        healthchecks = [
            str(sample.id)
            for sample, value in zip(samples, declared, strict=True)
            if value.get("healthcheck")
        ]
        multi_service = [
            str(sample.id)
            for sample in samples
            if len(sample.sandbox.config.services) != 1
        ]
        builds = sum(1 for service in services if service.build is not None)
        pinned_images = sum(1 for service in services if service.image)

        agent_timeouts = [
            (config.get("agent") or {}).get("timeout_sec") for config in harbor_configs
        ]
        verifier_timeouts = [
            (config.get("verifier") or {}).get("timeout_sec")
            for config in harbor_configs
        ]

        print()
        print(f"  [{variant}] {spec['package']}, {len(samples)} tasks")
        print(f"    declared cpus           : "
              f"{histogram(value.get('cpus') for value in declared)}")
        print(f"    granted cpus            : "
              f"{histogram(service.cpus for service in services)}")
        print(f"    declared memory_mb      : {histogram(declared_memory)}")
        print(f"    granted mem_limit       : "
              f"{histogram(str(service.mem_limit) for service in services)}")
        print(
            f"    memory floored to {MIN_MEMORY_MB} : {len(floored)} of "
            f"{len(samples)} tasks ({sum(declared_memory) / 1024 / 1024:.2f} TiB "
            f"declared, {sum(granted_memory) / 1024 / 1024:.2f} TiB granted)"
        )
        print(f"    declared storage_mb     : "
              f"{histogram(value.get('storage_mb') for value in declared)} "
              "(not wired up by the adapter)")
        print(f"    build_timeout_sec       : "
              f"{histogram(value.get('build_timeout_sec') for value in declared)} "
              "(not wired up either; these images are large)")
        print(f"    agent timeout_sec       : {histogram(agent_timeouts)}")
        print("                              NOT enforced by inspect_harbor")
        print(f"    verifier timeout_sec    : {histogram(verifier_timeouts)}")
        print("                              enforced, per sample, by the "
              "adapter's scorer")
        print(f"    compose services / task : "
              f"{histogram(len(s.sandbox.config.services) for s in samples)}")
        print(f"    services with a build   : {builds} of {len(services)} "
              "(no task ships a prebuilt docker_image, so every container is "
              "built locally)")
        print(f"    services with an image tag: {pinned_images} of {len(services)} "
              "(the adapter names the built layer hb__<task>)")
        print(f"    healthchecks declared   : {len(healthchecks)}")
        print(f"    compose command         : "
              f"{histogram(str(service.command) for service in services)}")

        require(
            not multi_service,
            f"{variant}: tasks with more than one compose service: "
            f"{multi_service[:5]}. The fidelity notes assume a single 'default' "
            "service per task",
        )
        require(
            not healthchecks,
            f"{variant}: tasks declaring a healthcheck change inspect's compose "
            "startup budget from the flat 600s default to start_period + 135s: "
            f"{healthchecks[:5]}. Re-derive the threshold before trusting a run",
        )
        require(
            len(floored) == len(samples),
            f"{variant}: the 6 GB floor applied to {len(floored)} of "
            f"{len(samples)} tasks, not all of them. The fidelity note in the "
            "task docstring says all 731, and a mixed picture means the tasks "
            "no longer declare a uniform 4096 MB",
        )
        _check_entrypoints(variant, samples, services)


def _check_entrypoints(
    variant: str, samples: Sequence[Sample], services: Sequence[Any]
) -> None:
    """The entrypoint question Scale's own README raises.

    Scale warns "bash runs by default in our images -- do not manually invoke
    bash", and reading the base image config out of the registry confirms it:
    21 of 22 sampled `jefzda/sweap-images` tags declare an `ENTRYPOINT`, 20 of
    them `["/bin/bash"]` and one `["/bin/sh"]`, and the remaining tag declares
    no entrypoint and `CMD ["bash"]` instead. The sample is the first two tags
    of each of the 11 source repositories in sorted tag order, read on
    2026-08-31; it is the one claim in this file that needs the registry, so no
    check reproduces it. The adapter sets no entrypoint on its service and hands it
    `command: tail -f /dev/null`, so an unreset `/bin/bash` entrypoint would
    receive `tail -f /dev/null` as arguments, try to run a script named `tail`,
    and exit -- taking the container with it before the react agent's bash tool
    ever reached it, on every sample.

    Both packagings already neutralise it, in different ways, and this is what
    checks that they still do: a Dockerfile ENTRYPOINT is safe here only if it
    is the empty reset or a script that ends by execing its arguments.
    """
    directives: Counter[str] = Counter()
    unsafe: list[str] = []
    for sample in samples:
        found = _entrypoint_directives(sample)
        require(
            len(found) == 1,
            f"{sample.id}: expected exactly one ENTRYPOINT directive, found "
            f"{len(found)}; the last one wins in Docker and this check reads "
            "them all",
        )
        directive = found[0]
        directives[directive] += 1
        if directive == SAFE_ENTRYPOINT_RESET:
            continue
        script_name = re.findall(r'"([^"]+)"', directive)
        script = (
            _task_dir(sample) / "environment" / Path(script_name[0]).name
            if script_name
            else None
        )
        if script is None or not script.is_file():
            unsafe.append(f"{sample.id}: {directive} (script not in the build context)")
            continue
        if not _read(script).rstrip().endswith(EXEC_ARGS):
            unsafe.append(f"{sample.id}: {directive} (does not exec its arguments)")

    overrides = [
        str(sample.id)
        for sample, service in zip(samples, services, strict=True)
        if getattr(service, "entrypoint", None)
    ]
    print(f"    Dockerfile ENTRYPOINT    : {histogram(directives.elements())}")
    print(f"    compose entrypoint set   : {len(overrides)} of {len(samples)} "
          "(the adapter sets none, so the image's runs)")
    print(f"    entrypoints that would swallow the command: {len(unsafe)}")
    require(
        not unsafe,
        f"{variant}: {len(unsafe)} tasks declare an entrypoint that neither "
        "resets the base image's nor execs its arguments. The adapter's "
        f"{EXPECTED_COMMAND!r} would never run and the container would exit at "
        f"startup: {unsafe[:3]}",
    )
    require(
        not overrides,
        f"{variant}: the adapter now sets a compose entrypoint on "
        f"{len(overrides)} services, which replaces the image's: "
        f"{overrides[:3]}",
    )


# ---------------------------------------------------------------------------
# Oracle plan
# ---------------------------------------------------------------------------


def _oracle_subset(samples: Sequence[Sample]) -> list[Sample]:
    """Five tasks per language, spread across that language's repositories."""
    return _spread(samples, ORACLE_SUBSET_SIZE // len(LANGUAGE_ORDER))


def check_oracle_plan() -> None:
    """Print the oracle sweeps' commands and their measured sizing. Runs nothing."""
    if not harbor_installed():
        raise Failure("inspect_harbor is not installed; see --check load")

    samples = quiet_samples("swe_bench_pro")
    verifier_timeouts = [
        float(
            (((s.metadata or {}).get("harbor_config") or {}).get("verifier") or {}).get(
                "timeout_sec"
            )
            or 0.0
        )
        for s in samples
    ]
    agent_timeouts = [
        float(
            (((s.metadata or {}).get("harbor_config") or {}).get("agent") or {}).get(
                "timeout_sec"
            )
            or 0.0
        )
        for s in samples
    ]
    # Inspect gives scoring `time_limit / 2` (run.py:2142) and the whole scorer
    # stack runs inside that, so the limit has to be twice the largest declared
    # verifier timeout PLUS the headroom the rest of the stack needs before it
    # stops being the thing that fails the sweep: twice 3000 exactly is a
    # verifier racing its own cancellation. It also has to clear the largest
    # declared agent timeout, for the oracle's own solve.sh; on this dataset the
    # declared timeouts are equal at 3000s, so the verifier constraint binds.
    # Computed rather than hardcoded so that this command and the task's own
    # starvation warning cannot recommend different numbers.
    time_limit = int(
        max(
            2 * (max(verifier_timeouts) + SCORING_OVERHEAD_SEC),
            max(agent_timeouts),
        )
    )
    require(
        time_limit >= RECOMMENDED_TIME_LIMIT_SEC,
        f"the sweep command would print -T time_limit={time_limit}, below the "
        f"{RECOMMENDED_TIME_LIMIT_SEC}s floor the task documents; the two are "
        "derived from the same constants, so one of them has drifted",
    )
    subset = _oracle_subset(samples)
    require(
        len(subset) == ORACLE_SUBSET_SIZE,
        f"the cross-language subset selected {len(subset)} tasks, expected "
        f"{ORACLE_SUBSET_SIZE}",
    )

    known = _shipped_oracle_rewards()
    print("  A. The 20-task cross-language oracle subset")
    print()
    print("    Tasks, five per language, round-robinned across each language's")
    print("    repositories. `known` is the reward the packagers' own shipped")
    print("    patched-oracle run recorded for that instance, where it ran one.")
    print(f"      {'language':9} {'repo':30} {'known':>5}  task")
    for sample in subset:
        config = _test_config(sample)
        name = str(sample.id).split("/", 1)[1]
        recorded = known.get(name)
        print(
            f"      {str(config.get('repo_language')):9} "
            f"{str(config.get('repo')):30.30} "
            f"{'-' if recorded is None else f'{recorded:.1f}':>5}  {name}"
        )
    overlap = {
        str(s.id).split("/", 1)[1]: known[str(s.id).split("/", 1)[1]]
        for s in subset
        if str(s.id).split("/", 1)[1] in known
    }
    if overlap:
        print()
        print(
            f"    {len(overlap)} of these {len(subset)} already have a recorded "
            f"oracle reward, and {sum(1 for v in overlap.values() if v != 1.0)} "
            "of those is not 1.0."
        )
        print("    That is a useful property, not a problem: a subset in which")
        print("    every task is expected to pass cannot tell a failing sweep")
        print("    from a broken one. Reproducing a known 0.0 here is evidence")
        print("    the seam is faithful; a NEW 0.0 is what has to be explained.")
    # `*/` rather than the variant's own prefix, so the identical command runs
    # the identical 20 instances in either variant. `n_tasks` cannot do this:
    # the two packages order the same 731 instances differently, and the first
    # 20 of each have no instance in common.
    bare = [str(sample.id).split("/", 1)[1] for sample in subset]
    # A YAML flow sequence of single-quoted patterns, not a bare comma list.
    # inspect runs a `-T` value through `yaml.safe_load` before it splits on
    # commas (`inspect_ai/_cli/util.py:214-227`), and a scalar that starts with
    # `*` is a YAML alias: `-T task_names=*/instance_x` dies in the CLI parser
    # with "expected alphabetic or numeric character". The list form parses to a
    # list and skips the comma-splitting entirely.
    argument = "task_names=[" + ",".join(f"'*/{name}'" for name in bare) + "]"
    # The claim that one command runs the same instances in both variants is
    # cheap to prove and expensive to get wrong (it is 20 large images), so it
    # is proved rather than printed, and proved through inspect's own argument
    # parser so that the spelling is checked along with the patterns.
    patterns = parse_cli_args([argument])["task_names"]
    require(
        isinstance(patterns, list) and len(patterns) == ORACLE_SUBSET_SIZE,
        f"inspect's own CLI parser turns the printed -T argument into "
        f"{patterns!r}, which is not a list of {ORACLE_SUBSET_SIZE} patterns",
    )
    for variant in VARIANTS:
        picked = sorted(
            str(s.id).split("/", 1)[1]
            for s in quiet_samples(variant, task_names=patterns)
        )
        require(
            picked == sorted(bare),
            f"the printed task_names patterns select {len(picked)} tasks in "
            f"{variant} rather than the subset's {ORACLE_SUBSET_SIZE}, so the "
            "command below would not run the same instances in both variants: "
            f"{sorted(set(picked) ^ set(bare))[:3]}",
        )
    print()
    print("    Command (needs a Docker daemon; builds one image per task):")
    print()
    print("      inspect eval \\")
    print("        inspect_evals/swe_bench_pro/swe_bench_pro.py@swe_bench_pro \\")
    print("        --solver inspect_harbor/oracle \\")
    print("        --model mockllm/model \\")
    print("        --max-samples 4 \\")
    print(f"        -T time_limit={time_limit} \\")
    # Double-quoted, because the patterns start with `*/` and an unquoted one
    # is a shell glob: bash would pass it through unmatched, and zsh would
    # refuse the command outright with "no matches found". Double rather than
    # single, because the patterns carry single quotes of their own for YAML.
    print(f'        -T "{argument}" \\')
    print("        --log-dir logs/swe-pro-oracle")
    print()
    print("    The names are written with a leading `*/` so that swapping the")
    print("    task file's `@swe_bench_pro` for `@swe_bench_pro_isolated` runs")
    print("    the same 20 instances in the isolated variant with no other")
    print("    edit, and as a YAML list because inspect yaml-parses a -T value")
    print("    before splitting it on commas, and a value starting with `*` is")
    print("    a YAML alias rather than a string (checked: the line above was")
    print("    round-tripped through inspect's own parser just now, and the")
    print("    patterns it produced select these 20 in both variants).")
    print("    Do not reach for -T n_tasks instead: the two packages")
    print("    order the same 731 instances differently, and the first 20 of")
    print("    each have no instance in common. The oracle applies the gold")
    print("    patch, so the two variants should agree; a task that passes in")
    print("    one and fails in the other is a packaging difference worth")
    print("    reading before any model run.")
    print()
    print("    Why each flag:")
    print("      --solver inspect_harbor/oracle runs each task's own solve.sh")
    print("        instead of a model. It ignores solve.sh's exit code and")
    print("        applies no timeout of its own, so the reward is the only")
    print("        signal and -T time_limit is the only bound on a stuck script.")
    print("      --model mockllm/model because the oracle never calls generate,")
    print("        but inspect still requires a --model. Nothing is spent.")
    print("      --max-samples bounds concurrent containers; each is granted")
    print(f"        {MIN_MEMORY_MB / 1024:.0f} GiB and one CPU.")
    print(f"      -T time_limit={time_limit} is twice the "
          f"{int(max(verifier_timeouts))}s verifier timeout every")
    print(f"        task declares plus {SCORING_OVERHEAD_SEC}s of headroom, "
          "because inspect spends half")
    print("        of a time_limit on scoring (run.py:2142) and the verifier is")
    print("        only part of what runs in there: the tests copy, the reward")
    print("        read, two cleanup execs and the diagnostics scorer share it.")
    print(f"        Twice 3000 exactly would give scoring "
          f"{max(verifier_timeouts):.0f}s, which is the")
    print("        verifier's own exec timeout, so a verifier at its cap loses")
    print("        the race. A cancelled verifier errors the sample instead of")
    print("        reporting verifier_failed, so it would look like a broken")
    print("        reference solution rather than a wrong flag. Per-sample")
    print(f"        wall-clock ceiling: {1.5 * time_limit:.0f}s.")
    print()
    print("    Expected result: reward 1.0 on most of the 20, verifier_failed 0.")
    print("    NOT 1.0 on all of them. The packagers' own patched-oracle run,")
    print("    shipped inside one of the task directories and printed by")
    print("    --check reward-semantics, scored 4 of 5. Explain each non-1.0")
    print("    before any paid run: a broken reference solution upstream is an")
    print("    upstream fact, and an adapter seam bug would silently depress")
    print("    every model's score, and they look identical from the mean.")

    print()
    print("  B. The full 731-task sweep")
    print()
    print("    Command: the same, without -T task_names and with a bigger box.")
    print("      inspect eval \\")
    print("        inspect_evals/swe_bench_pro/swe_bench_pro.py@swe_bench_pro \\")
    print("        --solver inspect_harbor/oracle --model mockllm/model \\")
    print(f"        --max-samples 4 -T time_limit={time_limit} \\")
    print("        --log-dir logs/swe-pro-oracle-full")
    print()
    _print_sizing(samples, subset, verifier_timeouts)
    print()
    print("  Nothing above was executed. No image was pulled and no container")
    print("  was started.")


def _print_sizing(
    samples: Sequence[Sample], subset: Sequence[Sample], verifier_timeouts: list[float]
) -> None:
    """Disk, image count and verifier hours, from measured per-repository sizes."""
    repos = Counter(str(_test_config(sample).get("repo")) for sample in samples)
    unknown = sorted(set(repos) - set(MEASURED_IMAGE_GB))
    require(
        not unknown,
        f"no measured image size for {unknown}; the sizing below would silently "
        "omit those repositories. Re-measure with the registry manifest API "
        "before quoting a number",
    )

    def total_gb(chosen: Sequence[Sample]) -> float:
        return sum(
            MEASURED_IMAGE_GB[str(_test_config(sample).get("repo"))]
            for sample in chosen
        )

    base_images_all = {_base_image(sample) for sample in samples}
    base_images_subset = {_base_image(sample) for sample in subset}
    require(
        all(image.startswith(f"{BASE_IMAGE_REPOSITORY}:") for image in base_images_all),
        "not every task builds on "
        f"{BASE_IMAGE_REPOSITORY}; the sizing assumes one Docker Hub repository",
    )
    # The "run both variants for the price of one pull" claim below is the
    # difference between a terabyte and two, so it is measured rather than
    # asserted in prose. It costs one cached rebuild and 731 file reads.
    isolated_bases = {_base_image(sample) for sample in quiet_samples(ISOLATED)}
    require(
        isolated_bases == base_images_all,
        "the two variants no longer build on the same base image tags "
        f"({len(base_images_all - isolated_bases)} only in the plain variant, "
        f"{len(isolated_bases - base_images_all)} only in the isolated one), so "
        "running both costs two full sets of pulls and the sizing below "
        "understates it by a factor of two",
    )

    print("    Sizing, from image manifests measured on 2026-08-31")
    print("    (4 tags sampled per source repository; sizes are compressed):")
    print()
    print(f"      {'repository':30} {'tasks':>6} {'GB/image':>9} {'GB total':>9}")
    for repo, count in sorted(repos.items()):
        size = MEASURED_IMAGE_GB[repo]
        print(f"      {repo:30.30} {count:>6} {size:>9.2f} {count * size:>9.0f}")
    full = total_gb(samples)
    small = total_gb(subset)
    print(f"      {'':30} {len(samples):>6} {'':>9} {full:>9.0f}")
    print()
    print(f"    unique base image tags        : {len(base_images_all)} for the "
          f"full sweep, {len(base_images_subset)} for the 20-task subset")
    print(f"    (measured just now: the two variants' {len(isolated_bases)} base "
          f"tags are the same set, so running both costs {len(base_images_all)} "
          f"pulls, not {2 * len(base_images_all)})")
    print(f"    20-task subset, compressed    : {small:.0f} GB down the wire")
    print(f"    20-task subset, unpacked      : about "
          f"{small * UNPACK_FACTOR:.0f} GB of disk")
    print(f"    full 731, compressed          : {full:,.0f} GB down the wire")
    print(f"    full 731, less layer sharing  : about "
          f"{full * (1 - MEASURED_LAYER_SHARING):,.0f} GB "
          f"({MEASURED_LAYER_SHARING:.0%} of layer bytes were shared between "
          "images of the same repository)")
    print(f"    full 731, unpacked on disk    : of the order of "
          f"{full * (1 - MEASURED_LAYER_SHARING) * UNPACK_FACTOR / 1000:.1f} TB, "
          f"assuming unpacking costs {UNPACK_FACTOR:g}x")
    print("    The full sweep is therefore a terabyte-scale operation and is not")
    print("    a thing to start on a laptop. Pull in batches, prune between")
    print("    them, and measure with `docker system df` after the first batch")
    print("    rather than trusting the multiplier above.")
    print()
    print(f"    verifier time, full sweep     : "
          f"{sum(verifier_timeouts) / 3600:,.0f} hours if every verifier ran to "
          "its declared 3000s timeout (they do not; this is the ceiling)")
    print(f"    verifier time, 20-task subset : "
          f"{len(subset) * max(verifier_timeouts) / 3600:.1f} hours at the same "
          "ceiling")
    print("    build time                    : one docker build per task on top")
    print("      of the pulled base, each declaring build_timeout_sec = 1800,")
    print("      which the adapter does not enforce")
    print("    task definitions              : about 60 MB per variant, already")
    print("      in ~/.cache/harbor")


# ---------------------------------------------------------------------------
# Scorer seam
# ---------------------------------------------------------------------------


class _StubHarborError(Exception):
    """Stands in for the adapter's own four exception classes.

    They are not imported: importing them would tie this check to the class
    names of a specific adapter release, and the point of the check is that the
    wrapper does not care which type came out. The real classes are
    `CopyTestsDirError`, `VerifierOutputParseError`, `RewardFileNotFoundError`
    and `RewardFileEmptyError`; `RewardFileNotFoundError` subclasses
    `FileNotFoundError`, which is why an `OSError` is in the list below.
    """


ADAPTER_FAILURES: tuple[Exception, ...] = (
    _StubHarborError("tests_dir not found in metadata"),
    _StubHarborError("Failed to copy tests to sandbox"),
    _StubHarborError("reward file is empty"),
    FileNotFoundError("Reward file not found. Test exit code: 2"),
    TimeoutError("Command timed out after 3000 seconds"),
    PermissionError("Permission denied"),
    RuntimeError("sandbox write_file failed"),
    OSError("connection to docker daemon lost"),
)
"""Every way the adapter's scorer can raise, in the shapes it raises them.

The first four stand for its own classes; the rest come out of the sandbox
`exec` beneath it. `TimeoutError` is the realistic one on this benchmark: the
verifier compiles a Go binary or builds a JavaScript bundle and then runs a
project test suite, inside a container with one CPU, against a 3000 second
budget.
"""

PASSTHROUGH_FAILURES: tuple[BaseException, ...] = (
    KeyboardInterrupt(),
    asyncio.CancelledError(),
    SystemExit(1),
)
"""What `harbor_reward` must *not* catch.

`except Exception` rather than a bare `except` is a deliberate choice in
`harbor_common.scorer`, and it is load-bearing here: with `time_limit` set,
inspect gives scoring half of it and cancels with `anyio`'s cancelled
exception, which is a `BaseException`. Swallowing that would turn a cancelled
scope into a reported reward and leave the cancellation unwound.
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
                tool_arguments={"cmd": "git diff"},
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
    """A synthetic sample state carrying a SWE-bench Pro task's metadata shape."""
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="scale-ai/instance_fixture",
        epoch=1,
        input="fixture problem statement",
        messages=list(messages),
        metadata={
            "task_name": "scale-ai/instance_fixture",
            "harbor_config": {
                "agent": {"timeout_sec": float(DECLARED_VERIFIER_TIMEOUT_SEC)},
                "verifier": {"timeout_sec": float(DECLARED_VERIFIER_TIMEOUT_SEC)},
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


def _tool_call_output(name: str, **arguments: Any) -> ModelOutput:
    return ModelOutput.for_tool_call(
        model="mockllm/model", tool_name=name, tool_arguments=arguments
    )


def check_guards() -> None:
    """The two refusals: an unpinned ref, and a missing optional dependency."""
    for digest in (SCALE_DIGEST, CAIS_DIGEST):
        require(
            require_pinned_ref(digest) == digest,
            f"the pinned digest {digest} was not accepted",
        )
    rejected = [
        "latest",
        "1",
        "v1.0",
        SCALE_DIGEST.upper(),
        CAIS_DIGEST.replace("sha256:", ""),
        SCALE_DIGEST[:-1],
        f"{CAIS_DIGEST} ",
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
    print(
        f"  require_pinned_ref accepts both pinned digests and refuses "
        f"{len(rejected)} non-digest refs,"
    )
    print("    including 'latest', a revision number, a tag and an uppercase digest")

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
    require(
        "swe_bench_pro" in message,
        "the missing-dependency error does not mention this eval among the "
        f"ones it applies to: {message}",
    )
    print("  require_harbor names the extra, the pin, the install command and "
          "this eval")


def check_construction_guards() -> None:
    """Every ValueError the two task functions promise, on both variants.

    Cheap: each of these raises before the adapter is asked for anything, so
    none of them loads a dataset.
    """
    cases: list[tuple[str, dict[str, Any]]] = [
        ("epochs must be at least 1", {"epochs": 0}),
        ("n_tasks must be at least 1", {"n_tasks": 0}),
        ("override_cpus must be at least 1", {"override_cpus": 0}),
        ("override_memory_mb must be at least 1", {"override_memory_mb": 0}),
        ("time_limit must be at least 1", {"time_limit": 0}),
        ("sandbox_env_name must be one of", {"sandbox_env_name": "k8s"}),
        ("ref must be a pinned content digest", {"ref": "latest"}),
    ]
    for variant, spec in VARIANTS.items():
        for expected, kwargs in cases:
            try:
                spec["factory"](**kwargs)
            except ValueError as error:
                require(
                    expected in str(error),
                    f"{variant}{kwargs}: expected a ValueError mentioning "
                    f"{expected!r}, got {error}",
                )
            else:
                raise Failure(f"{variant}{kwargs} did not raise")
    print(
        f"  {len(cases)} construction guards raise ValueError on both variants, "
        "each naming the"
    )
    print("    argument and the values it accepts, before any dataset is loaded")


def check_scorer() -> None:
    """The seam: every adapter raise becomes NaN plus a flag, never an error."""
    check_guards()
    check_construction_guards()
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
        "NaN/NaN with verifier_failed=1,"
    )
    print("    and the diagnostics scorer still ran afterwards")

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
    verifier_output = ""
    for value, expected_resolved, fractional in (
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 1.0),
    ):
        # SWE-bench Pro verifiers write a bare 0 or 1, so 0.5 cannot arrive from
        # this dataset today. It is fed in anyway: `resolved` is `reward == 1.0`
        # rather than the adapter's `reward > 0`, and the day a revision starts
        # reporting partial credit is the day that difference decides a headline.
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

    failure_keys = set(asyncio.run(_run_stack(ADAPTER_FAILURES[0], messages))[1].value)
    require(
        all(key_set == failure_keys for key_set in keys),
        "the diagnostics key set changes between a scored and an unscored "
        f"sample: {failure_keys ^ keys[0]}",
    )
    print(f"  diagnostics key set is identical either way: {sorted(failure_keys)}")

    _, diagnostics, _ = asyncio.run(_run_stack(Score(value=1.0), messages))
    require(
        diagnostics.value["declared_agent_timeout_sec"]
        == float(DECLARED_VERIFIER_TIMEOUT_SEC),
        "the task's declared (unenforced) agent timeout was not reported: "
        f"{diagnostics.value['declared_agent_timeout_sec']}",
    )
    print(
        "  declared_agent_timeout_sec is read off the task's own harbor_config: "
        f"{DECLARED_VERIFIER_TIMEOUT_SEC}"
    )

    check_agent_shape()


def _react_diagnostics(
    outputs: Sequence[ModelOutput], message_limit: int | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the real scorer stack under a real `react` agent, offline.

    The fixtures above hand the scorers a message list built by this file, and
    two of the diagnostics cannot be checked that way. `react` rewrites
    `state.messages` before scoring -- it deletes its own submit tool call
    (`inspect_ai/agent/_react.py:383-388`) -- so a hand-built list is a shape
    the shipping path never produces. Sample limits are the same: a
    `SampleLimitEvent` only exists inside a real sample.

    So this builds a one-sample task with no sandbox, drives it with mockllm and
    the real `harbor_reward` / `harbor_diagnostics`, and reads the diagnostics
    back out of the log. It needs no network, no Docker and no provider.
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
                dataset=[Sample(input="fix the issue", target="done", id="probe")],
                solver=react(tools=[probe_bash()]),
                scorer=[harbor_reward(_inner()), harbor_diagnostics()],
                message_limit=message_limit,
                name="swe_bench_pro_seam_probe",
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


def check_agent_shape() -> None:
    """The agent-shape diagnostics, under a real `react` agent on mockllm."""
    # `stop_reason` is a read-only property over the first choice, so the
    # truncated generation is made by setting it on the choice itself.
    truncated = _tool_call_output("probe_bash", cmd="go build ./...")
    truncated.choices[0].stop_reason = "max_tokens"
    values, metadata = _react_diagnostics(
        [
            _tool_call_output("probe_bash", cmd="git status"),
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


# ---------------------------------------------------------------------------


def check_verifier_shell() -> None:
    """The one override this seam applies to the adapter, and its blast radius.

    `inspect_harbor` 0.7.4 runs every verifier as `["bash", "-l", <test.sh>]`
    (`_harbor/scorer.py:100`) -- a fact `reward-semantics` above already prints,
    without knowing what it costs. `/etc/profile` in the `jefzda/sweap-images`
    images overwrites the image's own `ENV PATH` and drops `/usr/local/go/bin`,
    so on the adapter's invocation `go test` never runs, the benchmark's parser
    finds no results, and `test.sh`'s EXIT trap writes 0 -- with
    `verifier_failed = 0`. That is 280 of the 731 tasks scoring a confident zero
    that nothing in the results table distinguishes from a model failure.

    Three things are asserted here, and the first matters most:

    1. **The finding is still live on the pinned adapter.** If a bump fixed or
       moved it, the override becomes a silent no-op and the Go tasks go back to
       scoring zero while this seam still claims to have fixed it.
    2. **The rewrite is exactly as narrow as it claims.** Every other argv the
       adapter's scorer sends to the sandbox in the same function -- its two
       `mkdir -p` calls, its cleanup `unset`s -- must come through untouched, as
       must anything that merely looks similar.
    3. **The patch lands on the symbol the adapter actually calls**, checked by
       reaching for `inspect_harbor._harbor.scorer.sandbox` after installing and
       driving one `exec` through it.

    The fourth thing -- that a Go task's verifier really does see `go` under this
    invocation and really does not under the adapter's -- needs a container. It
    runs when a Docker daemon and a built Go task image are both present, and
    says it was skipped when they are not, because a check that quietly turns
    into nothing on the machine where it matters is worse than one that admits
    it did not run.
    """
    require(harbor_installed(), "inspect_harbor is not installed")

    source = adapter_verifier_source()
    require(
        adapter_uses_login_shell(),
        "inspect_harbor no longer runs the verifier as "
        f"{ADAPTER_LOGIN_SHELL_SOURCE}. Re-read _harbor/scorer.py: either this "
        "override is obsolete and should be deleted, or the invocation moved "
        "and the override is no longer neutralising it.",
    )
    require(
        "sandbox().exec(" in source,
        "the adapter no longer reaches the sandbox through the module-level "
        "`sandbox` symbol, which is the symbol this override patches",
    )
    print(f"    inspect-harbor {harbor_version()} still runs the verifier as")
    print(f"      {ADAPTER_LOGIN_SHELL_SOURCE}")

    rewritten = verifier_argv_without_login_shell(["bash", "-l", "/tests/test.sh"])
    require(
        rewritten == ["bash", "/tests/test.sh"],
        f"the verifier argv was not rewritten: got {rewritten!r}",
    )
    untouched = [
        ["mkdir", "-p", "/logs/agent"],
        ["mkdir", "-p", "/logs/verifier"],
        ["unset", "TEST_DIR"],
        ["bash", "-l"],
        ["bash", "-lc", "echo hi"],
        ["bash", "-l", "/tests/test.sh", "extra"],
        ["sh", "-l", "/tests/test.sh"],
        "bash -l /tests/test.sh",
    ]
    for argv in untouched:
        got = verifier_argv_without_login_shell(argv)
        require(
            got == argv,
            f"the rewrite touched an argv it should not have: {argv!r} -> {got!r}",
        )
    print(f"    it rewrites 1 argv shape and leaves {len(untouched)} lookalikes alone")

    calls: list[Any] = []

    class Recorder:
        async def exec(self, cmd: Any, *args: Any, **kwargs: Any) -> str:
            calls.append(cmd)
            return "recorded"

    from inspect_harbor._harbor import scorer as adapter_scorer

    install_image_path_verifier_shell()
    installed = adapter_scorer.sandbox
    try:
        adapter_scorer.sandbox = lambda *a, **k: ImagePathSandbox(Recorder())
        wrapped = adapter_scorer.sandbox()
        require(
            isinstance(wrapped, ImagePathSandbox),
            "the adapter's sandbox symbol is not wrapped after installing",
        )
        asyncio.run(wrapped.exec(["bash", "-l", "/tests/test.sh"]))
        asyncio.run(wrapped.exec(["mkdir", "-p", "/logs/verifier"]))
    finally:
        adapter_scorer.sandbox = installed
    require(
        calls == [["bash", "/tests/test.sh"], ["mkdir", "-p", "/logs/verifier"]],
        f"the sandbox saw {calls!r}, expected the verifier de-login-ed and the "
        "mkdir untouched",
    )
    print("    the adapter's own sandbox symbol delivers the rewritten argv")

    # The blast radius, stated as a fact rather than as an intention. The
    # adapter's tests copy and its two cleanups live in `_harbor/sandbox_utils`,
    # which imports `sandbox` into its OWN namespace, so patching the scorer's
    # symbol cannot reach them -- they keep the adapter's behaviour exactly.
    from inspect_harbor._harbor import sandbox_utils as adapter_sandbox_utils

    require(
        adapter_sandbox_utils.sandbox is not adapter_scorer.sandbox,
        "the override reached _harbor/sandbox_utils, which runs the tests copy "
        "and the two cleanups; it is only supposed to reach the verifier exec",
    )
    print("    _harbor/sandbox_utils is untouched: the tests copy and cleanups")
    print("      keep the adapter's own behaviour")

    image = built_go_task_image()
    if image is None:
        print(
            "    SKIPPED the PATH measurement: no Docker daemon, or no built Go "
            "task image on this host. Run this check on the box that runs the "
            "eval, once one Go task has been built."
        )
        return
    login = container_go_path(image, login_shell=True)
    image_path = container_go_path(image, login_shell=False)
    print(f"    {image}")
    print(f"      login shell (the adapter's) : go -> {login or 'NOT FOUND'}")
    print(f"      image PATH  (this seam's)   : go -> {image_path or 'NOT FOUND'}")
    require(
        image_path is not None,
        f"go is not on PATH in {image} even under the image's own environment, "
        "so this override is not the fix for whatever is wrong there",
    )
    require(
        login is None,
        f"go IS on PATH under the adapter's login shell in {image}, so the "
        "finding this override exists for does not reproduce here. Do not "
        "assume the override is harmless: work out what differs first.",
    )


def built_go_task_image() -> str | None:
    """A built SWE-bench Pro image for one of the four Go repositories, if any.

    Reads what Docker already has rather than pulling anything: this script's
    contract is that it pulls no images, and one is only present here because a
    run built it.
    """
    try:
        listed = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if listed.returncode != 0:
        return None
    for line in listed.stdout.splitlines():
        if line.startswith("hb__") and any(repo in line for repo in GO_REPOSITORIES):
            return line
    return None


def container_go_path(image: str, login_shell: bool) -> str | None:
    """Where `go` resolves in `image`, under a login shell or the image's own env."""
    flag = "-lc" if login_shell else "-c"
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "bash",
        image,
        flag,
        "command -v go || true",
    ]
    try:
        found = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    return found.stdout.strip() or None


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
        help="which check to run; repeatable. 'all' (the default) runs every one",
    )
    args = parser.parse_args()

    requested = args.checks or ["all"]
    selected: list[str] = []
    for name in requested:
        expanded = list(CHECKS) if name == "all" else [name]
        selected.extend(check for check in expanded if check not in selected)

    runners = {
        "load": check_load,
        "network": check_network,
        "reward-semantics": check_reward_semantics,
        "compose": check_compose,
        "oracle-plan": check_oracle_plan,
        "scorer": check_scorer,
        "verifier-shell": check_verifier_shell,
    }
    failures: list[str] = []
    for name in selected:
        print(f"[{name}]")
        try:
            runners[name]()
        # ValueError alongside Failure: the guards in harbor_common raise it (a
        # bad ref is the common case), and a refusal should read the same as
        # every other refusal rather than as a traceback.
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
