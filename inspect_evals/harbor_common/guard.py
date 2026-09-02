"""Two refusals that keep a harbor-backed run reproducible and legible.

`require_harbor` turns a missing optional dependency into a sentence that says
what to install and on which interpreter, instead of a bare
`ModuleNotFoundError` raised somewhere inside a task factory.

`require_pinned_ref` refuses a dataset reference that is not a content digest.
That refusal is the whole point of this module. `inspect_harbor`'s generated
task functions default to `ref="latest"` (verified in
`inspect_harbor/_tasks.py`, every generated function, `terminal_bench_2_1` at
`:3836`, and again on `_harbor/task.py:32,115` for `harbor` and
`load_harbor_tasks`), and harbor resolves a ref as a tag, a revision number, or
a digest (`harbor/db/client.py:206-266`). A tag moves. Two runs a month apart
under `latest` can be two different benchmarks reported under one name, and
nothing in the eval log would say so. A digest cannot move: harbor's registry
resolves it by `content_hash`, and each constituent task is separately pinned
by its own content hash, so pinning the dataset digest pins all of the task
bodies too.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import sys
import warnings

HARBOR_EXTRA = "harbor"
"""The optional-dependency group in pyproject.toml that installs the adapter."""

INSPECT_HARBOR_REQUIREMENT = "inspect-harbor==0.7.4"
"""The pinned adapter release.

Pinned rather than bounded because this repository's harbor evals are a thin
wrapper over third-party adapter code that was read line by line before it was
trusted: the scorer's raise semantics (`_harbor/scorer.py:25-37,63-181`), its
reward-file precedence (`:139-181`), its memory floor
(`_harbor/converters.py:64-70`) and its silent feature drops
(`_harbor/task.py:236-305`) are all recorded in the wrapper docstrings against
this exact version, at those lines. A range would let those statements go stale
without a diff, and the line anchors are what makes re-checking them on a bump
cheap enough to actually happen.
"""

MINIMUM_PYTHON = (3, 12)
"""inspect-harbor 0.7.4 declares `Requires-Python: >=3.12`.

The repository itself is `requires-python >=3.10`, which is why the adapter is
an extra and not a dependency: installing it into the base set would raise the
floor for every eval here, including the ones that need nothing from harbor.
"""

DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
"""The only ref shape this repository will pin to.

Lowercase hex only, and deliberately so: harbor resolves a digest with an
equality match on the stored `content_hash`, so an uppercase spelling of the
same digest does not resolve and would fail late, at download time, with a
"not found" error that reads like a yanked dataset.
"""


def harbor_installed() -> bool:
    """Whether `inspect_harbor` can be imported, without importing it.

    `find_spec` on a missing top-level package returns None rather than
    raising; a package whose own import machinery is broken raises, and that
    is reported as "not installed" too, because the actionable advice is the
    same.
    """
    try:
        return importlib.util.find_spec("inspect_harbor") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs only
        return False


def harbor_version() -> str:
    """The installed adapter version, or "not installed".

    Recorded in task metadata. The adapter is the thing between this repository
    and the benchmark, so which build of it produced a number is part of the
    number.
    """
    try:
        return importlib.metadata.version("inspect-harbor")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def require_harbor() -> None:
    """Refuse to build a harbor task without the adapter, and say what to do.

    Raises:
        ImportError: If `inspect_harbor` is not importable, with the install
            command and, on an interpreter below 3.12, the reason installing
            it there will not work either.
    """
    if harbor_installed():
        return

    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    lines = [
        "inspect_harbor is not installed, and the harbor-backed evals "
        "(terminal_bench_2, swe_bench_pro) are a wrapper around it.",
        f"Install the '{HARBOR_EXTRA}' extra, in three steps:",
        "    pip install -e .",
        f"    pip install {INSPECT_HARBOR_REQUIREMENT}   "
        "# this downgrades openai to 2.x",
        "    pip install 'openai>=3.1.0'         # put it back",
        "pip install -e '.[harbor]' does not resolve on its own: harbor "
        "depends on litellm, which caps openai below 3, and this repository "
        "needs openai 3 for the provider that `steered` is built on. Nothing "
        "on the harbor path imports either, so the combination works. See the "
        "comment on the extra in pyproject.toml.",
    ]
    if sys.version_info < MINIMUM_PYTHON:
        minimum = ".".join(str(part) for part in MINIMUM_PYTHON)
        lines.append(
            f"This interpreter is Python {running}, and inspect-harbor "
            f"requires >= {minimum}. The rest of this repository runs on 3.10, "
            "which is why harbor is an extra rather than a dependency: create "
            f"a Python {minimum}+ environment for the harbor evals."
        )
    raise ImportError("\n".join(lines))


def require_pinned_ref(ref: str, allow_unpinned: bool = False) -> str:
    """Refuse a dataset ref that is not a `sha256:...` content digest.

    Args:
        ref: The harbor dataset reference to check.
        allow_unpinned: Accept a tag, a revision number or `latest` anyway,
            with a warning. For deliberately looking at what moved since the
            pin, never for producing a number anyone will quote.

    Returns:
        The ref, unchanged, so this can be used inline where the ref is passed.

    Raises:
        ValueError: If `ref` is not a digest and `allow_unpinned` is False.
    """
    if DIGEST_PATTERN.match(ref):
        return ref

    if allow_unpinned:
        warnings.warn(
            f"Running against an unpinned harbor ref {ref!r}. harbor resolves "
            "this as a tag or a revision, and both can move: this run is not "
            "reproducible and its numbers are not comparable to a pinned run. "
            "Record what it resolved to before quoting anything from it.",
            UserWarning,
            stacklevel=2,
        )
        return ref

    raise ValueError(
        f"ref must be a pinned content digest of the form "
        f"'sha256:<64 lowercase hex chars>', got {ref!r}. "
        "harbor also accepts tags (including its own default, 'latest') and "
        "revision numbers, and both can point at different bytes tomorrow "
        "than they do today, which would make two runs of this eval "
        "incomparable without either of them saying so. Pass the digest the "
        "task module pins, or allow_unpinned=True to override deliberately."
    )
