"""One override of the adapter's verifier invocation, and the reason for it.

`inspect_harbor` 0.7.4 runs every task's verifier as a LOGIN shell:

    result = await sandbox().exec(
        ["bash", "-l", container_test_path], ...
    )                                   # inspect_harbor/_harbor/scorer.py:100

A login shell sources `/etc/profile`, and `/etc/profile` on the SWE-bench Pro
images (`jefzda/sweap-images`, one tag per task) overwrites the `PATH` the
image sets through `ENV`. Measured inside a built task image:

    non-login  PATH=/go/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:
                    /usr/sbin:/usr/bin:/sbin:/bin      -> /usr/local/go/bin/go
    login      PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:
                    /usr/sbin:/usr/bin:/sbin:/bin      -> go: not found

Go is installed at `/usr/local/go/bin`, which only the first of those contains.
So on every Go task the verifier's `go test` never runs, the benchmark's own
`parser.py` finds no results, and the shipped `tests/test.sh` EXIT trap writes
`0` to `/logs/verifier/reward.txt`.

**That zero is indistinguishable from a capability result.** `verifier_failed`
is 0, `unscored` is 0, `scorer_ran` is 1, the reward file exists and parses.
Nothing in the score record says the toolchain was missing. It affects all four
Go repositories in SWE-bench Pro -- flipt, teleport, vuls, navidrome -- which is
280 of the 731 tasks, 38% of the benchmark. It would depress every steering arm
equally, so it does not manufacture a dose-response curve; it makes 38% of the
measurement uninformative while looking like one.

## How it was established

Reproduced and narrowed on 2026-09-02 against
`scale-ai/instance_navidrome__navidrome-0130c6dc...`, whose recorded upstream
oracle reward is 1.0 and which this seam scored 0.0:

* the gold patch **is** applied at verify time and the gold test file **is**
  checked out (inspected in a surviving container from a
  `--no-sandbox-cleanup` run; `solution_exit_code` 0);
* the shipped `tests/test.sh`, run by hand in the same image at the same CPU
  and memory limits, gives `REWARD=1`, `RESULT: PASSED`, and an `output.json`
  of `{"name": "TestScanner", "status": "PASSED"}` -- so neither the task, nor
  the gold patch, nor the official parser is broken;
* the same verifier in the same container under `bash -l` gives `exit=1`,
  `REWARD=0`, `Passed tests: 0`, `Missing tests: ['TestScanner']`, reproducing
  the harness result exactly;
* two other explanations were tested and discarded: a cold Go build cache (a
  fresh 1-CPU/6 GB container completes the build and passes in 61 s) and an
  unset `HOME` breaking `GOCACHE` (`docker exec` sets `HOME=/root`).

## Why the fix lives here and not in the adapter

`inspect-harbor` is pinned at 0.7.4 precisely because this repository's harbor
wrappers make line-anchored claims about its source. Editing the installed
package would make the pin a lie, and bumping it would invalidate the audit
that findings 1-5 of `scorer.py` rest on. So the override is applied in this
seam, through the same `harbor_reward` wrapper that already exists, and it is
as narrow as it can be made: it removes the `-l` flag from **one** argv shape
and passes every other sandbox call through untouched.

## Terminal-Bench 2.1 is not affected, and this is not assumed

The same adapter scorer runs Terminal-Bench's verifiers, so the question had to
be answered rather than waved at. Evidence, from the 89-task oracle sweep of
2026-09-01 (`.work/runs/tb2-oracle/`, read back sample by sample):

* 86 of 89 tasks scored reward 1.0 under the adapter's own login shell. A PATH
  that dropped a toolchain would not produce that;
* **zero** of the 89 preserved verifier outputs contain `command not found`,
  `executable file not found`, or a missing-interpreter signature;
* the three that are not 1.0 are the documented task-side defects
  (`build-cython-ext`, `build-pov-ray`, `mcmc-sampling-stan`), and their
  verifier outputs are 5,629, 23,600 and 8,126 characters of real build and
  test output -- the verifiers ran. The SWE-bench Pro Go failures are the
  opposite shape: 264 characters saying nothing ran.

Terminal-Bench's images put their tooling on the stock `PATH`, so
`/etc/profile` takes nothing away. `terminal_bench_2` therefore keeps the
adapter's own invocation, and its 86/89 sweep stays the number this repository
quotes; changing the shell under it would have meant re-validating that sweep
for no benefit.
"""

from __future__ import annotations

import inspect
from typing import Any

LOGIN_SHELL_ARGV = ("bash", "-l")
"""The argv prefix the adapter uses for the verifier, and the only one rewritten.

`inspect_harbor/_harbor/scorer.py:100`. Matched as a whole prefix rather than by
searching for `-l` anywhere, so a verifier script that happens to take `-l` as
its own argument is never touched.
"""

ADAPTER_LOGIN_SHELL_SOURCE = '["bash", "-l", container_test_path]'
"""The literal this override exists to neutralise, asserted before patching.

If a later adapter stops running the verifier in a login shell, this string
stops appearing and `install_image_path_verifier_shell` raises instead of
silently doing nothing. A no-op override would be worse than no override: the
Go tasks would go back to scoring a confident zero and the seam would still
claim to have fixed it.
"""

_installed = False


def verifier_argv_without_login_shell(cmd: Any) -> Any:
    """Return `cmd` with the verifier's `-l` removed, or `cmd` unchanged.

    Pure, so the rewrite can be tested against the adapter without Docker, a
    container or a model. Rewrites exactly `["bash", "-l", <path>]` and nothing
    else: the adapter's other sandbox calls in the same scorer are
    `["mkdir", "-p", ...]` and the cleanup `unset`s, and none of them matches.

    Args:
        cmd: The argv the adapter passed to `sandbox().exec`. Anything that is
            not a 3-element sequence starting with `bash -l` is returned
            unchanged, including a plain string command.

    Returns:
        `["bash", <path>]` for the verifier invocation, `cmd` otherwise.
    """
    if isinstance(cmd, str) or not isinstance(cmd, (list, tuple)):
        return cmd
    if len(cmd) != 3:
        return cmd
    if tuple(cmd[:2]) != LOGIN_SHELL_ARGV:
        return cmd
    return [cmd[0], cmd[2]]


class ImagePathSandbox:
    """A sandbox that drops `-l` from the verifier's shell, and does nothing else.

    Delegates every attribute to the real sandbox environment, so it stays a
    drop-in wherever the adapter uses one, and overrides `exec` only to rewrite
    the one argv shape. Deliberately not a subclass of
    `inspect_ai.util.SandboxEnvironment`: the providers are concrete classes
    with their own constructors, and wrapping by delegation avoids depending on
    which provider is in play.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def exec(self, cmd: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.exec(
            verifier_argv_without_login_shell(cmd), *args, **kwargs
        )


def adapter_verifier_source() -> str:
    """The source of the adapter's scorer factory, for asserting the finding.

    Raises:
        ImportError: If `inspect_harbor` is not installed.
    """
    from inspect_harbor._harbor import scorer as adapter_scorer

    return inspect.getsource(adapter_scorer)


def adapter_uses_login_shell() -> bool:
    """Whether the pinned adapter still runs the verifier in a login shell."""
    return ADAPTER_LOGIN_SHELL_SOURCE in adapter_verifier_source()


def install_image_path_verifier_shell() -> None:
    """Make the adapter run the verifier with the image's own PATH.

    Patches the `sandbox` symbol in `inspect_harbor._harbor.scorer` so that the
    sandbox the adapter's scorer reaches for is wrapped in `ImagePathSandbox`.

    Called at scorer CONSTRUCTION time, not around each scoring call, and never
    undone. That is the point: inspect scores samples concurrently, so a
    context manager that unpatched on the way out would remove the override
    from one sample while another was still inside the adapter's scorer, and
    the resulting zeros would be unreproducible. Construction happens once, on
    the task-building thread, before any sample runs.

    Idempotent: installing twice would wrap the wrapper, which still works but
    makes the patch harder to reason about, so the second call returns.

    Raises:
        ImportError: If `inspect_harbor` is not installed.
        RuntimeError: If the installed adapter no longer contains the
            login-shell invocation this override exists to neutralise. Silence
            would be the dangerous outcome here; see
            `ADAPTER_LOGIN_SHELL_SOURCE`.
    """
    global _installed
    if _installed:
        return

    from inspect_harbor._harbor import scorer as adapter_scorer

    if not adapter_uses_login_shell():
        raise RuntimeError(
            "harbor_common.verifier_shell was asked to stop inspect_harbor "
            "running the verifier in a login shell, but the installed adapter "
            f"no longer contains {ADAPTER_LOGIN_SHELL_SOURCE!r}. Re-read "
            "`inspect_harbor/_harbor/scorer.py` before running SWE-bench Pro: "
            "either the bug is fixed upstream, in which case delete this "
            "override and the argument that selects it, or it moved, in which "
            "case this override is no longer neutralising it and every Go task "
            "will score a clean 0 that nothing in the results table flags."
        )

    original_sandbox = adapter_scorer.sandbox

    def sandbox_with_image_path(*args: Any, **kwargs: Any) -> ImagePathSandbox:
        return ImagePathSandbox(original_sandbox(*args, **kwargs))

    adapter_scorer.sandbox = sandbox_with_image_path  # type: ignore[assignment]
    _installed = True
