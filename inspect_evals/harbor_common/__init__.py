"""The audited seam onto `inspect_harbor`, shared by every harbor-backed eval.

Not an eval. `guard` refuses an unpinned dataset ref and a missing optional
dependency; `scorer` wraps the adapter's scorer so that a broken verifier is a
reported outcome instead of an errored sample; `verifier_shell` carries the one
place where this seam changes what the adapter does rather than only reading
it, because the adapter runs every verifier in a login shell
(`inspect_harbor/_harbor/scorer.py:100`) and `/etc/profile` in the SWE-bench Pro
images drops `/usr/local/go/bin` off `PATH`, so all 280 Go tasks score a clean 0
with `verifier_failed = 0`. Terminal-Bench 2.1 uses `guard` and `scorer` and is
measurably not affected by the third; SWE-bench Pro rides the same seam and
turns the override on.

Nothing here imports `inspect_harbor`, deliberately: this module has to stay
importable on an interpreter that does not have it, so that `require_harbor`
can be the thing that explains why.
"""

from .guard import (
    DIGEST_PATTERN,
    HARBOR_EXTRA,
    INSPECT_HARBOR_REQUIREMENT,
    MINIMUM_PYTHON,
    harbor_installed,
    harbor_version,
    require_harbor,
    require_pinned_ref,
)
from .scorer import (
    DEFAULT_SUBMIT_TOOL,
    REWARD_KEYS,
    TRUNCATION_STOP_REASONS,
    HarborState,
    harbor_diagnostics,
    harbor_reward,
)
from .verifier_shell import (
    ADAPTER_LOGIN_SHELL_SOURCE,
    LOGIN_SHELL_ARGV,
    ImagePathSandbox,
    adapter_uses_login_shell,
    install_image_path_verifier_shell,
    verifier_argv_without_login_shell,
)

__all__ = [
    "ADAPTER_LOGIN_SHELL_SOURCE",
    "DEFAULT_SUBMIT_TOOL",
    "DIGEST_PATTERN",
    "HARBOR_EXTRA",
    "HarborState",
    "INSPECT_HARBOR_REQUIREMENT",
    "ImagePathSandbox",
    "LOGIN_SHELL_ARGV",
    "MINIMUM_PYTHON",
    "REWARD_KEYS",
    "TRUNCATION_STOP_REASONS",
    "adapter_uses_login_shell",
    "harbor_diagnostics",
    "harbor_installed",
    "harbor_reward",
    "harbor_version",
    "install_image_path_verifier_shell",
    "require_harbor",
    "require_pinned_ref",
    "verifier_argv_without_login_shell",
]
