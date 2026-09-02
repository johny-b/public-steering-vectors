"""SWE-bench Pro through the audited `inspect_harbor` seam, in two variants.

`swe_bench_pro` is the plain Scale AI packaging; `swe_bench_pro_isolated` is
the CAIS anti-exploitation packaging of the same 731 instances, with the
repository's git history moved aside and GitHub blocked in `/etc/hosts`. Both
score through `harbor_common`, which is where every claim about the adapter's
behaviour was audited; this package adds no scoring logic of its own.

The constants are exported because the validation script pins against them and
because a reader asking "which digest produced this number" should be able to
answer it from the eval log's task metadata and one import.
"""

from .swe_bench_pro import (
    BASE_IMAGE_REPOSITORY,
    CAIS_DIGEST,
    CAIS_PACKAGE,
    DECLARED_VERIFIER_TIMEOUT_SEC,
    RECOMMENDED_TIME_LIMIT_SEC,
    SANDBOX_ENV_NAMES,
    SCALE_DIGEST,
    SCALE_PACKAGE,
    SCORING_OVERHEAD_SEC,
    SWE_BENCH_PRO_N_TASKS,
    swe_bench_pro,
    swe_bench_pro_isolated,
)

__all__ = [
    "BASE_IMAGE_REPOSITORY",
    "CAIS_DIGEST",
    "CAIS_PACKAGE",
    "DECLARED_VERIFIER_TIMEOUT_SEC",
    "RECOMMENDED_TIME_LIMIT_SEC",
    "SANDBOX_ENV_NAMES",
    "SCALE_DIGEST",
    "SCALE_PACKAGE",
    "SCORING_OVERHEAD_SEC",
    "SWE_BENCH_PRO_N_TASKS",
    "swe_bench_pro",
    "swe_bench_pro_isolated",
]
