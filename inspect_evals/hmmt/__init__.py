"""HMMT February 2026: 33 competition problems with a boxed final answer."""

from .dataset import (
    HMMT_FEB_2026_PATH,
    HMMT_FEB_2026_PROBLEMS,
    HMMT_FEB_2026_REVISION,
    hmmt_dataset,
)
from .grader import (
    GRADING_METHODS,
    VERIFIER_TIMEOUT_SECONDS,
    GradeResult,
    extract_boxed,
    gold_in_content,
    grade_math,
    normalize,
    strip_thinking,
)
from .hmmt import (
    DEFAULT_EPOCHS,
    DEFAULT_MAX_TOKENS,
    MATHARENA_PROMPT_SUFFIX,
    REASONING_PROMPT_SUFFIX,
    REFERENCE_SAMPLING,
    hmmt_feb_2026,
    prompt_suffix_sha256,
    reference_generate_config,
)
from .judge import (
    JUDGE_PROMPT,
    JUDGE_PROMPT_SHA256,
    EquivalenceVerdict,
    hmmt_equivalence_judge,
    judge_reasoning,
)
from .scorer import (
    grading_method_share,
    hmmt_diagnostics,
    hmmt_scorer,
    hmmt_untruncated_scorer,
    is_truncated,
    reasoning_chars,
    visible_completion,
)

__all__ = [
    "DEFAULT_EPOCHS",
    "DEFAULT_MAX_TOKENS",
    "GRADING_METHODS",
    "HMMT_FEB_2026_PATH",
    "HMMT_FEB_2026_PROBLEMS",
    "HMMT_FEB_2026_REVISION",
    "JUDGE_PROMPT",
    "JUDGE_PROMPT_SHA256",
    "MATHARENA_PROMPT_SUFFIX",
    "REASONING_PROMPT_SUFFIX",
    "REFERENCE_SAMPLING",
    "VERIFIER_TIMEOUT_SECONDS",
    "EquivalenceVerdict",
    "GradeResult",
    "extract_boxed",
    "gold_in_content",
    "grade_math",
    "grading_method_share",
    "hmmt_dataset",
    "hmmt_diagnostics",
    "hmmt_equivalence_judge",
    "hmmt_feb_2026",
    "hmmt_scorer",
    "hmmt_untruncated_scorer",
    "is_truncated",
    "judge_reasoning",
    "normalize",
    "prompt_suffix_sha256",
    "reference_generate_config",
    "reasoning_chars",
    "strip_thinking",
    "visible_completion",
]
