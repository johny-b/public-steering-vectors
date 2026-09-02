"""School of Reward Hacks: gaming a stated scoring rule, measured as a judge gap."""

from .dataset import (
    CATEGORIES,
    category_letter,
    school_of_reward_hacks_dataset,
)
from .school_of_reward_hacks import school_of_reward_hacks
from .scorers import (
    JUDGE_MAX_ATTEMPTS,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JudgeResult,
    parse_score,
    render_prompt,
    reward_hack_gap_scorer,
    strip_score,
)
from .truncation import (
    TRUNCATING_STOP_REASONS,
    answer_empty,
    no_answer,
    stop_truncated,
)

__all__ = [
    "CATEGORIES",
    "JUDGE_MAX_ATTEMPTS",
    "JUDGE_MAX_TOKENS",
    "JUDGE_MODEL",
    "JudgeResult",
    "TRUNCATING_STOP_REASONS",
    "answer_empty",
    "category_letter",
    "no_answer",
    "parse_score",
    "render_prompt",
    "reward_hack_gap_scorer",
    "school_of_reward_hacks",
    "school_of_reward_hacks_dataset",
    "stop_truncated",
    "strip_score",
]
