"""MACHIAVELLI: rewards versus ethics in text adventures (arXiv:2304.03279)."""

from .agent import AgentCounters, MachiavelliAgent, parse_choice, system_prompt
from .dataset import TEST_GAMES, machiavelli_dataset
from .engine import MachiavelliEnv, Thresholds, Trajectory
from .game_data import ensure_game_data, resolve_data_dir
from .machiavelli import DEFAULT_MAX_TOKENS, machiavelli
from .scorer import (
    mach_counts,
    mach_scores,
    machiavelli_diagnostics,
    machiavelli_scorer,
)
from .solver import MachiavelliState, machiavelli_solver

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "TEST_GAMES",
    "AgentCounters",
    "MachiavelliAgent",
    "MachiavelliEnv",
    "MachiavelliState",
    "Thresholds",
    "Trajectory",
    "ensure_game_data",
    "mach_counts",
    "mach_scores",
    "machiavelli",
    "machiavelli_dataset",
    "machiavelli_diagnostics",
    "machiavelli_scorer",
    "machiavelli_solver",
    "parse_choice",
    "resolve_data_dir",
    "system_prompt",
]
