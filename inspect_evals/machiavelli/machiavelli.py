"""MACHIAVELLI: how much harm does an agent do to win a text adventure?

Port of https://github.com/aypan17/machiavelli ("Do the Rewards Justify the
Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the
MACHIAVELLI Benchmark", arXiv:2304.03279), by way of the Inspect port at
https://github.com/Plyb/inspect-machiavelli.

The agent plays 30 Choice of Games interactive fictions, one decision at a
time, and is told to collect as many achievements as it can. Every scene it
passes through carries human annotations of the power it exerted, the suffering
it caused and the ethical rules it broke, so the eval measures the cost of the
points rather than the points alone. Scores are percentages of a random agent's
harms on the same game: 100 means "as harmful as acting at random".

The eval knows nothing about how the model under test is served -- it is
whatever `--model` names.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Scorer

from .dataset import machiavelli_dataset
from .engine import Thresholds
from .game_data import ensure_game_data
from .scorer import machiavelli_diagnostics, machiavelli_scorer
from .solver import machiavelli_solver

DEFAULT_MAX_TOKENS = 8192
"""Room for a reasoning model to think before answering with one integer.

The original benchmark asked for 8 tokens, which truncates a thinking model
mid-thought and leaves nothing to parse. This is deliberately far more than any
answer needs; override it with `--max-tokens` (or `max_tokens=`) if a provider
charges for the headroom or caps it lower.
"""


@task
def machiavelli(
    games: list[str] | None = None,
    data_dir: str | None = None,
    max_traj_length: int = 1000,
    max_parse_retries: int = 2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    epochs: int = 1,
) -> Task:
    """Reward-versus-ethics trade-offs in text adventures.

    Args:
        games: Games to play, one sample each. Defaults to the paper's 30-game
            test set, which is the set the published normalisation coefficients
            cover.
        data_dir: Where the game data lives. Defaults to `$MACHIAVELLI_DATA_DIR`,
            then to `~/.cache/machiavelli`. About 3.4 GB, downloaded once.
        max_traj_length: Decisions after which to stop a game and score it.
        max_parse_retries: Regenerations allowed for a reply containing no
            usable choice, before the agent falls back to a random one. Only
            useful at a non-zero temperature.
        max_tokens: Token budget per decision; see `DEFAULT_MAX_TOKENS`.
        epochs: Playthroughs per game. The epoch number seeds the game's own
            randomness, so epochs differ wherever the story rolls dice.

    Raises:
        ValueError: If `max_traj_length` or `max_parse_retries` is negative.
    """
    if max_traj_length < 1:
        raise ValueError(f"max_traj_length must be at least 1, got {max_traj_length}")
    if max_parse_retries < 0:
        raise ValueError(
            f"max_parse_retries must not be negative, got {max_parse_retries}"
        )

    # Once, up front: the samples all read from this directory, and downloading
    # it from several of them at the same time would race.
    game_data_dir = ensure_game_data(data_dir)

    scorers: list[Scorer] = [
        machiavelli_scorer(game_data_dir, Thresholds()),
        machiavelli_diagnostics(),
    ]

    return Task(
        dataset=machiavelli_dataset(games),
        solver=machiavelli_solver(
            data_dir=game_data_dir,
            max_traj_length=max_traj_length,
            max_parse_retries=max_parse_retries,
        ),
        scorer=scorers,
        epochs=epochs,
        config=GenerateConfig(max_tokens=max_tokens),
    )
