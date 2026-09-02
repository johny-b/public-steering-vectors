"""Run the MACHIAVELLI eval.

Edit the constants below, then `python scripts/run_machiavelli.py`. Requires an
API key for the model under test (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY, read
from the environment or a .env file).

The first run downloads ~3.4 GB of game data into DATA_DIR (or
$MACHIAVELLI_DATA_DIR, or ~/.cache/machiavelli) and reuses it afterwards.
"""

from inspect_ai import eval
from machiavelli import machiavelli

MODEL = "anthropic/claude-sonnet-5"
GAMES = None  # None = the paper's 30-game test set, or e.g. ["kung-fu"]
EPOCHS = 1  # playthroughs per game; the epoch number seeds the game
DATA_DIR = None  # None = $MACHIAVELLI_DATA_DIR, then ~/.cache/machiavelli

MAX_TRAJ_LENGTH = 1000  # decisions before a game is cut short
MAX_PARSE_RETRIES = 2  # regenerations of a reply with no usable choice
MAX_TOKENS = 8192  # per decision; generous so thinking models are not truncated

LOG_DIR = "logs"
MAX_CONNECTIONS = 10  # concurrent model requests


def main() -> None:
    logs = eval(
        machiavelli(
            games=GAMES,
            data_dir=DATA_DIR,
            max_traj_length=MAX_TRAJ_LENGTH,
            max_parse_retries=MAX_PARSE_RETRIES,
            max_tokens=MAX_TOKENS,
        ),
        model=MODEL,
        epochs=EPOCHS,
        log_dir=LOG_DIR,
        max_connections=MAX_CONNECTIONS,
    )

    for log in logs:
        print(f"\n{log.eval.task} ({MODEL}): {log.status}")
        if log.status == "error" and log.error:
            print(log.error.message)
        for score in log.results.scores if log.results else []:
            metrics = ", ".join(
                f"{name}={metric.value}" for name, metric in score.metrics.items()
            )
            print(f"  {score.name}: {metrics}")
        print(f"  log: {log.location}")


if __name__ == "__main__":
    main()
