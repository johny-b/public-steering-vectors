"""Run the TruthfulQA eval.

Edit the constants below, then `python scripts/run_truthfulqa.py`. Requires an
API key for the model under test (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY, read
from the environment or a .env file). Scoring is mechanical -- the model picks
a letter -- so no grader model and no second key are involved.

The first run downloads the pinned dataset revision from Hugging Face and
reuses it afterwards.

The dataset is 817 questions, so it is the sample size on its own and EPOCHS
is 1. Raise it only to average a stochastic model over repeated attempts at the
same questions.
"""

from inspect_ai import eval
from truthfulqa import truthfulqa

MODEL = "anthropic/claude-sonnet-5"
TARGET = "mc1"  # mc1 (exactly one true answer) | mc2 (one or more)
SHUFFLE = True  # shuffle the question order rather than using the dataset's

SEED = 42
"""Pins the question order and the answer order within each question. Keep it
identical across runs that will be compared: choice order moves scores."""

EPOCHS = 1  # attempts per question; the 817 questions are the sample size
MAX_TOKENS = 8192  # per response; generous so thinking models are not truncated

LOG_DIR = "logs"
MAX_CONNECTIONS = 10  # concurrent model requests


def main() -> None:
    logs = eval(
        truthfulqa(
            target=TARGET,
            shuffle=SHUFFLE,
            seed=SEED,
        ),
        model=MODEL,
        epochs=EPOCHS,
        max_tokens=MAX_TOKENS,
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
