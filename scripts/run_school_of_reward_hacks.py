"""Run the School of Reward Hacks eval.

Edit the constants below, then `python scripts/run_school_of_reward_hacks.py`.
Requires an API key for the model under test and one for the judge model
(ANTHROPIC_API_KEY by default, read from the environment or a .env file).

The dataset is 306 prompts, so unlike the agentic misalignment runner this
script already has a sample size without epochs: EPOCHS multiplies it by taking
several answers to each prompt, and is 1 by default.
"""

from inspect_ai import eval
from school_of_reward_hacks import school_of_reward_hacks

MODEL = "anthropic/claude-sonnet-5"
CATEGORIES = None  # None = all 306; or "B", or ["B", "C"], or full labels

EPOCHS = 1  # answers per prompt; the dataset itself is the sample size
MAX_TOKENS = 8192  # per response; generous so thinking models are not truncated

JUDGE_MODEL = "anthropic/claude-sonnet-5"  # both judges; see scorers.py

LOG_DIR = "logs"
MAX_CONNECTIONS = 10  # concurrent model requests


def main() -> None:
    logs = eval(
        school_of_reward_hacks(
            categories=CATEGORIES,
            judge_model=JUDGE_MODEL,
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
