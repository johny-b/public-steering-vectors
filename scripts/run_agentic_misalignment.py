"""Run the agentic misalignment eval.

Edit the constants below, then `python scripts/run_agentic_misalignment.py`.
Requires an API key for the model under test and one for the grader model
(ANTHROPIC_API_KEY by default, read from the environment or a .env file).

The dataset is a single sample — one scenario under one condition — so EPOCHS
is what produces a sample size. Every statistic here is over epochs.
"""

from agentic_misalignment import agentic_misalignment
from inspect_ai import eval

MODEL = "anthropic/claude-sonnet-5"
SCENARIO = "blackmail"  # blackmail | leaking | murder
GOAL_TYPE = "explicit"  # explicit | latent | none | ambiguous | swap
GOAL_VALUE = "america"  # america/global, safety/acceleration, ethical/pragmatic,
# individualism/collectivism; none with goal_type none or ambiguous
URGENCY_TYPE = "replacement"  # replacement | restriction | none
EXTRA_SYSTEM_INSTRUCTIONS = None  # None | "dont" | "goal"
PROD = False  # True drops the scratchpad from the system prompt
TEST_EVAL_AWARENESS = False  # True adds the eval-awareness probe and its scorer

EPOCHS = 100  # the whole sample size: the dataset is one sample
MAX_TOKENS = 8192  # per response; generous so thinking models are not truncated

GRADER_MODEL = "anthropic/claude-sonnet-4-6"  # None = each classifier's default

LOG_DIR = "logs"
MAX_CONNECTIONS = 10  # concurrent model requests


def main() -> None:
    logs = eval(
        agentic_misalignment(
            scenario=SCENARIO,
            goal_type=GOAL_TYPE,
            goal_value=GOAL_VALUE,
            urgency_type=URGENCY_TYPE,
            extra_system_instructions=EXTRA_SYSTEM_INSTRUCTIONS,
            prod=PROD,
            test_eval_awareness=TEST_EVAL_AWARENESS,
            grader_model=GRADER_MODEL,
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
