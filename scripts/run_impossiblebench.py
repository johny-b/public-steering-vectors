"""Run the Impossible-LiveCodeBench eval.

Edit the constants below, then run ``python scripts/run_impossiblebench.py``.
Docker and the model provider's API key are required.
"""

from impossiblebench import impossible_livecodebench
from impossiblebench.livecodebench_tasks import INSTRUCTION
from inspect_ai import eval

MODEL = "anthropic/claude-haiku-4-5"

SPLIT = "conflicting"  # original | oneoff | conflicting
AGENT_TYPE = "minimal"  # minimal | tools
MAX_ATTEMPTS = 10
ALLOW_TEST_MODIFICATIONS = True
INSTRUCTION_PROMPT = INSTRUCTION
LIMIT = 10  # None runs the complete split
SHUFFLE = False
MESSAGE_LIMIT = 50

LOG_DIR = "logs/impossible_livecodebench"
MAX_SANDBOXES = 5


def main() -> None:
    logs = eval(
        impossible_livecodebench(
            split=SPLIT,
            agent_type=AGENT_TYPE,
            max_attempts=MAX_ATTEMPTS,
            allow_test_modifications=ALLOW_TEST_MODIFICATIONS,
            instruction_prompt=INSTRUCTION_PROMPT,
            limit=LIMIT,
            shuffle=SHUFFLE,
            message_limit=MESSAGE_LIMIT,
        ),
        model=MODEL,
        log_dir=LOG_DIR,
        max_sandboxes=MAX_SANDBOXES,
    )

    for log in logs:
        print(f"\n{log.eval.task} ({MODEL}): {log.status}")
        if log.status == "error" and log.error:
            print(log.error.message)
        for score in (log.results.scores if log.results else []):
            metrics = ", ".join(
                f"{name}={metric.value}" for name, metric in score.metrics.items()
            )
            print(f"  {score.name}: {metrics}")
        print(f"  log: {log.location}")


if __name__ == "__main__":
    main()
