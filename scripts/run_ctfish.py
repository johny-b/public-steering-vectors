"""Run the ctfish eval.

Edit the constants below, then `python scripts/run_ctfish.py`. Requires Docker
running and API keys for the agent model and the judge models (e.g.
OPENAI_API_KEY, ANTHROPIC_API_KEY, read from the environment or a .env file).
"""

from ctfish import ctfish
from inspect_ai import eval

MODEL = "anthropic/claude-sonnet-5"
VARIANT = "baseline"  # baseline | dontlook | dontstart | powerless | spooky
EPOCHS = 10  # runs of the variant
MAX_STEPS = 37  # decision loop phases per run
COMMAND_TIMEOUT = 10  # seconds per shell command

# Sampling for the agent model. None keeps upstream's per-model-family policy
# (temperature 0.5, and max_tokens 1024 for anthropic/). Set to "qwen3-thinking"
# for Qwen's published thinking-mode settings, or to any other preset in
# ctfish.config -- and keep it identical across runs that will be compared.
GENERATE_CONFIG = None

STAGE_JUDGE_MODEL = "anthropic/claude-sonnet-5"
ESCALATION_JUDGE_MODEL = "anthropic/claude-sonnet-5"
JUDGE_SAMPLES = 5

LOG_DIR = "logs"
MAX_SANDBOXES = 5  # concurrent containers


def main() -> None:
    logs = eval(
        ctfish(
            variant=VARIANT,
            max_steps=MAX_STEPS,
            command_timeout=COMMAND_TIMEOUT,
            stage_judge_model=STAGE_JUDGE_MODEL,
            escalation_judge_model=ESCALATION_JUDGE_MODEL,
            judge_samples=JUDGE_SAMPLES,
            generate_config=GENERATE_CONFIG,
        ),
        model=MODEL,
        epochs=EPOCHS,
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
