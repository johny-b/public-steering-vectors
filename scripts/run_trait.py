"""Run the TRAIT personality eval.

Edit the constants below, then `python scripts/run_trait.py`. Needs an API key
for the model under test, an accepted Hugging Face gate for `mirlab/TRAIT`
(see the README), and a key for the extractor model that reads the replies.

The model is asked to answer each item and the answer is read back out of the
reply, which works on any provider. The paper's one-token administration is not
shipped; see the package docstring and PROVENANCE for why and what replaces it.

Before believing any score, read the diagnostics the run prints alongside it:
`refusals`, `truncated_generations`, `parse_failures` and
`channel_disagreements` near 0, and both `extractor_errors` and
`extractor_unparseable` at 0 -- either of the last two means the run had one
extraction channel rather than two. A steering condition that changes any of
them has changed how the instrument works, not what it measured.
"""

from __future__ import annotations

from inspect_ai import eval
from trait_bench import trait_generative

MODEL = "anthropic/claude-sonnet-5"

TRAITS = None  # None = all eight, or e.g. ["Agreeableness", "Psychopathy"]
ITEMS_PER_TRAIT = 100  # 1000 is the full test set; see the README on cost
SEED = 0  # the item draw and the per-item option shuffle
EXCLUDE_IDX = None  # None = the whole benchmark; see PROVENANCE before changing

MAX_TOKENS = 16384  # room to think and still answer; see DEFAULT_MAX_TOKENS
EXTRACTOR_MODEL = None  # None = $INSPECT_GRADER_MODEL, then claude-haiku-4-5
EPOCHS = 1  # replies per sample; worth raising, this administration samples

LOG_DIR = "logs"
MAX_CONNECTIONS = 10  # concurrent model requests


def main() -> None:
    task = trait_generative(
        traits=TRAITS,
        items_per_trait=ITEMS_PER_TRAIT,
        seed=SEED,
        max_tokens=MAX_TOKENS,
        extractor_model=EXTRACTOR_MODEL,
        epochs=EPOCHS,
        exclude_idx=EXCLUDE_IDX,
    )

    logs = eval(
        task,
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
            if score.unscored_samples:
                print(
                    f"    ({score.unscored_samples} of "
                    f"{score.scored_samples + score.unscored_samples} samples "
                    "could not be scored on this key)"
                )
        print(f"  log: {log.location}")


if __name__ == "__main__":
    main()
