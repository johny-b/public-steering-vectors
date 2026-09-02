"""HMMT February 2026: 33 competition problems with a single final answer each.

The Harvard-MIT Mathematics Tournament, February 2026, as released by MathArena
(https://matharena.ai, https://github.com/eth-sri/matharena) at
https://huggingface.co/datasets/MathArena/hmmt_feb_2026. The model is given one
problem and asked to put its final answer in a `\\boxed{}`; the answer is graded
mechanically against gold.

Why this benchmark, in a repository about steering vectors: it is a capability
measure that is hard, short-horizon and objectively gradable, so a steering
condition that damages reasoning shows up here as a number rather than as an
impression. 17 of the 33 answers are non-integer LaTeX (`\\sqrt{69}`,
`3+\\sqrt{11}`, `74^\\circ`), which is why the two AIME scorers already
published for Inspect cannot be reused: both assume an integer answer.

Read the results with the sample size in mind. n = 33 at 4 epochs is 132
generations, but the unit of independence is the problem, not the generation:
one problem is three percentage points. `scripts/run_hmmt.py` prints a
bootstrap interval over problems next to the metric, and that interval is the
one to quote.

The eval knows nothing about how the model under test is served -- it is
whatever `--model` names.

One fact about serving belongs here anyway, because this is the eval where it
bites: raise `client_timeout` on any long-generating provider. `DEFAULT_MAX_
TOKENS` is 81,920, and a reasoning model on a self-hosted server can spend well
over ten minutes on one problem. Inspect 0.3.259's OpenAI-compatible providers
-- including this repository's `steered/` provider, which subclasses
`OpenAICompatibleAPI` -- build their HTTP client from the OpenAI SDK's defaults
when no `client_timeout` is given, and those are `Timeout(connect=5.0,
read=600, write=600, pool=600)` (`_providers/openai_compatible.py`,
`_create_http_client` and `_create_client`). Past 600 seconds the request
raises `APITimeoutError`, the retry starts the same generation from scratch,
and it times out again: the run neither finishes nor fails, it loops and keeps
the GPU busy.

`GenerateConfig(timeout=...)` does not fix it. That value bounds the retry loop
through tenacity's `stop_after_delay` and never reaches the HTTP client. The
model argument is the only thing that moves the read timeout, as
`-M client_timeout=7200` or `get_model(..., client_timeout=7200.0)`.
`scripts/run_hmmt.py` sets it, and drops it for `anthropic/*`, whose provider
does not take the argument and does not carry the 600-second default either.
"""

from __future__ import annotations

import hashlib
from typing import Any

from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import Scorer

from .dataset import HMMT_FEB_2026_PATH, HMMT_FEB_2026_REVISION, hmmt_dataset
from .judge import hmmt_equivalence_judge
from .scorer import hmmt_diagnostics, hmmt_scorer, hmmt_untruncated_scorer

REASONING_PROMPT_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)
"""The default instruction, appended to the problem statement.

This is the wording behind the reference runs in the `capability-evals` results
of the internal `jevals` reference harness (a private steering-tools workspace;
its records are available to maintainers, not shipped) -- sha256 `e4af2c73…`,
recorded in its run configs -- whose unsteered avg@4 of 0.8409 sits next to the
84.3 MathArena publishes for the same model on this competition. That
independent agreement is the reason to keep this exact string rather than tidy
it: the number it produced is the number this eval's sanity band is drawn from.

The prompt is one of three things that number depends on. The other two are the
serving stack and `REFERENCE_SAMPLING`; all three have to match before a
comparison with 0.841 means anything.
"""

MATHARENA_PROMPT_SUFFIX = "\n\nPut your final answer within \\boxed{}."
"""MathArena's own instruction for this competition, offered as an alternative.

The wording is verbatim from `configs/competitions/hmmt/hmmt_feb_2026.yaml` in
eth-sri/matharena. One deviation, and it is deliberate: MathArena puts the
instruction *before* the problem (`runner.py` appends `"\\n\\n{problem}"` to the
instruction when the template has no `{problem}` placeholder), while this eval
appends it. Making it a prefix would need a different task argument for one
sentence's position; the position is documented instead, so both constants drop
into the same `prompt_suffix` slot and a run that used this one is identifiable
from `prompt_suffix_sha256` in the log.

It also asks for no reasoning. On a thinking model that costs nothing, but on a
non-thinking one it measures something else; the default above is the safer
comparison.
"""

DEFAULT_MAX_TOKENS = 81920
"""Output budget per problem, chosen from measurement rather than from taste.

At this budget the reference unsteered run still truncated 11 of 132
generations, with a mean completion of 39,672 tokens; the −2 steering arm
truncated 22. Lowering it would not measure mathematics, it would measure the
budget -- and `hmmt_diagnostics` reports `truncated_generations` precisely so
that a run where it *is* the budget cannot be read as a capability result.

Those counts are measurements under `REFERENCE_SAMPLING`, not constants. How
long a thinking model thinks is a function of its sampling policy, so a run
under a different one may truncate more or less at the same budget.
"""

DEFAULT_EPOCHS = 4
"""MathArena reports avg@4 for this competition, and so does this eval.

`Epochs(4, "mean")` reduces the four attempts at a problem to that problem's
success rate before any metric sees them, which is what makes the headline the
mean over *problems* rather than over generations.

Four attempts only mean four attempts if the provider samples. Against a server
whose default is greedy decoding, all four are the same generation: `correct`
quietly becomes pass@1 at four times the cost, and the item bootstrap narrows
because the attempts agree rather than because the model is sure. Nothing in
the metric table would look wrong. Pass a `generate_config` -- see
`REFERENCE_SAMPLING` -- rather than trusting a default you have not read.
"""

REFERENCE_SAMPLING: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
}
"""The sampling policy behind every reference number this eval quotes.

Recorded from `results/full/hmmt__nosteer/config.json` in the reference
harness's `capability-evals` runs: Qwen's published thinking-mode settings for
`Qwen/Qwen3.6-27B`. The 0.841 headline, the 0.876 untruncated accuracy, the
truncation counts in `DEFAULT_MAX_TOKENS` and the 0.75-0.92 sanity band in the
README are all measurements *of this policy*, and comparing against them under
a different one compares two things.

The task does not impose it. A default temperature is a property of the model
under test, not of the benchmark, and hard-coding Qwen's settings into an eval
that is supposed to be provider-agnostic would be the same mistake upstream
ctfish makes (see `inspect_evals/ctfish/config.py`). It is offered instead:
`hmmt_feb_2026(generate_config=reference_generate_config())`.
"""


def reference_generate_config() -> GenerateConfig:
    """`REFERENCE_SAMPLING` as a `GenerateConfig` that actually carries `top_k`.

    `top_k` is a real `GenerateConfig` field and it is *never read* on Inspect's
    OpenAI-compatible request path, which the `steered` provider and Inspect's
    own `vllm` provider both inherit -- so setting it looks right in the eval
    log and changes nothing about the request. It has to travel in `extra_body`,
    which is forwarded verbatim and which `SteeredAPI.completion_params` merges
    into rather than replaces, so the steering arguments and these coexist.

    `inspect_evals/ctfish/config.py` documents the whole mechanism and carries
    the general routing helper. This eval needs one policy rather than a preset
    system, so it builds that one here instead of taking a dependency on
    another eval's module.
    """
    params = dict(REFERENCE_SAMPLING)
    return GenerateConfig(
        temperature=params["temperature"],
        top_p=params["top_p"],
        extra_body={"top_k": params["top_k"]},
    )


@task
def hmmt_feb_2026(
    prompt_suffix: str = REASONING_PROMPT_SUFFIX,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    epochs: int = DEFAULT_EPOCHS,
    revision: str = HMMT_FEB_2026_REVISION,
    judge_model: str | Model | None = None,
    generate_config: GenerateConfig | None = None,
) -> Task:
    """HMMT February 2026, graded on the boxed final answer.

    Args:
        prompt_suffix: Appended to each problem statement. Defaults to
            `REASONING_PROMPT_SUFFIX`; `MATHARENA_PROMPT_SUFFIX` is the
            shorter upstream wording. Must ask for a `\\boxed{}`.
        max_tokens: Output budget per problem; see `DEFAULT_MAX_TOKENS`.
        epochs: Attempts per problem, averaged. See `DEFAULT_EPOCHS`.
        revision: Dataset commit to load. Change it only to compare against a
            different release, and say so when reporting.
        judge_model: Adds the optional format-blind equivalence judge, which
            regrades the answers strict grading rejected. `None` (the default)
            leaves it off; the strict grader is the eval's definition of
            correct and the judge is a diagnostic beside it.
        generate_config: Sampling policy for the model under test, merged over
            `max_tokens`. `None` (the default) leaves sampling to the provider,
            which keeps the eval provider-agnostic; a `max_tokens` set here
            wins over the `max_tokens` argument. Use
            `reference_generate_config()` to reproduce the conditions the
            numbers in the README were measured under, and read `DEFAULT_EPOCHS`
            before running four epochs against an unknown default.

    Raises:
        ValueError: If `epochs` or `max_tokens` is not positive, or if
            `prompt_suffix` never asks for a `\\boxed{}`.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
    if "\\boxed" not in prompt_suffix:
        # The grader reads the last \boxed{} and nothing else. A suffix that
        # never asks for one does not make the eval lenient, it makes every
        # sample a parse failure -- so refuse it here rather than at scoring
        # time, when 132 generations have already been paid for.
        raise ValueError(
            "prompt_suffix must ask for the answer in \\boxed{}; got "
            f"{prompt_suffix!r}. Use REASONING_PROMPT_SUFFIX or "
            "MATHARENA_PROMPT_SUFFIX."
        )

    scorers: list[Scorer] = [
        hmmt_scorer(),
        hmmt_untruncated_scorer(),
        hmmt_diagnostics(),
    ]
    if judge_model is not None:
        scorers.append(hmmt_equivalence_judge(judge_model))

    config = GenerateConfig(max_tokens=max_tokens).merge(
        generate_config or GenerateConfig()
    )

    return Task(
        dataset=hmmt_dataset(prompt_suffix=prompt_suffix, revision=revision),
        scorer=scorers,
        epochs=Epochs(epochs, "mean"),
        config=config,
        metadata={
            "dataset": HMMT_FEB_2026_PATH,
            "revision": revision,
            "prompt_suffix": prompt_suffix,
            # Recorded so two logs can be compared without trusting that the
            # default was in force in both: one changed word is one changed
            # benchmark, and the hash is what makes that visible.
            "prompt_suffix_sha256": prompt_suffix_sha256(prompt_suffix),
            "max_tokens": max_tokens,
            "judge_model": str(judge_model) if judge_model is not None else None,
            # The sampling policy the task asked for, so a log says what was
            # requested rather than leaving the reader to infer it. Empty means
            # the provider's defaults were accepted, which is a fact about the
            # run and belongs in the record.
            "generate_config": config.model_dump(exclude_none=True),
        },
    )


def prompt_suffix_sha256(prompt_suffix: str) -> str:
    """The hash recorded in task metadata, over the UTF-8 bytes of the suffix."""
    return hashlib.sha256(prompt_suffix.encode("utf-8")).hexdigest()
