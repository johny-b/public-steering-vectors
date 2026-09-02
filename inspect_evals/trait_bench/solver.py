"""Administering a TRAIT item: one generated reply, read back two ways.

`trait_answer_solver` runs after an ordinary generation and works out what the
reply answered, by the two independent channels in `extractor`. It sits in the
solver rather than in the scorer for one reason: two scorers need the result
(the behavioural one and the diagnostics one), and running the extraction
twice would double its cost and let the two scorers disagree about the same
reply.

The solver knows nothing about how the model is served. Thinking on or off,
steered or not, is a property of `--model` and its `-M` arguments; the
diagnostics are what make a mis-served run obvious.
"""

from __future__ import annotations

import logging

from inspect_ai.model import Model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import StoreModel, store_as
from pydantic import Field

from .extractor import (
    EXTRACTOR_MAX_TOKENS,
    Prompt,
    extract_answer,
    extractor_prompt,
    parse_answer_letter,
    read_reasoning,
    resolve_extractor_model,
    visible_text,
)

logger = logging.getLogger(__name__)


class TraitAnswerState(StoreModel):
    """What `trait_answer_solver` leaves behind for the scorers."""

    completion: str = Field(default="")
    """The visible reply, complete. Never truncated -- see the repository rule."""
    reasoning: str | None = Field(default=None)
    """The model's readable reasoning, kept separate and never shown to the
    extractor. `None`, never `""`, when none arrived; see `reasoning_unavailable`
    for why it did not and `extractor.read_reasoning` for the rule."""
    reasoning_unavailable: str | None = Field(default=None)
    """Why `reasoning` is null, or `None` when it is not. The two fields are
    always read together: a null with no reason beside it would be indexed as an
    oversight, and a `reasoning_chars` of zero cannot say whether the model did
    not think or the provider would not show what it thought."""
    reasoning_redacted: list[str] = Field(default_factory=list)
    """Encrypted thinking blocks, verbatim and in order. Never counted as
    reasoning and never sent to the extractor, but never discarded either: they
    are what the provider actually sent, and nothing else in the record can be
    used to re-derive them. Empty on the steered route, which returns its
    reasoning as text."""
    stop_reason: str = Field(default="")
    """Inspect's stop reason, so truncation is a fact rather than an inference."""
    exact_letter: str | None = Field(default=None)
    """The regex channel's answer, or None."""
    llm_letter: str | None = Field(default=None)
    """The extraction model's answer, or None (including its explicit "none")."""
    llm_refusal: bool | None = Field(default=None)
    """Whether the extractor called the reply a refusal; None if it never ran."""
    extractor_model: str = Field(default="")
    """Which model read the replies. Part of the metric's identity."""
    extractor_system_prompt: str = Field(default="")
    """The system message the extraction call sent, verbatim and never
    truncated. Constant across a run, and stored per sample anyway: a record
    that has to be joined against a module constant to be read is a record that
    stops being readable the first time the constant changes."""
    extractor_user_prompt: str = Field(default="")
    """The user message the extraction call sent, verbatim and never truncated:
    the options, then the whole reply. The half that varies per sample, and the
    half a verdict has to be audited against."""
    extractor_prompt_sha256: str = Field(default="")
    """Digest of both messages together; see `extractor.Prompt.sha256`. What
    makes two runs comparable without diffing kilobytes of reply text."""
    extractor_prompt_chars: int = Field(default=0)
    """Length of both messages together. Beside the digest for the reason every
    stored model text in this repository carries both."""
    extractor_response: str = Field(default="")
    """The extractor's raw completion, for when a verdict looks wrong."""
    extractor_reasoning: str | None = Field(default=None)
    """The extractor's own readable reasoning, or `None` when none arrived.

    Null on the package default, and that is a fact about the model rather than
    a gap in the recording: `claude-haiku-4-5` is pre-4.6 and will only summarize
    its thinking against a `budget_tokens` depth, which this eval may not set.
    The three fields here say which of the three cases happened -- readable
    reasoning, nothing returned, encrypted only -- because an earlier version of
    this store said `""` for all three and a base64 signature would have read as
    the extractor's thinking. See `extractor.read_reasoning`.
    """
    extractor_reasoning_unavailable: str | None = Field(default=None)
    """Why `extractor_reasoning` is null, or `None` when it is not."""
    extractor_reasoning_redacted: list[str] = Field(default_factory=list)
    """The extractor's encrypted thinking blocks, verbatim. Kept, never counted."""
    extractor_error: str | None = Field(default=None)
    """The error text if the extraction call failed, else None."""
    extractor_unparseable: bool = Field(default=False)
    """Whether the extractor answered in a shape the schema rejected.

    A separate fact from `extractor_error`, and the reason both exist. An
    error is a call that did not happen; this is a call that happened and came
    back unreadable -- a provider ignoring the response schema, a refusal from
    the extractor itself. Without it, a wholly broken extraction channel is
    indistinguishable from one that read every reply and found no answer, and
    the dual-extraction design would evaporate with nothing on the table
    saying so.
    """


@solver
def trait_answer_solver(
    extractor_model: str | Model | None = None,
    extractor_max_tokens: int | None = None,
) -> Solver:
    """Work out what a generated reply answered, by both channels.

    Args:
        extractor_model: Model for the reading channel, or `None` for
            `$INSPECT_GRADER_MODEL` and then the package default.
        extractor_max_tokens: Reply budget for the extractor, or `None` for
            `extractor.EXTRACTOR_MAX_TOKENS`.

    An extractor failure is recorded, not raised: the exact channel still has
    an answer for most replies, and losing a whole run because a second
    provider was unreachable would be a worse outcome than losing the channel
    that guards against format drift. `extractor_errors` is a diagnostic
    metric precisely so a run that quietly fell back to one channel is not
    mistaken for one that had two, and `extractor_unparseable` is its twin for
    the other way the channel can go: a call that returned, in a shape the
    verdict schema rejects. Both are recorded on the store, so neither can
    read on the table as "the extractor found no answer".
    """
    max_tokens = (
        EXTRACTOR_MAX_TOKENS if extractor_max_tokens is None else extractor_max_tokens
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        store = store_as(TraitAnswerState)
        message = state.output.message if state.output.choices else None
        store.completion = visible_text(message)
        reasoning = read_reasoning(message)
        store.reasoning = reasoning.text
        store.reasoning_unavailable = reasoning.unavailable
        store.reasoning_redacted = list(reasoning.redacted)
        store.stop_reason = (
            str(state.output.stop_reason) if state.output.choices else ""
        )
        store.exact_letter = parse_answer_letter(store.completion)

        options: list[str] = state.metadata["options"]
        try:
            verdict, raw, extractor_reasoning, prompt = await extract_answer(
                extractor_model, options, store.completion, max_tokens
            )
            _store_prompt(store, prompt)
            store.extractor_response = raw
            store.extractor_reasoning = extractor_reasoning.text
            store.extractor_reasoning_unavailable = extractor_reasoning.unavailable
            store.extractor_reasoning_redacted = list(extractor_reasoning.redacted)
            if verdict is None:
                store.extractor_unparseable = True
            else:
                store.llm_letter = None if verdict.answer == "none" else verdict.answer
                store.llm_refusal = verdict.refusal
        except Exception as error:  # noqa: BLE001 - a lost channel, not a lost run
            logger.warning(f"{state.sample_id}: extractor failed: {error}")
            store.extractor_error = str(error)
            # A failed call is the one whose prompt a reader most wants, so it
            # is rebuilt rather than left blank. `extractor_prompt` is pure in
            # its two arguments and is what the call itself used, so these are
            # the bytes that were sent and not a reconstruction of them.
            _store_prompt(store, extractor_prompt(options, store.completion))
            # The call never returned, so there is no reasoning to be had and no
            # provider setting to blame. Saying so here is what keeps a null
            # `extractor_reasoning` from being read as an extractor that ran and
            # thought in silence.
            store.extractor_reasoning_unavailable = (
                "the extraction call raised before returning; see extractor_error"
            )

        store.extractor_model = _extractor_model_name(extractor_model)
        return state

    return solve


def _store_prompt(store: TraitAnswerState, prompt: Prompt) -> None:
    """Record what the extraction call was sent, in full and with its shape."""
    store.extractor_system_prompt = prompt.system
    store.extractor_user_prompt = prompt.user
    store.extractor_prompt_sha256 = prompt.sha256
    store.extractor_prompt_chars = prompt.chars


def _extractor_model_name(extractor_model: str | Model | None) -> str:
    resolved = resolve_extractor_model(extractor_model)
    return resolved if isinstance(resolved, str) else str(resolved)
