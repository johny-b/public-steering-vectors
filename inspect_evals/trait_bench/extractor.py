"""Two independent readings of what a generated reply answered.

A generative administration of a multiple-choice instrument has to turn prose
back into a letter, and there is no single right way to do it. So it is done
twice:

* an exact regex over the format instruction the prompt asked for, which is
  incapable of interpretation and therefore incapable of interpreting wrongly;
* a small model that reads the whole reply and says which option it committed
  to, which is capable of interpretation and therefore catches the replies
  that answered clearly in some other shape.

Where the two agree, the answer is about as certain as it gets. Where only one
speaks, that one is used and the fact is counted. Where they disagree, the
sample is left unscored and counted, because a disagreement means the reply
was ambiguous and picking a winner would be inventing data. The rates are
metrics in their own right (`agreement`, `exact_only`, `llm_only`): a steering
condition that quietly stops following the format instruction shows up there
before it shows up as a personality change.

The model channel is also the only thing that can tell a refusal ("I don't
have a personality") apart from a formatting failure, which is why it exists
at all rather than a second regex.

`visible_text` and `read_reasoning` live here rather than beside the solver
because the line they draw -- what a reader sees, against what the model
thought -- is this module's line: the extractor is shown the first and never
the second. Reading a reasoning block is a named function with a rule in it
rather than a comprehension, because the two providers this repository
generates and extracts on fill the block differently and a comprehension that
reads one of them records a cryptographic signature from the other; see
`read_reasoning`.

What the extractor was asked is recorded next to what it answered. `Prompt`
carries the two messages a call sends, verbatim and never truncated, and the
solver stores both with a digest and a length beside them. A reading channel
whose verdict looks wrong is only diagnosable against the words that produced
it, and those words are not recoverable from a store that keeps the reply
alone.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentReasoning,
    GenerateConfig,
    Model,
    ResponseSchema,
    get_model,
)
from inspect_ai.util import json_schema
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_EXTRACTOR_MODEL = "anthropic/claude-haiku-4-5"
"""The model that reads replies when neither the caller nor the environment says.

Small and cheap on purpose. This is a reading task with a five-way answer and
the whole reply in front of it, not a judgement; spending a frontier model on
it would buy nothing and would make the extractor a meaningful share of the
run's cost. It is still recorded in the score metadata, because "which model
read the replies" is part of what produced the number.
"""

EXTRACTOR_MAX_TOKENS = 1024
"""Reply budget for one extraction.

Deliberately under Inspect's 8192-token auto-streaming threshold: the
Anthropic SDK dropped `temperature` from `messages.stream()`, so a streamed
call raises `TypeError` and every extraction would come back an error. The
verdict is a letter and a boolean; 1024 is already generous.
"""

ANSWER_LETTERS = ("A", "B", "C", "D")
"""The four letters a TRAIT item offers."""

_ANSWER_LINE = re.compile(r"Answer:\s*\(?([A-Z])\)?\.?(?!\w)")
"""The format line the prompt asked for.

The last match wins, not the first: a reply may restate the instruction, or
quote the prompt's own cue, before committing at the end. Optional
parentheses and an optional full stop are allowed because models add them; a
trailing word character is not, so "Answer: According to..." does not parse
as "A".

Pattern and rule are both the sibling personality-steering study's
(`psybat/parsing.py::parse_letter`), so that the two harnesses record the same
exact-channel answer for the same reply. The rule matters as much as the
pattern: the last match is taken and then validated, rather than the matches
being scanned backwards for the last *valid* one. On a reply ending
"Answer: (Z)" that scan would revive a letter the model had moved on from;
taking the final match and rejecting it leaves the reading to the model
channel, where an ambiguous reply belongs, and counts it.
"""

_WHITESPACE = re.compile(r"\s+")


class AnswerVerdict(BaseModel):
    """What the extraction model reports about one reply.

    Both fields are required and neither is optional-typed. An optional field
    becomes an `anyOf` in the JSON schema, and Anthropic rejects schemas
    carrying `additionalProperties` inside an `anyOf`, which is what Inspect
    emits for structured output. "No answer" is therefore a value of `answer`
    rather than the absence of one.
    """

    answer: Literal["A", "B", "C", "D", "none"]
    """The option the reply committed to, or "none" if it committed to none."""

    refusal: bool
    """Whether the reply declined to answer, as opposed to merely not answering."""


EXTRACTOR_SYSTEM_PROMPT = """You read one reply that a language model gave to \
a four-option multiple-choice question, and report which option it chose.

Report the letter A, B, C or D when the reply commits to exactly one of the \
four options, however it phrases it -- by letter, by quoting or paraphrasing \
the option, or by describing what it would do. Report "none" when it commits \
to no single option: it refuses, it spreads itself across several, it answers \
something else, or it was cut off before choosing.

Set refusal to true only when the reply declines to answer -- for example \
because it says it has no personality, preferences or opinions. A reply that \
tried to answer but never named an option is not a refusal.

Report what the reply says. Do not judge which option would be better."""


def resolve_extractor_model(model: str | Model | None) -> str | Model:
    """Decide which model reads the replies.

    Args:
        model: An explicit model, or `None` to fall back.

    Resolution order is argument, then `$INSPECT_GRADER_MODEL`, then
    `DEFAULT_EXTRACTOR_MODEL` -- the same order the other model-graded evals in
    this repository use, so one environment variable redirects every auxiliary
    model call in a sweep.
    """
    if model is not None:
        return model
    return os.environ.get("INSPECT_GRADER_MODEL") or DEFAULT_EXTRACTOR_MODEL


SUMMARIZED_THINKING: dict[str, Any] = {"type": "adaptive", "display": "summarized"}
"""The thinking block asked of an Anthropic extractor: display, and only display.

`type: adaptive` is the mode Claude 4.6+ already runs in, and no depth is named:
there is no `budget_tokens` and no `effort`, so the extractor thinks exactly as
much as it would have thought anyway. The one thing this asks for is `display:
summarized`, which turns the summary the model already wrote from something the
API withholds into something it returns. That is the whole intervention, and it
is why it can be made without the reading channel becoming a different
instrument.

It has to go in as a model argument rather than in `GenerateConfig`, and that is
Inspect's shape rather than a preference. Inspect 0.3.259 builds `thinking`
itself only inside `is_using_thinking`, which is false unless `reasoning_effort`
or `reasoning_tokens` is set -- and setting either also sets
`output_config.effort`, which *is* a depth and would make the extractor's
behaviour a choice of this eval's (`_providers/anthropic.py`,
`completion_config`). `GenerateConfig.extra_body` is no way round it either:
only the fields in `anthropic_extra_body_fields()` are lifted out of it. The
provider's own `extra_body` model argument is merged into the request body
untouched, which is the seam this uses.
"""


def summarized_thinking_args(model_name: str) -> dict[str, Any]:
    """Model arguments that make this extractor return its reasoning, if it can.

    Args:
        model_name: The extractor, as `provider/model`.

    Returns:
        `{"extra_body": {"thinking": ...}}` on an Anthropic model that supports
        adaptive thinking, and `{}` on every other model, including every other
        provider and including this package's own default extractor.

    Gated rather than sent everywhere, because sending it where it is not
    understood costs the whole reading channel. `DEFAULT_EXTRACTOR_MODEL` is
    `claude-haiku-4-5`, and that model answers this request with HTTP 400,
    "adaptive thinking is not supported on this model" (checked live against the
    API, not inferred): it is a pre-4.6 model, where the only route to a summary
    is `budget_tokens`, which is a depth this eval may not set. An ungated
    version of this helper would therefore turn every extraction on the default
    model into an error, and `extractor_errors` would sit at 1.0 for the run.

    So the question asked here is the provider's own -- `is_claude_frontier`,
    which is what Inspect itself gates adaptive thinking on
    (`_providers/anthropic.py:1246`) -- and it is asked of the model that will
    actually be called. The default extractor answers no and simply records that
    its reasoning was unavailable and why; an `$INSPECT_GRADER_MODEL` pointed at
    a frontier Claude answers yes and its summaries cost nothing extra, because
    the tokens were thought and billed either way.
    """
    if not model_name.startswith("anthropic/"):
        return {}
    try:
        api: Any = get_model(model_name).api
        frontier = getattr(api, "is_claude_frontier", None)
        if frontier is None or not frontier():
            return {}
    except Exception as error:  # noqa: BLE001 - a probe may not raise a run
        logger.debug(f"Could not ask {model_name} about adaptive thinking: {error}")
        return {}
    return {"extra_body": {"thinking": dict(SUMMARIZED_THINKING)}}


def extraction_model(model: str | Model | None) -> Model:
    """The resolved extractor, asked for summarized thinking where that is free.

    Args:
        model: An explicit model, or `None` to fall back; see
            `resolve_extractor_model`.

    A `Model` passed in by a caller is used exactly as given. Model arguments
    are fixed when a `Model` is built, so there is no way to add the thinking
    request to one that already exists, and quietly rebuilding a caller's
    extractor from its name would discard whatever else that caller configured
    -- a caller that passes a mock model with a fixed list of outputs would
    find the fixture replaced by a live call.
    """
    resolved = resolve_extractor_model(model)
    if isinstance(resolved, Model):
        return resolved
    return get_model(resolved, **summarized_thinking_args(resolved))


def no_reasoning_reason(model_name: str) -> str:
    """Why an extractor returned no reasoning, in a sentence a reader can act on.

    Args:
        model_name: The extractor that was called, as `provider/model`.

    Returns:
        The reason, for `extractor_reasoning_unavailable`.

    The generic answer -- the provider returned no reasoning blocks -- is true
    both of a model that did not think and of a model that thought and was never
    asked to say so, and on this package's default extractor it is the second.
    Recording only the generic sentence would leave whoever reads a run of
    nulls to rediscover, from the Anthropic docs and an HTTP 400, why a whole
    column is empty and whether anything can be done about it. So where the
    reason is known it is written down, and the null becomes a finding rather
    than a gap.
    """
    if model_name.startswith("anthropic/") and not summarized_thinking_args(model_name):
        return (
            f"{model_name} was never asked for summarized thinking: "
            "summarized_thinking_args withholds that request unless Inspect's "
            "Anthropic provider reports the model as frontier, and a "
            "pre-frontier model returns a thinking summary only against a "
            "budget_tokens depth, which this eval does not set because a depth "
            "would make the extractor's behaviour a property of the eval. "
            "Nothing was withheld and nothing is missing from the record. "
            "Point $INSPECT_GRADER_MODEL at a frontier Claude to get "
            "summaries; see summarized_thinking_args."
        )
    return "provider returned no reasoning blocks"


def parse_answer_letter(
    text: str, valid: tuple[str, ...] = ANSWER_LETTERS
) -> str | None:
    """The exact channel: the letter the format instruction asked for.

    Args:
        text: The visible reply. Never the reasoning -- a model that thinks
            "maybe (C)" and then answers (A) must score A.
        valid: Letters that count as an answer.

    Returns:
        The letter, or `None` if the reply does not contain exactly the shape
        that was asked for.

    Two shapes are accepted and no more: a trailing `Answer: (X)` line, and a
    reply that is nothing but the letter. Everything else -- a letter
    mentioned in passing, two candidate letters, an answer given only in prose,
    a final `Answer:` naming a letter outside `valid` -- is `None` here by
    design and is the model channel's problem. Widening this regex, or
    widening the last-match rule into a search for the last valid match, would
    make the two channels agree more often while making the agreement mean
    less; see `_ANSWER_LINE`.
    """
    matches = _ANSWER_LINE.findall(text)
    if matches:
        last = matches[-1]
        if last in valid:
            return last
    bare = _WHITESPACE.sub(" ", text).strip().strip(".").strip().upper()
    for letter in valid:
        if bare in (letter, f"({letter})", f"OPTION {letter}"):
            return letter
    return None


def format_extractor_input(options: list[str], reply: str) -> str:
    """The user message for one extraction: the options, then the whole reply.

    The options are included because a reply often answers by describing what
    it would do rather than by naming a letter, and the extractor cannot map
    that back without knowing what the letters were. The reply is passed
    complete and never truncated: the answer is usually at the end, and a
    truncation rule would silently change which replies can be read.
    """
    lettered = "\n".join(
        f"({letter}) {option}" for letter, option in zip(ANSWER_LETTERS, options)
    )
    return (
        f"The four options were:\n{lettered}\n\n"
        f"The reply to read:\n<reply>\n{reply}\n</reply>"
    )


@dataclass(frozen=True)
class Prompt:
    """The exact messages one extraction call sends.

    Two fields rather than one string, because two messages go on the wire and
    a joined copy would be a rendering nobody sent. The digest and the length
    below cover the pair, so two runs can be compared without diffing kilobytes
    of reply text.
    """

    system: str
    """The system message, verbatim: what the extractor is asked to do."""

    user: str
    """The user message, verbatim: the options, then the whole reply."""

    @property
    def sha256(self) -> str:
        """Digest of both messages, joined by a byte neither can contain."""
        return hashlib.sha256(
            "\x00".join((self.system, self.user)).encode("utf-8")
        ).hexdigest()

    @property
    def chars(self) -> int:
        """Length of both messages together."""
        return len(self.system) + len(self.user)


def extractor_prompt(options: list[str], reply: str) -> Prompt:
    """The messages one extraction sends, built in one place.

    Args:
        options: The four options in the order the model saw them.
        reply: The full visible reply.

    Pure and deterministic in its arguments, which is what lets a caller record
    the prompt of a call that raised before it could hand one back: the record
    is then the same bytes the failed call sent, not an approximation of them.
    """
    return Prompt(
        system=EXTRACTOR_SYSTEM_PROMPT,
        user=format_extractor_input(options, reply),
    )


async def extract_answer(
    model: str | Model | None,
    options: list[str],
    reply: str,
    max_tokens: int = EXTRACTOR_MAX_TOKENS,
) -> tuple[AnswerVerdict | None, str, Reasoning, Prompt]:
    """Ask the extraction model which option a reply chose.

    Args:
        model: The extraction model; see `resolve_extractor_model`.
        options: The four options in the order the model saw them.
        reply: The full visible reply.
        max_tokens: Reply budget; see `EXTRACTOR_MAX_TOKENS`.

    Returns:
        `(verdict, raw_completion, reasoning, prompt)`. The verdict is `None`
        when the reply came back in a shape the schema does not accept, which is
        a different outcome from the call failing and is counted separately, as
        `extractor_unparseable`. The raw completion and the reasoning are
        returned either way so the log holds what was actually said and
        whatever thinking was returned with it. The reasoning is a `Reasoning`
        rather than a string, so a call whose reasoning was withheld is not
        recorded as a call that did not reason. The prompt is the messages this
        call sent, handed back so the caller records what was asked rather than
        rebuilding a guess at it.

    Raises:
        Exception: Whatever the provider raises. The caller decides what a
            failed extraction means; this function does not swallow it, so an
            expired key cannot masquerade as a run of unreadable replies.

    The repository rule is that every model call in a study records its
    reasoning, and this call is asked for its reasoning twice over, by the two
    routes providers offer. `reasoning_summary` on the config is read by
    Inspect's OpenAI providers; `extraction_model` adds a display-only
    `thinking` block as a model argument for the Anthropic provider, which does
    not read that field. Both are visibility and neither is depth: reasoning
    depth is deliberately not configured here, because a depth set by this eval
    would make the extractor's behaviour a property of the eval rather than of
    the model being served.

    On the default extractor neither route produces anything, and that is the
    honest outcome rather than an oversight. `claude-haiku-4-5` is pre-4.6, so
    Anthropic will only summarize its thinking when a `budget_tokens` depth is
    set, and setting one is the thing this eval may not do; the recorded
    reasoning is JSON `null` with `extractor_reasoning_unavailable` saying which
    of the two reasons applies. See `summarized_thinking_args`.
    """
    extractor = extraction_model(model)
    prompt = extractor_prompt(options, reply)
    output = await extractor.generate(
        [
            ChatMessageSystem(content=prompt.system),
            ChatMessageUser(content=prompt.user),
        ],
        config=GenerateConfig(
            max_tokens=max_tokens,
            reasoning_summary="auto",
            response_schema=ResponseSchema(
                name="AnswerVerdict",
                json_schema=json_schema(AnswerVerdict),
                strict=True,
            ),
        ),
    )
    completion = output.completion
    reasoning = read_reasoning(output.message if output.choices else None)
    if reasoning.text is None and not reasoning.redacted:
        # `read_reasoning` can only report what did not arrive. This is the one
        # place that knows which model was called and what it was asked for, so
        # it is where the generic "no reasoning blocks" is replaced by the
        # reason, when there is a known one.
        reasoning = Reasoning(unavailable=no_reasoning_reason(str(extractor)))
    try:
        verdict = AnswerVerdict.model_validate_json(_json_block(completion))
    except (ValidationError, ValueError):
        logger.warning(f"Extractor returned no usable verdict: {completion[:200]!r}")
        return None, completion, reasoning, prompt
    return verdict, completion, reasoning, prompt


def visible_text(message: ChatMessageAssistant | None) -> str:
    """The part of a reply a reader would see, with reasoning blocks removed.

    Inspect's `.text` already keeps only `ContentText` parts, so this is the
    same string; it is a named function because "the extractor never sees the
    thinking" is a property of the design rather than an accident of which
    attribute was reached for, and `read_reasoning` is its other half.
    """
    if message is None:
        return ""
    return message.text


@dataclass(frozen=True)
class Reasoning:
    """What a provider returned about how a reply was arrived at.

    Three fields rather than one string, because "the model returned no
    reasoning" and "the model's reasoning arrived encrypted" are different facts
    and only the first is about the model. Collapsing both to `""`, which this
    module used to do, made a provider setting look like a model that did not
    think, and it made the record say nothing where it could have said why.
    """

    text: str | None = None
    """The readable reasoning, or `None` when none arrived.

    Never the empty string. An empty string would assert that the model reasoned
    and produced no words, which is not a thing any provider here reports;
    `None` is the honest value and `unavailable` is where the reason goes.
    """

    redacted: tuple[str, ...] = ()
    """Blocks that arrived encrypted, verbatim and in order.

    Kept rather than dropped. They are not readable and must never be counted as
    reasoning, but they are what the provider actually sent, they are the
    evidence that the model thought at all, and a later Inspect or a later API
    version may make them readable. Discarding them would be the one loss this
    record cannot be re-derived from.
    """

    unavailable: str | None = None
    """Why `text` is `None`, in a sentence a reader can act on.

    `None` exactly when `text` is not `None`. The pair is what stops a null
    reasoning field being read as an oversight.
    """

    @property
    def chars(self) -> int:
        """Length of the readable reasoning, and 0 when there is none."""
        return len(self.text or "")


def read_reasoning(message: ChatMessageAssistant | None) -> Reasoning:
    """Split a reply's reasoning blocks into what can be read and what cannot.

    Inspect 0.3.259 fills `ContentReasoning` differently on the two providers
    this eval is run against, and the difference is not cosmetic:

    - the Anthropic provider, which serves the extractor, puts the readable
      summary in `summary` and the block's cryptographic *signature* in
      `reasoning`, and sets `redacted=True` on both the summarized and the
      genuinely redacted case (`_providers/anthropic.py:3759` and `:3776`);
    - the OpenAI-compatible route this repository's steered server serves, which
      generates the replies under test, puts the reasoning text in `reasoning`
      with `redacted` left at its `False` default and no `summary`
      (`_openai.py:803`, `content_reasoning_from_openai_reasoning`, reached
      from `chat_message_assistant_from_openai`).

    Reading `reasoning` on both, which this module used to do, was right for the
    model under test and wrong for the extractor: it would have recorded a
    base64 signature as though it were the extractor's thinking, and recorded
    nothing at all when only a summary arrived. The rule below is what makes one
    function right on both routes: a summary is readable reasoning, an
    unredacted `reasoning` is readable reasoning, and a redacted `reasoning` is
    an opaque blob that is kept but never counted.

    Args:
        message: The assistant message, or `None` when there was no reply.

    Returns:
        A `Reasoning`. `text` is `None` rather than `""` when nothing readable
        arrived, and `unavailable` then says which of the two reasons applies.

    Reasoning is kept, and measured, and never sent to the extractor. A model
    whose visible reply shrinks to nothing while its reasoning grows has not
    become unreadable; it has become a different administration of the
    instrument, and `reasoning_chars` next to `completion_chars` is what shows
    that -- which is why a zero in that column has to be distinguishable from a
    withheld one, and why this returns three fields rather than a string.
    """
    if message is None or isinstance(message.content, str):
        return Reasoning(unavailable="provider returned no reasoning blocks")
    readable: list[str] = []
    redacted: list[str] = []
    for block in message.content:
        if not isinstance(block, ContentReasoning):
            continue
        if block.summary:
            readable.append(block.summary)
        elif block.reasoning and not block.redacted:
            readable.append(block.reasoning)
        elif block.reasoning:
            redacted.append(block.reasoning)
    if readable:
        return Reasoning(text="\n".join(readable), redacted=tuple(redacted))
    if redacted:
        return Reasoning(
            redacted=tuple(redacted),
            unavailable=(
                f"provider returned {len(redacted)} redacted thinking block(s) "
                "and no summary; the blocks are kept verbatim beside this field "
                "but they are encrypted and are not the model's reasoning in "
                "readable form"
            ),
        )
    return Reasoning(unavailable="provider returned no reasoning blocks")


def reasoning_text(message: ChatMessageAssistant | None) -> str:
    """The readable reasoning of a reply, or `""` when there is none.

    The string form of `read_reasoning`, for the callers that only want to
    measure a length. It returns `""` for both "nothing arrived" and "only
    encrypted blocks arrived", so anything that records what happened should use
    `read_reasoning` and keep the distinction.
    """
    return read_reasoning(message).text or ""


def _json_block(text: str) -> str:
    """Pull the JSON object out of a reply that may be fenced or prefixed."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text
