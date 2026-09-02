"""An optional format-blind regrade of the answers strict grading rejected.

Off by default, and a diagnostic view only. The mechanical grader in
`grader.py` is the eval's definition of correct; this scorer exists to say how
much of a run's failure is the model getting the mathematics wrong and how much
is it declining to put the answer in a box. Where the two numbers differ, the
difference is a formatting story.

The prompt is taken from `jevals/judge.py` in the internal `jevals` reference
harness (a private steering-tools workspace; its records are available to
maintainers, not shipped), kept word for word -- including the clause that
refuses a rounded decimal, which is what stops the judge from handing back the
credit `gold_in_content` deliberately withholds. Three deviations from the
harness's implementation, all argued at the point they occur:
the reply and the problem are sent in full rather than clipped, the verdict is
requested as structured output with the free-text fallbacks kept underneath,
and the judge's own reasoning is requested and stored.

Every verdict carries `judge_reasoning` in its score metadata: the judge's
summarised reasoning in full, or `None` when the provider returned none. The
house rule is that a model call's reasoning is captured or its absence is
recorded, and both halves matter -- a judge whose reasoning is billed and
discarded is a judge nobody can audit.

Every sample that reached the judge also carries `judge_prompt`, the rendered
prompt in full, with `judge_prompt_sha256` and `judge_prompt_chars` beside it.
`JUDGE_PROMPT_SHA256` pins the template; this pins what was substituted into
it, which is the problem, the gold answer and the final channel of a reply
whose stripping rule is the subtlest thing in this file. The two samples that
never reach the judge carry none of the three, because no prompt was rendered
for them.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Literal

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    ResponseSchema,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import json_schema
from pydantic import BaseModel, ValidationError

from .grader import grade_math, strip_thinking
from .scorer import visible_completion

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MAX_TOKENS = 4096
"""Reply budget for one verdict.

Deliberately below Inspect's 8192-token auto-streaming threshold: Anthropic's
SDK dropped `temperature` from `messages.stream()`, so a streamed judge call
raises TypeError and every sample silently scores an error instead of a
verdict. Same reason as the ctfish stage judge; a verdict is two short fields.
"""

JUDGE_PROMPT = (
    "You are grading a model's answer to a question. Decide ONLY whether the "
    "final answer stated in the response is equivalent to the gold answer. "
    "Ignore formatting, missing boxes, extra prose, and reasoning quality. "
    "Mathematical or semantic equivalence counts (e.g. 0.5 vs 1/2, 'option B' "
    "vs 'B'). A rounded decimal is NOT equivalent to an exact non-terminating "
    "value unless the exact value is also stated. If the response states no "
    "final answer, it is not equivalent."
    "\n\nQUESTION:\n{problem}"
    "\n\nGOLD ANSWER: {gold}"
    "\n\nMODEL RESPONSE (final channel only):\n{content}"
    '\n\nReply with strict JSON: {{"equivalent": true|false, '
    '"reason": "<one sentence>"}}'
)
"""The reference harness's judge prompt, verbatim.

Not reworded, for the same reason the dataset is pinned: this prompt is the
instrument, and the reference verdicts were produced with it. Only the source
formatting differs -- the one long paragraph is split across adjacent string
literals to stay inside the line limit, and `--check grader` asserts the
assembled string still hashes to sha256 `02574357…`, which is the copy in
`jevals/judge.py`. The response schema below constrains the shape of the reply
without changing a word of what is asked.
"""

JUDGE_PROMPT_SHA256 = (
    "02574357a53cb1615beb8899b0793fe0ef4165b3e72163b153cb4bac70a5fa37"
)
"""sha256 of `JUDGE_PROMPT`, so a reflow that changed a byte fails a check."""


class EquivalenceVerdict(BaseModel):
    """The judge's answer. `reason` is required so a verdict is always auditable."""

    equivalent: bool
    reason: str


@scorer(metrics={"*": [mean(), stderr()]})
def hmmt_equivalence_judge(
    model: str | Model | None = None,
    temperature: float | None = 0.0,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    reasoning_summary: Literal["none", "concise", "detailed", "auto"] | None = None,
) -> Scorer:
    """Regrade strict failures format-blind.

    Args:
        model: The judge. `None` uses Inspect's default model, which is the
            model under test -- almost never what you want, so pass one.
        temperature: Judge sampling temperature. One sample is taken per
            failure and nothing averages its variance away, so this defaults to
            0. `None` inherits the provider's default.
        max_tokens: Reply budget per verdict; see `DEFAULT_JUDGE_MAX_TOKENS`.
        reasoning_summary: How hard to ask the provider for a summary of the
            judge's own reasoning. `None`, the default, is a request rather
            than a refusal: Inspect's OpenAI path reads it as "find out",
            probing once whether the organisation may have reasoning summaries
            and then sending `summary="auto"` if it may and `"none"` if it may
            not. Forcing `"auto"` skips that probe and makes every judge call
            fail for an unverified organisation — asking harder would get less.
            The Anthropic path in inspect_ai 0.3.259 sends the
            summarised-thinking display only alongside `reasoning_effort`,
            which this eval will not set on someone else's judge, so an
            Anthropic judge returns no reasoning and `judge_reasoning` is
            `None`. That is recorded, not hidden.

    Reports:

    * `judge_correct` -- strict correct, or the judge calling the stated
      answer equivalent. NaN when the judge failed, so a broken judge shrinks
      the denominator instead of silently scoring zeros.
    * `judge_recovered` -- answers the judge credited that strict grading did
      not. This is the size of the formatting effect, and it is the number to
      quote.
    * `judge_errors` -- judge calls that raised or came back unreadable.

    Two samples are never sent to the judge, and both exclusions save a call
    whose answer is already determined. A sample the strict grader marked
    correct cannot change. A reply whose final channel is empty has no stated
    answer, and the prompt's own last sentence says an answer that is not
    stated is not equivalent -- so it is scored as a non-recovery here rather
    than paid for. The reference harness's `judge_dir` makes the same second
    exclusion; on the reference arms it is 11, 7 and 19 of the records.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # The prompt promises the judge the final channel only, so it has to be
        # the final channel. `visible_completion` already excludes a separated
        # `ContentReasoning` block, but an unmatched inline tag survives it --
        # a Qwen-style bare `</think>`, or a reply cut off mid-revision -- and
        # a draft `\boxed{}` on the wrong side of that tag is exactly what the
        # strict grader refused. Sending it would let the judge hand back
        # credit for scratch work under the name `judge_recovered`.
        completion = strip_thinking(visible_completion(state))
        strict = grade_math(visible_completion(state), target.text)
        if strict.correct:
            return Score(
                value={
                    "judge_correct": 1.0,
                    "judge_recovered": 0.0,
                    "judge_errors": 0.0,
                },
                answer=strict.extracted,
                explanation="Strict grading already correct; judge not called.",
                metadata={"judged": False, "strict_correct": True},
            )

        if not completion.strip():
            return Score(
                value={
                    "judge_correct": 0.0,
                    "judge_recovered": 0.0,
                    "judge_errors": 0.0,
                },
                explanation="Empty final channel; no answer to judge.",
                metadata={
                    "judged": False,
                    "strict_correct": False,
                    "empty_completion": True,
                },
            )

        problem = (state.metadata or {}).get("problem") or state.input_text
        prompt = JUDGE_PROMPT.format(
            problem=problem, gold=target.text, content=completion
        )
        # Recorded before the call, so the sample whose judge raised keeps the
        # prompt that raised it. Every path below that made a call carries these
        # three keys; the two paths above it, which made none, carry neither a
        # prompt nor a digest of one.
        sent = {
            "judge_prompt": prompt,
            "judge_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "judge_prompt_chars": len(prompt),
        }

        try:
            output = await get_model(model).generate(
                [ChatMessageUser(content=prompt)],
                config=GenerateConfig(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_summary=reasoning_summary,
                    response_schema=ResponseSchema(
                        name="EquivalenceVerdict",
                        json_schema=json_schema(EquivalenceVerdict),
                        strict=True,
                    ),
                ),
            )
        except Exception as e:  # noqa: BLE001 - never discard a sample over a judge
            logger.warning(f"hmmt_equivalence_judge failed: {e}")
            return Score(
                value={
                    "judge_correct": float("nan"),
                    "judge_recovered": float("nan"),
                    "judge_errors": 1.0,
                },
                explanation=f"Judge call failed: {e}",
                metadata={
                    "judged": False,
                    "strict_correct": False,
                    "judge_reasoning": None,
                    **sent,
                },
            )

        reasoning = judge_reasoning(output)
        verdict = _parse_verdict(output.completion)
        if verdict is None:
            return Score(
                value={
                    "judge_correct": float("nan"),
                    "judge_recovered": float("nan"),
                    "judge_errors": 1.0,
                },
                # The reply itself is on the metadata below, whole. Nothing
                # here holds a clipped copy of model output: the house rule
                # allows truncation in a display layer, and `explanation` is a
                # stored field, not one.
                explanation="Judge reply could not be read; see judge_reply.",
                metadata={
                    "judged": True,
                    "strict_correct": False,
                    "judge_reply": output.completion,
                    "judge_reasoning": reasoning,
                    **sent,
                },
            )

        equivalent, reason = verdict
        return Score(
            value={
                "judge_correct": float(equivalent),
                "judge_recovered": float(equivalent),
                "judge_errors": 0.0,
            },
            answer=strict.extracted,
            explanation=f"Judge: equivalent={equivalent} — {reason}",
            metadata={
                "judged": True,
                "strict_correct": False,
                "judge_reason": reason,
                "judge_reply": output.completion,
                "judge_reasoning": reasoning,
                **sent,
            },
        )

    return score


def judge_reasoning(output: ModelOutput) -> str | None:
    """The judge's own summarised reasoning, or None if it returned none.

    `None` is recorded rather than omitted: "this provider does not return
    reasoning" and "this call did no reasoning" both need to be readable from
    the log, and an absent key is neither. The text is stored whole.
    """
    message = output.message
    if not isinstance(message, ChatMessageAssistant) or isinstance(
        message.content, str
    ):
        return None
    blocks = [
        block.reasoning for block in message.content if block.type == "reasoning"
    ]
    return "\n".join(blocks) if blocks else None


def _parse_verdict(text: str) -> tuple[bool, str] | None:
    """Read a verdict out of a judge reply, or return None if there isn't one.

    Three attempts, in order of how much they trust the reply. The last one
    exists because the judge is asked for a one-sentence reason about LaTeX,
    and an unescaped `\\frac` makes the JSON invalid while leaving the boolean
    -- the part that is actually scored -- perfectly readable.
    """
    try:
        verdict = EquivalenceVerdict.model_validate_json(_json_block(text))
        return verdict.equivalent, verdict.reason
    except (ValidationError, ValueError):
        pass

    match = re.search(r'"equivalent"\s*:\s*(true|false)', text)
    if match:
        return (
            match.group(1) == "true",
            "reason unreadable (probably LaTeX escapes); boolean recovered",
        )
    return None


def _json_block(text: str) -> str:
    """Pull the JSON object out of a reply that may be fenced or prefixed with prose."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text
