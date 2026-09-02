"""What counts as an answer that never arrived -- one definition, two readers.

The model under test is a thinking model with a 32768-token budget, and under
steering it can spend the whole of it deliberating and return nothing visible.
That reply comes back in one of two shapes: a `stop_reason` of `max_tokens` (the
request's own budget ran out) or `model_length` (the server's context did), or a
visible completion that is empty or whitespace once the reasoning blocks are set
aside -- the model thought until there was nothing left to say.

Either shape is a GENERATION failure, not a bad answer. An answer that never
arrived is not an answer worth zero: handed the empty string, both judges score
it near the bottom, and the sample lands as `metric=0, quality=0, gap=0` -- a
real-looking data point manufactured out of a failed generation. The bias is not
even across a steering sweep, because truncation concentrates in exactly the
arms that make the model think longest, so those arms would be handed a score
change that no behavioural difference produced.

Two places act on that and they have to agree exactly:

* `scorers.py` calls `no_answer` before it calls the judges. When it is true it
  calls neither, and records `metric`, `quality` and `gap` as NaN with the
  `no_answer` flag set, so a run logs "no answer" as its own category rather
  than as a score of zero.
* `scripts/analyze_school_of_reward_hacks.py` calls the same predicate when
  reading logs and excludes those samples from every mean. That is what makes
  runs graded BEFORE the scorer knew how -- the nine-arm sweep already
  collected -- analysable at all, and it stays there so the rule remains
  revisable over collected logs without re-running GPU hours or judge calls.

The predicate lives here rather than inside either of them so that the scorer's
flag and the analysis's exclusion can never drift into meaning two different
things. The analysis keeps calling its column `truncated`, the name it has had
since it was the only place that knew about this; it is this predicate under an
older name, not a second rule.

Everything here takes a `ModelOutput`, which is what both callers hold:
`TaskState.output` while scoring, `EvalSample.output` while reading a log.
"""

from __future__ import annotations

from inspect_ai.model import ModelOutput

TRUNCATING_STOP_REASONS = frozenset({"max_tokens", "model_length"})
"""Stop reasons that mean the reply was cut off rather than finished.

`max_tokens` is the request's own budget and `model_length` is the server's
context window. Neither says anything about what the model wanted to write; both
say the text stopped for a reason outside the model's control.
"""


def stop_reason(output: ModelOutput) -> str | None:
    """The first choice's stop reason, or None when there is no choice at all.

    A `ModelOutput` with no choices has no `stop_reason` to read -- the property
    would raise -- so the absence is reported rather than guessed at. Such an
    output is still caught by `answer_empty`, since its completion is "".
    """
    if not output.choices:
        return None
    return str(output.stop_reason)


def stop_truncated(output: ModelOutput) -> bool:
    """True when generation stopped at the token budget or the context limit."""
    reason = stop_reason(output)
    return reason is not None and reason in TRUNCATING_STOP_REASONS


def answer_empty(output: ModelOutput) -> bool:
    """True when the visible completion is empty or nothing but whitespace.

    `ModelOutput.completion` is the text content of the final assistant message,
    and `ChatMessage.text` concatenates the `text` blocks only -- reasoning
    arrives beside them as `ContentReasoning`, whether from Anthropic's thinking
    or from a vLLM server's `reasoning_content`, and is not part of it. So a
    reply that is 32k tokens of thinking and nothing else is empty here, which is
    exactly the case this exists to catch.
    """
    return (output.completion or "").strip() == ""


def no_answer(output: ModelOutput) -> bool:
    """True when this reply carries nothing gradable: either half is enough.

    The union rather than one or the other: a reply cut off at the budget is not
    a whole answer even when some text arrived, and a reply that stopped cleanly
    with nothing visible is not an answer at all. The two halves stay separately
    available above, because "ran out of budget mid-answer" and "thought until
    there was nothing left to say" are different facts about the steering.
    """
    return stop_truncated(output) or answer_empty(output)
