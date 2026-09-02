"""Pull the final answer out of a reply and decide whether it matches gold.

A port of the grader in the internal `jevals` reference harness (a private
steering-tools workspace; its records are available to maintainers, not
shipped), from `jevals/grading.py`, which produced the HMMT Feb 2026 reference
runs this eval is cross-validated against
(`scripts/validate_hmmt.py --check regrade`). Keeping its semantics is the
point: a grader that disagrees with those runs would make the saved records
useless as an audit, and would silently redefine the benchmark.

The rules, in order:

1. Grade only the *final* channel. A `\\boxed{}` written while the model is
   still thinking is a draft, not an answer.
2. Take the **last** `\\boxed{...}`, with the closing brace found by counting
   rather than by regex -- `\\boxed{\\frac{1}{2}}` nests, and no regular
   expression gets that right.
3. Normalise display conventions that carry no mathematical content
   (`\\dfrac` vs `\\frac`, `\\left`/`\\right`, thin spaces, `1{,}234`).
4. Compare cheaply first and symbolically last -- exact string, then integer,
   then `math-verify` -- and record which of those decided, so the results
   table says how much machinery the verdicts needed.

No answer at all is a *parse failure*, reported separately from a wrong
answer. The headline accuracy still scores it zero, which is how the published
numbers treat a missing answer, but the two are never conflated in the
diagnostics. A symbolic comparison that could not be made is a third outcome
again, `verifier_failed`, and it is never folded into "incorrect": half this
dataset's answers are decided by `math-verify`, so a broken verifier looks
exactly like a model that lost half its mathematics.

Two deliberate departures from the reference harness's function, both argued
where they occur: `strip_thinking` also refuses an unterminated thinking block,
and `gold_in_content` refuses a match that sits inside a longer number. Neither
changes a single verdict on the 396 saved reference records -- `--check regrade`
is what says so -- and both close a failure the reference data does not happen
to contain.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

GRADING_METHODS = ("exact", "int", "math-verify", "none")
"""Every value `GradeResult.method` can be reported as.

`none` is the label for "no `\\boxed{}` was found", which is what
`GradeResult.method is None` means. Listed here so `grading_method_share()`
can emit a complete, fixed key set even for a run in which some method never
fires -- Inspect needs the key set to be the same for every run.
"""


def strip_thinking(text: str) -> str:
    """Return only the part of *text* that is the model's final channel.

    Three representations reach this function, and all three have to end up at
    the same answer. Verified against inspect_ai 0.3.259's OpenAI-compatible
    path (`inspect_ai.model._openai`), which is what the `steered` provider and
    every other vLLM route inherit:

    * The server splits reasoning into `reasoning_content` (or `reasoning`, or
      `reasoning_details`). Inspect turns that into a `ContentReasoning` block,
      and `ModelOutput.completion` -- the only thing this eval ever grades --
      excludes it. There is no tag left to strip and this function is a no-op.
      This is the case the reference runs were produced under.
    * The reply carries a matched `<think>...</think>` pair inline (a server
      with no reasoning parser). Inspect's `parse_content_with_reasoning`
      lifts it out into a `ContentReasoning` block before the eval sees it, so
      again there is nothing to strip.
    * The reply carries an *unmatched* tag. Inspect leaves those alone --
      verified, `parse_content_with_reasoning` matches only a closed pair --
      and they arrive here as plain text. Two ways that happens: Qwen-style
      chat templates open `<think>` in the prompt, so the completion ends with
      a bare `</think>` and no opener; and a generation truncated mid-thought
      has an opener and no closer.

    So: take everything after the last `</think>` if there is one, and then, in
    whatever remains, drop everything from an unclosed `<think>` onwards --
    that block never ends, so it is unterminated reasoning and only what
    precedes it is final.

    Both steps run, in that order, and that matters. A model that answers,
    reconsiders and is then cut off mid-revision produces
    `<think>…</think> answer <think> wait, maybe \\boxed{99}` -- a closed pair
    followed by an unclosed one. Testing for the closer first and returning
    immediately would hand the draft `\\boxed{99}` back as the final answer,
    which is the exact case this rule exists to refuse. Inspect's
    `parse_content_with_reasoning` lifts out only the first matched pair, so a
    two-block reply reaches this function in precisely that shape.

    The unclosed-opener rule is this port's one addition to the reference
    harness's function. The reference runs were produced under the
    separated-reasoning representation -- 0 of the 396 saved records contain
    either tag -- so the addition cannot change any of the verdicts
    `--check regrade` compares against, and it keeps rule 1 true when the
    reasoning is inline.
    """
    tail = text.rsplit("</think>", 1)[1] if "</think>" in text else text
    return tail.split("<think>", 1)[0] if "<think>" in tail else tail


def extract_boxed(text: str) -> str | None:
    """Return the contents of the last `\\boxed{...}` in *text*, brace-balanced.

    A regex with a greedy or a lazy group gets nested braces wrong
    (`\\boxed{\\frac{1}{2}}` truncates to `\\frac{1`), so the closing brace is
    found by counting depth. An unbalanced box -- the reply was cut off inside
    it -- returns `None`, the same as no box at all.
    """
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def normalize(s: str) -> str:
    """Strip display conventions that carry no mathematical content.

    Deliberately conservative: everything removed here is either pure
    typesetting (`\\left`, `\\!`, `\\dfrac`) or a thousands separator. Anything
    that could change the value is left for `math-verify` to decide.
    """
    s = s.strip()
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\ ", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    # 1{,}234 and 1,234 are display conventions for 1234.
    s = s.replace("{,}", "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    return s


_NUMERIC_NEEDLE = re.compile(r"-?[\d.]+")


def gold_in_content(content: str, gold: str) -> bool:
    """True when the normalised gold appears verbatim in the final channel.

    This is the lenient view reported *beside* the strict verdict, never
    instead of it: a reply graded wrong under the `\\boxed{}` rule still gets
    lenient credit when the exactly-correct gold is written somewhere in the
    final channel -- a boxless final answer, or a rounded decimal boxed next to
    the exact value (`\\frac{875}{128} \\approx \\boxed{6.84}`). Both sides go
    through the same `normalize` as strict grading, so a `\\dfrac` or spacing
    variant of the gold still hits while a rounded decimal alone does not.

    The rule was verified against the reference harness's manual review of the
    −2 steering arm: exactly 7 hits among its 43 non-correct records and 0
    among the unsteered arm's.

    A bare substring test is not enough, and this is the port's one deviation
    here. `normalize` deletes all whitespace, so the gold `10` occurs inside
    `110`, `2010` and `10.5`, and nine of the 33 answers are three characters
    or fewer. Measured over the 396 reference records, asking how often a
    record wins lenient credit for a gold belonging to a *different* problem:
    2.38% with a bare substring test. Requiring the match not to be part of a
    longer number takes that to 1.50% and changes none of the reference
    verdicts -- 0, 2 and 7 hits across the three arms, the same records, which
    is what keeps this comparable with their `metrics.json`.

    It is still not clean. Two-digit numeric golds can be written anywhere in a
    page of prose about a different quantity, and no boundary rule fixes that.
    `lenient` is a formatting diagnostic, not an accuracy. Read it as an upper
    bound, and trust it less the shorter the answers are.
    """
    needle = normalize(gold)
    if not needle:
        # An empty needle would match everything; refuse to credit it.
        return False
    haystack = normalize(strip_thinking(content))
    numeric = _NUMERIC_NEEDLE.fullmatch(needle) is not None
    for match in re.finditer(re.escape(needle), haystack):
        before = haystack[match.start() - 1] if match.start() else ""
        after = haystack[match.end()] if match.end() < len(haystack) else ""
        if before.isdigit() or after.isdigit():
            # Part of a longer number: `10` inside `110` is not the answer 10.
            continue
        if numeric and (
            before == "."
            or (after == "." and _next_is_digit(haystack, match.end()))
        ):
            # `3.10` and `10.5` are not the answer 10 either.
            continue
        return True
    return False


def _next_is_digit(text: str, index: int) -> bool:
    """Whether the character after `text[index]` is a digit."""
    return index + 1 < len(text) and text[index + 1].isdigit()


def _as_int(s: str) -> int | None:
    try:
        return int(normalize(s))
    except ValueError:
        return None


VERIFIER_TIMEOUT_SECONDS = 30.0
"""How long one symbolic comparison may take before it is abandoned.

Generous, because the alternative is worse. math-verify's own default is 5
seconds and it is the budget the reference runs graded under, but a sympy
comparison that is merely slow must not be recorded as a wrong answer, and 30
seconds is far beyond anything the reference records need (the whole 396-record
regrade finishes in seconds). A comparison that exceeds it is reported as
`verifier_failed`, never as incorrect.
"""


def _run_bounded(work: Callable[[], T], seconds: float) -> T:
    """Run `work` on a worker thread and give up on it after `seconds`.

    math-verify implements its own timeouts with `signal.alarm`, which raises
    `ValueError` outside the main thread -- so `parse(..., parsing_timeout=5)`
    turns every symbolic comparison into an exception the moment the grader is
    called from a thread, an event loop's worker pool, or anything else that is
    not `__main__`. Its timeouts are therefore disabled at the call site and the
    budget is imposed here instead, where it works on any thread.

    A thread that overruns cannot be killed in Python, so it is left running as
    a daemon and the process exits without waiting for it. That is the cost of
    the guarantee; it is bounded by one abandoned thread per undecidable
    expression, and `verifier_failed` counts them.

    math-verify logs a warning the first time its timeout is disabled ("you
    must provide the logic for timeout interuption yourself"). Expect it once
    per process, and read it as confirmation rather than as a problem: this
    function is that logic.
    """
    result: list[T] = []
    failure: list[BaseException] = []

    def target() -> None:
        try:
            result.append(work())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            failure.append(exc)

    thread = threading.Thread(target=target, daemon=True, name="hmmt-math-verify")
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(f"symbolic comparison exceeded {seconds}s")
    if failure:
        raise failure[0]
    return result[0]


def _mathverify_equal(pred: str, gold: str) -> bool | None:
    """Symbolic equivalence via math-verify, or `None` when it could not decide.

    Three outcomes, and the third is the point. `True` and `False` are verdicts.
    `None` means the verifier raised, timed out, or produced nothing to compare
    -- which is not the same as "the answer is wrong", and is never recorded as
    if it were. 17 of the 33 HMMT Feb 2026 answers are non-integer LaTeX, so a
    verifier that silently fails takes half the dataset's credit with it and the
    run reads as a capability drop.

    `ImportError` is deliberately not swallowed, for the same reason: an
    environment without `math-verify` should stop the run, not quietly halve it.

    `parsing_timeout` and `timeout_seconds` are both switched off. Passing
    `None` makes math-verify's `timeout` decorator a no-op rather than a
    `signal.alarm`, which is what makes this callable off the main thread;
    `_run_bounded` supplies a real budget in its place.
    """
    from math_verify import parse, verify

    def compare() -> bool | None:
        gold_parsed = parse(f"${gold}$", parsing_timeout=None)
        if not gold_parsed:
            # The gold comes from the dataset. A gold the verifier cannot read
            # is the verifier's problem, not the model's.
            return None
        pred_parsed = parse(f"${pred}$", parsing_timeout=None)
        if not pred_parsed:
            # The prediction does not parse as an expression at all. That is a
            # verdict -- the model boxed something that is not the answer.
            return False
        return bool(verify(gold_parsed, pred_parsed, timeout_seconds=None))

    try:
        return _run_bounded(compare, VERIFIER_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - one undecidable expression, not a bad env
        logger.warning(
            f"math-verify could not compare {pred!r} with {gold!r}: {exc!r}. "
            "Recorded as verifier_failed, not as an incorrect answer."
        )
        return None


@dataclass(frozen=True)
class GradeResult:
    """What the grader decided, and how it decided it."""

    extracted: str | None
    """The contents of the last `\\boxed{}` in the final channel, or None."""

    correct: bool
    """Whether `extracted` matches gold. False whenever nothing was extracted."""

    method: str | None
    """"exact", "int", "math-verify", or None when there was nothing to grade.

    `method == "math-verify"` with `correct == False` means the full cascade
    ran and rejected the answer; `method is None` means no answer was found at
    all. Those are different failures and the diagnostics keep them apart.
    """

    verifier_failed: bool = False
    """Whether the symbolic verifier raised or timed out instead of deciding.

    `correct` is False in that case, because a verdict has to be something, but
    the two are not the same event and the diagnostics report this one
    separately. A run whose `verifier_failed` is non-zero has a broken
    environment, not a weakened model.
    """


def grade_math(reply_content: str, gold: str) -> GradeResult:
    """Grade a reply's final answer against *gold*.

    Args:
        reply_content: The model's visible reply. Pass `ModelOutput.completion`
            (reasoning blocks already excluded); `strip_thinking` handles the
            inline-tag representations.
        gold: The dataset's answer, as LaTeX.

    Cheap deterministic comparisons run first and symbolic equivalence last, so
    the recorded method says how much machinery the verdict needed. A run whose
    `math-verify` share jumps is a run whose answers changed shape, which is
    worth knowing before reading anything into the accuracy.
    """
    extracted = extract_boxed(strip_thinking(reply_content))
    if extracted is None:
        return GradeResult(None, False, None)
    if not normalize(gold):
        # No gold to compare against. Unreachable at the pinned revision --
        # `dataset.py` rejects a blank answer at load time -- but an empty
        # `normalize(gold)` would otherwise match an empty `\boxed{}` and score
        # it correct, which is the one way this grader could invent credit.
        return GradeResult(extracted, False, None)
    if normalize(extracted) == normalize(gold):
        return GradeResult(extracted, True, "exact")
    pred_int, gold_int = _as_int(extracted), _as_int(gold)
    if pred_int is not None and gold_int is not None:
        return GradeResult(extracted, pred_int == gold_int, "int")
    equal = _mathverify_equal(extracted, gold)
    return GradeResult(extracted, equal is True, "math-verify", equal is None)
