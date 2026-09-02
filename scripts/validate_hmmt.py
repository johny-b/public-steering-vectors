"""Check the HMMT eval without spending anything on the model under test.

Four checks, none of which needs an API key:

    python scripts/validate_hmmt.py --check grader     # offline
    python scripts/validate_hmmt.py --check dataset    # needs Hugging Face
    python scripts/validate_hmmt.py --check mockllm    # needs the dataset
    python scripts/validate_hmmt.py --check regrade --records <dir>

`--check all` runs the first three.

The strongest of them is `regrade`. The `capability-evals` runs of the internal
`jevals` reference harness (a private steering-tools workspace; its records are
available to maintainers, not shipped) saved every completion they graded, so
this eval's grader can be pointed at them and its verdicts diffed against
theirs, one record at a time, for free. A grader that has quietly drifted --
looser, stricter, or differently confused by LaTeX -- shows up as a
disagreement count before any money is spent, and before any number from this
eval is compared with theirs. It gates on five comparisons, not one: the
verdicts, the extracted strings, the lenient-hit count, the accuracy and the
item bootstrap.

`grader` needs no network, but the check that every gold answer grades itself
does need the dataset, so it lives in `dataset`. This eval does not vendor the
answer key: `dataset` pins it by sha256 instead.

House precedent for a purpose-built validation script rather than unit tests:
`scripts/machiavelli_random_agent.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from hmmt import (
    GRADING_METHODS,
    HMMT_FEB_2026_PROBLEMS,
    HMMT_FEB_2026_REVISION,
    JUDGE_PROMPT,
    JUDGE_PROMPT_SHA256,
    MATHARENA_PROMPT_SUFFIX,
    REASONING_PROMPT_SUFFIX,
    extract_boxed,
    gold_in_content,
    grade_math,
    hmmt_dataset,
    hmmt_feb_2026,
    prompt_suffix_sha256,
    strip_thinking,
)

REASONING_SUFFIX_SHA256 = (
    "e4af2c73eebe21e7b31b929aad972d553a3143da0cd5b76cd51e42468a32286d"
)
"""The hash recorded in the reference harness's run configs.

Asserted here so a whitespace edit to `REASONING_PROMPT_SUFFIX` cannot silently
break comparability with the runs this eval's sanity band comes from.
"""

EXPECTED_NON_INTEGER_ANSWERS = 17
"""17 of the 33 answers are non-integer LaTeX.

This is the fact that rules out both published AIME scorers, so it is asserted
rather than remembered: if a revision ever made every answer an integer, the
symbolic half of the grading cascade would stop being exercised and nobody
would notice.
"""

MIN_REGRADE_AGREEMENT_RATIO = 130 / 132
"""Share of a reference arm's records that must agree before the grader is trusted.

A ratio rather than a count. The plan's "≥130/132" is a proportion, and only
the three known arms happen to hold 132 records: pointed at a partial run, a
re-run at a different `samples_per_problem`, or a 33-record smoke, an absolute
threshold fails a perfect regrade; pointed at a larger directory it passes on a
third of the verdicts agreeing. Both directions were reproduced before this was
changed.
"""

REFERENCE_RECORD_FIELDS = ("key", "item_id", "content", "gold", "correct", "truncated")
"""Fields `--check regrade` reads from every reference record.

Checked once, by name, before the loop. A reference run written by a later
version of the reference harness should say which field it stopped carrying,
not raise a KeyError from inside a 400-iteration loop.
"""

ANSWER_KEY_SHA256 = (
    "eb35eb8245934bea78fb38599e4484c363a7b59648d01c949fd7d2e7a10d4947"
)
"""sha256 over the answer key at the pinned revision, as a drift detector.

Computed over `"{problem_idx}\\t{answer}"` lines, sorted by problem index and
joined with newlines. A digest rather than the 33 answers themselves: the
dataset is CC BY-NC-SA 4.0 and this eval's claim, repeated in `dataset.py` and
the README, is that it is never vendored. Copying the answer key into a file
bound for a public repository would make that claim false, and a hand-typed
second copy is in any case a weaker check than a hash — it can be mistyped in
the same way the thing it is checking can.

The cost is that the 33-gold self-grade needs the dataset, so it lives in
`--check dataset` rather than in the offline `--check grader`. `--check all`
runs both.
"""


# --------------------------------------------------------------------------
# --check grader
# --------------------------------------------------------------------------

BOXED_IN_THINKING_CLOSED = (
    "<think>Let me guess \\boxed{99} for now.</think>"
    "After checking, the answer is \\boxed{42}."
)
BOXED_IN_THINKING_CLOSER_ONLY = (
    "Let me guess \\boxed{99} for now.</think>The answer is \\boxed{42}."
)
BOXED_IN_THINKING_UNCLOSED = "<think>Let me guess \\boxed{99} and keep going, but I ran"
BOXED_IN_SECOND_THINKING_UNCLOSED = (
    "<think>First pass.</think>The answer is 42."
    "<think>Wait, on reflection maybe \\boxed{99} and then I ran"
)
"""A closed thinking block, an answer, then a second block that never closes.

The shape a reasoning model produces when it starts revising and is cut off,
and the one a closer-first `strip_thinking` credits the draft from.
"""

GRADER_FIXTURES: list[tuple[str, str, str, bool, str | None, bool]] = [
    # (name, completion, gold, correct, method, lenient)
    (
        "dfrac vs frac",
        "So the answer is \\boxed{\\dfrac{7}{2}}.",
        "\\frac{7}{2}",
        True,
        "exact",
        True,
    ),
    (
        "frac vs dfrac gold",
        "\\boxed{\\frac{7}{2}}",
        "\\dfrac{7}{2}",
        True,
        "exact",
        True,
    ),
    (
        "1{,}234 thousands separator",
        "\\boxed{1{,}234}",
        "1234",
        True,
        "exact",
        True,
    ),
    (
        "1,234 thousands separator",
        "\\boxed{1,234}",
        "1234",
        True,
        "exact",
        True,
    ),
    (
        "degrees dropped",
        "\\boxed{74}",
        "74^\\circ",
        True,
        "math-verify",
        True,
    ),
    (
        "degrees braced",
        "\\boxed{74^{\\circ}}",
        "74^\\circ",
        True,
        "math-verify",
        True,
    ),
    (
        "degrees spelled",
        "\\boxed{74\\degree}",
        "74^\\circ",
        True,
        "math-verify",
        True,
    ),
    (
        "sqrt vs fractional exponent",
        "\\boxed{69^{1/2}}",
        "\\sqrt{69}",
        True,
        "math-verify",
        True,
    ),
    (
        "rounded decimal is not the exact value",
        "The value is \\frac{875}{128} \\approx \\boxed{6.84}.",
        "\\frac{875}{128}",
        False,
        "math-verify",
        # Lenient credit is deserved here and only here: the exact gold IS
        # written in the final channel, immediately before the rounded box.
        True,
    ),
    (
        "rounded decimal alone gets nothing",
        "\\boxed{6.84}",
        "\\frac{875}{128}",
        False,
        "math-verify",
        False,
    ),
    (
        "exact decimal expansion is credited",
        # 875/128 is 6.8359375 exactly, so math-verify is right to credit it.
        # See the note in check_grader() about the plan's wording here.
        "\\boxed{6.8359375}",
        "\\frac{875}{128}",
        True,
        "math-verify",
        True,
    ),
    (
        "nested braces",
        "\\boxed{\\frac{1}{2}}",
        "\\frac{1}{2}",
        True,
        "exact",
        True,
    ),
    (
        "nested braces, symbolic gold",
        "\\boxed{\\frac{\\sqrt{1740}}{3}}",
        "\\frac{\\sqrt{1740}}{3}",
        True,
        "exact",
        True,
    ),
    (
        "last box wins",
        "First I thought \\boxed{7}, but actually \\boxed{48}.",
        "48",
        True,
        "exact",
        True,
    ),
    (
        "wrong answer is wrong",
        "\\boxed{7}",
        "48",
        False,
        "int",
        False,
    ),
    (
        "no box is a parse failure, not a wrong answer",
        "The answer is 48.",
        "48",
        False,
        None,
        # Lenient does credit it: the gold is there, unboxed. That is exactly
        # the gap between the strict and lenient views.
        True,
    ),
    (
        "empty completion",
        "",
        "48",
        False,
        None,
        False,
    ),
    (
        "unbalanced box (cut off mid-answer)",
        "The answer is \\boxed{\\frac{1}{2",
        "\\frac{1}{2}",
        False,
        None,
        False,
    ),
    (
        "boxed draft in closed thinking must not win",
        BOXED_IN_THINKING_CLOSED,
        "42",
        True,
        "exact",
        True,
    ),
    (
        "boxed draft in closed thinking is not the answer",
        BOXED_IN_THINKING_CLOSED,
        "99",
        False,
        "int",
        False,
    ),
    (
        "boxed draft before a bare closing tag must not win",
        BOXED_IN_THINKING_CLOSER_ONLY,
        "42",
        True,
        "exact",
        True,
    ),
    (
        "boxed draft before a bare closing tag is not the answer",
        BOXED_IN_THINKING_CLOSER_ONLY,
        "99",
        False,
        "int",
        False,
    ),
    (
        "boxed draft in unclosed thinking is not an answer at all",
        BOXED_IN_THINKING_UNCLOSED,
        "99",
        False,
        None,
        False,
    ),
    (
        "boxed draft in an unclosed SECOND thinking block is not an answer",
        BOXED_IN_SECOND_THINKING_UNCLOSED,
        "99",
        False,
        None,
        False,
    ),
    (
        "the answer before an unclosed second block still counts leniently",
        BOXED_IN_SECOND_THINKING_UNCLOSED,
        "42",
        False,
        None,
        True,
    ),
    (
        "empty gold is never credited leniently",
        "anything at all",
        "",
        False,
        None,
        False,
    ),
    (
        "empty gold does not match an empty box either",
        "\\boxed{}",
        "",
        False,
        None,
        False,
    ),
    (
        # normalize() deletes whitespace, so a bare substring test credits the
        # gold 10 to a reply whose only number is 110.
        "a short numeric gold does not match inside a longer number",
        "After simplifying we get 110.",
        "10",
        False,
        None,
        False,
    ),
    (
        "a short numeric gold does not match inside a decimal",
        "The ratio is 10.5 exactly.",
        "10",
        False,
        None,
        False,
    ),
    (
        "a short numeric gold still matches when it stands alone",
        "The answer is 10.",
        "10",
        False,
        None,
        True,
    ),
]


def check_grader() -> bool:
    """The fixture table and the grader's helpers, offline.

    The check that matters most — every gold answer grading itself — needs the
    dataset's answers, and this eval does not vendor them, so it runs in
    `--check dataset` instead. `--check all` runs both.

    One deviation from the plan's fixture list, recorded here because it is a
    correction rather than an omission. The plan asks that `\\frac{875}{128}`
    versus `6.8359375` be wrong under strict grading *and* uncredited by
    math-verify. It is wrong under strict grading, but 875/128 is 6.8359375
    exactly, so math-verify credits it and is right to. The case the plan
    means -- and the one that actually occurs in the reference harness's
    records, named in the `gold_in_content` docstring -- is the *rounded*
    decimal 6.84, which is tested above and is correctly refused by both. The
    exact expansion is tested too, with the verdict it deserves.
    """
    ok = True

    print(f"grader: {len(GRADER_FIXTURES)} fixtures")
    for name, completion, gold, want_correct, want_method, want_lenient in (
        GRADER_FIXTURES
    ):
        result = grade_math(completion, gold)
        lenient = result.correct or gold_in_content(completion, gold)
        problems = []
        if result.correct != want_correct:
            problems.append(f"correct={result.correct}, expected {want_correct}")
        if result.method != want_method:
            problems.append(f"method={result.method!r}, expected {want_method!r}")
        if lenient != want_lenient:
            problems.append(f"lenient={lenient}, expected {want_lenient}")
        if problems:
            ok = False
            print(f"  FAIL {name}: {'; '.join(problems)}")
        else:
            print(f"  ok   {name}")

    print("grader: symbolic grading off the main thread")
    # math-verify implements its timeouts with signal.alarm, which raises
    # outside the main thread. Under the library's defaults every symbolic
    # comparison would come back False there, silently, and 17 of the 33
    # answers are decided symbolically -- so the failure would read as the
    # model losing half the dataset. This is the fixture that stops that from
    # ever regressing quietly.
    symbolic = [("\\boxed{69^{1/2}}", "\\sqrt{69}"), ("\\boxed{0.4}", "\\frac{2}{5}")]
    for label, runner in [("main thread", _run_here), ("worker thread", _run_threaded)]:
        verdicts = runner(
            lambda: [grade_math(pred, gold) for pred, gold in symbolic]
        )
        credited = sum(1 for verdict in verdicts if verdict.correct)
        failed = sum(1 for verdict in verdicts if verdict.verifier_failed)
        if credited != len(symbolic) or failed:
            ok = False
            print(
                f"  FAIL {label}: {credited}/{len(symbolic)} credited, "
                f"{failed} verifier failures"
            )
        else:
            print(f"  ok   {label}: {credited}/{len(symbolic)} credited")

    print("grader: helpers")
    helper_cases = [
        (
            "strip_thinking closed",
            strip_thinking(BOXED_IN_THINKING_CLOSED).strip(),
            "After checking, the answer is \\boxed{42}.",
        ),
        (
            "strip_thinking closer only",
            strip_thinking(BOXED_IN_THINKING_CLOSER_ONLY).strip(),
            "The answer is \\boxed{42}.",
        ),
        (
            "strip_thinking unclosed",
            strip_thinking(BOXED_IN_THINKING_UNCLOSED).strip(),
            "",
        ),
        (
            "strip_thinking unclosed second block",
            strip_thinking(BOXED_IN_SECOND_THINKING_UNCLOSED).strip(),
            "The answer is 42.",
        ),
        ("strip_thinking plain", strip_thinking("plain text"), "plain text"),
        (
            "extract_boxed nested",
            extract_boxed("\\boxed{\\frac{1}{2}}"),
            "\\frac{1}{2}",
        ),
        ("extract_boxed spaced", extract_boxed("\\boxed {5}"), "5"),
        ("extract_boxed missing", extract_boxed("no box here"), None),
    ]
    for name, got, want in helper_cases:
        if got != want:
            ok = False
            print(f"  FAIL {name}: {got!r}, expected {want!r}")
        else:
            print(f"  ok   {name}")

    print("grader: prompt suffixes")
    got_sha = prompt_suffix_sha256(REASONING_PROMPT_SUFFIX)
    if got_sha != REASONING_SUFFIX_SHA256:
        ok = False
        print(
            f"  FAIL REASONING_PROMPT_SUFFIX sha256 is {got_sha}, "
            f"reference runs used {REASONING_SUFFIX_SHA256}"
        )
    else:
        print(
            f"  ok   REASONING_PROMPT_SUFFIX sha256 {got_sha[:12]}… "
            "matches the reference runs"
        )
    judge_sha = hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()
    if judge_sha != JUDGE_PROMPT_SHA256:
        ok = False
        print(
            f"  FAIL JUDGE_PROMPT sha256 is {judge_sha}, "
            f"expected {JUDGE_PROMPT_SHA256}"
        )
    else:
        print(f"  ok   JUDGE_PROMPT sha256 {judge_sha[:12]}… matches jevals/judge.py")
    for name, suffix in [
        ("REASONING_PROMPT_SUFFIX", REASONING_PROMPT_SUFFIX),
        ("MATHARENA_PROMPT_SUFFIX", MATHARENA_PROMPT_SUFFIX),
    ]:
        if "\\boxed" not in suffix:
            ok = False
            print(f"  FAIL {name} does not ask for \\boxed{{}}")
        else:
            print(f"  ok   {name} asks for \\boxed{{}}")

    print("grader: task construction refuses a suffix with no box")
    try:
        hmmt_feb_2026(prompt_suffix="\n\nAnswer briefly.")
    except ValueError as e:
        print(f"  ok   ValueError: {e}")
    else:
        ok = False
        print("  FAIL a suffix that never asks for a box was accepted")

    return ok


def _run_here(work: Any) -> Any:
    """Call `work` on this thread."""
    return work()


def _run_threaded(work: Any) -> Any:
    """Call `work` on a worker thread and hand back what it returned."""
    box: list[Any] = []
    failure: list[BaseException] = []

    def target() -> None:
        try:
            box.append(work())
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            failure.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return box[0]


def answer_key_sha256(golds: dict[str, str]) -> str:
    """The digest `ANSWER_KEY_SHA256` pins, over `{idx}\\t{answer}` lines."""
    lines = [f"{idx}\t{golds[idx]}" for idx in sorted(golds, key=int)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# --check dataset
# --------------------------------------------------------------------------


def check_dataset() -> bool:
    """Load the pinned revision and assert everything the eval relies on."""
    ok = True
    dataset = hmmt_dataset(prompt_suffix=REASONING_PROMPT_SUFFIX)
    print(f"dataset: {len(dataset)} samples at revision {HMMT_FEB_2026_REVISION}")

    if len(dataset) != HMMT_FEB_2026_PROBLEMS:
        ok = False
        print(f"  FAIL expected {HMMT_FEB_2026_PROBLEMS} samples")
    else:
        print(f"  ok   {HMMT_FEB_2026_PROBLEMS} samples")

    ids = [str(sample.id) for sample in dataset]
    if len(set(ids)) != len(ids):
        ok = False
        print(f"  FAIL duplicate sample ids: {[i for i in ids if ids.count(i) > 1]}")
    else:
        print("  ok   sample ids unique")

    non_integer = 0
    subjects: Counter[str] = Counter()
    for sample in dataset:
        target = sample.target if isinstance(sample.target, str) else sample.target[0]
        metadata = sample.metadata or {}
        for field in ("problem_idx", "problem", "problem_type"):
            if field not in metadata:
                ok = False
                print(f"  FAIL sample {sample.id} has no {field} in metadata")
        if not target.strip():
            ok = False
            print(f"  FAIL sample {sample.id} has an empty answer")
        try:
            int(target.strip())
        except ValueError:
            non_integer += 1
        problem = str(metadata.get("problem", ""))
        if not sample.input.endswith(REASONING_PROMPT_SUFFIX):
            ok = False
            print(f"  FAIL sample {sample.id} input does not end with the suffix")
        if not sample.input.startswith(problem):
            ok = False
            print(f"  FAIL sample {sample.id} input does not start with the problem")
        for subject in metadata.get("problem_type", []):
            subjects[subject] += 1
            if subject != subject.strip():
                ok = False
                print(f"  FAIL sample {sample.id} subject {subject!r} is not stripped")

    if non_integer != EXPECTED_NON_INTEGER_ANSWERS:
        ok = False
        print(
            f"  FAIL {non_integer} non-integer answers, "
            f"expected {EXPECTED_NON_INTEGER_ANSWERS}"
        )
    else:
        print(f"  ok   {non_integer} non-integer answers")
    print("  ok   problem_type tags stripped")
    print(f"  subjects: {dict(sorted(subjects.items()))}")

    golds = {
        str(sample.id): (
            sample.target if isinstance(sample.target, str) else sample.target[0]
        )
        for sample in dataset
    }

    print("dataset: the answer key is the one the pinned revision published")
    got_digest = answer_key_sha256(golds)
    if got_digest != ANSWER_KEY_SHA256:
        ok = False
        print(
            f"  FAIL answer key sha256 is {got_digest}, expected {ANSWER_KEY_SHA256}. "
            "An answer changed under a revision this eval claims to have pinned."
        )
    else:
        print(f"  ok   answer key sha256 {got_digest[:12]}…")

    print("dataset: every gold answer self-grades")
    # The check that matters most. If the grader cannot credit the dataset's
    # own answer written back verbatim, nothing else it says means anything.
    for problem_idx, gold in sorted(golds.items(), key=lambda kv: int(kv[0])):
        result = grade_math(f"The answer is \\boxed{{{gold}}}.", gold)
        if not (result.correct and result.method == "exact"):
            ok = False
            print(
                f"  FAIL problem {problem_idx} gold {gold!r}: "
                f"correct={result.correct} method={result.method!r}"
            )
    print(f"  {len(golds)} golds checked")

    return ok


# --------------------------------------------------------------------------
# --check mockllm
# --------------------------------------------------------------------------

EXPECTED_METRIC_KEYS = {
    "hmmt_scorer": {"correct", "lenient", "boxed_found", *GRADING_METHODS},
    "hmmt_untruncated_scorer": {"correct"},
    "hmmt_diagnostics": {
        "truncated_generations",
        "max_tokens_stops",
        "model_length_stops",
        "parse_failures",
        "empty_completions",
        "completion_tokens",
        "reasoning_present",
        "verifier_failed",
    },
    "hmmt_equivalence_judge": {"judge_correct", "judge_recovered", "judge_errors"},
}
"""Every metric key the eval must report, per scorer.

Asserted end to end rather than by reading the scorers, because the thing that
actually breaks is the wiring: a key that is only set on some branch, or a
metric spec that does not expand the way it looks like it does.
"""


def check_mockllm() -> bool:
    """Run the whole task against a scripted model and read the metric table.

    The scripted model answers most problems correctly, and is made to fail in
    one specific way per problem so that every branch of every scorer fires in
    a single run: a truncated generation with no answer, a correct answer with
    no box, a wrong answer, an empty reply, a reply whose reasoning arrives
    separated, a reply the provider reported no usage for, and a reply with an
    inline unmatched `</think>` and a draft answer before it. The equivalence
    judge runs too, against the same mock, whose replies it cannot read --
    which is how the `judge_errors` path gets exercised without an API key --
    and every prompt it is sent is captured and inspected.
    """
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessage,
        ChatMessageAssistant,
        Content,
        ContentReasoning,
        ContentText,
        ModelOutput,
        ModelUsage,
        StopReason,
        get_model,
    )

    ok = True
    dataset = hmmt_dataset(prompt_suffix=REASONING_PROMPT_SUFFIX)
    gold_by_input = {}
    for sample in dataset:
        target = sample.target if isinstance(sample.target, str) else sample.target[0]
        gold_by_input[sample.input] = (str(sample.id), target)

    truncated_id, unboxed_id, wrong_id, empty_id, reasoning_id = "1", "2", "3", "4", "5"
    no_usage_id, inline_thinking_id = "6", "7"
    draft_answer = "-987654321"
    hidden_draft = "-123456789"
    """A number that must never leave the model's thinking block.

    Distinct from `draft_answer`, which is the wrong-answer sample's real boxed
    answer and legitimately reaches the judge. Only the inline-thinking reply
    writes this one, and only before its `</think>`, so finding it in a judge
    prompt means the thinking block was sent under a heading that promised the
    final channel.
    """
    judge_prompts: list[str] = []

    def reply(
        content: str | list[Content],
        stop_reason: StopReason = "stop",
        usage: bool = True,
    ) -> ModelOutput:
        """One scripted reply, with usage filled in unless asked otherwise.

        mockllm only invents usage on its iterator path; a callable's output
        is returned untouched, so `completion_tokens` would silently read 0
        unless it is set here. Every reply claims the same 100 output tokens,
        which makes the expected mean a number rather than a sum of lengths --
        and one reply reports no usage at all, which is what exercises the
        NaN branch that keeps "not measured" apart from "nothing generated".
        """
        return ModelOutput(
            model="mockllm",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=content, model="mockllm"),
                    stop_reason=stop_reason,
                )
            ],
            usage=(
                ModelUsage(input_tokens=10, output_tokens=100, total_tokens=110)
                if usage
                else None
            ),
        )

    def respond(
        messages: list[ChatMessage], tools: object, tool_choice: object, config: object
    ) -> ModelOutput:
        text = messages[-1].text
        if "GOLD ANSWER:" in text:
            # A judge call. The mock cannot produce a verdict, which is the
            # point: judge_errors is a first-class outcome, not a crash. The
            # prompt is kept so the assertions below can read what was sent.
            judge_prompts.append(text)
            return reply("I cannot say.")
        problem_id, gold = gold_by_input[text]
        if problem_id == truncated_id:
            return reply("Let me work through this. First we observe", "max_tokens")
        if problem_id == unboxed_id:
            return reply(f"The final answer is {gold}.")
        if problem_id == wrong_id:
            return reply(f"Therefore \\boxed{{{draft_answer}}}.")
        if problem_id == empty_id:
            return reply("")
        if problem_id == reasoning_id:
            # A separated reasoning block, the representation the vLLM route
            # produces. Its draft answer must not be graded, and it is what
            # makes `reasoning_present` non-zero.
            return reply(
                [
                    ContentReasoning(
                        reasoning=f"Maybe \\boxed{{{draft_answer}}}? Let me redo it."
                    ),
                    ContentText(text=f"Hence the answer is \\boxed{{{gold}}}."),
                ]
            )
        if problem_id == no_usage_id:
            return reply(f"Hence the answer is \\boxed{{{gold}}}.", usage=False)
        if problem_id == inline_thinking_id:
            # The Qwen-style representation: a bare closing tag with no opener,
            # which Inspect leaves in the completion. The draft before it must
            # reach neither the grader nor the judge.
            return reply(
                f"Draft: the answer might be \\boxed{{{hidden_draft}}}.</think>"
                "\n\nFinal: I am not sure."
            )
        return reply(f"Hence the answer is \\boxed{{{gold}}}.")

    epochs = 2
    model = get_model("mockllm/model", custom_outputs=respond)
    logs = inspect_eval(
        hmmt_feb_2026(epochs=epochs, max_tokens=256, judge_model=model),
        model=model,
        log_dir=None,
        display="none",
        score_display=False,
    )
    log = logs[0]
    print(f"mockllm: status={log.status}")
    if log.status != "success":
        ok = False
        if log.error:
            print(f"  FAIL {log.error.message}")
        return ok

    # Inspect reports a dict-valued score as one EvalScore per key, named for
    # the key, holding mean/stderr; a metric that itself returned a dict (here
    # grading_method_share) puts its keys on the metric names instead. Both
    # shapes are flattened into one {scorer: {key: mean}} table.
    seen: dict[str, dict[str, float]] = {}
    for score in log.results.scores if log.results else []:
        bucket = seen.setdefault(score.scorer or score.name, {})
        for metric_ in score.metrics.values():
            if metric_.name == "mean":
                bucket[score.name] = metric_.value
            elif metric_.name != "stderr":
                bucket[metric_.name] = metric_.value

    for scorer_name, expected in EXPECTED_METRIC_KEYS.items():
        got = set(seen.get(scorer_name, {}))
        missing = expected - got
        if missing:
            ok = False
            print(
                f"  FAIL {scorer_name} is missing {sorted(missing)} "
                f"(has {sorted(got)})"
            )
        else:
            print(f"  ok   {scorer_name}: {sorted(got)}")

    metadata = log.eval.metadata or {}
    for key in (
        "dataset",
        "revision",
        "prompt_suffix",
        "prompt_suffix_sha256",
        "max_tokens",
    ):
        if key not in metadata:
            ok = False
            print(f"  FAIL task metadata has no {key}")
    if metadata.get("prompt_suffix_sha256") != REASONING_SUFFIX_SHA256:
        ok = False
        print("  FAIL task metadata prompt_suffix_sha256 is not the reference hash")
    else:
        print("  ok   task metadata records the pinned revision and the suffix hash")

    scorer_values = seen.get("hmmt_scorer", {})
    diagnostics = seen.get("hmmt_diagnostics", {})
    n = HMMT_FEB_2026_PROBLEMS
    # Five problems are answered wrongly: truncated, unboxed, wrong, empty and
    # the one whose only box sits before an inline `</think>`. Four of the five
    # have no boxed answer in their final channel; only the unboxed one writes
    # the gold in plain text, so only it earns lenient credit.
    expectations = [
        ("hmmt_scorer correct", scorer_values.get("correct"), (n - 5) / n),
        ("hmmt_scorer lenient", scorer_values.get("lenient"), (n - 4) / n),
        ("hmmt_scorer boxed_found", scorer_values.get("boxed_found"), (n - 4) / n),
        ("hmmt_scorer none share", scorer_values.get("none"), 4 / n),
        ("diagnostics truncated", diagnostics.get("truncated_generations"), 1 / n),
        ("diagnostics max_tokens_stops", diagnostics.get("max_tokens_stops"), 1 / n),
        ("diagnostics model_length_stops", diagnostics.get("model_length_stops"), 0.0),
        ("diagnostics parse_failures", diagnostics.get("parse_failures"), 4 / n),
        ("diagnostics empty_completions", diagnostics.get("empty_completions"), 1 / n),
        # The separated-reasoning reply and the inline-tag one.
        ("diagnostics reasoning_present", diagnostics.get("reasoning_present"), 2 / n),
        # The mean skips the problem whose provider reported no usage; a 0
        # there would have dragged it to 96.97.
        ("diagnostics completion_tokens", diagnostics.get("completion_tokens"), 100.0),
        ("diagnostics verifier_failed", diagnostics.get("verifier_failed"), 0.0),
        # 32 problems scored; the truncated one is excluded, and of the
        # remaining 32 four replies are incorrect.
        (
            "untruncated correct",
            seen.get("hmmt_untruncated_scorer", {}).get("correct"),
            (n - 5) / (n - 1),
        ),
        # Four of the five non-correct samples reach the judge, which cannot be
        # read; the empty reply is not sent at all.
        (
            "judge judge_errors",
            seen.get("hmmt_equivalence_judge", {}).get("judge_errors"),
            4 / n,
        ),
    ]
    for name, got, want in expectations:
        if got is None or not math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9):
            ok = False
            print(f"  FAIL {name} = {got}, expected {want}")
        else:
            print(f"  ok   {name} = {got:.4f}")

    print("mockllm: completion_tokens tells 'not measured' from 'nothing generated'")
    # The mean alone cannot prove this: a provider that reported 0 output
    # tokens for one problem and 100 for the rest would still be a mean of
    # 96.97, not 100, but a run where every problem reported 100 would look
    # identical to this one. `unscored_samples` on the key's own row is what
    # says the missing measurement was dropped rather than counted as zero.
    token_rows = [
        score
        for score in (log.results.scores if log.results else [])
        if score.scorer == "hmmt_diagnostics" and score.name == "completion_tokens"
    ]
    unscored = sum(score.unscored_samples or 0 for score in token_rows)
    # One, not two: the epoch reducer runs first, so both of the no-usage
    # reply's epochs reduce to a single NaN for that problem.
    if len(token_rows) != 1 or unscored != 1:
        ok = False
        print(
            f"  FAIL {len(token_rows)} completion_tokens row(s) reporting "
            f"{unscored} unscored samples, expected 1 row and 1"
        )
    else:
        print("  ok   the no-usage problem is unscored for this key, not zero")

    print("mockllm: what the judge was actually sent")
    if len(judge_prompts) != 4 * epochs:
        ok = False
        print(
            f"  FAIL judge was called {len(judge_prompts)} times, expected "
            f"{4 * epochs} (five wrong answers, minus the empty reply)"
        )
    else:
        print(
            f"  ok   judge called {len(judge_prompts)} times, "
            "never on an empty reply"
        )
    leaked = [prompt for prompt in judge_prompts if hidden_draft in prompt]
    if leaked:
        ok = False
        print(
            f"  FAIL {len(leaked)} judge prompt(s) carried a draft answer from "
            "before a </think> tag, under a heading promising the final channel"
        )
    else:
        print("  ok   no judge prompt carried a pre-</think> draft answer")

    print("mockllm: the known metric-less score row")
    empty_rows = [
        (score.scorer, score.name)
        for score in (log.results.scores if log.results else [])
        if not score.metrics
    ]
    if empty_rows != [("hmmt_scorer", "hmmt_scorer")]:
        ok = False
        print(
            f"  FAIL metric-less rows are {empty_rows}, expected exactly "
            "[('hmmt_scorer', 'hmmt_scorer')] — see hmmt_scorer's docstring"
        )
    else:
        print("  ok   exactly the one documented empty row")

    return ok


# --------------------------------------------------------------------------
# --check regrade
# --------------------------------------------------------------------------


def check_regrade(records_dir: Path) -> bool:
    """Re-grade a saved reference run and diff the verdicts against its own.

    Args:
        records_dir: A `capability-evals` output directory holding
            `records.jsonl` (and, where it exists, `metrics.json`).

    Every record carries the completion the reference harness graded, the gold,
    its verdict and which method decided it. Regrading the same text with this
    eval's grader and comparing is as close to a controlled experiment on the
    grader as this repository can get for free.

    Five things are compared and all five can fail the check: the verdicts, the
    extracted strings, the lenient-hit count, the record-level accuracy and the
    item bootstrap. Anything printed as a bare number is printed because it has
    no counterpart to be checked against, and the summary line at the end names
    which sub-checks were gating.
    """
    records_path = records_dir / "records.jsonl"
    if not records_path.exists():
        print(f"regrade: FAIL no records.jsonl in {records_dir}")
        return False

    records = [
        json.loads(line)
        for line in records_path.read_text().splitlines()
        if line.strip()
    ]
    ok_records = [r for r in records if r.get("error") is None]
    print(
        f"regrade: {records_dir.name} — {len(records)} records, "
        f"{len(ok_records)} without errors"
    )
    if not ok_records:
        print("  FAIL no error-free records to regrade")
        return False

    absent = sorted(
        {
            field
            for field in REFERENCE_RECORD_FIELDS
            for r in ok_records
            if field not in r
        }
    )
    if absent:
        print(
            f"  FAIL reference records are missing {absent}; this check reads "
            f"{list(REFERENCE_RECORD_FIELDS)} and cannot run without them"
        )
        return False

    failed_checks: list[str] = []
    agree = 0
    n_correct = 0
    extracted_agree = 0
    disagreements: list[dict[str, Any]] = []
    method_changes: list[dict[str, Any]] = []
    for record in ok_records:
        result = grade_math(record["content"], record["gold"])
        n_correct += int(result.correct)
        if result.extracted == (record.get("extracted") or None):
            extracted_agree += 1
        if result.correct == bool(record["correct"]):
            agree += 1
        else:
            disagreements.append({"record": record, "result": result})
        if record.get("method") != result.method:
            method_changes.append({"record": record, "result": result})

    ratio = agree / len(ok_records)
    print(f"  agreement: {agree}/{len(ok_records)} = {ratio:.4f}")
    if disagreements:
        print(f"  {len(disagreements)} disagreement(s):")
        for item in disagreements:
            record, result = item["record"], item["result"]
            print(
                f"    {record['key']} (problem {record['item_id']}): "
                f"theirs correct={record['correct']} method={record.get('method')!r} "
                f"extracted={record.get('extracted')!r} | "
                f"ours correct={result.correct} method={result.method!r} "
                f"extracted={result.extracted!r} | gold={record['gold']!r} | "
                f"diagnosis: {_diagnose(record, result)}"
            )
    else:
        print("  no disagreements")

    if method_changes:
        print(
            f"  {len(method_changes)} record(s) decided by a different "
            "cascade rung (verdict unchanged):"
        )
        for item in method_changes:
            record, result = item["record"], item["result"]
            print(
                f"    {record['key']}: theirs {record.get('method')!r} "
                f"-> ours {result.method!r}"
            )

    print(f"  extracted-string agreement: {extracted_agree}/{len(ok_records)}")
    if extracted_agree != len(ok_records):
        failed_checks.append("extracted strings")

    lenient_hits = sorted(
        r["key"]
        for r in ok_records
        if not r["correct"] and gold_in_content(r["content"], r["gold"])
    )
    print(
        f"  lenient regrade credits {len(lenient_hits)} of the non-correct "
        f"records: {lenient_hits}"
    )

    from run_hmmt import item_bootstrap_ci

    by_item: dict[str, list[int]] = {}
    by_item_untruncated: dict[str, list[int]] = {}
    for record in ok_records:
        correct = int(grade_math(record["content"], record["gold"]).correct)
        by_item.setdefault(record["item_id"], []).append(correct)
        if not record["truncated"]:
            by_item_untruncated.setdefault(record["item_id"], []).append(correct)

    ours_accuracy = n_correct / len(ok_records)
    untruncated = [r for r in ok_records if not r["truncated"]]
    ours_untruncated_record = (
        sum(grade_math(r["content"], r["gold"]).correct for r in untruncated)
        / len(untruncated)
        if untruncated
        else float("nan")
    )
    ours_untruncated_problem = (
        sum(sum(v) / len(v) for v in by_item_untruncated.values())
        / len(by_item_untruncated)
        if by_item_untruncated
        else float("nan")
    )
    # The two statistics that both answer to "accuracy excluding truncations".
    # `hmmt_untruncated_scorer` reports the problem-weighted one, because the
    # epoch reducer runs before any metric; the reference harness's metrics.json
    # reports the record-weighted one. Printing both here is what keeps the gap
    # from being discovered as a four-point regression on someone's first run.
    print(
        f"  untruncated accuracy over {len(untruncated)} records: "
        f"{ours_untruncated_record:.6f} record-weighted "
        f"(the reference harness's statistic), {ours_untruncated_problem:.6f} "
        f"problem-weighted over {len(by_item_untruncated)} problems "
        "(hmmt_untruncated_scorer's)"
    )

    metrics_path = records_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

        theirs_lenient = metrics.get("n_gold_in_content_among_incorrect")
        if theirs_lenient is None:
            print("  NOTE their metrics.json records no lenient-hit count to compare")
        elif theirs_lenient != len(lenient_hits):
            failed_checks.append("lenient hits")
            print(
                f"  FAIL their metrics.json recorded {theirs_lenient} lenient "
                f"hits, this regrade found {len(lenient_hits)}"
            )
        else:
            print(f"  ok   lenient hits match their metrics.json ({theirs_lenient})")

        theirs_accuracy = metrics.get("accuracy")
        if theirs_accuracy is None:
            print("  NOTE their metrics.json records no accuracy to compare")
        elif not math.isclose(theirs_accuracy, ours_accuracy, abs_tol=1e-9):
            failed_checks.append("record-level accuracy")
            print(
                f"  FAIL accuracy: theirs {theirs_accuracy}, ours {ours_accuracy}"
            )
        else:
            print(f"  ok   record-level accuracy reproduces theirs: {ours_accuracy}")

        theirs_untruncated = metrics.get("accuracy_excl_truncated")
        if theirs_untruncated is not None:
            if math.isclose(theirs_untruncated, ours_untruncated_record, abs_tol=1e-9):
                print(
                    "  ok   record-weighted untruncated accuracy reproduces "
                    f"theirs: {theirs_untruncated}"
                )
            else:
                failed_checks.append("untruncated accuracy")
                print(
                    f"  FAIL accuracy_excl_truncated: theirs {theirs_untruncated}, "
                    f"ours {ours_untruncated_record}"
                )

        # The interval run_hmmt.py prints has to be the same statistic as the
        # one in their metrics.json, or the two cannot be read side by side.
        # Regrading their records gives the per-problem rates to check it on.
        interval = item_bootstrap_ci([sum(v) / len(v) for v in by_item.values()])
        theirs_ci = metrics.get("ci95_item_bootstrap")
        if interval is None or theirs_ci is None:
            print("  NOTE no item bootstrap to compare")
        elif [round(x, 12) for x in interval] == [round(x, 12) for x in theirs_ci]:
            print(f"  ok   item bootstrap reproduces theirs: {theirs_ci}")
        else:
            failed_checks.append("item bootstrap")
            print(f"  FAIL item bootstrap {list(interval)} vs theirs {theirs_ci}")
    else:
        print("  NOTE no metrics.json; only the per-record verdicts were gated")

    truncated_no_box = sum(
        1
        for r in ok_records
        if r["truncated"] and extract_boxed(strip_thinking(r["content"])) is None
    )
    print(
        f"  {sum(1 for r in ok_records if r['truncated'])} truncated, "
        f"{truncated_no_box} of them with no boxed answer"
    )

    if ratio < MIN_REGRADE_AGREEMENT_RATIO:
        failed_checks.insert(0, "verdicts")
        print(
            f"  FAIL agreement {agree}/{len(ok_records)} = {ratio:.4f} is below "
            f"the {MIN_REGRADE_AGREEMENT_RATIO:.4f} target"
        )

    if failed_checks:
        print(f"  gating sub-checks that failed: {', '.join(failed_checks)}")
        return False
    print(
        "  gating sub-checks all passed: verdicts, extracted strings, lenient "
        "hits, record-level accuracy, untruncated accuracy, item bootstrap"
    )
    return True


def _diagnose(record: dict[str, Any], result: Any) -> str:
    """A one-line reading of why two graders disagreed on one record."""
    if result.extracted != (record.get("extracted") or None):
        return "different answers extracted from the same completion"
    if result.method != record.get("method"):
        return (
            "same answer, different cascade rung "
            f"({record.get('method')!r} vs {result.method!r})"
        )
    return "same answer and rung, different verdict — math-verify behaved differently"


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        required=True,
        choices=["dataset", "grader", "mockllm", "regrade", "all"],
        help="Which check to run. 'all' runs dataset, grader and mockllm.",
    )
    parser.add_argument(
        "--records",
        type=Path,
        action="append",
        help="A capability-evals output directory for --check regrade. Repeatable.",
    )
    args = parser.parse_args()

    results: list[tuple[str, bool]] = []
    if args.check in ("grader", "all"):
        results.append(("grader", check_grader()))
    if args.check in ("dataset", "all"):
        results.append(("dataset", check_dataset()))
    if args.check in ("mockllm", "all"):
        results.append(("mockllm", check_mockllm()))
    if args.check == "regrade":
        if not args.records:
            parser.error("--check regrade needs at least one --records <dir>")
        for records_dir in args.records:
            results.append((f"regrade:{records_dir.name}", check_regrade(records_dir)))

    print()
    for name, passed in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    return 0 if all(passed for _, passed in results) else 1


if __name__ == "__main__":
    sys.exit(main())
