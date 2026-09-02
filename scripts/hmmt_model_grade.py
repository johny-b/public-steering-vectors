"""Grade finished HMMT replies with a model, beside the strict grader.

The strict grader reads the last `\\boxed{}` in a reply and marks a reply with
no box as wrong. A model that writes the right answer in prose, or in a box it
never closes, is therefore scored the same as one that got it wrong. This
script asks a small model the one question the strict grader cannot answer:
does this reply commit to the reference answer, whatever the formatting?

It runs only over replies that finished on their own (stop reason not in the
truncation set), because a cut-off reply has no final answer to read. For each
such reply it records the full grader prompt, the full grader completion, the
grader's reasoning (or null with the reason the provider gave none), the
verdict, and the strict grader's verdict from the log beside it. Nothing is
truncated.

    python scripts/hmmt_model_grade.py <log root> --out <dir> \\
        [--grader anthropic/claude-haiku-4-5]

`<log root>` holds one directory per steering strength (`s-0.5`, `s0.0`,
`s+0.3`, ...), each with one successful HMMT log. The output directory gets one
`<strength>.jsonl` of per-attempt records and a `summary.json` with, per
strength, the strict and the model-graded accuracy over finished attempts,
each averaged per problem first and bootstrapped over problems.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)

TRUNCATION_STOP_REASONS = ("max_tokens", "model_length")
DEFAULT_GRADER = "anthropic/claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You compare the final answer of a mathematics reply with a reference "
    "answer. Do not solve the problem, do not verify the reply's reasoning, "
    "and do not explain. Judge one thing only: does the reply commit to a "
    "single final answer that is mathematically the same value as the "
    "reference, ignoring formatting, notation, LaTeX, units written in words "
    "and trailing remarks? A reply that hedges between several values, gives a "
    "range, or never commits to one final answer does not match. Output exactly "
    'one line of JSON and nothing else: {"final_answer": "<the answer the '
    'reply commits to, as written, or null>", "match": true or false}'
)

RETRY_NUDGE = (
    "That was not a single line of JSON. Output exactly one line: "
    '{"final_answer": "<answer or null>", "match": true or false}'
)

USER_TEMPLATE = (
    "Problem:\n{problem}\n\n"
    "Reference answer:\n{gold}\n\n"
    "Reply to judge (its final answer may be anywhere in it):\n"
    "<<<\n{reply}\n>>>"
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_verdict(text: str) -> tuple[bool | None, str | None, bool]:
    """(match, final_answer, parsed).

    Decodes a JSON object starting at every `{` in the text, last one first,
    so LaTeX braces inside the answer string cannot cut the object short.
    """
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for i in reversed(starts):
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("match"), bool):
            ans = obj.get("final_answer")
            return obj["match"], (None if ans is None else str(ans)), True
    return None, None, False


def reasoning_of(output) -> tuple[str | None, str | None]:
    """(reasoning text, reason it is null). Redacted blocks count as absent."""
    try:
        parts = output.message.content
    except AttributeError:
        return None, "no message content"
    if isinstance(parts, str):
        return None, "provider returned no reasoning blocks"
    for part in parts:
        if getattr(part, "type", None) == "reasoning":
            summary = getattr(part, "summary", None)
            if summary:
                return summary, None
            if getattr(part, "redacted", False):
                return None, "provider returned an encrypted reasoning block only"
            text = getattr(part, "reasoning", None)
            if text:
                return text, None
    return None, "provider returned no reasoning blocks"


def visible_text(sample) -> str:
    """The reply's final channel: text parts of the last assistant message."""
    out = sample.output
    if out is None:
        return ""
    return out.completion or ""


def strength_of(log) -> float | None:
    args = log.eval.model_args or {}
    val = args.get("steer_strength")
    return float(val) if val is not None else None


async def grade_log(log, grader, concurrency: int, out_path: Path) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    records: list[dict] = []

    async def one(sample):
        stop = sample.output.stop_reason if sample.output else None
        strict = sample.scores.get("hmmt_scorer") if sample.scores else None
        strict_val = strict.value if strict else None
        if isinstance(strict_val, dict):
            strict_correct = strict_val.get("correct")
            boxed_found = strict_val.get("boxed_found")
        else:
            strict_correct, boxed_found = strict_val, None
        base = {
            "sample_id": sample.id,
            "epoch": sample.epoch,
            "gold": sample.target
            if isinstance(sample.target, str)
            else "\n".join(sample.target),
            "stop_reason": stop,
            "truncated": stop in TRUNCATION_STOP_REASONS,
            "strict_correct": strict_correct,
            "boxed_found": boxed_found,
            "strict_extracted": strict.answer if strict else None,
        }
        if stop in TRUNCATION_STOP_REASONS:
            return {
                **base,
                "graded": False,
                "reason": "truncated; no final answer to read",
            }
        reply = visible_text(sample)
        problem = (
            sample.input
            if isinstance(sample.input, str)
            else "\n".join(getattr(m, "text", str(m)) for m in sample.input)
        )
        user = USER_TEMPLATE.format(problem=problem, gold=base["gold"], reply=reply)
        messages = [
            ChatMessageSystem(content=SYSTEM_PROMPT),
            ChatMessageUser(content=user),
        ]
        async with sem:
            output = await grader.generate(
                messages, config=GenerateConfig(max_tokens=600)
            )
        completion = output.completion or ""
        match, final_answer, parsed = parse_verdict(completion)
        first_completion = None
        if not parsed:
            # One retry with the model's own reply in context and a nudge to
            # the format; the first reply is kept beside the second.
            async with sem:
                output = await grader.generate(
                    messages + [output.message, ChatMessageUser(content=RETRY_NUDGE)],
                    config=GenerateConfig(max_tokens=200),
                )
            first_completion = completion
            completion = output.completion or ""
            match, final_answer, parsed = parse_verdict(completion)
        reasoning, why_null = reasoning_of(output)
        return {
            **base,
            "graded": True,
            "grader_model": str(grader),
            "grader_prompt_system": SYSTEM_PROMPT,
            "grader_prompt_user": user,
            "grader_prompt_sha256": sha(SYSTEM_PROMPT + "\n" + user),
            "grader_prompt_chars": len(SYSTEM_PROMPT) + 1 + len(user),
            "grader_completion": completion,
            "grader_completion_sha256": sha(completion),
            "grader_completion_chars": len(completion),
            "grader_completion_first_attempt": first_completion,
            "grader_reasoning": reasoning,
            "grader_reasoning_unavailable": why_null,
            "grader_stop_reason": output.stop_reason,
            "model_match": match,
            "model_final_answer": final_answer,
            "verdict_parsed": parsed,
            "reply_chars": len(reply),
            "reply_sha256": sha(reply),
        }

    results = await asyncio.gather(*(one(s) for s in log.samples))
    records.extend(results)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def bootstrap_problem_mean(
    per_problem: dict, key: str, resamples: int, seed: int
) -> dict:
    """Mean of per-problem means and a percentile interval over problems."""
    ids = sorted(per_problem)
    means = []
    for pid in ids:
        vals = [r[key] for r in per_problem[pid] if r[key] is not None]
        if vals:
            means.append(sum(vals) / len(vals))
    if not means:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_problems": 0}
    rng = random.Random(seed)
    boots = []
    for _ in range(resamples):
        draw = [means[rng.randrange(len(means))] for _ in means]
        boots.append(sum(draw) / len(draw))
    boots.sort()
    lo = boots[int(0.025 * resamples)]
    hi = boots[min(resamples - 1, int(0.975 * resamples))]
    return {
        "mean": sum(means) / len(means),
        "ci_low": lo,
        "ci_high": hi,
        "n_problems": len(means),
        "ci_method": "bootstrap over problems",
        "resamples": resamples,
    }


NO_ANSWER = (None, "", "null", "None")


def outcome_of(rec: dict) -> str:
    """One of five mutually exclusive outcomes for an attempt.

    cut_off: the generation hit the token limit. correct_boxed: the strict
    grader scored it correct. correct_unboxed: the strict grader found no
    matching box but the model grader read a committed answer that matches.
    wrong_answer: committed to a single answer that does not match. no_answer:
    finished without committing to one final answer. A grader verdict that
    could not be parsed even after a retry is counted as no_answer, and the
    number of such verdicts is reported beside the table.
    """
    if not rec.get("graded"):
        return "cut_off"
    if rec.get("strict_correct") == 1.0:
        return "correct_boxed"
    if not rec.get("verdict_parsed"):
        return "no_answer"
    if rec.get("model_final_answer") in NO_ANSWER:
        return "no_answer"
    return "correct_unboxed" if rec.get("model_match") is True else "wrong_answer"


OUTCOMES = ["correct_boxed", "correct_unboxed", "wrong_answer", "no_answer", "cut_off"]


def summarise(records: list[dict], resamples: int, seed: int) -> dict:
    finished = [r for r in records if r.get("graded")]
    per_problem: dict = {}
    for r in finished:
        r["_strict"] = (
            float(r["strict_correct"]) if r["strict_correct"] is not None else None
        )
        r["_model"] = (
            float(r["model_match"]) if isinstance(r["model_match"], bool) else None
        )
        per_problem.setdefault(r["sample_id"], []).append(r)
    boxed = [r for r in finished if r.get("boxed_found") == 1.0]
    agree = sum(
        1 for r in boxed if r["_model"] is not None and r["_model"] == r["_strict"]
    )
    return {
        "n_attempts": len(records),
        "n_finished": len(finished),
        "n_truncated": len(records) - len(finished),
        "n_verdicts_unparsed": sum(1 for r in finished if not r["verdict_parsed"]),
        "strict": bootstrap_problem_mean(per_problem, "_strict", resamples, seed),
        "model_grader": bootstrap_problem_mean(
            per_problem, "_model", resamples, seed + 1
        ),
        "agreement_on_boxed": {
            "n": len(boxed),
            "agree": agree,
            "share": (agree / len(boxed)) if boxed else None,
        },
        "rescued": sum(
            1 for r in finished if r["_strict"] == 0.0 and r["_model"] == 1.0
        ),
        "lost": sum(1 for r in finished if r["_strict"] == 1.0 and r["_model"] == 0.0),
        "reasoning_returned": sum(1 for r in finished if r.get("grader_reasoning")),
        "outcomes": {
            k: sum(1 for r in records if outcome_of(r) == k) for k in OUTCOMES
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--grader", default=os.environ.get("INSPECT_GRADER_MODEL") or DEFAULT_GRADER
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--retry-unparsed",
        action="store_true",
        help="re-grade only the stored records whose verdict did not parse, then rebuild summary.json",
    )
    parser.add_argument(
        "--reparse",
        action="store_true",
        help=(
            "re-read the stored grader replies in --out and rebuild "
            "summary.json without calling the grader"
        ),
    )
    args = parser.parse_args()

    root = Path(args.log_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.retry_unparsed:
        grader = get_model(args.grader)
        summary = {"grader": args.grader, "strengths": {}}
        for path in sorted(out.glob("s*.jsonl")):
            records = [json.loads(line) for line in path.open(encoding="utf-8")]
            todo = [
                r for r in records if r.get("graded") and not r.get("verdict_parsed")
            ]

            async def redo(rec):
                messages = [
                    ChatMessageSystem(content=rec["grader_prompt_system"]),
                    ChatMessageUser(content=rec["grader_prompt_user"]),
                ]
                output = await grader.generate(
                    messages, config=GenerateConfig(max_tokens=600)
                )
                completion = output.completion or ""
                match, ans, parsed = parse_verdict(completion)
                if not parsed:
                    output = await grader.generate(
                        messages
                        + [output.message, ChatMessageUser(content=RETRY_NUDGE)],
                        config=GenerateConfig(max_tokens=200),
                    )
                    completion = output.completion or ""
                    match, ans, parsed = parse_verdict(completion)
                rec["grader_completion_first_attempt"] = rec.get("grader_completion")
                rec["grader_completion"] = completion
                rec["grader_completion_sha256"] = sha(completion)
                rec["grader_completion_chars"] = len(completion)
                rec["model_match"], rec["model_final_answer"], rec["verdict_parsed"] = (
                    match,
                    ans,
                    parsed,
                )

            async def redo_all():
                await asyncio.gather(*(redo(r) for r in todo))

            asyncio.run(redo_all())
            with path.open("w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats = summarise(records, args.resamples, args.seed)
            stats["strength"] = float(path.stem[1:])
            summary["strengths"][path.stem] = stats
            print(
                f"{path.stem}: retried {len(todo)}, still unparsed {stats['n_verdicts_unparsed']}"
            )
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        return 0
    if args.reparse:
        summary = {"grader": args.grader, "strengths": {}}
        for path in sorted(out.glob("s*.jsonl")):
            records = [json.loads(line) for line in path.open(encoding="utf-8")]
            for rec in records:
                if rec.get("graded"):
                    match, ans, parsed = parse_verdict(rec.get("grader_completion", ""))
                    (
                        rec["model_match"],
                        rec["model_final_answer"],
                        rec["verdict_parsed"],
                    ) = match, ans, parsed
            with path.open("w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats = summarise(records, args.resamples, args.seed)
            stats["strength"] = float(path.stem[1:])
            summary["strengths"][path.stem] = stats
            print(f"{path.stem}: reparsed, unparsed {stats['n_verdicts_unparsed']}")
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        return 0
    grader = get_model(args.grader)
    summary: dict = {"grader": args.grader, "strengths": {}}
    for sdir in sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.startswith("s")
    ):
        logs = [
            info
            for info in list_eval_logs(str(sdir))
            if read_eval_log(info.name, header_only=True).status == "success"
        ]
        if len(logs) != 1:
            print(f"{sdir.name}: {len(logs)} success logs, skipped", file=sys.stderr)
            continue
        log = read_eval_log(logs[0].name)
        strength = strength_of(log)
        records = asyncio.run(
            grade_log(log, grader, args.concurrency, out / f"{sdir.name}.jsonl")
        )
        stats = summarise(records, args.resamples, args.seed)
        stats["strength"] = strength
        stats["log"] = logs[0].name
        summary["strengths"][sdir.name] = stats
        s, m = stats["strict"], stats["model_grader"]
        print(
            f"{sdir.name}: finished {stats['n_finished']}/{stats['n_attempts']}  "
            f"strict {s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]  "
            f"model {m['mean']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]  "
            f"rescued {stats['rescued']} lost {stats['lost']} "
            f"unparsed {stats['n_verdicts_unparsed']}"
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
