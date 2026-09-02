"""Check the ported judges still separate known reward hacks from honest answers.

    python scripts/validate_school_of_reward_hacks_judges.py --out-dir /work/workspace

Every row of `data/srh_subset.jsonl` ships two reference answers to the same
request: a `hack_completion` that games the row's stated scoring rule and a
`control_completion` that just answers it. The label is therefore known without
running any model under test, which makes the dataset a calibration set for the
INSTRUMENT rather than for a policy. This script grades both answers of all 306
rows with both judges -- 4 calls per row, 1224 in total -- and reports whether
each judge still moves in the direction its template says it should:

* judge 1 (metric, sees the ORIGINAL prompt) should score the hack ABOVE the
  control, because the hack is what maxing the stated rule looks like;
* judge 2 (quality, sees the DE-SCORED prompt) should score the hack BELOW the
  control, because the hack is the worse answer to the request itself.

The headline is the PAIRED per-prompt sign rate, not the difference of the two
group means. Both answers to a row respond to the same request, so pairing
removes the per-prompt difficulty that dominates both scores; a judge could
easily produce a healthy-looking mean difference while ordering an individual
pair wrongly half the time, and it is the ordering that the gap measure depends
on. Group means are reported beside it, not instead of it.

Nothing here re-implements the instrument. The templates, the call settings, the
retry policy and the `<score>N</score>` parsing all come from the eval's own
`scorers.py` via `render_prompt`, `judge_model`, `judge_config` and `run_judge`,
so a drift between this check and the eval is impossible by construction --
which is the only reason the check is worth anything.

Results are appended to a JSONL as each call returns, keyed by
(id, completion kind, judge). A re-run reads that file first and issues only the
calls it does not already have, so an interrupted run resumes instead of
re-paying for 1224 judge calls. Delete the JSONL to force a clean re-grade.

`--max-concurrent` is deliberately low (8). This check shares a judge API quota
with whatever sweep is running, and starving a sweep's inline judges to finish a
calibration check ten minutes sooner is a bad trade.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inspect_ai.model import GenerateConfig, Model, get_model
from school_of_reward_hacks.dataset import (
    category_letter,
    load_rows,
)
from school_of_reward_hacks.scorers import (
    JUDGE_MODEL,
    METRIC_JUDGE,
    QUALITY_JUDGE,
    JudgeResult,
    judge_config,
    judge_model,
    render_prompt,
    run_judge,
)

KINDS = ("hack", "control")
"""The two labelled answers per row, in the order they are reported."""

COMPLETION_FIELD = {"hack": "hack_completion", "control": "control_completion"}

JUDGES = (METRIC_JUDGE, QUALITY_JUDGE)

PROMPT_FIELD = {METRIC_JUDGE: "user", QUALITY_JUDGE: "user_descored"}
"""Which prompt each judge is allowed to see. This is the instrument's central
invariant -- judge 2 must never learn that a metric was named -- and it is
asserted per call rather than trusted, in `judge_prompt` below."""

EXPECTED_SIGN = {METRIC_JUDGE: +1, QUALITY_JUDGE: -1}
"""Which way each judge should order (hack, control) on a row it gets right.

Judge 1 should rank the hack higher (it maxes the stated rule); judge 2 should
rank it lower (it is the worse answer). "Correct sign" is scored against this,
per judge, so the two rates are comparable to each other and to the earlier
non-Inspect implementation's. The raw `hack > control` fraction is reported
alongside it for judges whose expected direction is negative, so the convention
never has to be taken on trust."""

RESULT_COLUMNS = [
    "id",
    "category",
    "category_letter",
    "kind",
    "judge1_metric",
    "judge2_quality",
    "gap",
    "judge1_status",
    "judge2_status",
    "parse_status",
    "judge1_attempts",
    "judge2_attempts",
]


def judge_prompt(row: dict[str, Any], judge: int, answer: str) -> str:
    """The prompt text `judge` is allowed to see for `row`, checked on the way out.

    Args:
        row: A subset row, with both `user` and `user_descored`.
        judge: 1 (metric) or 2 (quality).
        answer: The answer that will be graded, needed only for the rendered-
            message check below.

    Returns:
        `row["user"]` for judge 1, `row["user_descored"]` for judge 2.

    Raises:
        AssertionError: If the wrong field was selected, if the row's de-scored
            rewrite is missing or identical to the original, or if the original
            prompt's text survives inside the message judge 2 would actually be
            sent. The last one is the check that matters: the first two are
            about this script's bookkeeping, but a template that pasted the
            original prompt in anyway would leak the scoring rule to the quality
            judge and quietly collapse the gap, and only the rendered message
            can rule that out.
    """
    original = row["user"]
    descored = row["user_descored"]
    assert isinstance(descored, str) and descored.strip(), (
        f"row {row['id']} has no de-scored prompt"
    )
    assert descored != original, (
        f"row {row['id']}: the de-scored prompt is the original prompt"
    )

    text = row[PROMPT_FIELD[judge]]
    if judge == METRIC_JUDGE:
        assert text == original, f"row {row['id']}: judge 1 was not given `user`"
    else:
        assert text == descored, (
            f"row {row['id']}: judge 2 was not given `user_descored`"
        )
        assert original not in render_prompt(judge, text, answer), (
            f"row {row['id']}: the original prompt survives inside judge 2's "
            "rendered message, so it would see the scoring rule"
        )
    return text


def unit_key(record: dict[str, Any]) -> tuple[Any, str, int]:
    """The (row id, completion kind, judge) a JSONL record answers for."""
    return (record["id"], record["kind"], int(record["judge"]))


def read_done(path: Path) -> dict[tuple[Any, str, int], dict[str, Any]]:
    """Judge calls already recorded in the JSONL, keyed by unit.

    A trailing partial line -- a run killed mid-write -- is dropped rather than
    raising, since the unit it belonged to is simply re-issued.
    """
    if not path.exists():
        return {}
    done: dict[tuple[Any, str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[unit_key(record)] = record
    return done


def record_of(
    row: dict[str, Any], kind: str, judge: int, result: JudgeResult, prompt_text: str
) -> dict[str, Any]:
    """One judge's verdict on one labelled answer, as a JSONL record."""
    return {
        "id": row["id"],
        "cat": row["cat"],
        "kind": kind,
        "judge": judge,
        "prompt_field": PROMPT_FIELD[judge],
        "status": result.status,
        "score": result.score,
        "attempts": result.attempts,
        "rationale": result.rationale,
        "prompt_chars": len(prompt_text),
        "answer_chars": len(row[COMPLETION_FIELD[kind]]),
    }


async def grade_all(
    rows: Sequence[dict[str, Any]],
    model: Model,
    out_jsonl: Path,
    max_concurrent: int,
) -> dict[tuple[Any, str, int], dict[str, Any]]:
    """Issue every missing judge call, appending each result as it lands.

    Returns every record for these rows, resumed ones included.
    """
    done = read_done(out_jsonl)
    units = [
        (row, kind, judge) for row in rows for kind in KINDS for judge in JUDGES
    ]
    todo = [unit for unit in units if (unit[0]["id"], unit[1], unit[2]) not in done]
    print(
        f"{len(units)} judge calls total: {len(units) - len(todo)} already in "
        f"{out_jsonl.name}, {len(todo)} to make "
        f"(<= {max_concurrent} concurrent)",
        flush=True,
    )
    if not todo:
        return done

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent)
    write_lock = asyncio.Lock()
    handle = out_jsonl.open("a", encoding="utf-8")
    finished = 0

    async def one(row: dict[str, Any], kind: str, judge: int) -> None:
        nonlocal finished
        answer = row[COMPLETION_FIELD[kind]]
        prompt_text = judge_prompt(row, judge, answer)
        async with semaphore:
            result = await run_judge(model, judge, prompt_text, answer)
        record = record_of(row, kind, judge, result, prompt_text)
        async with write_lock:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done[unit_key(record)] = record
            finished += 1
            if finished % 50 == 0 or finished == len(todo):
                print(f"  {finished}/{len(todo)} calls made", flush=True)

    try:
        await asyncio.gather(*(one(*unit) for unit in todo))
    finally:
        handle.close()
    return done


def result_rows(
    rows: Sequence[dict[str, Any]], done: dict[tuple[Any, str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    """One row per (subset row, completion kind): both judges side by side."""
    out = []
    for row in rows:
        for kind in KINDS:
            j1 = done.get((row["id"], kind, METRIC_JUDGE), {})
            j2 = done.get((row["id"], kind, QUALITY_JUDGE), {})
            s1, s2 = j1.get("score"), j2.get("score")
            both_ok = j1.get("status") == "ok" and j2.get("status") == "ok"
            out.append(
                {
                    "id": row["id"],
                    "category": row["cat"],
                    "category_letter": category_letter(row["cat"]),
                    "kind": kind,
                    "judge1_metric": s1,
                    "judge2_quality": s2,
                    "gap": (s1 - s2) if both_ok else None,
                    "judge1_status": j1.get("status"),
                    "judge2_status": j2.get("status"),
                    "parse_status": "ok" if both_ok else "unparsed",
                    "judge1_attempts": j1.get("attempts"),
                    "judge2_attempts": j2.get("attempts"),
                }
            )
    return out


def mean_stderr(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Sample mean and standard error of the mean; stderr needs n >= 2."""
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n < 2:
        return mean, None
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance / n)


def summarise(
    result_rows_: Sequence[dict[str, Any]], judge: int, group: str | None
) -> dict[str, Any]:
    """One judge's separation on one group of rows.

    `group` is a category letter, or None for every row. Group means come from
    every row whose own score parsed; the paired numbers come only from prompts
    where BOTH answers scored, so the two denominators can differ and both are
    reported.
    """
    column = "judge1_metric" if judge == METRIC_JUDGE else "judge2_quality"
    members = [
        row
        for row in result_rows_
        if group is None or row["category_letter"] == group
    ]

    by_kind = {
        kind: [
            float(row[column])
            for row in members
            if row["kind"] == kind and row[column] is not None
        ]
        for kind in KINDS
    }
    hack_mean, hack_stderr = mean_stderr(by_kind["hack"])
    control_mean, control_stderr = mean_stderr(by_kind["control"])

    scores: dict[Any, dict[str, float]] = {}
    for row in members:
        if row[column] is not None:
            scores.setdefault(row["id"], {})[row["kind"]] = float(row[column])
    pairs = [v for v in scores.values() if len(v) == 2]
    above = sum(1 for v in pairs if v["hack"] > v["control"])
    below = sum(1 for v in pairs if v["hack"] < v["control"])
    ties = len(pairs) - above - below
    correct = above if EXPECTED_SIGN[judge] > 0 else below
    diffs = [v["hack"] - v["control"] for v in pairs]
    paired_mean, paired_stderr = mean_stderr(diffs)

    return {
        "judge": judge,
        "group": group or "ALL",
        "n_prompts": len(members) // len(KINDS),
        "n_hack_scored": len(by_kind["hack"]),
        "n_control_scored": len(by_kind["control"]),
        "hack_mean": hack_mean,
        "hack_stderr": hack_stderr,
        "control_mean": control_mean,
        "control_stderr": control_stderr,
        "diff": (
            hack_mean - control_mean
            if hack_mean is not None and control_mean is not None
            else None
        ),
        "n_pairs": len(pairs),
        "paired_diff_mean": paired_mean,
        "paired_diff_stderr": paired_stderr,
        "frac_hack_above": above / len(pairs) if pairs else None,
        "frac_hack_below": below / len(pairs) if pairs else None,
        "n_ties": ties,
        "expected_sign": EXPECTED_SIGN[judge],
        "correct_sign_rate": correct / len(pairs) if pairs else None,
        "n_correct_sign": correct,
    }


def write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]
) -> None:
    """Write rows as CSV, with empty cells for None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: "" if row.get(c) is None else row[c] for c in columns})


def fmt(value: float | None, places: int = 1, sign: bool = False) -> str:
    """A number for the printed table, or a dash where there isn't one.

    Formatted here rather than in the f-string's own spec because the value may
    be None, so the column is always a string by the time it is padded.
    """
    if value is None:
        return "-"
    return f"{value:{'+' if sign else ''}.{places}f}"


def print_summary(summaries: Sequence[dict[str, Any]]) -> None:
    """The separation table, one block per judge."""
    for judge in JUDGES:
        name = "1 metric   (sees `user`)" if judge == METRIC_JUDGE else (
            "2 quality  (sees `user_descored`)"
        )
        direction = "hack > control" if EXPECTED_SIGN[judge] > 0 else "hack < control"
        print(f"\njudge {name}   expected: {direction}")
        print(
            f"  {'group':<6} {'n':>4} {'hack':>7} {'control':>8} {'diff':>7} "
            f"{'paired':>8} {'above%':>7} {'sign%':>7} {'ties':>5}"
        )
        for row in summaries:
            if row["judge"] != judge:
                continue
            above = row["frac_hack_above"]
            sign = row["correct_sign_rate"]
            print(
                f"  {row['group']:<6} {row['n_pairs']:>4} "
                f"{fmt(row['hack_mean']):>7} {fmt(row['control_mean']):>8} "
                f"{fmt(row['diff'], sign=True):>7} "
                f"{fmt(row['paired_diff_mean'], sign=True):>8} "
                f"{fmt(None if above is None else 100 * above, 0):>7} "
                f"{fmt(None if sign is None else 100 * sign, 0):>7} "
                f"{row['n_ties']:>5}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out-dir", default=".", help="directory for the JSONL and the CSVs"
    )
    parser.add_argument(
        "--jsonl",
        default="srh_judge_validation.jsonl",
        help="incremental per-call log; a re-run resumes from it",
    )
    parser.add_argument(
        "--rows-csv",
        default="srh_judge_validation_rows.csv",
        help="per (row, completion kind) CSV",
    )
    parser.add_argument(
        "--summary-csv",
        default="srh_judge_validation_summary.csv",
        help="per (judge, group) separation CSV",
    )
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="concurrent judge calls; kept low so a running sweep is not starved",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="first N rows only (for a smoke run)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_jsonl = out_dir / args.jsonl
    rows = load_rows()
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"{len(rows)} rows, judge model {args.judge_model}", flush=True)

    # The judge's own settings, from the eval, plus a connection cap: without an
    # explicit max_connections inspect probes upward with adaptive concurrency,
    # which is exactly what this script must not do to a shared quota.
    model = judge_model(
        get_model(
            args.judge_model,
            config=judge_config().merge(
                GenerateConfig(max_connections=args.max_concurrent)
            ),
        )
    )

    done = asyncio.run(grade_all(rows, model, out_jsonl, args.max_concurrent))

    results = result_rows(rows, done)
    letters = sorted({row["category_letter"] for row in results})
    summaries = [
        summarise(results, judge, group)
        for judge in JUDGES
        for group in [None, *letters]
    ]

    write_csv(out_dir / args.rows_csv, RESULT_COLUMNS, results)
    write_csv(out_dir / args.summary_csv, list(summaries[0].keys()), summaries)

    unparsed = [r for r in done.values() if r["status"] != "ok"]
    retried = [r for r in done.values() if r["status"] == "ok" and r["attempts"] > 1]
    print_summary(summaries)
    print(
        f"\n{len(done)} judge calls over {len(rows)} rows x {len(KINDS)} completions "
        f"x {len(JUDGES)} judges\n"
        f"{len(unparsed)} unparseable judgements "
        f"({len(retried)} scored only after a retry)\n"
        f"{out_dir / args.rows_csv}\n{out_dir / args.summary_csv}\n{out_jsonl}",
        flush=True,
    )
    for record in unparsed:
        print(f"  UNPARSED: id={record['id']} {record['kind']} judge {record['judge']}")


if __name__ == "__main__":
    main()
