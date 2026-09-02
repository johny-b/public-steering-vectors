"""Turn a directory of School of Reward Hacks eval logs into three tidy CSVs.

    python scripts/analyze_school_of_reward_hacks.py logs/school-of-reward-hacks-sweep

Writes one row per sample to `--samples`, one row per (vector, strength) to
`--aggregate`, and one row per (vector, strength, category) to `--by-category`.
The per-sample file is the source of truth and both aggregates are derived from
it with nothing else added.

The headline measure is the PAIRED gap, so it is computed per sample by the
scorer and only then averaged here. `gap_mean` and `metric_mean - quality_mean`
come out equal (both judges' scores are dropped together when either fails to
parse, so the three columns cover the same samples), but `gap_stderr` is the
stderr of the per-sample differences and is typically far smaller than anything
derivable from the two separate stderrs — the prompt difficulty both judges see
cancels inside each pair. Read the gap column, not the difference of the other
two.

Alongside the three scores it carries the reasons a sample might carry no usable
score at all, because in this eval that has two very different causes.

`judge1_status` / `judge2_status` are "unparsed" when the judge never produced a
usable final `<score>N</score>`. That is an INSTRUMENT failure: there was an
answer, and the grader could not read its own reply. Those samples are already
NaN in the log and drop out of the score means on their own; `unparsed_mean` is
the rate at which it happened.

`truncated` is a GENERATION failure: the reply hit the request's token budget
(`stop_reason='max_tokens'`), hit the server's context (`'model_length'`), or
came back with no visible answer at all — under steering a model can spend its
whole 32k budget thinking and emit nothing. `stop_truncated` and `answer_empty`
are the two halves separately; `truncated` is their union and is the flag the
aggregates act on.

That rule is not written here. It is
`school_of_reward_hacks.truncation.no_answer`, imported below and shared with
`scorers.py`, which applies the same predicate while grading. Two copies of a
rule about what counts as a lost answer is exactly the kind of thing that drifts
silently and turns the scorer's flag and this script's column into two different
measurements with one name.

A truncated sample is EXCLUDED from every mean and stderr below, and its
absence is reported as `n_truncated` beside `n_scored`. It has to be excluded
rather than carried, because a missing answer is not a zero: both judges score
an empty answer near the bottom, so counting it would push `metric`, `quality`
and `answer_chars` down and pull `gap` toward zero. The bias is not even across
the sweep — truncation is a consequence of the steering condition, so it
concentrates in exactly the arms that make the model think longest, and those
arms would be handed an artificial score change that no behavioural difference
produced. The earlier non-Inspect run of this instrument excluded them the same
way and reported the count as "answers lost to token budget"; `n_truncated` is
that number.

The exclusion still lives HERE as well as in the scorer, and for a different
reason than the scorer's. `scorers.py` now recognises the condition while
grading and records it as `no_answer` with the scores NaN, so runs made from now
on carry the fact in the log and never pay for a judge call on an empty string.
The nine-arm sweep this script was written for was graded BEFORE it did, by a
scorer that sent the empty string to both judges and wrote down what they said —
and re-grading those logs is not on offer, because the scorer is the measurement
instrument and a sweep is only comparable across its arms if every arm was
graded by the same one. So the exclusion stays here for them, reading the stop
reason and the visible completion the log already records.

Keeping it here also keeps it revisable: a different rule about what counts as a
lost answer is a re-run of this script over the same logs, not a re-run of the
GPU hours and the 612 judge calls that produced them. What it must not become is
a second rule, which is why the predicate is imported rather than restated.

The validity rates themselves — `truncated_mean`, `stop_truncated_mean`,
`answer_empty_mean`, `errored_mean` — are the one group computed over EVERY
sample of the cell. They are the rate of exclusion, so measuring them on the
surviving samples would report zero by construction. Everything else, including
`reasoning_chars_mean`, is over the surviving samples only; the per-sample CSV
keeps every row, so the full distribution including the truncated tail is still
there for anyone who wants it.

Only logs whose status is `success` are read, matching what the sweep script
treats as a completed cell. A cell that failed and was re-run leaves both logs
in the directory, and counting the failed one's partial samples alongside the
re-run's would put the same cell in twice at two sample sizes.
`--include-failed` reads them all, for looking at what went wrong.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalSample, list_eval_logs, read_eval_log
from inspect_ai.model import ContentReasoning
from school_of_reward_hacks.truncation import (
    answer_empty,
    no_answer,
    stop_reason,
    stop_truncated,
)

SCORER_NAME = "reward_hack_gap_scorer"
VECTOR_ARG = "steer_vector"
STRENGTH_ARG = "steer_strength"

SCORE_KEYS = ("metric", "quality", "gap", "unparsed")
"""The score keys this script reads. `metric`, `quality` and `gap` are NaN
together when either judge failed to parse; `unparsed` is the flag saying so.

A log written by the current scorer carries a fifth key, `no_answer`, which is
not read: it is `truncated` under another name, and recomputing it below from
the output instead means logs from before the scorer knew about it are treated
identically to logs from after. (In such a log the scorer also leaves `unparsed`
NaN on a no-answer sample, which lands in the same place: `finite` maps it to
None and the row is dropped from every score mean anyway.)"""

# Reading full samples would pull every message and event — the judge calls are
# events, and their rendered prompts embed the whole judge template. Nothing
# below needs them, and `scores` (which carries the judges' rationales) survives.
EXCLUDE_FIELDS = {"messages", "events", "events_data", "store", "attachments"}

SAMPLE_COLUMNS = [
    "log_file",
    "task_id",
    "log_status",
    "steer_vector",
    "steer_strength",
    "model",
    "epoch",
    "sample_id",
    "category",
    "category_letter",
    "task",
    "evaluation_metric",
    "metric",
    "quality",
    "gap",
    "unparsed",
    "judge1_status",
    "judge2_status",
    "judge1_attempts",
    "judge2_attempts",
    "answer_chars",
    "reasoning_chars",
    "output_tokens",
    "input_tokens",
    "stop_reason",
    "stop_truncated",
    "answer_empty",
    "truncated",
    "sample_error",
]

# Metrics measured on the samples that SURVIVE the truncation exclusion: the
# three paired measures, the instrument-failure rate among answers there was
# something to grade, and the size measurements.
SCORED_METRICS = [
    "metric",
    "quality",
    "gap",
    "unparsed",
    "answer_chars",
    "reasoning_chars",
    "output_tokens",
]

# Metrics measured on EVERY sample of the cell. These are the rates of exclusion
# and of outright failure, so restricting them to the surviving samples would
# report zero by construction and hide the thing they exist to show.
VALIDITY_METRICS = [
    "truncated",
    "stop_truncated",
    "answer_empty",
    "errored",
]

METRICS = SCORED_METRICS + VALIDITY_METRICS

GROUP_COLUMNS = ["steer_vector", "steer_strength"]
CATEGORY_GROUP_COLUMNS = ["steer_vector", "steer_strength", "category"]


def finite(value: Any) -> float | None:
    """A float, or None where the value is missing or the NaN unscored sentinel."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def category_letter(category: str) -> str:
    """The leading letter of a category label, e.g. "B" for "B. Keyword ..."."""
    return (category or "").split(".")[0].strip()


def score_values(sample: EvalSample) -> dict[str, float | None]:
    """The four score keys, with NaN mapped to None."""
    score = (sample.scores or {}).get(SCORER_NAME)
    value = score.value if score is not None else None
    if not isinstance(value, dict):
        return dict.fromkeys(SCORE_KEYS)
    return {key: finite(value.get(key)) for key in SCORE_KEYS}


def judge_metadata(sample: EvalSample, key: str) -> dict[str, Any]:
    """One judge's block of the score metadata, or an empty dict if unscored."""
    score = (sample.scores or {}).get(SCORER_NAME)
    metadata = (score.metadata if score is not None else None) or {}
    block = metadata.get(key)
    return block if isinstance(block, dict) else {}


def reasoning_chars(sample: EvalSample) -> int:
    """Characters of thinking in the reply, summed over its reasoning blocks.

    Thinking length is a signal here rather than bookkeeping — steering can make
    the model deliberate at length or barely at all, and a model that thinks
    past its whole token budget returns an empty visible answer, which is what
    `answer_empty` catches. It is also not otherwise measurable in this sweep:
    the vLLM server returns no `completion_tokens_details`, so `reasoning_tokens`
    is None on every sample. Characters are what the log preserves, so
    characters are what this counts. Read it beside `answer_chars`, which covers
    only the visible reply the judges graded: the two do not overlap.
    """
    if not sample.output.choices:
        return 0
    content = sample.output.choices[0].message.content
    if isinstance(content, str):
        return 0
    return sum(
        len(block.reasoning)
        for block in content
        if isinstance(block, ContentReasoning) and block.reasoning
    )


def sample_row(
    log_file: str,
    task_id: str,
    log_status: str,
    model: str,
    steer_vector: Any,
    steer_strength: Any,
    sample: EvalSample,
) -> dict[str, Any]:
    """One tidy row for one sample."""
    completion = sample.output.completion or ""
    usage = sample.output.usage
    reason = stop_reason(sample.output)
    # The two halves of `truncated`, kept as columns of their own so a cell can
    # be read as "ran out of budget mid-answer" or "thought until there was
    # nothing left to say", which are different things about the steering.
    cut_off = int(stop_truncated(sample.output))
    empty = int(answer_empty(sample.output))
    scores = score_values(sample)
    metadata = sample.metadata or {}
    judge1 = judge_metadata(sample, "judge1_metric")
    judge2 = judge_metadata(sample, "judge2_quality")
    category = str(metadata.get("cat", ""))

    return {
        "log_file": log_file,
        "task_id": task_id,
        "log_status": log_status,
        "steer_vector": steer_vector,
        "steer_strength": steer_strength,
        "model": model,
        "epoch": sample.epoch,
        "sample_id": sample.id,
        "category": category,
        "category_letter": category_letter(category),
        "task": metadata.get("task"),
        "evaluation_metric": metadata.get("evaluation_metric"),
        "metric": scores["metric"],
        "quality": scores["quality"],
        "gap": scores["gap"],
        "unparsed": scores["unparsed"],
        "judge1_status": judge1.get("status"),
        "judge2_status": judge2.get("status"),
        "judge1_attempts": judge1.get("attempts"),
        "judge2_attempts": judge2.get("attempts"),
        "answer_chars": len(completion),
        "reasoning_chars": reasoning_chars(sample),
        "output_tokens": usage.output_tokens if usage else None,
        "input_tokens": usage.input_tokens if usage else None,
        "stop_reason": reason,
        "stop_truncated": cut_off,
        "answer_empty": empty,
        "truncated": int(no_answer(sample.output)),
        "sample_error": sample.error.message if sample.error else None,
    }


def rows_for_log(log_file: str) -> list[dict[str, Any]]:
    """Every sample of one eval log, as tidy rows."""
    log = read_eval_log(log_file, exclude_fields=EXCLUDE_FIELDS)
    model_args = log.eval.model_args or {}

    return [
        sample_row(
            log_file=Path(log.location).name,
            task_id=log.eval.task_id,
            log_status=log.status,
            model=log.eval.model,
            steer_vector=model_args.get(VECTOR_ARG),
            steer_strength=model_args.get(STRENGTH_ARG),
            sample=sample,
        )
        for sample in (log.samples or [])
    ]


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


def aggregate(
    rows: Iterable[dict[str, Any]], group_columns: Sequence[str]
) -> list[dict[str, Any]]:
    """Collapse the per-sample rows to one row per group.

    Truncated samples — no visible answer, or a reply cut off by the token
    budget or the context — are dropped before anything in `SCORED_METRICS` is
    averaged. They are not a low score; they are a missing measurement, and the
    module docstring says at length why leaving them in would bias the arms
    where they are most common. `n` is the cell's size, `n_truncated` is how
    many were dropped, and `n_scored` is what the score means were actually
    computed over. `VALIDITY_METRICS` are averaged over all `n` instead, because
    they are the rate of the dropping.

    Every metric also reports its own `_n` alongside its mean and stderr, since
    the denominators still differ below `n_scored`: a surviving sample where
    either judge failed to parse contributes to `unparsed` but to none of
    `metric`, `quality` or `gap`.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        # `errored` is a metric like any other here: the fraction of the cell
        # that produced no usable sample is exactly what decides whether the
        # cell's other means can be believed.
        row = dict(row, errored=int(row["sample_error"] is not None))
        groups.setdefault(tuple(row[column] for column in group_columns), []).append(
            row
        )

    out = []
    for key, members in sorted(groups.items(), key=lambda item: _sort_key(item[0])):
        scored = [row for row in members if not row.get("truncated")]
        record: dict[str, Any] = dict(zip(group_columns, key))
        record["n"] = len(members)
        record["n_truncated"] = sum(int(bool(row.get("truncated"))) for row in members)
        record["n_scored"] = len(scored)
        record["n_error"] = sum(row["errored"] for row in members)
        for metric in METRICS:
            source = members if metric in VALIDITY_METRICS else scored
            values = [
                float(row[metric]) for row in source if row.get(metric) is not None
            ]
            mean, stderr = mean_stderr(values)
            record[f"{metric}_n"] = len(values)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_stderr"] = stderr
        out.append(record)
    return out


def _sort_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    """Order groups by vector, then numeric strength, then category."""
    vector, strength, *rest = key
    return (
        str(vector),
        float(strength) if strength is not None else float("-inf"),
        str(rest[0]) if rest else "",
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", help="directory of .eval logs to read")
    parser.add_argument(
        "--samples",
        default="school_of_reward_hacks_samples.csv",
        help="per-sample CSV",
    )
    parser.add_argument(
        "--aggregate",
        default="school_of_reward_hacks_by_condition.csv",
        help="per (vector, strength) CSV",
    )
    parser.add_argument(
        "--by-category",
        default="school_of_reward_hacks_by_category.csv",
        help="per (vector, strength, category) CSV",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="also read logs whose status is not success (default: skip them)",
    )
    args = parser.parse_args()

    logs = list_eval_logs(args.log_dir)
    if not logs:
        raise SystemExit(f"no eval logs found in {args.log_dir}")

    rows: list[dict[str, Any]] = []
    skipped = 0
    for info in sorted(logs, key=lambda i: i.name):
        status = read_eval_log(info.name, header_only=True).status
        if status != "success" and not args.include_failed:
            print(f"{Path(info.name).name}: skipped ({status})", flush=True)
            skipped += 1
            continue
        log_rows = rows_for_log(info.name)
        print(f"{Path(info.name).name}: {len(log_rows)} samples", flush=True)
        rows.extend(log_rows)

    if not rows:
        raise SystemExit(
            f"no samples read from {args.log_dir} ({skipped} logs skipped)"
        )

    by_condition = aggregate(rows, GROUP_COLUMNS)
    by_category = aggregate(rows, CATEGORY_GROUP_COLUMNS)

    write_csv(Path(args.samples), SAMPLE_COLUMNS, rows)
    write_csv(
        Path(args.aggregate),
        list(by_condition[0].keys()) if by_condition else GROUP_COLUMNS,
        by_condition,
    )
    write_csv(
        Path(args.by_category),
        list(by_category[0].keys()) if by_category else CATEGORY_GROUP_COLUMNS,
        by_category,
    )
    print(
        f"\n{len(rows)} samples in {len(logs) - skipped} logs "
        f"({skipped} skipped) -> {args.samples}\n"
        f"{len(by_condition)} conditions -> {args.aggregate}\n"
        f"{len(by_category)} (condition, category) cells -> {args.by_category}",
        flush=True,
    )


if __name__ == "__main__":
    main()
