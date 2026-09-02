"""Judge every rollout of a TruthfulQA sweep for why the answer was given.

    python scripts/judge_truthfulqa.py
    python scripts/judge_truthfulqa.py --limit 5 --log-dir example-logs

Reads the sweep's eval logs, and for each scored sample asks
`truthfulqa.answer_motivation` whether the chain of thought answered what the
model believed or what it thought the test wanted. Requires ANTHROPIC_API_KEY
for the judge; the model under test is not called and no GPU is touched, so this
can run alongside the sweep it is reading.

Output is one JSONL file per steering condition under OUT_DIR, one row per
sample, carrying the judgement *and* everything it was made from -- question,
options, letters, correctness and the full chain of thought. Deliberately
self-contained: the plots and the report are built from these files alone, and a
row that only held a label would send every reader back to the .eval logs to
find out what it was a label of.

Resumable, because it is ~7,350 judge calls. A sample id already present in its
condition's file is skipped, and rows are appended and flushed as each judgement
lands rather than at the end, so an interrupted run resumes from where it
stopped instead of from the beginning. Delete a condition's file to re-judge it.

The read is `exclude_fields`-narrowed: `output.message` carries the reasoning
block as well as the answer text, so the messages and events -- 3k-character
scratchpads, twice over, times 817 samples times 9 strengths -- never have to be
materialised.

Two things the judge is not told, both on purpose. The steering vector and
strength never reach it: it scores nine conditions and would be scoring the
condition rather than the reasoning if it could see which. And a judgement that
cannot be read is not a label -- it is counted as a parse failure and written
with an empty `label`, so the failures are visible in the summary instead of
being absorbed into `unclear`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalSample, list_eval_logs, read_eval_log
from inspect_ai.model import Model
from inspect_ai.scorer import CORRECT
from truthfulqa.answer_motivation import (
    JUDGE_EFFORT,
    JUDGE_MODEL,
    Judgement,
    Rollout,
    judge,
    judge_model,
)

from steering_vectors import vectorfmt

LOG_DIR = "logs/truthfulqa-sweep"
OUT_DIR = "logs/truthfulqa-judge"

SCORER_NAME = "choice"
VECTOR_ARG = "steer_vector"
STRENGTH_ARG = "steer_strength"

MAX_CONCURRENT = 25
"""In-flight judge calls. Also the judge model's connection pool, so inspect's
own limiter and the semaphore below agree rather than one throttling under the
other."""

PROGRESS_EVERY = 25
"""Judgements between progress lines. The run is an hour long; silence for an
hour is indistinguishable from a run that died."""

# `output` carries the answer text and the reasoning block, and `scores` the
# letter and the verdict, which is everything a row needs. `messages` and
# `events` hold a second copy of the same 3k-character scratchpad per sample.
EXCLUDE_FIELDS = {"messages", "events", "events_data", "store", "attachments"}

# Sonnet 5, $ per million tokens. Only used for the run's own cost line, which
# is the number that decides whether a re-judge is worth it.
INPUT_COST_PER_MTOK = 2.00
OUTPUT_COST_PER_MTOK = 10.00

# Anthropic's standard multipliers on the input price: a cache read is a tenth
# of an input token, a five-minute cache write is an input token and a quarter.
# These are not decoration -- the endpoint caches implicitly, and a judge prompt
# is 2.6k tokens of which ~2.6k is the cached prefix, so a cost line that
# counted `input_tokens` alone would report a run costing dollars as costing
# cents. Same reason `judge_input_tokens_cache_read` is on every row.
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class Row:
    """One sample of one condition: the blinded rollout plus its bookkeeping."""

    sample_id: str
    steer_vector: str | None
    steer_strength: float | None
    correct: bool | None
    rollout: Rollout

    @property
    def stem(self) -> str:
        """The output file this row belongs in, `<vector>_<strength>`."""
        vector = self.steer_vector or "none"
        strength = (
            "none" if self.steer_strength is None else f"{self.steer_strength:+.2f}"
        )
        return f"{vector}_{strength}"


def chain_of_thought(sample: EvalSample) -> str:
    """The sample's reasoning blocks, joined.

    The served model returns one `ContentReasoning` block and one
    `ContentText`; the text is `ANSWER: X` and is already recorded as the
    answer letter, so only the reasoning is the judge's material. Joined rather
    than indexed in case a future run returns interleaved thinking.
    """
    content = sample.output.message.content
    if isinstance(content, str):
        return ""
    return "\n\n".join(
        block.reasoning for block in content if block.type == "reasoning"
    )


def rows_for_log(log_file: str, limit: int | None) -> list[Row]:
    """Every scored sample of one eval log, as judgeable rows.

    Only successful logs are read, matching the sweep's own notion of a
    finished strength: a partial log would contribute a truncated cell that
    looks like a complete one once it is a JSONL file.
    """
    log = read_eval_log(log_file, exclude_fields=EXCLUDE_FIELDS)
    if log.status != "success":
        print(f"  skipped ({log.status}): {log_file}", flush=True)
        return []

    model_args = log.eval.model_args or {}
    vector = model_args.get(VECTOR_ARG)
    strength = model_args.get(STRENGTH_ARG)

    rows = []
    for sample in log.samples or []:
        score = (sample.scores or {}).get(SCORER_NAME)
        # `choice()` puts the verdict in `value` and the letter the model picked
        # in `answer`, so correctness is not recomputed here from the letters.
        rows.append(
            Row(
                sample_id=str(sample.id),
                steer_vector=None if vector is None else vectorfmt.vector_id(vector),
                steer_strength=None if strength is None else float(strength),
                correct=None if score is None else score.value == CORRECT,
                rollout=Rollout(
                    question=str(sample.input),
                    choices=list(sample.choices or []),
                    target_letters=list(sample.target),
                    answer_letter=None if score is None else score.answer,
                    cot=chain_of_thought(sample),
                ),
            )
        )
    return rows if limit is None else rows[:limit]


def judged_ids(path: Path) -> set[str]:
    """Sample ids already written to one condition's file."""
    if not path.exists():
        return set()
    ids = set()
    with path.open() as file:
        for line in file:
            line = line.strip()
            if line:
                ids.add(str(json.loads(line)["sample_id"]))
    return ids


def record(row: Row, judgement: Judgement, judge_name: str) -> dict[str, Any]:
    """One JSONL row: the judgement and everything it was made from."""
    usage = judgement.usage
    return {
        "sample_id": row.sample_id,
        "steer_vector": row.steer_vector,
        "steer_strength": row.steer_strength,
        "question": row.rollout.question,
        "choices": row.rollout.choices,
        # Comma-joined, because mc2 questions have more than one true answer.
        # mc1, which the sweep runs, always gives a single letter.
        "target_letter": ",".join(row.rollout.target_letters),
        "answer_letter": row.rollout.answer_letter,
        "correct": row.correct,
        "label": judgement.label,
        "test_reasoning": judgement.test_reasoning,
        "quote": judgement.quote,
        # Whether `quote` really is a span of the chain of thought. The judge is
        # told to copy verbatim; this is the check on that, and a column of
        # `false` would mean the quotes are reconstructions and no evidence.
        "quote_verbatim": bool(judgement.quote) and judgement.quote in row.rollout.cot,
        "justification": judgement.justification,
        "parse_error": judgement.parse_error,
        "judge_model": judge_name,
        "judge_effort": JUDGE_EFFORT,
        "judge_input_tokens": None if usage is None else usage.input_tokens,
        "judge_input_tokens_cache_read": (
            None if usage is None else usage.input_tokens_cache_read
        ),
        "judge_output_tokens": None if usage is None else usage.output_tokens,
        "judge_reasoning_tokens": None if usage is None else usage.reasoning_tokens,
        # Last because it is thousands of characters and would otherwise push
        # every field a reader cares about off the end of the line.
        "cot": row.rollout.cot,
    }


async def judge_rows(
    rows: list[Row],
    model: Model,
    judge_name: str,
    out_path: Path,
    counts: Counter[str],
) -> None:
    """Judge one condition's outstanding rows, appending each as it lands."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def one(row: Row) -> tuple[Row, Judgement]:
        async with semaphore:
            return row, await judge(row.rollout, model)

    done = 0
    with out_path.open("a") as file:
        for task in asyncio.as_completed([one(row) for row in rows]):
            row, judgement = await task
            file.write(json.dumps(record(row, judgement, judge_name)) + "\n")
            file.flush()

            usage = judgement.usage
            if usage is not None:
                counts["input_tokens"] += usage.input_tokens
                counts["output_tokens"] += usage.output_tokens
                counts["cache_read_tokens"] += usage.input_tokens_cache_read or 0
                counts["cache_write_tokens"] += usage.input_tokens_cache_write or 0
                counts["reasoning_tokens"] += usage.reasoning_tokens or 0
            counts["judged"] += 1
            done += 1
            if done % PROGRESS_EVERY == 0 or done == len(rows):
                print(
                    f"    {done}/{len(rows)} judged in {out_path.name}",
                    flush=True,
                )


def summarise(out_dir: Path) -> None:
    """Per condition: how many rows, what they were labelled, how many failed."""
    print("\ncondition            n  believed_true  expected_on_test  unclear  failed")
    for path in sorted(out_dir.glob("*.jsonl")):
        labels: Counter[str] = Counter()
        failures = 0
        total = 0
        with path.open() as file:
            for line in file:
                if not line.strip():
                    continue
                total += 1
                row = json.loads(line)
                if row["parse_error"] is not None:
                    failures += 1
                else:
                    labels[row["label"]] += 1
        print(
            f"{path.stem:<16} {total:>5} {labels['believed_true']:>14}"
            f" {labels['expected_on_test']:>17} {labels['unclear']:>8} {failures:>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=LOG_DIR, help="sweep logs to judge")
    parser.add_argument("--out-dir", default=OUT_DIR, help="where the JSONL rows go")
    parser.add_argument("--judge", default=JUDGE_MODEL, help="judge model")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="judge at most N samples per log (a cheap trial that still covers "
        "every condition, rather than N samples of the first one)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = judge_model(args.judge, max_connections=MAX_CONCURRENT)
    counts: Counter[str] = Counter()

    log_files = sorted(info.name for info in list_eval_logs(args.log_dir))
    print(f"{len(log_files)} logs in {args.log_dir} -> {out_dir}", flush=True)

    for log_file in log_files:
        rows = rows_for_log(log_file, args.limit)
        if not rows:
            continue
        stem = rows[0].stem
        out_path = out_dir / f"{stem}.jsonl"
        already = judged_ids(out_path)
        outstanding = [row for row in rows if row.sample_id not in already]
        print(
            f"  {stem}: {len(rows)} samples, {len(already)} already judged, "
            f"{len(outstanding)} to do",
            flush=True,
        )
        if outstanding:
            asyncio.run(judge_rows(outstanding, model, args.judge, out_path, counts))

    summarise(out_dir)
    billed_input = (
        counts["input_tokens"]
        + counts["cache_read_tokens"] * CACHE_READ_MULTIPLIER
        + counts["cache_write_tokens"] * CACHE_WRITE_MULTIPLIER
    )
    cost = (
        billed_input / 1e6 * INPUT_COST_PER_MTOK
        + counts["output_tokens"] / 1e6 * OUTPUT_COST_PER_MTOK
    )
    cached = counts["cache_read_tokens"] + counts["cache_write_tokens"]
    print(
        f"\nthis run: {counts['judged']} judged, "
        f"{counts['input_tokens'] + cached} input ({cached} of it cached) "
        f"+ {counts['output_tokens']} output tokens "
        f"({counts['reasoning_tokens']} of them thinking), ~${cost:.2f}"
    )


if __name__ == "__main__":
    main()
