"""Turn a directory of Terminal-Bench 2.1 eval logs into two tidy CSVs.

    python scripts/analyze_terminal_bench.py logs/tb2-sweep

Writes one row per sample to `--samples` and one row per (vector, strength) to
`--aggregate`. The per-sample file is the source of truth and the aggregate is
derived from it with nothing else added, so every number in a plot can be
traced back to a task name.

The thing to get right in this eval is the denominator. A sample whose
verifier could not be run is scored NaN on `reward` and `resolved`, not zero,
because an unreachable container is not a failed task. So:

* `resolved_mean` here is a mean over the samples that produced a reward, and
  `resolved_n` is how many that was. Compare it against `n`.
* `verifier_failed_mean` is the share of the cell that produced no reward at
  all. Read it first. Under a steering sweep it is not automatically noise:
  a condition that makes the agent fill the disk, or leave a process holding
  the container, breaks verifiers at a rate that varies with the condition.
  It is also the only reading of that share that survives epochs above 1:
  inspect's epoch reducer drops a NaN epoch rather than carrying it into the
  reduced score, so `reward.unscored_samples` in an eval log undercounts
  verifier failures whenever a task was attempted more than once. The
  per-sample CSV here is written before any reduction, so it does not.
* `reward_fractional_mean` should be exactly 0 on Terminal-Bench 2.1, whose 89
  verifiers all write a bare 0 or 1. Anything else means partial credit has
  appeared and `resolved = reward == 1.0` has become a choice rather than an
  identity.

`tool_calls`, `messages` and `submitted` are carried alongside because the
same `resolved = 0` means different things with 3 tool calls and with 300.

Only logs whose status is `success` are read, matching what a sweep treats as
a completed cell; `--include-failed` reads them all, for looking at what went
wrong. Logs from a different eval are skipped by name and counted: a sweep
directory shared with another eval would otherwise contribute rows with every
metric empty, and those rows are indistinguishable from verifier failures in
the closing summary.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalSample, list_eval_logs, read_eval_log

TASK_NAME = "terminal_bench_2"
"""The eval whose logs this reads. Named so a shared log directory can be
told apart from a mixed one rather than silently averaged together."""

REWARD_SCORER = "harbor_reward"
DIAGNOSTICS_SCORER = "harbor_diagnostics"

VECTOR_ARG = "steer_vector"
STRENGTH_ARG = "steer_strength"

# Reading full samples would pull every message and event: an agentic sample is
# hundreds of tool calls, times 89 tasks, times the cells in a sweep. The
# scorers already reduced everything below to numbers at scoring time.
EXCLUDE_FIELDS = {"messages", "events", "events_data", "store", "attachments"}

REWARD_KEYS = ["reward", "resolved"]

DIAGNOSTIC_KEYS = [
    "verifier_failed",
    "unscored",
    "scorer_ran",
    "reward_fractional",
    "messages",
    "tool_calls",
    "submitted",
    "agent_limit_hit",
    "truncated_generations",
    "max_tokens_stops",
    "model_length_stops",
    "generations",
    "solution_exit_code",
    "declared_agent_timeout_sec",
    "verifier_output_chars",
]

SAMPLE_COLUMNS = [
    "log_file",
    "task_id",
    "log_status",
    "model",
    "steer_vector",
    "steer_strength",
    "epoch",
    "sample_id",
    "task_name",
    *REWARD_KEYS,
    *DIAGNOSTIC_KEYS,
    "verifier_error_type",
    "verifier_exit_code",
    "adapter_answer",
    "limit_types",
    "total_time",
    "working_time",
    "sample_error",
]

METRICS = [*REWARD_KEYS, *DIAGNOSTIC_KEYS, "errored"]

GROUP_COLUMNS = ["steer_vector", "steer_strength"]


def _score_dict(sample: EvalSample, scorer: str, keys: Sequence[str]) -> dict[str, Any]:
    """One scorer's keys for a sample, with None where it did not produce a number.

    NaN becomes None on the way out. A NaN in a CSV is a value that silently
    poisons whatever reads it; None is an empty cell, which is what an
    unscorable sample deserves, and the count of them is recoverable from
    `verifier_failed` and from the `_n` columns in the aggregate.
    """
    score = (sample.scores or {}).get(scorer)
    value = score.value if score is not None else None
    if not isinstance(value, dict):
        return {key: None for key in keys}
    out: dict[str, Any] = {}
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, (int, float)) and not math.isnan(float(raw)):
            out[key] = float(raw)
        else:
            out[key] = None
    return out


def _score_metadata(sample: EvalSample, scorer: str, key: str) -> Any:
    score = (sample.scores or {}).get(scorer)
    return (score.metadata or {}).get(key) if score is not None else None


def sample_row(
    log_file: str,
    task_id: str,
    log_status: str,
    model: str,
    steer_vector: Any,
    steer_strength: Any,
    sample: EvalSample,
) -> dict[str, Any]:
    """One row per scored sample, with the scorers' numbers as they were logged."""
    limits = _score_metadata(sample, DIAGNOSTICS_SCORER, "limit_types")
    return {
        "log_file": log_file,
        "task_id": task_id,
        "log_status": log_status,
        "model": model,
        "steer_vector": steer_vector,
        "steer_strength": steer_strength,
        "epoch": sample.epoch,
        "sample_id": sample.id,
        # The bare task name, so this joins to an official harbor run's trial
        # directories (which are named `<task-name>__<id>`) without the org
        # prefix that harbor's sample ids carry.
        "task_name": str(sample.id).split("/")[-1],
        **_score_dict(sample, REWARD_SCORER, REWARD_KEYS),
        **_score_dict(sample, DIAGNOSTICS_SCORER, DIAGNOSTIC_KEYS),
        "verifier_error_type": _score_metadata(
            sample, DIAGNOSTICS_SCORER, "verifier_error_type"
        ),
        "verifier_exit_code": _score_metadata(
            sample, DIAGNOSTICS_SCORER, "verifier_exit_code"
        ),
        "adapter_answer": _score_metadata(sample, DIAGNOSTICS_SCORER, "adapter_answer"),
        "limit_types": ",".join(limits) if limits else None,
        "total_time": sample.total_time,
        "working_time": sample.working_time,
        "sample_error": sample.error.message if sample.error else None,
    }


def rows_for_log(log_file: str) -> list[dict[str, Any]]:
    """Every sample in one eval log, as rows."""
    log = read_eval_log(log_file, exclude_fields=EXCLUDE_FIELDS)
    model_args = log.eval.model_args or {}
    return [
        sample_row(
            log_file=Path(log_file).name,
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


def aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the per-sample rows to one row per (vector, strength)."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        row = dict(row, errored=int(row["sample_error"] is not None))
        groups.setdefault(tuple(row[column] for column in GROUP_COLUMNS), []).append(
            row
        )

    out = []
    for key, members in sorted(groups.items(), key=lambda item: _sort_key(item[0])):
        record: dict[str, Any] = dict(zip(GROUP_COLUMNS, key))
        record["n"] = len(members)
        record["n_error"] = sum(row["errored"] for row in members)
        record["n_tasks"] = len({row["task_name"] for row in members})
        for metric in METRICS:
            values = [
                float(row[metric]) for row in members if row.get(metric) is not None
            ]
            mean, stderr = mean_stderr(values)
            # `_n` is not decoration: for reward and resolved it is the
            # denominator, and it is smaller than `n` exactly when a verifier
            # failed. A mean without it cannot be read.
            record[f"{metric}_n"] = len(values)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_stderr"] = stderr
        out.append(record)
    return out


def _sort_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    """Order groups by vector, then numeric strength."""
    vector, strength = key
    return (
        str(vector),
        float(strength) if strength is not None else float("-inf"),
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
        "--samples", default="terminal_bench_2_samples.csv", help="per-sample CSV"
    )
    parser.add_argument(
        "--aggregate",
        default="terminal_bench_2_by_condition.csv",
        help="per (vector, strength) CSV",
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
    foreign = 0
    for info in sorted(logs, key=lambda i: i.name):
        header = read_eval_log(info.name, header_only=True)
        # A log from another eval has neither of our scorers, so every metric
        # in it would come out empty and be counted as a verifier failure. The
        # header read has already happened, so this costs nothing.
        if header.eval.task != TASK_NAME:
            print(
                f"{Path(info.name).name}: skipped (task {header.eval.task!r}, "
                f"not {TASK_NAME!r})",
                flush=True,
            )
            foreign += 1
            continue
        if header.status != "success" and not args.include_failed:
            print(f"{Path(info.name).name}: skipped ({header.status})", flush=True)
            skipped += 1
            continue
        log_rows = rows_for_log(info.name)
        print(f"{Path(info.name).name}: {len(log_rows)} samples", flush=True)
        rows.extend(log_rows)

    if not rows:
        raise SystemExit(
            f"no samples read from {args.log_dir} ({skipped} logs skipped as "
            f"unsuccessful, {foreign} as belonging to another eval)"
        )

    grouped = aggregate(rows)
    aggregate_columns = list(grouped[0].keys()) if grouped else GROUP_COLUMNS

    write_csv(Path(args.samples), SAMPLE_COLUMNS, rows)
    write_csv(Path(args.aggregate), aggregate_columns, grouped)

    unscored = sum(1 for row in rows if row["reward"] is None)
    print(
        f"\n{len(rows)} samples in {len(logs) - skipped - foreign} logs "
        f"({skipped} skipped as unsuccessful, {foreign} as belonging to another "
        f"eval) -> {args.samples}\n"
        f"{len(grouped)} conditions -> {args.aggregate}\n"
        f"{unscored} of those {len(rows)} sample(s) produced no reward "
        "(verifier failures); they are empty cells here and are out of every "
        "mean above",
        flush=True,
    )


if __name__ == "__main__":
    main()
