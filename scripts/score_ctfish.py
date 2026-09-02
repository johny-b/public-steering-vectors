"""The judge pass: re-score saved ctfish agent-pass logs with the LLM judges.

    python scripts/score_ctfish.py logs/ctfish --output-dir logs/ctfish-scored \
        --stage-judge-model anthropic/claude-sonnet-5 \
        --escalation-judge-model anthropic/claude-sonnet-5 \
        --judge-temperature 0 --min-judge-entries 0

Takes log files and/or directories of them, applies the full scorer set
(`won_game`, `moves_submitted`, `engine_assisted_moves`,
`truncated_generations`, `stage_judge`, `escalation_judge`) and writes the
result to a *separate* directory. The input logs are never modified: the agent
pass costs GPU hours, the judge pass costs an API call, and only one of those is
cheap to redo -- so the expensive artefact is treated as read-only.

Scoring never touches the sandbox (every scorer reads `state.store`), so this
needs no Docker and can run on a different machine from the agent pass, while
the agent pass is still running.

Requires the judge models' API keys, and nothing else.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from urllib.parse import unquote, urlparse

from ctfish.scorer import (
    engine_assisted_moves,
    escalation_judge,
    moves_submitted,
    stage_judge,
    truncated_generations,
    won_game,
)
from inspect_ai import score
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log, write_eval_log


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-score saved ctfish logs with the judges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Log files, or directories searched recursively for them.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory the scored copies are written to. Must not be an input.",
    )
    parser.add_argument("--stage-judge-model", default="anthropic/claude-haiku-4-5")
    parser.add_argument("--escalation-judge-model", default="anthropic/claude-sonnet-5")
    parser.add_argument(
        "--judge-samples",
        type=int,
        default=1,
        help="Independent stage-judge samples per run, reduced by majority vote.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for both judges. Pin it: judge variance "
        "between runs can be as large as the effect being measured.",
    )
    parser.add_argument(
        "--min-judge-entries",
        type=int,
        default=0,
        help="0 judges every run, however short. Anything higher leaves short "
        "runs unscored, and whatever made them short decides which ones.",
    )
    parser.add_argument(
        "--action",
        choices=["overwrite", "append"],
        default="overwrite",
        help="'overwrite' replaces the agent pass's deterministic scores with "
        "this pass's (identical values, recomputed); 'append' keeps both under "
        "deduped names.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the primary model reconstructed from the log header. Set "
        "this (e.g. mockllm/model) to score on a host where the agent-pass "
        "provider is not installed. The judges are unaffected either way.",
    )
    parser.add_argument(
        "--no-judges",
        action="store_true",
        help="Recompute only the deterministic scorers (no API keys needed).",
    )
    parser.add_argument(
        "--escalation-judge-temperature",
        type=float,
        default=None,
        help="Temperature for the escalation judge; omit for models that "
        "reject the parameter (claude-sonnet-5 does).",
    )
    parser.add_argument(
        "--escalation-judge-prefill",
        action="store_true",
        default=False,
        help='Seed the escalation judge with "Key Observations:". Off by '
        "default: claude-sonnet-5 rejects assistant prefill.",
    )
    parser.add_argument("--display", default="plain")
    return parser.parse_args(argv)


def collect_logs(inputs: list[str]) -> list[str]:
    """Every log file named directly or found under a named directory."""
    found: list[str] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found += [info.name for info in list_eval_logs(str(path), recursive=True)]
        elif path.exists():
            found.append(str(path))
        else:
            # A typo'd path that silently contributes nothing would show up as a
            # group with no data rather than as an error.
            raise FileNotFoundError(raw)
    # list_eval_logs on overlapping inputs can return the same file twice, and
    # scoring it twice would double the judge bill for no extra information.
    return sorted(dict.fromkeys(found))


def local_path(source: str) -> Path | None:
    """The local filesystem path of a log, or None if it lives somewhere else.

    `list_eval_logs` hands back `file://...` URIs, not paths, so comparing them
    to a directory with `Path` silently never matches -- which would have made
    the "do not write into your own input" guard below quietly inert.
    """
    parsed = urlparse(source)
    if parsed.scheme in ("", "file"):
        return Path(unquote(parsed.path) if parsed.scheme == "file" else source)
    return None


def output_path(source: str, output_dir: Path, index: int) -> Path:
    """Where a scored copy goes: the source's basename, deduped if it collides."""
    name = Path(source).name
    candidate = output_dir / name
    if candidate.exists():
        candidate = output_dir / f"{index:03d}_{name}"
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    scorers = [
        won_game(),
        moves_submitted(),
        engine_assisted_moves(),
        truncated_generations(),
    ]
    if not args.no_judges:
        scorers += [
            stage_judge(
                model=args.stage_judge_model,
                samples=args.judge_samples,
                min_entries=args.min_judge_entries,
                temperature=args.judge_temperature,
            ),
            escalation_judge(
                model=args.escalation_judge_model,
                min_entries=args.min_judge_entries,
                # Verified against the live endpoint: claude-sonnet-5 answers
                # HTTP 400 to *both* `temperature` ("deprecated for this model")
                # and an assistant prefill ("does not support assistant message
                # prefill"), so this judge is run without either while the stage
                # judge stays pinned at 0.
                temperature=args.escalation_judge_temperature,
                prefill=args.escalation_judge_prefill,
            ),
        ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = collect_logs(args.inputs)
    resolved_out = output_dir.resolve()
    for source in sources:
        source_path = local_path(source)
        if source_path is not None and source_path.resolve().parent == resolved_out:
            raise ValueError(
                f"--output-dir {output_dir} contains input log {source}; "
                "scoring must not overwrite the agent pass."
            )
    print(f"scoring {len(sources)} log(s) -> {output_dir}", flush=True)

    failures = 0
    for index, source in enumerate(sources):
        print(f"\n[{index + 1}/{len(sources)}] {source}", flush=True)
        try:
            log: EvalLog = read_eval_log(source)
            if not log.samples:
                print("  no samples; skipped", flush=True)
                continue
            scored = score(
                log,
                scorers,
                action=args.action,
                model=args.model,
                display=args.display,
                copy=True,
            )
            destination = output_path(source, output_dir, index)
            write_eval_log(scored, str(destination))
            for entry in scored.results.scores if scored.results else []:
                metrics = ", ".join(
                    f"{name}={metric.value}" for name, metric in entry.metrics.items()
                )
                print(f"  {entry.name}: {metrics}", flush=True)
            print(f"  wrote {destination}", flush=True)
        except Exception:  # noqa: BLE001 - one bad log must not lose the rest
            failures += 1
            print(f"  FAILED to score {source}:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()

    if failures:
        print(f"\n{failures} of {len(sources)} log(s) failed to score", flush=True)
        return 1
    print(f"\nscored {len(sources)} log(s) into {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
