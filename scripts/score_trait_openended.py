"""The judge pass: score saved open-ended generation logs into a new directory.

    python scripts/score_trait_openended.py logs --output-dir logs/judged \
        --grader-model anthropic/claude-sonnet-5 --judge-temperature 0

Takes log files and/or directories of them, applies both scorers
(`trait_openended_scorer`, `trait_openended_diagnostics`) and writes the result
to a *separate* directory. The input logs are never modified: the generation
pass can cost GPU hours on a steered server, the judge pass costs API calls, and
only one of those is cheap to redo -- so the expensive artefact is treated as
read-only. The script refuses outright to write into a directory it is reading
from.

Nothing here touches the model under test. The scorers read the reply out of the
saved sample, so this runs without the generation provider, on another machine,
while the generation pass is still going.

Two things worth pausing over before a re-judge:

  A judge model is part of the metric's identity. Re-scoring one condition with
  a different judge than another and comparing the two numbers is not a
  comparison. Every score carries `judge_model` and `rubric_sha256`; use them.

  Comparing two judges on the same replies means chaining the passes, not
  running this twice over the same input. `--action append` keeps whatever
  scores the input already had and adds this pass's under deduped names
  (`trait_openended_scorer`, `trait_openended_scorer1`), so the way to get both
  judges into one log is generation log -> judged-a -> judged-b:

      python scripts/score_trait_openended.py logs --output-dir logs/judge-a \
          --grader-model anthropic/claude-sonnet-5
      python scripts/score_trait_openended.py logs/judge-a \
          --output-dir logs/judge-a-and-b --grader-model openai/gpt-5.6

  Running this twice over the same *unjudged* logs gives two separate logs with
  one judge each, which is fine for comparing aggregates but does not put the
  two verdicts side by side on a sample. `--action overwrite` replaces the
  earlier pass instead.

Requires an API key for the judge model, and nothing else.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from urllib.parse import unquote, urlparse

from inspect_ai import score
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log, write_eval_log
from trait_openended import (
    clear_judge_cache,
    trait_openended_diagnostics,
    trait_openended_scorer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge saved open-ended trait-expression logs.",
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
        help="Directory the judged copies are written to. Must not be an input.",
    )
    parser.add_argument(
        "--grader-model",
        default=None,
        help="Judge model; default resolves $INSPECT_GRADER_MODEL then the "
        "package default (anthropic/claude-sonnet-5).",
    )
    parser.add_argument(
        "--judge-samples",
        type=int,
        default=1,
        help="Judge calls per reply, reduced by median score and refusal vote.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Judge sampling temperature to request. Requested is not applied: "
        "Inspect's Anthropic provider drops the parameter for every Claude "
        "4.7+ model, including the default judge, so that judge runs "
        "unpinned. Every score records the requested and the applied value "
        "separately, and judge_temperature is null when nothing was sent.",
    )
    parser.add_argument(
        "--no-judge-temperature",
        action="store_true",
        help="Send no temperature at all, leaving the provider's own.",
    )
    parser.add_argument(
        "--action",
        choices=["append", "overwrite"],
        default="append",
        help="'append' keeps any existing scores under deduped names, which is "
        "how two judges are compared on the same replies; 'overwrite' "
        "replaces them.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model reconstructed from the log header. Set this "
        "(e.g. mockllm/model) to judge on a host where the generation "
        "provider is not installed. The judge is unaffected either way.",
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
            # A typo'd path that silently contributes nothing would show up as
            # a group with no data rather than as an error.
            raise FileNotFoundError(raw)
    # list_eval_logs on overlapping inputs can return the same file twice, and
    # judging it twice would double the bill for no extra information.
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
    """Where a judged copy goes: the source's basename, deduped if it collides."""
    name = Path(source).name
    candidate = output_dir / name
    if candidate.exists():
        candidate = output_dir / f"{index:03d}_{name}"
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    temperature = None if args.no_judge_temperature else args.judge_temperature
    scorers = [
        trait_openended_scorer(
            grader_model=args.grader_model,
            judge_samples=args.judge_samples,
            judge_temperature=temperature,
        ),
        trait_openended_diagnostics(
            grader_model=args.grader_model,
            judge_samples=args.judge_samples,
            judge_temperature=temperature,
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
                "judging must not overwrite the generation pass."
            )
    print(f"judging {len(sources)} log(s) -> {output_dir}", flush=True)

    failures = 0
    for index, source in enumerate(sources):
        print(f"\n[{index + 1}/{len(sources)}] {source}", flush=True)
        # The judge cache is scoped to one log. Two logs can hold the same
        # sample ids, and with `--judge-samples > 1` two logs holding the same
        # reply must buy their own verdicts or the variance the flag was raised
        # to measure is copied from the first log to the second.
        clear_judge_cache()
        try:
            log: EvalLog = read_eval_log(source)
            if not log.samples:
                print("  no samples; skipped", flush=True)
                continue
            judged = score(
                log,
                scorers,
                action=args.action,
                model=args.model,
                display=args.display,
                copy=True,
            )
            destination = output_path(source, output_dir, index)
            write_eval_log(judged, str(destination))
            for entry in judged.results.scores if judged.results else []:
                metrics = ", ".join(
                    f"{name}={metric.value}" for name, metric in entry.metrics.items()
                )
                print(f"  {entry.scorer}/{entry.name}: {metrics}", flush=True)
            print(f"  wrote {destination}", flush=True)
        except Exception:  # noqa: BLE001 - one bad log must not lose the rest
            failures += 1
            print(f"  FAILED to judge {source}:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()

    if failures:
        print(f"\n{failures} of {len(sources)} log(s) failed to judge", flush=True)
        return 1
    print(f"\njudged {len(sources)} log(s) into {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
