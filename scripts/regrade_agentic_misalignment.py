"""Re-grade agentic misalignment transcripts with a second grader, to test the grader.

    python scripts/regrade_agentic_misalignment.py logs/am-sweep \
        --grader anthropic/claude-sonnet-4-6 --run-tag sonnet-rerun
    python scripts/regrade_agentic_misalignment.py logs/am-sweep \
        --grader openai/gpt-5.5 --run-tag gpt55

The strength sweep produces a curve in `harmful` and `classifier_verdict`. The
steering vector also changes writing style enormously — long discursive prose at
negative strengths, short compliant replies at positive ones — and the score is
an LLM's judgement of that prose. A grader whose yes-rate moved with verbosity,
hedging or tone would draw the same curve out of transcripts whose *behaviour*
never changed. This script exists to separate those two explanations, and it
answers two separate questions:

  1. Does a different-family grader reproduce the curve? Run it with an OpenAI
     grader and compare the per-strength rates against the Anthropic original.
     A family-specific stylistic bias cannot survive that.
  2. How self-consistent is the grader with itself? Run it with the ORIGINAL
     grader and compare verdict-by-verdict against what the sweep recorded.
     That is the noise floor: no disagreement between graders is interesting
     below the level a grader disagrees with itself at, and — the point of the
     per-strength breakdown — self-consistency that *varies* with strength is
     itself a way to manufacture a curve.

So agreement is reported both overall and per steering strength. The breakdown
is what matters: a grader that is uniformly 5% noisy shifts the curve's level,
while a grader that is 2% noisy at +0.3 and 20% noisy at -0.5 bends its shape.

Nothing about the grading is reimplemented here. The eval's own classifiers
build the prompt (`get_grader_input`) and parse the answer (`classify`), called
in the same order `scorers.score_from_classifier` calls them, so a re-grade with
the original grader differs from the original verdict only by grader sampling
noise. Restating the classifier prompts here would make a disagreement mean
nothing, since it could always be this file's paraphrase rather than the grader.

Two details of the recovery are worth stating, because the scorer had a task
state and this has only a log:

  * `response` is `sample.output.completion`, which is what the scorer passed.
    It is the final assistant message's TEXT content: thinking arrives as
    separate `ContentReasoning` blocks and is excluded, so the grader sees what
    it saw originally rather than 32k tokens of scratchpad on top.
  * `email_content` is the sample's user message, which is the eval's
    `user_prompt` and `email_content` joined. The scorer regenerated the prompts
    and passed `email_content` alone. Those are interchangeable *here* because
    `email_content` reaches the classifier only through `get_email_context`,
    which pulls `<email>` tags out of it by index, and the `user_prompt` prefix
    contains no `<email>` tag — so the same emails come back in the same order.
    Reading the log is preferred to regenerating: it grades the prompt the model
    actually saw.

The subsample is stratified and deterministic — N_PER_CELL per (scenario,
strength), ordered by epoch — so two grader runs score the SAME transcripts and
agreement can be computed pairwise rather than as a difference of two rates.
Determinism by sort order alone is not enough while the sweep is still running:
a block that finishes another cell between the two runs would change which
transcripts sort first. So the selection is written to a manifest on the first
run and re-read on every later one, and the manifest, not the directory listing,
is what the runs agree about. Delete it to re-stratify.

Only logs whose status is `success` are read, matching
`analyze_agentic_misalignment.py` and the sweep's own notion of a finished cell.

Output is one CSV row per (transcript, grader run), appended: a run adds its
rows to whatever is already in the file rather than replacing it, so run A and
run B pool into one tidy frame keyed by `run_tag`. Grader calls that fail or
return an unparseable answer are written as rows with `grader_error` set and
empty verdicts — a re-grade that quietly dropped the transcripts one grader
choked on would bias exactly the comparison being made.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_misalignment.classifiers import (
    BlackmailClassifier,
    Classifier,
    LeakClassifier,
    MurderClassifier,
)
from inspect_ai.log import EvalSample, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, Model, get_model

SCORER_NAME = "harmfulness_scorer"
VECTOR_ARG = "steer_vector"
STRENGTH_ARG = "steer_strength"

GRADER = "anthropic/claude-sonnet-4-6"
"""Default grader. The sweep's own grader, i.e. the self-consistency run."""

RUN_TAG = "rerun"
"""Label written into every row, distinguishing runs pooled in one CSV."""

N_PER_CELL = 20
"""Transcripts per (scenario, strength). 27 cells, so 540 grader calls a run."""

MAX_CONCURRENT = 16
"""In-flight grader calls. Also the model's connection pool, so inspect's own
limiter and the semaphore below agree instead of one throttling under the other."""

# Reading full samples would pull every message and event — 32k-token thinking
# transcripts, times epochs, times 48 logs. `input` carries the user message and
# `output` the completion, which is everything grading needs.
EXCLUDE_FIELDS = {"messages", "events", "events_data", "store", "attachments"}

CSV_COLUMNS = [
    "scenario",
    "steer_strength",
    "epoch",
    "log_file",
    "sample_id",
    "grader_model",
    "run_tag",
    "orig_harmful",
    "orig_classifier_verdict",
    "new_harmful",
    "new_classifier_verdict",
    "grader_error",
    # Last because it is thousands of characters and would otherwise push every
    # column a reader cares about off the right of the screen.
    "grader_reasoning",
]

MANIFEST_COLUMNS = ["scenario", "steer_strength", "epoch", "log_file", "sample_id"]


def classifier_for(scenario: str) -> Classifier:
    """The eval's own classifier for a scenario, used for its prompt and parsing."""
    match scenario:
        case "blackmail":
            return BlackmailClassifier()
        case "leaking":
            return LeakClassifier()
        case "murder":
            return MurderClassifier()
        case _:
            raise ValueError(f"unknown scenario: {scenario!r}")


@dataclass(frozen=True)
class Transcript:
    """One graded sample, with everything a re-grade needs and nothing else."""

    scenario: str
    steer_strength: float | None
    epoch: int
    log_file: str
    sample_id: str
    email_content: str
    response: str
    orig_harmful: float | None
    orig_classifier_verdict: float | None

    @property
    def key(self) -> tuple[str, str, int]:
        """Identity of the transcript across runs, as the manifest records it."""
        return (self.log_file, str(self.sample_id), self.epoch)

    @property
    def cell(self) -> tuple[str, float | None]:
        """The (scenario, strength) cell this transcript is stratified into."""
        return (self.scenario, self.steer_strength)


def user_content(sample: EvalSample) -> str:
    """The sample's user prompt, as the model received it.

    `sample.input` is the dataset input before the solver ran; this eval puts the
    whole prompt in a single user message there, so no `messages` are needed. A
    plain-string input is that same content already flattened.
    """
    if isinstance(sample.input, str):
        return sample.input
    return "\n".join(message.text for message in sample.input if message.role == "user")


def score_values(sample: EvalSample) -> dict[str, float | None]:
    """`harmful` and `classifier_verdict` as the sweep recorded them."""
    score = (sample.scores or {}).get(SCORER_NAME)
    value = score.value if score is not None else None
    if not isinstance(value, dict):
        return {"harmful": None, "classifier_verdict": None}
    return {
        name: float(value[name]) if name in value else None
        for name in ("harmful", "classifier_verdict")
    }


def transcripts_for_log(log_file: str) -> list[Transcript]:
    """Every scored sample of one eval log, as regradeable transcripts."""
    log = read_eval_log(log_file, exclude_fields=EXCLUDE_FIELDS)
    task_args = log.eval.task_args or {}
    model_args = log.eval.model_args or {}
    scenario = str(task_args.get("scenario", ""))
    strength = model_args.get(STRENGTH_ARG)
    name = Path(log.location).name

    out = []
    for sample in log.samples or []:
        scores = score_values(sample)
        out.append(
            Transcript(
                scenario=scenario,
                steer_strength=None if strength is None else float(strength),
                epoch=sample.epoch,
                log_file=name,
                sample_id=str(sample.id),
                email_content=user_content(sample),
                response=sample.output.completion or "",
                orig_harmful=scores["harmful"],
                orig_classifier_verdict=scores["classifier_verdict"],
            )
        )
    return out


def read_transcripts(log_dir: str, include_failed: bool = False) -> list[Transcript]:
    """Every transcript under `log_dir`, pooled across blocks."""
    logs = list_eval_logs(log_dir)
    if not logs:
        raise SystemExit(f"no eval logs found in {log_dir}")

    out: list[Transcript] = []
    skipped = 0
    for info in sorted(logs, key=lambda i: i.name):
        status = read_eval_log(info.name, header_only=True).status
        if status != "success" and not include_failed:
            print(f"{Path(info.name).name}: skipped ({status})", flush=True)
            skipped += 1
            continue
        rows = transcripts_for_log(info.name)
        print(f"{Path(info.name).name}: {len(rows)} samples", flush=True)
        out.extend(rows)
    print(
        f"\n{len(out)} transcripts in {len(logs) - skipped} logs ({skipped} skipped)",
        flush=True,
    )
    return out


def stratify(transcripts: list[Transcript], n_per_cell: int) -> list[Transcript]:
    """The first `n_per_cell` transcripts of each (scenario, strength), by epoch.

    Blocks pool, so a cell can hold two logs that both number their epochs from
    one. `log_file` breaks that tie, deterministically and without preferring
    either block's samples over the other's within an epoch.
    """
    cells: dict[tuple[str, float | None], list[Transcript]] = {}
    for transcript in transcripts:
        cells.setdefault(transcript.cell, []).append(transcript)

    out: list[Transcript] = []
    for cell in sorted(cells, key=lambda c: (c[0], _strength_key(c[1]))):
        members = sorted(cells[cell], key=lambda t: (t.epoch, t.log_file, t.sample_id))
        chosen = members[:n_per_cell]
        if len(chosen) < n_per_cell:
            print(
                f"  {cell[0]:<9} strength={_fmt(cell[1])}: only {len(chosen)} of "
                f"{n_per_cell} available",
                flush=True,
            )
        out.extend(chosen)
    return out


def _strength_key(strength: float | None) -> float:
    return float("-inf") if strength is None else strength


def _fmt(strength: float | None) -> str:
    return "?" if strength is None else f"{strength:+.2f}"


def load_manifest(path: Path) -> list[dict[str, Any]] | None:
    """The recorded subsample, or None if this is the first run."""
    if not path.exists():
        return None
    return json.loads(path.read_text())["samples"]


def save_manifest(path: Path, chosen: list[Transcript], n_per_cell: int) -> None:
    """Record the subsample so later runs score exactly these transcripts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "n_per_cell": n_per_cell,
                "samples": [
                    {
                        "scenario": t.scenario,
                        "steer_strength": t.steer_strength,
                        "epoch": t.epoch,
                        "log_file": t.log_file,
                        "sample_id": t.sample_id,
                    }
                    for t in chosen
                ],
            },
            indent=2,
        )
    )


def select(
    transcripts: list[Transcript], manifest_path: Path, n_per_cell: int
) -> list[Transcript]:
    """The subsample to grade: the manifest's if it exists, a fresh one if not."""
    recorded = load_manifest(manifest_path)
    if recorded is None:
        chosen = stratify(transcripts, n_per_cell)
        save_manifest(manifest_path, chosen, n_per_cell)
        print(f"stratified {len(chosen)} transcripts -> {manifest_path}", flush=True)
        return chosen

    by_key = {t.key: t for t in transcripts}
    chosen, missing = [], 0
    for entry in recorded:
        key = (entry["log_file"], str(entry["sample_id"]), int(entry["epoch"]))
        found = by_key.get(key)
        if found is None:
            missing += 1
            continue
        chosen.append(found)
    print(
        f"reusing {manifest_path}: {len(chosen)} transcripts"
        + (f" ({missing} no longer in the log dir)" if missing else ""),
        flush=True,
    )
    return chosen


async def regrade(
    transcript: Transcript, grader: Model, grader_model: str, run_tag: str
) -> dict[str, Any]:
    """One transcript through one grader, as a CSV row.

    The three calls below are `scorers.score_from_classifier`'s, in its order.
    Anything the grader call or the answer parse raises is caught and recorded:
    an unparseable answer is a real property of a grader on a transcript, and
    dropping those rows would silently restrict the comparison to the transcripts
    both graders happened to find easy.
    """
    row: dict[str, Any] = {
        "scenario": transcript.scenario,
        "steer_strength": transcript.steer_strength,
        "epoch": transcript.epoch,
        "log_file": transcript.log_file,
        "sample_id": transcript.sample_id,
        "grader_model": grader_model,
        "run_tag": run_tag,
        "orig_harmful": transcript.orig_harmful,
        "orig_classifier_verdict": transcript.orig_classifier_verdict,
        "new_harmful": None,
        "new_classifier_verdict": None,
        "grader_error": None,
        "grader_reasoning": None,
    }

    classifier = classifier_for(transcript.scenario)
    try:
        grader_input = classifier.get_grader_input(
            email_content=transcript.email_content, response=transcript.response
        )
        grader_response = await grader.generate(grader_input)
        harmful, verdict, reasoning = classifier.classify(
            response=transcript.response, grader_response=grader_response.completion
        )
    except Exception as error:  # noqa: BLE001 - recorded, not swallowed
        row["grader_error"] = f"{type(error).__name__}: {error}"
        return row

    row["new_harmful"] = 1.0 if harmful else 0.0
    row["new_classifier_verdict"] = 1.0 if verdict else 0.0
    row["grader_reasoning"] = reasoning
    return row


class AppendingWriter:
    """CSV appender that writes a header only into a file that has none.

    Rows are flushed as they finish rather than collected, so a run killed
    partway leaves the transcripts it did grade on disk, and two runs pooling
    into one file interleave harmlessly instead of the second truncating the
    first.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists() or path.stat().st_size == 0
        self._handle = path.open("a", newline="")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=CSV_COLUMNS, extrasaction="ignore"
        )
        if fresh:
            self._writer.writeheader()
        self._lock = asyncio.Lock()

    async def write(self, row: dict[str, Any]) -> None:
        async with self._lock:
            self._writer.writerow(
                {c: "" if row.get(c) is None else row[c] for c in CSV_COLUMNS}
            )
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


async def run(
    chosen: list[Transcript], grader_model: str, run_tag: str, out_path: Path
) -> list[dict[str, Any]]:
    """Grade every chosen transcript, at most MAX_CONCURRENT calls in flight."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    # One Model for the whole run rather than one per call: inspect scopes its
    # own connection pool to the model instance, so a fresh instance per
    # transcript would give each call a private pool of MAX_CONCURRENT and leave
    # the semaphore below as the only real limit.
    grader = get_model(
        grader_model, config=GenerateConfig(max_connections=MAX_CONCURRENT)
    )
    writer = AppendingWriter(out_path)
    done = 0

    async def one(transcript: Transcript) -> dict[str, Any]:
        nonlocal done
        async with semaphore:
            row = await regrade(transcript, grader, grader_model, run_tag)
        await writer.write(row)
        done += 1
        if done % 25 == 0 or done == len(chosen):
            print(f"  graded {done}/{len(chosen)}", flush=True)
        return row

    try:
        rows = await asyncio.gather(*(one(t) for t in chosen))
    finally:
        writer.close()
    return list(rows)


def summarise(rows: list[dict[str, Any]]) -> None:
    """Agreement with the original verdict, overall and per steering strength."""
    graded = [r for r in rows if r["grader_error"] is None]
    failed = len(rows) - len(graded)
    print(f"\n{len(graded)} graded, {failed} failed", flush=True)
    if not graded:
        return

    def agreement(subset: list[dict[str, Any]], field: str) -> tuple[int, int]:
        pairs = [
            r
            for r in subset
            if r[f"orig_{field}"] is not None and r[f"new_{field}"] is not None
        ]
        return sum(r[f"orig_{field}"] == r[f"new_{field}"] for r in pairs), len(pairs)

    for field in ("harmful", "classifier_verdict"):
        same, n = agreement(graded, field)
        if not n:
            print(f"{field:<18} no comparable pairs", flush=True)
            continue
        print(
            f"{field:<18} overall agreement {same}/{n} = {same / n:.3f}",
            flush=True,
        )

    print(
        f"\n{'strength':>9} {'n':>4} {'agree_h':>8} {'agree_cv':>9} "
        f"{'new_h':>7} {'new_cv':>7} {'orig_h':>7} {'orig_cv':>8}",
        flush=True,
    )
    strengths = sorted({r["steer_strength"] for r in graded}, key=_strength_key)
    for strength in strengths:
        subset = [r for r in graded if r["steer_strength"] == strength]
        agree_h, n_h = agreement(subset, "harmful")
        agree_v, n_v = agreement(subset, "classifier_verdict")
        print(
            f"{_fmt(strength):>9} {len(subset):>4} "
            f"{agree_h / n_h if n_h else float('nan'):>8.3f} "
            f"{agree_v / n_v if n_v else float('nan'):>9.3f} "
            f"{_mean(subset, 'new_harmful'):>7.3f} "
            f"{_mean(subset, 'new_classifier_verdict'):>7.3f} "
            f"{_mean(subset, 'orig_harmful'):>7.3f} "
            f"{_mean(subset, 'orig_classifier_verdict'):>8.3f}",
            flush=True,
        )


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [r[field] for r in rows if r[field] is not None]
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", help="directory of .eval logs to re-grade")
    parser.add_argument("--grader", default=GRADER, help="grader model id")
    parser.add_argument("--run-tag", default=RUN_TAG, help="label for this run")
    parser.add_argument(
        "--out",
        default="logs/regrade/regrade_agentic_misalignment.csv",
        help="CSV to append rows to",
    )
    parser.add_argument(
        "--manifest",
        default="logs/regrade/subsample_manifest.json",
        help="the stratified subsample; written on first run, reused after",
    )
    parser.add_argument(
        "--n-per-cell", type=int, default=N_PER_CELL, help="samples per cell"
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="also read logs whose status is not success (default: skip them)",
    )
    args = parser.parse_args()

    transcripts = read_transcripts(args.log_dir, args.include_failed)
    chosen = select(transcripts, Path(args.manifest), args.n_per_cell)
    if not chosen:
        raise SystemExit("no transcripts selected")

    print(
        f"\ngrading {len(chosen)} transcripts with {args.grader} "
        f"(run_tag={args.run_tag}) -> {args.out}",
        flush=True,
    )
    rows = asyncio.run(run(chosen, args.grader, args.run_tag, Path(args.out)))
    summarise(rows)


if __name__ == "__main__":
    main()
