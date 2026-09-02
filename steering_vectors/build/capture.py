"""Capture-side subprocess orchestration and strict collection.

Ported from local ``v2-steering-tools`` capture builder (commits ea65b2f,
a486b53, d732a8c, 6508dab, 6678408).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..capture_engine import config
from ..core import canonjson, modelprofile, paths
from . import worker as workerlib
from .derive import CaptureReport, MeanAccumulator

PROFILE = modelprofile.PROFILE
LOG_TAIL_LINES = 40


class CaptureError(RuntimeError):
    """A capture cannot be trusted, so derivation must stop."""


def worker_command(
    spec_path: str | os.PathLike[str], *, executable: str | None = None
) -> list[str]:
    return [
        sys.executable if executable is None else executable,
        "-m",
        "steering_vectors.build.worker",
        str(spec_path),
    ]


def worker_environment(
    capture_dir: str | os.PathLike[str],
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    return config.capture_environment(capture_dir, base=base)


def check_worker_result(
    result: Mapping[str, Any], *, n_prompts: int
) -> dict[str, Any]:
    required = (
        "n_prompts",
        "n_checked",
        "records",
        "assertions",
        "capture_indices",
    )
    for key in required:
        if key not in result:
            raise CaptureError(f"worker result has no {key!r}")
    if int(result["n_prompts"]) != int(n_prompts):
        raise CaptureError("worker and parent describe different prompt counts")
    if n_prompts < 1 or int(result["n_checked"]) != int(n_prompts):
        raise CaptureError(
            f"{result['n_checked']} of {n_prompts} prompts were checked"
        )
    if len(result["records"]) != n_prompts:
        raise CaptureError("worker record count does not match prompt count")
    return dict(result)


def check_capture_startup(
    record: Mapping[str, Any] | None, capture_dir: str | os.PathLike[str]
) -> None:
    if record is None:
        raise CaptureError("capture plugin emitted no startup record")
    wanted = str(Path(capture_dir).resolve())
    expected = {
        "mode": config.MODE_CAPTURE,
        "steering": False,
        "capturing": True,
        "capture_dir": wanted,
        "num_layers": PROFILE.n_layers,
        "patched_blocks": PROFILE.n_layers,
        "enforce_eager": True,
        "blocks_path": PROFILE.decoder_blocks_path,
        "vllm_version": PROFILE.verified_vllm_version,
    }
    problems = [
        f"{key}={record.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if record.get(key) != value
    ]
    if problems:
        raise CaptureError("capturing engine mismatch: " + "; ".join(problems))


def engine_description(record: Mapping[str, Any]) -> str:
    return (
        f"vLLM {record.get('vllm_version')} offline LLM("
        f"dtype={PROFILE.dtype}, max_num_seqs=1, enable_prefix_caching=False, "
        "enforce_eager=True)"
    )


def collect(
    capture_dir: str | os.PathLike[str],
    result: Mapping[str, Any],
    *,
    accumulator: MeanAccumulator | None = None,
    delete: bool = True,
) -> MeanAccumulator:
    directory = Path(capture_dir)
    total = accumulator or MeanAccumulator()
    started = total.count
    pids: set[int] = set()
    for entry in result["records"]:
        index = int(entry["capture_index"])
        array, sidecar = config.load_capture(directory, index)
        if int(sidecar["num_tokens"]) != int(entry["n_prompt_tokens"]):
            raise CaptureError(f"capture {index} token count disagrees with worker")
        pids.add(int(sidecar["pid"]))
        if len(pids) > 1:
            raise CaptureError("capture files were written by more than one process")
        total.add(array)
        if delete:
            config.capture_array_path(directory, index).unlink(missing_ok=True)
            config.capture_sidecar_path(directory, index).unlink(missing_ok=True)
    if delete:
        left = config.capture_indices(directory)
        if left:
            raise CaptureError(f"unaccounted capture files remain: {left[:8]}")
    if total.count - started != len(result["records"]):
        raise CaptureError("not every worker record was accumulated")
    return total


def capture_side(
    prompts: Sequence[Any],
    work_dir: str | os.PathLike[str],
    tag: str,
    *,
    model: str = PROFILE.model_id,
    max_model_len: int = 4096,
    base_env: Mapping[str, str] | None = None,
    announce: Any = None,
) -> tuple[MeanAccumulator, CaptureReport]:
    if not prompts:
        raise CaptureError(f"side {tag!r} has no prompts")
    say = (lambda _: None) if announce is None else announce
    work = paths.ensure_dir(Path(work_dir))
    capture_dir = paths.ensure_dir(work / f"cap_{tag}")
    spec = workerlib.WorkerSpec(
        prompts=list(prompts),
        capture_dir=str(capture_dir),
        result_path=str(work / f"result_{tag}.json"),
        model=model,
        max_model_len=max_model_len,
    )
    spec_path = spec.write(work / f"spec_{tag}.json")
    log_path = work / f"log_{tag}.txt"
    say(f"[capture:{tag}] {len(prompts)} prompts -> {log_path}")
    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as handle:
        completed = subprocess.run(
            worker_command(spec_path),
            env=worker_environment(capture_dir, base=base_env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=paths.repo_root(),
            check=False,
        )
    seconds = time.monotonic() - started
    if completed.returncode != 0:
        tail = log_path.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]
        raise CaptureError(
            f"capture worker {tag!r} exited {completed.returncode}:\n"
            + "\n".join(tail)
        )
    result = check_worker_result(
        canonjson.read_json(spec.result_path), n_prompts=len(prompts)
    )
    lines = log_path.read_text(errors="replace").splitlines()
    record = config.find_startup_record(lines)
    check_capture_startup(record, capture_dir)
    assert record is not None
    accumulator = collect(capture_dir, result)
    report = CaptureReport(
        engine=engine_description(record),
        n_prompts=len(prompts),
        n_checked=int(result["n_checked"]),
        capture_files_from_engine_init=int(
            result["capture_files_from_engine_init"]
        ),
        seconds=seconds,
        max_model_len=int(result["max_model_len"]),
        prompt_token_lengths=list(result["prompt_token_lengths"]),
        templated_first_prompt=str(result["templated_first_prompt"]),
        assertions=str(result["assertions"]),
        vllm_version=str(record["vllm_version"]),
    )
    return accumulator, report
