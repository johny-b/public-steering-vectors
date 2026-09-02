"""Capture-only contract shared by the launcher, worker, and vLLM plugin.

Ported from local ``v2-steering-tools`` capture/config code (notably commits
ea65b2f, a486b53, d732a8c, 6508dab, and 6678408). This module is stdlib-only;
NumPy is imported only by functions that read or write capture arrays.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core import canonjson, modelprofile

PROFILE = modelprofile.PROFILE

ENV_MODE = "STEERING_VECTORS_CAPTURE_MODE"
ENV_CAPTURE_DIR = "STEERING_VECTORS_CAPTURE_DIR"
MODE_CAPTURE = "capture"
ENV_NAMES = (ENV_MODE, ENV_CAPTURE_DIR)
# These were part of the environment used by the source builder. ``spawn`` is
# especially important: vLLM worker processes must import the general plugin
# themselves before model construction.
CAPTURE_ENGINE_ENV = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_LOGGING_LEVEL": "INFO",
    "VLLM_DISABLE_COMPILE_CACHE": "1",
}
# Clear legacy v2 steering variables from the worker environment. They are not
# understood or implemented here, but inheriting one could activate a separately
# installed runtime-steering plugin while this package captures.
LEGACY_STEER_ENV_NAMES = (
    "STEER_MODE",
    "STEER_VECTOR",
    "STEER_LAYER",
    "STEER_STRENGTH",
    "STEER_CAPTURE_DIR",
    "STEER_CONTROL_PORT",
    "STEER_CONTROL_HOST",
    "STEER_METRICS_URL",
    "STEER_QUIESCE_MS",
    "STEER_PROBE",
    "STEER_PROBE_LAYER",
)
LOG_TAG = "STEERING_CAPTURE_CONFIG"
LOG_LOGGER_NAME = "vllm.steering_vectors_capture"

CAPTURE_STEM_DIGITS = 6
CAPTURE_ARRAY_SUFFIX = ".npy"
CAPTURE_SIDECAR_SUFFIX = ".json"
SIDECAR_FIELDS = (
    "index",
    "array",
    "shape",
    "num_tokens",
    "num_positions",
    "pid",
)


class CaptureConfigError(ValueError):
    """A capture configuration or published capture cannot be trusted."""


@dataclass(frozen=True)
class CaptureConfig:
    """Validated configuration for one capture-only engine process."""

    capture_dir: str
    mode: str = MODE_CAPTURE

    def __post_init__(self) -> None:
        if self.mode != MODE_CAPTURE:
            raise CaptureConfigError(f"mode must be {MODE_CAPTURE!r}, got {self.mode!r}")
        if not self.capture_dir:
            raise CaptureConfigError(f"{ENV_CAPTURE_DIR} must be a non-empty path")

    @property
    def capturing(self) -> bool:
        return True

    def environment(self) -> dict[str, str]:
        return {
            ENV_MODE: MODE_CAPTURE,
            ENV_CAPTURE_DIR: str(Path(self.capture_dir).resolve()),
        }


def read_config(env: Mapping[str, str] | None = None) -> CaptureConfig | None:
    """Parse capture configuration; absent mode is deliberately inert."""
    values = os.environ if env is None else env
    mode = (values.get(ENV_MODE) or "").strip().lower()
    if not mode or mode == "off":
        return None
    if mode != MODE_CAPTURE:
        raise CaptureConfigError(
            f"{ENV_MODE}={mode!r}; this plugin is capture-only and accepts "
            f"only {MODE_CAPTURE!r}"
        )
    directory = (values.get(ENV_CAPTURE_DIR) or "").strip()
    if not directory:
        raise CaptureConfigError(f"{ENV_MODE}=capture requires {ENV_CAPTURE_DIR}")
    return CaptureConfig(capture_dir=directory)


def capture_environment(
    directory: str | os.PathLike[str],
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """A clean child environment containing only this capture request."""
    env = dict(os.environ if base is None else base)
    for name in (*ENV_NAMES, *LEGACY_STEER_ENV_NAMES):
        env.pop(name, None)
    env.update(CAPTURE_ENGINE_ENV)
    env.update(CaptureConfig(str(directory)).environment())
    return env


def residual_stream(hidden_states: Any, residual: Any) -> Any:
    """True residual at block input for Qwen3.6-27B's fused-add convention."""
    return hidden_states if residual is None else hidden_states + residual


class CaptureAccumulator:
    """Collect one row per block, in order, for one complete forward."""

    def __init__(self, *, num_layers: int = PROFILE.n_layers) -> None:
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise CaptureConfigError("a capture must span at least one block")
        self.completed = 0
        self.discarded = 0
        self._rows: list[Any] = []
        self._num_tokens = 0
        self._num_positions = -1

    def add(
        self, index: int, row: Any, *, num_tokens: int, num_positions: int = -1
    ) -> tuple[list[Any], int, int] | None:
        index = int(index)
        if index != len(self._rows):
            if index == 0:
                self.discarded += 1
                self._rows = []
            else:
                raise CaptureConfigError(
                    f"block {index} recorded while {len(self._rows)} rows are held"
                )
        self._rows.append(row)
        self._num_tokens = int(num_tokens)
        self._num_positions = int(num_positions)
        if len(self._rows) < self.num_layers:
            return None
        rows, self._rows = self._rows, []
        self.completed += 1
        return rows, self._num_tokens, self._num_positions


def capture_stem(index: int) -> str:
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise CaptureConfigError(f"capture index must be non-negative, got {index!r}")
    return f"{index:0{CAPTURE_STEM_DIGITS}d}"


def capture_array_path(directory: str | os.PathLike[str], index: int) -> Path:
    return Path(directory) / f"{capture_stem(index)}{CAPTURE_ARRAY_SUFFIX}"


def capture_sidecar_path(directory: str | os.PathLike[str], index: int) -> Path:
    return Path(directory) / f"{capture_stem(index)}{CAPTURE_SIDECAR_SUFFIX}"


def validate_sidecar(
    record: Mapping[str, Any], *, where: str = "capture sidecar"
) -> dict[str, Any]:
    missing = [key for key in SIDECAR_FIELDS if key not in record]
    extra = [key for key in record if key not in SIDECAR_FIELDS]
    if missing or extra:
        raise CaptureConfigError(f"{where}: missing {missing}; unexpected {extra}")
    for key in ("index", "num_tokens", "num_positions", "pid"):
        if not isinstance(record[key], int) or isinstance(record[key], bool):
            raise CaptureConfigError(f"{where}: {key} must be an int")
    if list(record["shape"]) != list(PROFILE.deltas_shape):
        raise CaptureConfigError(
            f"{where}: shape {record['shape']} is not {list(PROFILE.deltas_shape)}"
        )
    expected = capture_stem(int(record["index"])) + CAPTURE_ARRAY_SUFFIX
    if record["array"] != expected:
        raise CaptureConfigError(f"{where}: array must be {expected!r}")
    if int(record["num_tokens"]) < 1:
        raise CaptureConfigError(f"{where}: num_tokens must be positive")
    return dict(record)


def write_capture(
    directory: str | os.PathLike[str],
    index: int,
    rows: Any,
    *,
    num_tokens: int,
    num_positions: int = -1,
    pid: int | None = None,
) -> tuple[Path, Path]:
    """Atomically publish float32 rows, then the validating sidecar."""
    import numpy as np

    array = np.asarray(rows, dtype=np.float32)
    try:
        PROFILE.check_deltas_shape(array.shape, what="capture")
    except ValueError as exc:
        raise CaptureConfigError(str(exc)) from exc
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    array_path = capture_array_path(out, index)
    sidecar_path = capture_sidecar_path(out, index)
    temporary = out / f".{array_path.name}.tmp"
    try:
        with open(temporary, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, array_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    record = {
        "index": int(index),
        "array": array_path.name,
        "shape": list(array.shape),
        "num_tokens": int(num_tokens),
        "num_positions": int(num_positions),
        "pid": os.getpid() if pid is None else int(pid),
    }
    validate_sidecar(record, where=str(sidecar_path))
    canonjson.write_json(sidecar_path, record, canonical=True)
    return array_path, sidecar_path


def read_sidecar(path: str | os.PathLike[str]) -> dict[str, Any]:
    return validate_sidecar(canonjson.read_json(path), where=str(path))


def capture_indices(directory: str | os.PathLike[str]) -> list[int]:
    out = Path(directory)
    if not out.is_dir():
        return []
    indices: list[int] = []
    for sidecar in sorted(out.glob(f"*{CAPTURE_SIDECAR_SUFFIX}")):
        stem = sidecar.stem
        if not stem.isdigit():
            continue
        index = int(stem)
        if not capture_array_path(out, index).is_file():
            raise CaptureConfigError(f"{sidecar} has no published array beside it")
        indices.append(index)
    return indices


def load_capture(directory: str | os.PathLike[str], index: int) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    record = read_sidecar(capture_sidecar_path(directory, index))
    array = np.load(capture_array_path(directory, index), allow_pickle=False)
    if tuple(array.shape) != tuple(record["shape"]):
        raise CaptureConfigError(f"capture {index}: array and sidecar shapes differ")
    return array, record


def purge_captures(directory: str | os.PathLike[str]) -> list[str]:
    out = Path(directory)
    if not out.is_dir():
        return []
    removed: list[str] = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.suffix in (
            CAPTURE_ARRAY_SUFFIX,
            CAPTURE_SIDECAR_SUFFIX,
        ):
            path.unlink()
            removed.append(path.name)
    return removed


def check_blocks_found(*, n_blocks: int, declared_layers: int, blocks_path: str) -> None:
    if n_blocks != declared_layers or n_blocks != PROFILE.n_layers:
        raise CaptureConfigError(
            f"found {n_blocks} blocks at {blocks_path!r}; checkpoint declares "
            f"{declared_layers} and profile requires {PROFILE.n_layers}"
        )


def check_hidden_size(hidden_size: int) -> None:
    if int(hidden_size) != PROFILE.hidden_size:
        raise CaptureConfigError(
            f"model hidden size {hidden_size} is not {PROFILE.hidden_size}"
        )


def check_pipeline_parallel(size: int) -> None:
    if int(size) != 1:
        raise CaptureConfigError("capture requires pipeline_parallel_size=1")


def check_capture_is_eager(enforce_eager: bool) -> None:
    if not enforce_eager:
        raise CaptureConfigError("capture requires enforce_eager=True")


def startup_line(**fields: Any) -> str:
    return f"{LOG_TAG} {json.dumps(fields, sort_keys=True, separators=(',', ':'))}"


def find_startup_record(lines: Iterable[str]) -> dict[str, Any] | None:
    for line in lines:
        marker = line.find(LOG_TAG)
        if marker < 0:
            continue
        payload = line[marker + len(LOG_TAG) :].strip()
        try:
            record = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            return record
    return None
