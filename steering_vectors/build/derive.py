"""Streaming difference-of-means and vector-directory writer.

Capture/derivation methodology is ported from local ``v2-steering-tools``
(ea65b2f, a486b53, d732a8c, 6508dab, 6678408): float32 captures, float64
streaming sums, positive mean minus negative mean, and float32 output arrays.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .. import vectorfmt
from ..core import canonjson, clock, digest, modelprofile, paths, provenance
from . import prompts as promptlib

PROFILE = modelprofile.PROFILE
VectorFormatError = vectorfmt.VectorFormatError


@dataclass
class CaptureReport:
    engine: str
    n_prompts: int
    n_checked: int
    capture_files_from_engine_init: int
    seconds: float
    max_model_len: int
    prompt_token_lengths: Sequence[int]
    templated_first_prompt: str
    assertions: str
    vllm_version: str

    def as_meta(self) -> dict[str, Any]:
        vectorfmt.check_every_one_of(
            self.n_checked, self.n_prompts, what="capture assertions"
        )
        if len(self.prompt_token_lengths) != self.n_prompts:
            raise VectorFormatError("capture token-length count does not match prompts")
        return {
            "n_prompts": self.n_prompts,
            "n_checked": self.n_checked,
            "capture_files_from_engine_init": self.capture_files_from_engine_init,
            "seconds": round(float(self.seconds), 1),
            "max_model_len": self.max_model_len,
            **promptlib.token_length_facts(self.prompt_token_lengths),
        }


class MeanAccumulator:
    """Float64 running sums for float32 per-prompt activations and norms."""

    def __init__(
        self,
        n_layers: int = PROFILE.n_layers,
        hidden_size: int = PROFILE.hidden_size,
    ) -> None:
        import numpy as np

        self.n_layers = int(n_layers)
        self.hidden_size = int(hidden_size)
        self._sum = np.zeros((self.n_layers, self.hidden_size), dtype=np.float64)
        self._norm_sum = np.zeros(self.n_layers, dtype=np.float64)
        self.count = 0

    def add(self, activation: Any) -> None:
        import numpy as np

        array = np.asarray(activation)
        expected = (self.n_layers, self.hidden_size)
        if tuple(array.shape) != expected:
            raise VectorFormatError(
                f"activation has shape {tuple(array.shape)}, expected {expected}"
            )
        wide = array.astype(np.float64)
        if not bool(np.all(np.isfinite(wide))):
            raise VectorFormatError("capture contains non-finite activations")
        self._sum += wide
        self._norm_sum += np.linalg.norm(wide, axis=1)
        self.count += 1

    def mean(self) -> Any:
        if self.count < 1:
            raise VectorFormatError("no activations were accumulated")
        return self._sum / float(self.count)

    def norm_sums(self) -> Any:
        if self.count < 1:
            raise VectorFormatError("no activations were accumulated")
        return self._norm_sum.copy()


@dataclass
class Derivation:
    layer: int
    deltas: Any
    vector: Any
    per_layer_delta_norm: list[float]
    per_layer_mean_activation_norm: list[float]
    n_pos: int
    n_neg: int

    @property
    def vector_norm(self) -> float:
        return self.per_layer_delta_norm[self.layer]

    @property
    def activation_norm_at_layer(self) -> float:
        return self.per_layer_mean_activation_norm[self.layer]

    @property
    def ratio(self) -> float:
        return self.vector_norm / self.activation_norm_at_layer


def derive(
    positive: MeanAccumulator, negative: MeanAccumulator, layer: int
) -> Derivation:
    import numpy as np

    PROFILE.check_layer(layer, what="derivation layer")
    if (positive.n_layers, positive.hidden_size) != (
        negative.n_layers,
        negative.hidden_size,
    ):
        raise VectorFormatError("positive and negative capture shapes differ")
    deltas = (positive.mean() - negative.mean()).astype(np.float32)
    vector = deltas[layer].copy()
    pooled = (positive.norm_sums() + negative.norm_sums()) / float(
        positive.count + negative.count
    )
    result = Derivation(
        layer=int(layer),
        deltas=deltas,
        vector=vector,
        per_layer_delta_norm=[
            float(np.linalg.norm(row.astype(np.float64))) for row in deltas
        ],
        per_layer_mean_activation_norm=[float(value) for value in pooled],
        n_pos=positive.count,
        n_neg=negative.count,
    )
    if result.vector_norm == 0.0:
        raise VectorFormatError(f"difference of means at layer {layer} is zero")
    if result.activation_norm_at_layer == 0.0:
        raise VectorFormatError(f"activation norm at layer {layer} is zero")
    return result


def save_npy_atomic(path: Path, array: Any) -> str:
    import numpy as np

    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.sha256_file(path)


def default_command() -> str:
    import sys

    return "build-steering-vector " + " ".join(
        shlex.quote(argument) for argument in sys.argv[1:]
    )


def next_free_id(vectors_dir: str | Path | None = None) -> int:
    root = Path(vectors_dir) if vectors_dir is not None else paths.vectors_dir()
    existing = (
        [
            int(child.name)
            for child in root.iterdir()
            if vectorfmt.ID_PATTERN.fullmatch(child.name)
        ]
        if root.is_dir()
        else []
    )
    return max(existing, default=0) + 1


def _meta(
    *,
    vector_id: int,
    name: str,
    description: str,
    model: str,
    result: Derivation,
    positive_prompts: Sequence[Any],
    negative_prompts: Sequence[Any],
    positive_capture: CaptureReport,
    negative_capture: CaptureReport,
    command: str,
    digests: dict[str, str],
) -> dict[str, Any]:
    if positive_capture.engine != negative_capture.engine:
        raise VectorFormatError("positive and negative captures used different engines")
    if positive_capture.assertions != negative_capture.assertions:
        raise VectorFormatError("positive and negative capture assertions differ")
    if (len(positive_prompts), len(negative_prompts)) != (
        result.n_pos,
        result.n_neg,
    ):
        raise VectorFormatError("prompt counts do not match accumulated activations")
    meta = {
        "id": int(vector_id),
        "id_str": f"{int(vector_id):04d}",
        "name": name,
        "description": description,
        "model": model,
        "layer": result.layer,
        "n_layers": PROFILE.n_layers,
        "hidden_size": PROFILE.hidden_size,
        "position_convention": vectorfmt.POSITION_CONVENTION,
        "sign_convention": vectorfmt.SIGN_CONVENTION,
        "n_pos": len(positive_prompts),
        "n_neg": len(negative_prompts),
        "prompt_format": {
            "positive": promptlib.format_summary(positive_prompts),
            "negative": promptlib.format_summary(negative_prompts),
        },
        "vector_norm": result.vector_norm,
        "activation_norm_at_layer": result.activation_norm_at_layer,
        "vector_norm_over_activation_norm": result.ratio,
        "per_layer_delta_norm": result.per_layer_delta_norm,
        "per_layer_mean_activation_norm": result.per_layer_mean_activation_norm,
        "created_at": clock.utc_now_iso(),
        "git_sha": provenance.repo_git_sha(),
        "vllm_version": positive_capture.vllm_version,
        "templated_first_positive_prompt": positive_capture.templated_first_prompt,
        "positive_jsonl_sha256": digests["positive"],
        "negative_jsonl_sha256": digests["negative"],
        "vector_npy_sha256": digests["vector"],
        "deltas_npy_sha256": digests["deltas"],
        "command": command,
        "capture": {
            "engine": positive_capture.engine,
            "positive": positive_capture.as_meta(),
            "negative": negative_capture.as_meta(),
            "assertions": positive_capture.assertions,
        },
    }
    return vectorfmt.validate_meta(meta, where="assembled meta.json")


def render_card(
    meta: dict[str, Any], positive: Sequence[Any], negative: Sequence[Any]
) -> str:
    """Render a compact human-readable companion to the existing format."""
    return "\n".join(
        [
            f"# vector {meta['id_str']} — `{meta['name']}`",
            "",
            meta["description"],
            "",
            f"- model: `{meta['model']}` (BF16)",
            f"- layer: {meta['layer']} (input of decoder block)",
            f"- prompts: {meta['n_pos']} positive / {meta['n_neg']} negative",
            f"- vector norm: {meta['vector_norm']:.6f}",
            f"- activation norm: {meta['activation_norm_at_layer']:.6f}",
            f"- relative magnitude: {meta['vector_norm_over_activation_norm']:.6f}",
            f"- engine: {meta['capture']['engine']}",
            f"- command: `{meta['command']}`",
            "",
            "## Positive prompts",
            "",
            promptlib.preview_block(positive),
            "",
            "## Negative prompts",
            "",
            promptlib.preview_block(negative),
            "",
            f"`{vectorfmt.DELTAS_NAME}` contains all layer deltas; "
            f"`{vectorfmt.VECTOR_NAME}` is row {meta['layer']}.",
            "",
        ]
    )


def write_vector_directory(
    *,
    vector_id: int,
    name: str,
    description: str,
    model: str,
    layer: int,
    positive_source: str | Path,
    negative_source: str | Path,
    positive: MeanAccumulator,
    negative: MeanAccumulator,
    positive_capture: CaptureReport,
    negative_capture: CaptureReport,
    command: str | None = None,
    vectors_dir: str | Path | None = None,
) -> Path:
    root = Path(vectors_dir) if vectors_dir is not None else paths.vectors_dir()
    vdir = root / vectorfmt.vector_id(vector_id)
    if vdir.exists():
        raise VectorFormatError(f"{vdir} already exists; vector ids are never overwritten")
    result = derive(positive, negative, layer)
    vdir.mkdir(parents=True, exist_ok=False)
    positive_prompts = promptlib.copy_prompt_file(
        positive_source, vdir / vectorfmt.POSITIVE_NAME
    )
    negative_prompts = promptlib.copy_prompt_file(
        negative_source, vdir / vectorfmt.NEGATIVE_NAME
    )
    digests = {
        "positive": digest.sha256_file(vdir / vectorfmt.POSITIVE_NAME),
        "negative": digest.sha256_file(vdir / vectorfmt.NEGATIVE_NAME),
        "deltas": save_npy_atomic(vdir / vectorfmt.DELTAS_NAME, result.deltas),
        "vector": save_npy_atomic(vdir / vectorfmt.VECTOR_NAME, result.vector),
    }
    meta = _meta(
        vector_id=vector_id,
        name=name,
        description=description,
        model=model,
        result=result,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        positive_capture=positive_capture,
        negative_capture=negative_capture,
        command=command or default_command(),
        digests=digests,
    )
    canonjson.write_json(vdir / vectorfmt.META_NAME, meta)
    canonjson.write_text_atomic(
        vdir / vectorfmt.README_NAME,
        render_card(meta, positive_prompts, negative_prompts),
    )
    read_back = vectorfmt.read_meta(vdir)
    vector = vectorfmt.load_vector(vdir, read_back)
    deltas = vectorfmt.load_deltas(vdir, read_back)
    vectorfmt.check_vector_is_delta_row(vector, deltas, read_back["layer"])
    vectorfmt.require_per_prompt_verification(read_back, what=f"vector {meta['id_str']}")
    return vdir
