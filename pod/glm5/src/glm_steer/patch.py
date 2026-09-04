"""Global-strength activation steering for GLM-5.3-Flash under vLLM.

One steering strength per server process, fixed at launch from the environment.
There are no per-request buffers, no per-token row/alpha tables and no
`_prepare_inputs` patch: the entire mechanism is a constant vector added to the
four hyper-connection streams entering one decoder layer, on every token,
prefill included.

Why a class substitution and not a forward hook: a hook is a host-side callback,
and whether `torch.compile` absorbs it into the graph is an implementation
detail of the compiler.  If a version stops absorbing it, the hook still runs
during CUDA-graph *capture* and silently stops running on *replay* -- a server
that looks steered and is not.  Overriding `forward` on a substituted class puts
the arithmetic where the compiler must trace it to compile the model at all.

Where the delta goes.  A GLM-5.3-Flash decoder layer in this wheel takes
`(positions, hidden_states, residual, post, comb)` and `residual` is the
`[num_tokens, 4, hidden]` hyper-connection stream tensor.  `post`/`comb` are
deferred: the *materialised* streams entering layer L are
`mhc_post(hidden_states, residual, post, comb)`, i.e.
`out_j = post_j * x + sum_i comb_ij * r_i`.  Because the Sinkhorn loop's last
operation is the column normalisation, `sum_i comb_ij == 1` (measured to
1.1e-06), so adding the same delta to every row of `residual` adds exactly that
delta to every materialised stream:

    sum_i comb_ij (r_i + d) = out_j + d * sum_i comb_ij = out_j + d

which is the perturbation the vector was captured against, and the one the
numerical probe validated to 0.45-0.66% (bf16 rounding).

Environment:
    GLM_STEER_ENABLE=1          gate; anything else and this module does nothing
    GLM_STEER_VECTOR_DIR=DIR    directory holding vector.npy and meta.json
    GLM_STEER_STRENGTH=FLOAT    dimensionless; 0.2 perturbs the stream mean by
                                20% of the measured activation norm at the
                                injection layer.  Required even when zero, so a
                                control run is a deliberate act rather than a
                                forgotten variable.

Strength 0 still executes the addition.  Skipping it would make the strength-0
arm a different code path from the steered arms, which is exactly the thing a
control is supposed to rule out.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

_TAG = "glm_steer"


def _log(message: str) -> None:
    print(f"[{_TAG}][pid {os.getpid()}] {message}", flush=True)


class _State:
    meta: dict[str, Any] | None = None
    vector: np.ndarray | None = None
    strength: float | None = None
    scale: float | None = None
    layer: int | None = None
    delta: torch.Tensor | None = None
    hc_mult: int | None = None
    hidden: int | None = None
    installed: bool = False


_st = _State()


# --------------------------------------------------------------------------
# vector loading
# --------------------------------------------------------------------------


def _load_vector(vector_dir: str) -> tuple[dict[str, Any], np.ndarray]:
    root = Path(vector_dir)
    meta_path, vec_path = root / "meta.json", root / "vector.npy"
    for path in (meta_path, vec_path):
        if not path.is_file():
            raise RuntimeError(f"{_TAG}: {path} does not exist")

    meta = json.loads(meta_path.read_text())
    raw = vec_path.read_bytes()

    # The vector is the experiment.  If the bytes on the pod are not the bytes
    # the metadata describes, every number downstream is about something else.
    expected = meta.get("vector_npy_sha256")
    if expected:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"{_TAG}: vector.npy sha256 {actual} != meta.json's {expected}"
            )

    vector = np.load(vec_path)
    if vector.dtype != np.float32 or vector.ndim != 1:
        raise RuntimeError(
            f"{_TAG}: expected 1-D float32 vector, got {vector.ndim}-D {vector.dtype}"
        )
    if int(vector.shape[0]) != int(meta["hidden_size"]):
        raise RuntimeError(
            f"{_TAG}: vector length {vector.shape[0]} != meta hidden_size "
            f"{meta['hidden_size']}"
        )

    norm = float(np.linalg.norm(vector.astype(np.float64)))
    claimed = float(meta["vector_norm"])
    if abs(norm - claimed) > 1e-6 * max(1.0, claimed):
        raise RuntimeError(
            f"{_TAG}: recomputed vector norm {norm!r} != meta vector_norm {claimed!r}"
        )

    if meta.get("steer_site") != "stream_mean_input":
        raise RuntimeError(
            f"{_TAG}: this patch injects at a block input; vector declares "
            f"steer_site={meta.get('steer_site')!r}"
        )
    return meta, vector


# --------------------------------------------------------------------------
# start-up guards
# --------------------------------------------------------------------------


def _refuse_unsafe_parallelism(vllm_config: Any) -> None:
    """Refuse configurations under which the injection would be wrong or absent.

    Each of these produces a *silently* wrong run rather than a crash, which is
    the failure mode worth spending start-up time on.
    """
    parallel = vllm_config.parallel_config

    if getattr(vllm_config, "speculative_config", None) is not None:
        raise RuntimeError(
            f"{_TAG}: speculative decoding / MTP is on.  The draft layer is a "
            "separate module outside the 45-layer stack, so drafted tokens "
            "would bypass the steering while verified ones did not."
        )
    if getattr(parallel, "use_sequence_parallel_moe", False):
        raise RuntimeError(f"{_TAG}: sequence parallelism (MoE) is on")
    compilation = getattr(vllm_config, "compilation_config", None)
    pass_config = getattr(compilation, "pass_config", None)
    if getattr(pass_config, "enable_sp", False):
        raise RuntimeError(f"{_TAG}: sequence-parallel compile pass is on")
    if getattr(parallel, "use_ubatching", False):
        raise RuntimeError(f"{_TAG}: microbatching (DBO) is on")
    if int(getattr(parallel, "pipeline_parallel_size", 1)) != 1:
        raise RuntimeError(
            f"{_TAG}: pipeline parallelism is on, so a global layer index is "
            "ambiguous in this process"
        )


def _check_model_matches(model_config: Any, meta: dict[str, Any]) -> Any:
    text_config = model_config.hf_text_config

    n_layers = int(text_config.num_hidden_layers)
    hidden = int(text_config.hidden_size)
    if n_layers != int(meta["n_layers"]):
        raise RuntimeError(
            f"{_TAG}: loaded model has {n_layers} layers, vector was built "
            f"against {meta['n_layers']}"
        )
    if hidden != int(meta["hidden_size"]):
        raise RuntimeError(
            f"{_TAG}: loaded model hidden_size {hidden}, vector expects "
            f"{meta['hidden_size']}"
        )

    hc_mult = getattr(text_config, "mhc_num_residual_streams", None)
    if hc_mult is not None and int(hc_mult) != int(meta["hc_mult"]):
        raise RuntimeError(
            f"{_TAG}: loaded model has {hc_mult} hyper-connection streams, "
            f"vector expects {meta['hc_mult']}"
        )

    # meta['model'] is a hub id; the pod serves a local directory.  Compare the
    # last path component, which is what the download preserved.
    want = str(meta["model"]).rstrip("/").split("/")[-1].lower()
    got = os.path.basename(str(model_config.model).rstrip("/")).lower()
    if want != got:
        raise RuntimeError(
            f"{_TAG}: serving {got!r} but the vector was measured in {want!r}"
        )
    return text_config


def _decoder_stack(model: torch.nn.Module) -> tuple[str, Any]:
    """The one module that owns both an embedding table and a layer list.

    Deliberately *not* an enumeration of `nn.ModuleList`s: this model keeps
    `_active_layers`, a second list holding the same 45 layer objects, and a
    vision tower with a `blocks` list.  The embed_tokens co-occurrence picks out
    `language_model.model` and nothing else.
    """
    matches = [
        (name or "<root>", module)
        for name, module in model.named_modules()
        if isinstance(getattr(module, "layers", None), torch.nn.ModuleList)
        and hasattr(module, "embed_tokens")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{_TAG}: expected exactly one decoder stack, found "
            f"{[name for name, _ in matches]}"
        )
    return matches[0]


# --------------------------------------------------------------------------
# the substituted layer
# --------------------------------------------------------------------------

_STEERED: dict[type, type] = {}


def _steered_class(base: type) -> type:
    cached = _STEERED.get(base)
    if cached is not None:
        return cached

    class SteeredGlmDecoderLayer(base):  # type: ignore[misc, valid-type]
        _glm_steer_marker = True

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            delta = _st.delta
            if delta is None:
                raise RuntimeError(f"{_TAG}: layer ran before the delta existed")

            if "residual" in kwargs:
                residual = kwargs["residual"]
                where: Any = "residual"
            elif len(args) >= 3:
                residual = args[2]
                where = 2
            else:
                raise RuntimeError(
                    f"{_TAG}: cannot locate the residual streams; "
                    f"{len(args)} positional args, keywords {sorted(kwargs)}"
                )

            # None here would mean the target is layer 0 (which calls hc_expand
            # instead of receiving streams) or that the signature moved.  Either
            # way the steering would be silently absent, so refuse.
            if residual is None:
                raise RuntimeError(
                    f"{_TAG}: residual is None at layer {_st.layer}; this patch "
                    "must not target layer 0 and the layer signature must carry "
                    "the streams"
                )
            if (
                residual.ndim != 3
                or int(residual.shape[-2]) != _st.hc_mult
                or int(residual.shape[-1]) != _st.hidden
            ):
                raise RuntimeError(
                    f"{_TAG}: expected residual [tokens, {_st.hc_mult}, "
                    f"{_st.hidden}], got {tuple(residual.shape)}"
                )

            # Out of place: the caller's tensor may be a CUDA-graph buffer that
            # other layers still read.  Broadcasting [hidden] over [T, 4, hidden]
            # adds the same delta to all four streams at every token position.
            steered = residual + delta
            if where == "residual":
                kwargs = dict(kwargs)
                kwargs["residual"] = steered
            else:
                mutable = list(args)
                mutable[2] = steered
                args = tuple(mutable)
            return super().forward(*args, **kwargs)

    SteeredGlmDecoderLayer.__name__ = f"Steered{base.__name__}"
    SteeredGlmDecoderLayer.__qualname__ = SteeredGlmDecoderLayer.__name__
    _STEERED[base] = SteeredGlmDecoderLayer
    return SteeredGlmDecoderLayer


# --------------------------------------------------------------------------
# installation
# --------------------------------------------------------------------------


def _install(runner: Any) -> None:
    meta, vector = _st.meta, _st.vector
    assert meta is not None and vector is not None

    _refuse_unsafe_parallelism(runner.vllm_config)
    text_config = _check_model_matches(runner.model_config, meta)

    layer_index = int(meta["layer"])
    n_layers = int(text_config.num_hidden_layers)
    if not 0 < layer_index < n_layers:
        raise RuntimeError(
            f"{_TAG}: layer {layer_index} out of range for {n_layers} layers "
            "(layer 0 receives no streams and cannot be a target)"
        )

    name, stack = _decoder_stack(runner.model)
    layers = stack.layers
    if len(layers) != n_layers:
        raise RuntimeError(
            f"{_TAG}: stack {name!r} has {len(layers)} layers, config says {n_layers}"
        )
    start = int(getattr(stack, "start_layer", 0))
    end = int(getattr(stack, "end_layer", n_layers))
    if (start, end) != (0, n_layers):
        raise RuntimeError(
            f"{_TAG}: this process holds pipeline shard {start}:{end}"
        )

    target = layers[layer_index]
    if getattr(type(target), "_glm_steer_marker", False):
        raise RuntimeError(
            f"{_TAG}: layer {layer_index} is already steered; a second "
            "substitution would inject the delta twice"
        )

    _st.hc_mult = int(meta["hc_mult"])
    _st.hidden = int(meta["hidden_size"])
    _st.layer = layer_index

    # Allocate before substituting, and before any dummy run or graph capture,
    # so the tensor the graph records is the tensor every replay reads.
    delta_f64 = float(_st.strength) * float(_st.scale) * vector.astype(np.float64)
    dtype = runner.model_config.dtype
    _st.delta = torch.from_numpy(delta_f64.astype(np.float32)).to(
        device=runner.device, dtype=dtype
    )

    target.__class__ = _steered_class(type(target))

    # `_active_layers` is a second ModuleList over the *same* objects.  Rebinding
    # __class__ on the object reaches both lists; what must be checked is that
    # exactly one distinct object ended up steered.
    steered_ids = {
        id(m) for m in layers if getattr(type(m), "_glm_steer_marker", False)
    }
    active = getattr(stack, "_active_layers", None)
    if active is not None:
        steered_ids |= {
            id(m) for m in active if getattr(type(m), "_glm_steer_marker", False)
        }
        if active[layer_index] is not target:
            raise RuntimeError(
                f"{_TAG}: _active_layers[{layer_index}] is a different object "
                f"from layers[{layer_index}]; the alias assumption is wrong"
            )
    if len(steered_ids) != 1:
        raise RuntimeError(
            f"{_TAG}: {len(steered_ids)} distinct layer objects are steered, want 1"
        )

    if getattr(target, "is_sequence_parallel", False):
        raise RuntimeError(f"{_TAG}: target layer reports is_sequence_parallel")

    added_norm = float(torch.linalg.vector_norm(_st.delta.float()).item())
    exact_norm = float(np.linalg.norm(delta_f64))
    activation = float(meta["activation_norm_at_layer"])
    status_path = os.environ.get("GLM_STEER_STATUS_FILE")
    if status_path:
        # Append, one line per TP worker.  The launcher counts these; a server
        # whose workers did not all install is a server that is partly steered,
        # which is worse than one that is not steered at all.
        with open(status_path, "a") as handle:
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "layer": layer_index,
                        "strength": _st.strength,
                        "scale": _st.scale,
                        "delta_norm": added_norm,
                        "activation_norm": activation,
                        "ratio": added_norm / activation,
                        "class": type(target).__name__,
                        "stack": name,
                        "device": str(runner.device),
                    }
                )
                + "\n"
            )

    _log(
        f"installed: stack={name!r} layer={layer_index} class={type(target).__name__} "
        f"strength={_st.strength:+.6g} scale={_st.scale:.9g} "
        f"|v|={float(np.linalg.norm(vector.astype(np.float64))):.9g} "
        f"|delta|_fp64={exact_norm:.9g} |delta|_as_{str(dtype).split('.')[-1]}"
        f"={added_norm:.9g} activation_norm={activation:.9g} "
        f"|delta|/activation={added_norm / activation:.9g} "
        f"active_layers_aliased={active is not None} device={runner.device}"
    )
    _st.installed = True


# This wheel ships TWO distinct GPUModelRunner classes.  The legacy
# `vllm.v1.worker.gpu_model_runner.GPUModelRunner` is the one the pod server was
# written against; the one actually instantiated at runtime is
# `vllm.v1.worker.gpu.model_runner.GPUModelRunner` (the log line
# "Model loading took ... seconds" comes out of gpu/model_runner.py:409).  They
# are different objects with different `load_model` functions, so patching only
# the first produces a server that starts, serves, and is not steered.  Patch
# every one that exists and let the marker check in `_install` catch the case
# where more than one is somehow used.
_RUNNER_MODULES = (
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu_model_runner",
)


def _patch_runner() -> None:
    import importlib

    patched, seen = [], set()
    for module_name in _RUNNER_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - a missing variant is fine
            _log(
                f"runner module {module_name} not importable "
                f"({type(exc).__name__}); skipping"
            )
            continue
        runner = getattr(module, "GPUModelRunner", None)
        if runner is None or id(runner) in seen:
            continue
        seen.add(id(runner))

        original = runner.load_model
        if getattr(original, "_glm_steer_patched", False):
            _log(f"{module_name}.GPUModelRunner.load_model already patched")
            patched.append(module_name)
            continue

        def make(original_load: Any) -> Any:
            def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = original_load(self, *args, **kwargs)
                _install(self)
                return result

            load_model._glm_steer_patched = True  # type: ignore[attr-defined]
            return load_model

        runner.load_model = make(original)  # type: ignore[method-assign]
        patched.append(module_name)

    if not patched:
        raise RuntimeError(
            f"{_TAG}: found no GPUModelRunner to patch in {_RUNNER_MODULES}; "
            "refusing to start a server that would silently be unsteered"
        )
    _log(f"patched load_model on: {', '.join(patched)}")


def apply() -> None:
    """Read the environment, load the vector, arm the load_model patch."""
    if os.environ.get("GLM_STEER_ENABLE") != "1":
        return

    vector_dir = os.environ.get("GLM_STEER_VECTOR_DIR")
    if not vector_dir:
        raise RuntimeError(f"{_TAG}: GLM_STEER_VECTOR_DIR must be set")
    raw_strength = os.environ.get("GLM_STEER_STRENGTH")
    if raw_strength is None or raw_strength == "":
        raise RuntimeError(
            f"{_TAG}: GLM_STEER_STRENGTH must be set explicitly, even to 0"
        )
    strength = float(raw_strength)
    if not np.isfinite(strength):
        raise RuntimeError(f"{_TAG}: GLM_STEER_STRENGTH={raw_strength!r} is not finite")

    meta, vector = _load_vector(vector_dir)
    norm = float(np.linalg.norm(vector.astype(np.float64)))

    # scale turns a dimensionless strength into stream units: strength 1.0 adds
    # a delta whose norm equals the mean activation norm measured at this layer,
    # so 0.2 is 20% of a typical activation there.  Absolute norms are not
    # comparable across layers in this model (they grow ~2700x with depth), so
    # the ratio is the only portable handle.
    _st.meta = meta
    _st.vector = vector
    _st.strength = strength
    _st.scale = float(meta["activation_norm_at_layer"]) / norm

    _patch_runner()
    _log(
        f"armed: vector_dir={vector_dir} id={meta.get('id_str')} "
        f"name={meta.get('name')} layer={meta.get('layer')} "
        f"strength={strength:+.6g} scale={_st.scale:.9g}"
    )
