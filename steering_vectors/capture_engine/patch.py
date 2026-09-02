"""Build and install the vLLM 0.26.0 capture architecture replacement.

This is the capture-only subset of the local ``v2-steering-tools`` plugin
(ea65b2f, a486b53, d732a8c, 6508dab, 6678408). It intentionally uses vLLM's
private model registry: that is the proven pre-weight-load replacement point.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..core.modelprofile import PROFILE
from . import config, hook
from .recorder import Recorder

_LOGGER: Any = None


def logger() -> Any:
    global _LOGGER
    if _LOGGER is None:
        from vllm.logger import init_logger

        _LOGGER = init_logger(config.LOG_LOGGER_NAME)
    return _LOGGER


def resolve_attribute_path(root: Any, path: str) -> Any:
    node = root
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def find_decoder_blocks(model: Any) -> tuple[str, Any]:
    """Locate only the profiled decoder stack; do not guess another tower."""
    path = PROFILE.decoder_blocks_path
    blocks = resolve_attribute_path(model, path)
    try:
        valid = blocks is not None and len(blocks) > 0 and hasattr(blocks[0], "forward")
    except (TypeError, AttributeError, IndexError, KeyError):
        valid = False
    if not valid:
        raise config.CaptureConfigError(
            f"no decoder blocks on {type(model).__name__} at verified path {path!r}"
        )
    return path, blocks


def _vllm_version() -> str:
    from vllm import __version__

    return str(__version__)


def base_architecture_class() -> type:
    """Resolve the original class from vLLM's private 0.26.0 registry."""
    version = _vllm_version()
    if version != PROFILE.verified_vllm_version:
        raise config.CaptureConfigError(
            f"capture plugin was verified with vLLM {PROFILE.verified_vllm_version}, "
            f"not {version}; private architecture APIs may differ"
        )
    from vllm.model_executor.models.registry import _VLLM_MODELS

    entry = _VLLM_MODELS.get(PROFILE.architecture)
    if entry is None:
        raise config.CaptureConfigError(
            f"vLLM has no registry entry for {PROFILE.architecture!r}"
        )
    module_name, class_name = entry
    expected_module = PROFILE.architecture_module.rsplit(".", 1)[-1]
    if module_name != expected_module:
        raise config.CaptureConfigError(
            f"{PROFILE.architecture!r} resolves to module {module_name!r}, "
            f"expected {expected_module!r}"
        )
    module = __import__(PROFILE.architecture_module, fromlist=[class_name])
    return getattr(module, class_name)


def build_capture_architecture_class(cfg: config.CaptureConfig) -> type:
    base = base_architecture_class()

    class CaptureArchitecture(base):  # type: ignore[misc, valid-type]
        _sv_capture_architecture = True

        # vLLM inspects these parameter names to select its modern init path.
        def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            install(self, cfg, vllm_config=vllm_config)

    CaptureArchitecture.__name__ = f"Capture{base.__name__}"
    CaptureArchitecture.__qualname__ = CaptureArchitecture.__name__
    return CaptureArchitecture


def _text_config(hf_config: Any) -> Any:
    inner = getattr(hf_config, "text_config", None)
    if inner is not None and getattr(inner, "num_hidden_layers", None) is not None:
        return inner
    return hf_config


def install(
    model: Any, cfg: config.CaptureConfig, *, vllm_config: Any = None
) -> dict[str, Any]:
    """Patch every decoder block and log exactly what was installed."""
    if vllm_config is None:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
    hf_config = _text_config(vllm_config.model_config.hf_config)
    enforce_eager = bool(getattr(vllm_config.model_config, "enforce_eager", False))
    config.check_pipeline_parallel(
        vllm_config.parallel_config.pipeline_parallel_size
    )
    config.check_capture_is_eager(enforce_eager)

    blocks_path, blocks = find_decoder_blocks(model)
    config.check_blocks_found(
        n_blocks=len(blocks),
        declared_layers=int(hf_config.num_hidden_layers),
        blocks_path=blocks_path,
    )
    config.check_hidden_size(int(hf_config.hidden_size))
    recorder = Recorder(cfg.capture_dir, num_layers=len(blocks))
    patched = []
    for index, block in enumerate(blocks):
        if hook.is_capture_class(hook.attach(block, index=index, recorder=recorder)):
            patched.append(index)
    if patched != list(range(len(blocks))):
        raise config.CaptureConfigError(
            f"capture patch installed on {patched}, expected every decoder block"
        )

    model._sv_capture_recorder = recorder
    info = {
        "mode": config.MODE_CAPTURE,
        "steering": False,
        "capturing": True,
        "capture_dir": str(Path(cfg.capture_dir).resolve()),
        "num_layers": len(blocks),
        "hidden_size": int(hf_config.hidden_size),
        "blocks_path": blocks_path,
        "patched_blocks": len(patched),
        "enforce_eager": enforce_eager,
        "vllm_version": _vllm_version(),
        "pid": os.getpid(),
    }
    logger().info("%s", config.startup_line(**info))
    return info
