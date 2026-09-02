"""Inert-unless-requested vLLM general plugin for capture-only replacement."""

from __future__ import annotations

import os
from typing import Any

from ..core.modelprofile import PROFILE
from . import config

CAPTURE_CLASS_NAME = "CaptureArchitecture"
_REGISTERED = False


def register() -> None:
    """Register the replacement architecture in every vLLM process."""
    global _REGISTERED
    cfg = config.read_config(os.environ)
    if cfg is None or _REGISTERED:
        return
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        PROFILE.architecture, f"{__name__}:{CAPTURE_CLASS_NAME}"
    )
    _REGISTERED = True


def __getattr__(name: str) -> Any:
    """Lazily import vLLM/torch-dependent replacement code."""
    if name != CAPTURE_CLASS_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    cfg = config.read_config(os.environ)
    if cfg is None:
        raise RuntimeError("capture architecture requested without capture environment")
    from .patch import build_capture_architecture_class

    return build_capture_architecture_class(cfg)
