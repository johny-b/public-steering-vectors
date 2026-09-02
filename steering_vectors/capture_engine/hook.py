"""Capture-only decoder-block class substitution.

Ported from ``v2-steering-tools`` commit lineage ea65b2f..6678408. The patched
forward records the true residual stream before the original block runs.
"""

from __future__ import annotations

from typing import Any

from . import config

_CAPTURE_CLASSES: dict[type, type] = {}


def capture_block_class(base: type) -> type:
    """Return a cached subclass whose forward records block input."""
    cached = _CAPTURE_CLASSES.get(base)
    if cached is not None:
        return cached

    class CaptureDecoderBlock(base):  # type: ignore[misc, valid-type]
        def forward(  # type: ignore[override]
            self,
            positions: Any = None,
            hidden_states: Any = None,
            residual: Any = None,
            **kwargs: Any,
        ) -> Any:
            if hidden_states is None:
                raise config.CaptureConfigError(
                    "captured decoder block was called without hidden_states"
                )
            stream = config.residual_stream(hidden_states, residual)
            self._sv_capture_recorder.record(
                self._sv_capture_layer,
                stream,
                num_positions=-1 if positions is None else int(positions.shape[0]),
            )
            return super().forward(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                **kwargs,
            )

    CaptureDecoderBlock.__name__ = f"Capture{base.__name__}"
    CaptureDecoderBlock.__qualname__ = CaptureDecoderBlock.__name__
    _CAPTURE_CLASSES[base] = CaptureDecoderBlock
    return CaptureDecoderBlock


def attach(block: Any, *, index: int, recorder: Any) -> type:
    """Attach recorder state and substitute only the instance's class."""
    block._sv_capture_layer = int(index)
    block._sv_capture_recorder = recorder
    if not is_capture_class(type(block)):
        block.__class__ = capture_block_class(type(block))
    return type(block)


def is_capture_class(cls: type) -> bool:
    return cls in _CAPTURE_CLASSES.values()
