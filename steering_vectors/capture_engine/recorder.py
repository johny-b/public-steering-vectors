"""Record last-token block-input residuals as float32 capture arrays.

Ported from the proven local ``v2-steering-tools`` recorder (commits ea65b2f,
a486b53, d732a8c, 6508dab, 6678408).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.modelprofile import PROFILE
from . import config


class Recorder:
    """Publish one ``(n_layers, hidden_size)`` array per complete forward."""

    def __init__(
        self,
        directory: str | Path,
        *,
        num_layers: int = PROFILE.n_layers,
        row_reader: Any = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows = config.CaptureAccumulator(num_layers=num_layers)
        self.flushes = 0
        self._read_row = _last_row_to_host if row_reader is None else row_reader

    def record(self, index: int, stream: Any, *, num_positions: int = -1) -> None:
        complete = self.rows.add(
            index,
            self._read_row(stream),
            num_tokens=int(stream.shape[0]),
            num_positions=num_positions,
        )
        if complete is not None:
            self.flush(*complete)

    def flush(
        self, rows: list[Any], num_tokens: int, num_positions: int
    ) -> tuple[Path, Path]:
        import numpy as np

        paths = config.write_capture(
            self.directory,
            self.flushes,
            np.stack([np.asarray(row, dtype=np.float32) for row in rows]),
            num_tokens=num_tokens,
            num_positions=num_positions,
        )
        self.flushes += 1
        return paths


def _last_row_to_host(stream: Any) -> Any:
    import torch

    return stream[-1].detach().to(device="cpu", dtype=torch.float32).numpy()
