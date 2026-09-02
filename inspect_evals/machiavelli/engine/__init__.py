"""The MACHIAVELLI game engine, vendored from the original benchmark.

This subpackage is deliberately free of any Inspect or model-provider imports:
it is a ChoiceScript interpreter (`player`) and a step/observe wrapper around it
(`machiavelli_env`), and can be driven by anything -- including the random agent
used to validate the port.
"""

from __future__ import annotations

from .machiavelli_env import MachiavelliEnv, Trajectory, TrajectoryStep
from .types import (
    Achievement,
    GameAnnotations,
    GameInfo,
    GameMetadata,
    GameSource,
    GameState,
    GameTreeNode,
    MachiavelliEngineError,
    MoralityAnnotation,
    NormalizationCoeffs,
    SceneAnnotations,
    Thresholds,
)

__all__ = [
    "Achievement",
    "GameAnnotations",
    "GameInfo",
    "GameMetadata",
    "GameSource",
    "GameState",
    "GameTreeNode",
    "MachiavelliEngineError",
    "MachiavelliEnv",
    "MoralityAnnotation",
    "NormalizationCoeffs",
    "SceneAnnotations",
    "Thresholds",
    "Trajectory",
    "TrajectoryStep",
]
