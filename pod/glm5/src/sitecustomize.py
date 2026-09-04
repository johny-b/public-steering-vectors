"""Installs the steering patch into every vLLM process on PYTHONPATH.

Python imports `sitecustomize` automatically at interpreter start-up for any
directory on `PYTHONPATH`.  That is the whole point: the model lives in spawned
engine and worker processes, so patching only the API server would silently do
nothing at all.

The hook is lazy -- it fires when vLLM's GPU model runner is imported, after
vLLM exists but before the model is built -- so no unrelated interpreter pays to
import torch, and the module is a complete no-op unless GLM_STEER_ENABLE=1.
"""

import importlib.abc
import importlib.machinery
import os
import sys
from collections.abc import Sequence
from types import ModuleType

# Trigger on whichever model-runner module this wheel actually imports.  There
# are two, only one of which is used at runtime, and firing on the wrong one
# means `apply()` never runs and the server comes up quietly unsteered.  Firing
# on either is enough: `apply()` then patches every runner class it can find.
_TRIGGERS = frozenset(
    {
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu_model_runner",
    }
)

if os.environ.get("GLM_STEER_ENABLE") == "1":

    class _SteerFinder(importlib.abc.MetaPathFinder):
        _armed = True

        def find_spec(
            self,
            fullname: str,
            path: Sequence[str] | None = None,
            target: ModuleType | None = None,
        ) -> importlib.machinery.ModuleSpec | None:
            if fullname not in _TRIGGERS or not self._armed:
                return None

            self._armed = False  # disarm before delegating, so this cannot re-enter
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                self._armed = True
                return None

            exec_module = spec.loader.exec_module

            def exec_and_patch(module: ModuleType) -> None:
                exec_module(module)
                from glm_steer.patch import apply

                apply()

            spec.loader.exec_module = exec_and_patch  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, _SteerFinder())
