"""Global-strength activation steering for GLM-5.3-Flash under vLLM.

`sitecustomize.py` calls :func:`glm_steer.patch.apply` once vLLM's model runner
module has been imported. Nothing here is imported at package-import time, so a
process that inherits `PYTHONPATH` without meaning to pays nothing.
"""
