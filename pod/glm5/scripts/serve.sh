#!/usr/bin/env bash
# GLM-5.3-Flash served by vLLM, steered at ONE GLOBAL STRENGTH fixed at launch.
#
#   MODEL=/models/GLM-5.3-Flash \
#   GLM_STEER_VECTOR_DIR=pod/glm5/vectors/GLM-5.3-flash-0007 \
#   GLM_STEER_STRENGTH=0.2 \
#     pod/glm5/scripts/serve.sh
#
# GLM_STEER_STRENGTH is mandatory and takes three kinds of value:
#
#   off   the patch is never imported, PYTHONPATH is not set and no class is
#         substituted. This is the *unpatched* baseline, the control for the
#         question "does loading the patch change generation at all".
#   0     patched, delta is the zero vector, and the addition is still executed
#         and still traced. This is the *steering* control: same code path as a
#         steered arm, same compiled graph, zero perturbation.
#   0.2   patched and steered.
#
# Every other flag is byte-identical across the modes on purpose: the only
# difference between the baseline and the strength-0 server must be the patch.
# See pod/glm5/README.md for what each variable means and why.
set -euo pipefail

glm5_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

: "${MODEL:?MODEL must be set (path to the GLM-5.3-Flash checkpoint)}"
: "${GLM_STEER_STRENGTH:?set GLM_STEER_STRENGTH (a float, or 'off' for the unpatched baseline)}"
: "${GLM_STEER_VECTOR_DIR:=$glm5_dir/vectors/GLM-5.3-flash-0007}"
export GLM_STEER_VECTOR_DIR

# vLLM's compile-cache key does not know about a substituted decoder-layer class,
# so without this a steered engine can replay an artefact compiled without the
# steering -- unsteered generation reporting success. Disabled on the baseline
# too, so all three modes compile by the same route.
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}

if [ "$GLM_STEER_STRENGTH" = "off" ]; then
  unset GLM_STEER_ENABLE GLM_STEER_STRENGTH GLM_STEER_STATUS_FILE
  echo "[serve] UNPATCHED baseline: no PYTHONPATH, no sitecustomize, no substitution"
else
  export GLM_STEER_ENABLE=1
  # sitecustomize.py lives here and is auto-imported by every interpreter that
  # inherits PYTHONPATH, including the engine and worker processes vLLM spawns.
  # Deliberately a directory of its own, NOT pod/src: two sitecustomize.py files
  # cannot coexist on one PYTHONPATH, and only the first found would run.
  export PYTHONPATH=$glm5_dir/src${PYTHONPATH:+:$PYTHONPATH}
  # Written by each worker the moment its decoder layer is substituted. The
  # launcher refuses the server unless one line per TP rank appears: a server
  # that came up unsteered is otherwise indistinguishable from one that works.
  : "${GLM_STEER_STATUS_FILE:=${GLM_STEER_LOG_DIR:-/tmp/glm5-steer}/steer_status_${GLM_STEER_TAG:-run}.jsonl}"
  export GLM_STEER_STATUS_FILE
  mkdir -p "$(dirname "$GLM_STEER_STATUS_FILE")"
  rm -f "$GLM_STEER_STATUS_FILE"
  echo "[serve] STEERED: strength=$GLM_STEER_STRENGTH vector_dir=$GLM_STEER_VECTOR_DIR status=$GLM_STEER_STATUS_FILE"
fi

# NOTE: deliberately NO --speculative-config      -> MTP off. The draft layer sits
#       outside the 45-layer stack, so drafted tokens would bypass the steering
#       while verified ones did not. The patch also refuses to start under it.
#       deliberately NO sequence-parallel flags   -> likewise refused by the patch.
#       prefix caching left at its default (on): one strength per process, so
#       every entry in the cache was produced under the strength that reads it.
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "${SERVED_MODEL_NAME:-$(basename "$MODEL")}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-4}" \
  --max-model-len "${MAX_MODEL_LEN:-40960}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --attention-backend "${ATTENTION_BACKEND:-FLASH_ATTN_MLA_SPARSE}" \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --seed "${SEED:-0}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  "$@"
