#!/usr/bin/env bash
# Block until the server is serving, then REFUSE it unless the steering
# installed in EVERY tensor-parallel worker.
#
#   pod/glm5/scripts/await_ready.sh pp2 0.2 4
#
# Exit 0 only when the server is both up and demonstrably in the state that was
# asked for. A vLLM server that came up with the patch silently absent answers
# requests perfectly well and produces unsteered generations under a steered
# label, which is the one failure this whole stack is built to make impossible.
set -uo pipefail
tag=${1:?usage: await_ready.sh <tag> <strength|off> [n_tp]}
strength=${2:?usage: await_ready.sh <tag> <strength|off> [n_tp]}
ntp=${3:-4}
log_dir=${GLM_STEER_LOG_DIR:-/tmp/glm5-steer}
log=$log_dir/serve_${tag}.log
status=$log_dir/steer_status_${tag}.jsonl

until grep -qaE "Application startup complete|Traceback|Engine core initialization failed|EngineDeadError" "$log"; do
  sleep 15
done

if ! grep -qa "Application startup complete" "$log"; then
  echo "SERVER_FAILED"
  grep -aoE "(RuntimeError|Error)[^|]{0,200}" "$log" | tail -5
  exit 1
fi

if [ "$strength" = "off" ]; then
  # The baseline's claim is that nothing was patched, and the only evidence for
  # it is the absence of the patch's own log lines.
  if grep -qa "glm_steer" "$log"; then
    echo "BASELINE_CONTAMINATED: glm_steer ran in a server that should be unpatched"
    exit 1
  fi
  echo "READY_UNPATCHED (verified: zero glm_steer log lines)"
  exit 0
fi

n=$(wc -l < "$status" 2>/dev/null || echo 0)
if [ "$n" -ne "$ntp" ]; then
  echo "STEERING_NOT_INSTALLED: $n/$ntp workers reported installation"
  echo "  -> refusing this server; it would have served UNSTEERED output."
  exit 1
fi
got=$(python3 -c "
import json
rows = [json.loads(line) for line in open('$status')]
print(sorted({r['strength'] for r in rows}),
      sorted({r['layer'] for r in rows}),
      round(rows[0]['ratio'], 6))
")
echo "READY_STEERED workers=$n/$ntp strengths/layers/ratio=$got"
