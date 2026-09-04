#!/usr/bin/env bash
# Start one server in the background and record its pid.
#
#   MODEL=/models/GLM-5.3-Flash pod/glm5/scripts/launch.sh 0.2 pp2
#
# The tag names the run: it selects the log file, the pid file and the status
# file, so several strengths served one after another leave separate evidence.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
strength=${1:?usage: launch.sh <strength|off> <tag>}
tag=${2:?usage: launch.sh <strength|off> <tag>}
log_dir=${GLM_STEER_LOG_DIR:-/tmp/glm5-steer}
mkdir -p "$log_dir"
log=$log_dir/serve_${tag}.log
: > "$log"
GLM_STEER_TAG="$tag" GLM_STEER_STRENGTH="$strength" GLM_STEER_LOG_DIR="$log_dir" \
  nohup "$here/serve.sh" >>"$log" 2>&1 &
pid=$!
echo "$pid" > "$log_dir/serve_${tag}.pid"
echo "launched pid=$pid tag=$tag strength=$strength log=$log"
