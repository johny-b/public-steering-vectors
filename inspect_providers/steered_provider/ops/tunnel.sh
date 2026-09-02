#!/usr/bin/env bash
# Idempotent SSH tunnel from a local port to the pod's vLLM server.
#
# The server binds to the pod's loopback interface, so it is not reachable
# without this. The provider talks to http://127.0.0.1:$LOCAL_PORT/v1.
#
#   export POD_HOST=root@1.2.3.4
#   .../ops/tunnel.sh [ensure|status|restart|stop|keepalive]
#
# ensure (the default) brings the tunnel up or no-ops, status exits 1 if it is
# down, and keepalive reopens it whenever it drops.
set -uo pipefail

POD_HOST=${POD_HOST:?POD_HOST must be set, e.g. root@1.2.3.4}
POD_SSH_PORT=${POD_SSH_PORT:-22}
POD_SSH_KEY=${POD_SSH_KEY:-}
LOCAL_PORT=${LOCAL_PORT:-8000}
REMOTE_PORT=${REMOTE_PORT:-8000}
KEEPALIVE_INTERVAL=${KEEPALIVE_INTERVAL:-20}

FORWARD="127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"

is_up() {
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' \
       "http://127.0.0.1:${LOCAL_PORT}/v1/models" 2>/dev/null)" = "200" ]
}

tunnel_pids() {
  # ps is absent in some slim containers; fall back to scanning /proc so that
  # kill_tunnel can still clear a wedged forwarder holding the local port.
  if command -v ps >/dev/null 2>&1; then
    ps -Ao pid=,command= 2>/dev/null | awk -v fwd="$FORWARD" -v host="$POD_HOST" \
      'index($0, "ssh") && index($0, fwd) && index($0, host) { print $1 }'
    return 0
  fi
  local d pid cmd
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$cmd" in
      *ssh*" -N "*"$FORWARD"*"$POD_HOST"*|*ssh*"$FORWARD"*"$POD_HOST"*)
        case "$cmd" in *tunnel.sh*) continue ;; esac
        echo "$pid"
        ;;
    esac
  done
}

kill_tunnel() {
  local pids
  pids=$(tunnel_pids)
  [ -z "$pids" ] && return 0
  echo "tunnel: killing forwarder(s):" $pids
  kill $pids 2>/dev/null
  sleep 1
  pids=$(tunnel_pids)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  return 0
}

start_tunnel() {
  local key_args=()
  if [ -n "$POD_SSH_KEY" ]; then
    key_args=(-i "$POD_SSH_KEY" -o IdentitiesOnly=yes)
  fi
  ssh ${key_args[@]+"${key_args[@]}"} \
      -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \
      -f -N -L "$FORWARD" -p "$POD_SSH_PORT" "$POD_HOST" || return 1
  local i
  for i in $(seq 1 20); do
    is_up && return 0
    sleep 1
  done
  return 1
}

ensure() {
  if is_up; then
    echo "tunnel: already up"
    return 0
  fi
  # Not serving, but a wedged forwarder may still be holding the local port.
  kill_tunnel
  if start_tunnel; then
    echo "tunnel: up (/v1/models 200); pids:" $(tunnel_pids)
    return 0
  fi
  echo "tunnel: FAILED to reach http://127.0.0.1:${LOCAL_PORT}/v1/models" >&2
  return 1
}

case "${1:-ensure}" in
  ensure)
    ensure
    ;;
  status)
    if is_up; then
      echo "tunnel: UP; pids:" $(tunnel_pids)
    else
      echo "tunnel: DOWN; pids:" $(tunnel_pids)
      exit 1
    fi
    ;;
  restart)
    kill_tunnel
    ensure
    ;;
  stop)
    kill_tunnel
    echo "tunnel: stopped"
    ;;
  keepalive)
    while true; do
      ensure >/dev/null 2>&1
      sleep "$KEEPALIVE_INTERVAL"
    done
    ;;
  *)
    echo "usage: $0 [ensure|status|restart|stop|keepalive]" >&2
    exit 2
    ;;
esac
