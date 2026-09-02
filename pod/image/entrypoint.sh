#!/usr/bin/env bash
#
# Container start for the pod image.
#
# Everything here is something that cannot be decided at build time: which
# driver the host turned out to have, which key RunPod wants installed, whether
# the volume already carries state from a previous session. It ends in
# `exec sshd -D`, which is what keeps the container alive — RunPod gives this
# image no start command, so if this script returns, the pod dies.
#
# It runs again on every pod resume, so every step must be idempotent.
#
# Modes (used by the smoke test; neither is used in production):
#   --check-syntax   parse and exit 0
#   --dry-run        do all the setup, print the decisions, do not exec sshd
#                    (also honoured via POD_ENTRYPOINT_DRYRUN=1)

set -uo pipefail

log()  { printf '[pod-entrypoint] %s\n' "$*" >&2; }
warn() { printf '[pod-entrypoint] WARNING: %s\n' "$*" >&2; }

case "${1:-}" in
  --check-syntax) log "syntax ok"; exit 0 ;;
  --dry-run)      POD_ENTRYPOINT_DRYRUN=1; shift ;;
esac
DRYRUN=${POD_ENTRYPOINT_DRYRUN:-0}

# CUDA 13.0's minimum driver is 580.65.06 — not "any 580". A host on 580.42 has
# the right major version and still cannot run a cu130 build, so the comparison
# below is on (major, minor) rather than on the major alone.
readonly CUDA13_MIN_DRIVER=580.65
readonly COMPAT_DIR=/usr/local/cuda-13.0/compat

# ---------------------------------------------------------------------------
# 1. /workspace layout
# ---------------------------------------------------------------------------
# A redirect into a missing directory fails outright, so create the two
# directories anything on this pod is likely to write to before sshd accepts a
# login.
for d in /workspace/hf /workspace/logs; do
  mkdir -p "$d" 2>/dev/null || warn "could not create $d"
done

# uv's cache lives on the volume so a resumed pod reinstalls from cache. If the
# volume is absent or read-only, fall back to the base image's location rather
# than making every uv command fail.
if ! mkdir -p "${UV_CACHE_DIR:-/workspace/.uv-cache}" 2>/dev/null \
   || ! [ -w "${UV_CACHE_DIR:-/workspace/.uv-cache}" ]; then
  warn "UV_CACHE_DIR=${UV_CACHE_DIR:-} not writable; falling back to /opt/uv/cache"
  export UV_CACHE_DIR=/opt/uv/cache
  mkdir -p "$UV_CACHE_DIR" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 2. CUDA forward compatibility
# ---------------------------------------------------------------------------
#
# The wheel is torch 2.13.0+cu130. A host driver older than CUDA 13.0's minimum
# cannot run it; the fix is NVIDIA's forward-compatible libcuda, which this
# image carries and which only has to be first on LD_LIBRARY_PATH.
#
# It is a decision rather than an unconditional export because the compat
# library is actively wrong in two situations:
#
#   - the host driver already supports CUDA 13, where loading an older
#     user-mode libcuda over a newer kernel driver fails;
#   - the GPU is a consumer part, where forward compatibility is unsupported
#     and surfaces as CUDA error 803. vLLM's own helper carries the same
#     warning.
#
# POD_CUDA_COMPAT=on|off overrides the detection.

driver_version() {
  # /proc first: it is injected by the container runtime alongside the driver
  # itself and needs no binary. nvidia-smi is the fallback.
  if [ -r /proc/driver/nvidia/version ]; then
    sed -n 's/.*Kernel Module *\([0-9][0-9.]*\).*/\1/p' /proc/driver/nvidia/version | head -1
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1
  fi
}

gpu_name() {
  command -v nvidia-smi >/dev/null 2>&1 &&
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
}

decide_compat() {
  case "${POD_CUDA_COMPAT:-auto}" in
    off|0) echo "off:forced-by-POD_CUDA_COMPAT"; return ;;
    on|1)  echo "on:forced-by-POD_CUDA_COMPAT";  return ;;
  esac

  [ -d "$COMPAT_DIR" ] || { echo "off:no-compat-dir-in-image"; return; }

  local drv major name
  drv=$(driver_version)
  if [ -z "$drv" ]; then
    # No GPU visible at all (CPU box, or the runtime did not inject the
    # driver). Enabling compat here would be harmless but misleading; report
    # the reason instead and let the failure be about the missing GPU.
    echo "off:no-nvidia-driver-visible"; return
  fi
  name=$(gpu_name)
  case "$name" in
    *GeForce*|*TITAN*)
      echo "off:consumer-gpu-forward-compat-unsupported($name)"; return ;;
  esac

  # Numeric compare on (major, minor) — `sort -V` would do too, but this keeps
  # the comparison explicit and needs no subshell per field.
  local dmaj dmin rmaj rmin
  IFS=. read -r dmaj dmin _ <<< "$drv"
  IFS=. read -r rmaj rmin   <<< "$CUDA13_MIN_DRIVER"
  dmaj=${dmaj:-0}; dmin=${dmin:-0}
  if [ "$dmaj" -lt "$rmaj" ] 2>/dev/null ||
     { [ "$dmaj" -eq "$rmaj" ] && [ "$dmin" -lt "$rmin" ]; } 2>/dev/null; then
    echo "on:driver-$drv-older-than-cuda-13.0-minimum-${CUDA13_MIN_DRIVER}"
  else
    echo "off:driver-$drv-already-supports-cuda-13.0"
  fi
}

COMPAT_DECISION=$(decide_compat)
log "gpu: driver=$(driver_version || echo none) name=$(gpu_name || echo none)"
log "cuda forward-compat: ${COMPAT_DECISION}"

if [ "${COMPAT_DECISION%%:*}" = "on" ]; then
  # First on the path, and exported before anything is exec'd, so that every
  # process vLLM spawns — API server, engine core, each worker — starts with it
  # already in place. Setting it here rather than inside python also sidesteps
  # the question of whether a mid-process mutation of LD_LIBRARY_PATH reaches
  # the dynamic loader at all.
  export LD_LIBRARY_PATH="${COMPAT_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  # Tell vLLM's own helper the same thing, so its independent attempt at this
  # agrees with ours instead of fighting it. It returns early when the
  # directory is already at the front, so this is a no-op that documents intent.
  export VLLM_ENABLE_CUDA_COMPATIBILITY=1
  export VLLM_CUDA_COMPATIBILITY_PATH="$COMPAT_DIR"
  log "prepended $COMPAT_DIR to LD_LIBRARY_PATH ($(cat /etc/pod-cuda-compat-version 2>/dev/null || echo 'unknown libcuda'))"
else
  export VLLM_ENABLE_CUDA_COMPATIBILITY=0
fi

# ---------------------------------------------------------------------------
# 3. Publish the environment where ssh sessions can see it
# ---------------------------------------------------------------------------
#
# sshd does not give a session its own environment. Two files, because two
# different mechanisms are needed:
#
#   /etc/pod-env.sh   sourced by /etc/profile.d and /root/.bashrc. Full dump,
#                     shell-quoted, mode 0600 because RunPod may have injected
#                     tokens (HF_TOKEN, RUNPOD_*) into this container's env.
#   /etc/environment  read by pam_env, which is the ONLY one of the three hooks
#                     that reaches `ssh pod '<command>'` — a non-interactive,
#                     non-login shell sources no profile and no bashrc. An
#                     explicit allowlist of non-secret values, mode 0644.
#
# Both are rewritten every start, because the compat decision above is part of
# what they carry.

ENV_DUMP_SKIP='PWD OLDPWD SHLVL _ HOME USER LOGNAME SHELL TERM MAIL HOSTNAME
               POD_ENTRYPOINT_DRYRUN'
ETC_ENV_ALLOW='PATH LD_LIBRARY_PATH HF_HOME CUDA_HOME CUDA_VERSION
               VLLM_ENABLE_CUDA_COMPATIBILITY VLLM_CUDA_COMPATIBILITY_PATH
               UV_CACHE_DIR UV_LINK_MODE UV_HTTP_TIMEOUT UV_INDEX_STRATEGY
               UV_OVERRIDE UV_PYTHON_INSTALL_DIR
               POD_CUDA_COMPAT PYTHONUNBUFFERED'

python3 - "$ENV_DUMP_SKIP" "$ETC_ENV_ALLOW" <<'PY'
import os, re, shlex, sys

skip  = set(sys.argv[1].split())
allow = sys.argv[2].split()
ident = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

# Full dump for the shell hooks.
with open('/etc/pod-env.sh', 'w') as f:
    f.write("# Written by pod-entrypoint.sh at container start. Do not edit.\n")
    for k, v in sorted(os.environ.items()):
        if k in skip or k.startswith(('SSH_', 'BASH_FUNC_')) or not ident.match(k):
            continue
        f.write(f"export {k}={shlex.quote(v)}\n")
os.chmod('/etc/pod-env.sh', 0o600)

# Allowlisted, non-secret subset for pam_env. pam_env expands $ and ` in
# values, so anything containing them is dropped rather than mangled; those
# variables still reach login and interactive shells through the file above.
with open('/etc/environment', 'w') as f:
    f.write("# Written by pod-entrypoint.sh at container start. Read by pam_env,\n"
            "# which is what gets these into a non-login `ssh pod <command>`.\n")
    for k in allow:
        v = os.environ.get(k)
        if v is None or '\n' in v or '$' in v or '`' in v or '"' in v or '\\' in v:
            continue
        f.write(f'{k}="{v}"\n')
os.chmod('/etc/environment', 0o644)
PY
[ -s /etc/environment ] || warn "/etc/environment came out empty; ssh commands may not see PATH"
log "published environment to /etc/pod-env.sh (0600) and /etc/environment (0644)"

# ---------------------------------------------------------------------------
# 4. ssh: authorized key and host keys
# ---------------------------------------------------------------------------
mkdir -p /root/.ssh && chmod 700 /root/.ssh
touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys

if [ -n "${PUBLIC_KEY:-}" ]; then
  # RunPod may pass several keys, newline-separated. Append only the ones that
  # are not already there: this script runs again on every pod resume, and
  # authorized_keys growing a duplicate line per restart is untidy at best.
  added=0
  while IFS= read -r key; do
    key=$(printf '%s' "$key" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
    [ -z "$key" ] && continue
    grep -qxF "$key" /root/.ssh/authorized_keys || { printf '%s\n' "$key" >> /root/.ssh/authorized_keys; added=$((added+1)); }
  done <<< "${PUBLIC_KEY}"
  log "installed $added key(s) from \$PUBLIC_KEY into /root/.ssh/authorized_keys"
else
  # Not fatal: a pod can still be reached through RunPod's web terminal, and
  # failing to boot because of a missing env var would be worse than being
  # unreachable by ssh. But say it loudly, because "connection refused" is
  # otherwise a ten-minute mystery.
  warn '$PUBLIC_KEY is empty — no key installed, ssh will reject every login.'
fi

# Host keys on the volume, so a stopped-and-resumed pod keeps its identity and
# the client is not greeted with REMOTE HOST IDENTIFICATION HAS CHANGED.
# /root is container disk and would lose them on every stop.
HOSTKEY_STORE=/workspace/.ssh_host_keys
if mkdir -p "$HOSTKEY_STORE" 2>/dev/null && [ -w "$HOSTKEY_STORE" ]; then
  if compgen -G "$HOSTKEY_STORE/ssh_host_*" >/dev/null; then
    cp -a "$HOSTKEY_STORE"/ssh_host_* /etc/ssh/ && log "restored host keys from $HOSTKEY_STORE"
  fi
  ssh-keygen -A >/dev/null 2>&1 || warn "ssh-keygen -A failed"
  cp -a /etc/ssh/ssh_host_* "$HOSTKEY_STORE"/ 2>/dev/null || true
else
  warn "$HOSTKEY_STORE not writable; host keys will change on every restart"
  ssh-keygen -A >/dev/null 2>&1 || warn "ssh-keygen -A failed"
fi
chmod 600 /etc/ssh/ssh_host_*_key 2>/dev/null || true
mkdir -p /run/sshd

/usr/sbin/sshd -t || { warn "sshd config test FAILED — see above"; }

# ---------------------------------------------------------------------------
# 5. Hand over to sshd
# ---------------------------------------------------------------------------
if [ "$DRYRUN" = "1" ]; then
  log "dry run: not starting sshd"
  exit 0
fi

log "starting sshd on port 22 (foreground; this is what keeps the pod alive)"
# -D: stay in the foreground.  -e: log to stderr, so `docker logs` and the
# RunPod log pane show authentication failures instead of swallowing them into
# a syslog that is not running in this container.
exec /usr/sbin/sshd -D -e
