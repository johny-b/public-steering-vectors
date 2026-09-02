#!/usr/bin/env bash
#
# Everything about this image that can be checked without a GPU.
#
#   docker run --rm --entrypoint bash <image> /usr/local/bin/smoke_cpu.sh
#
# Exits non-zero on the first failure, with a line saying what was expected and
# why it matters. It cannot tell you that the server will serve — that needs a
# GPU host — but every failure it *can* catch is one that would otherwise show
# up as a stack trace deep in a log on a machine that costs money per hour.
#
# It is destructive in small ways (it dry-runs the entrypoint, which rewrites
# /etc/environment and /etc/pod-env.sh), so run it in a throwaway container.

set -uo pipefail

STEP=0
ok()   { STEP=$((STEP+1)); printf '  ok %2d  %s\n' "$STEP" "$*"; }
fail() { printf '\nFAIL: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "$1 is not on PATH. $2"; }

echo "=== pod image: CPU smoke test ==="

# --- 1. platform ------------------------------------------------------------
[ "$(uname -m)" = "x86_64" ] || fail "expected x86_64, got $(uname -m). The vLLM wheels in this image are linux/amd64 only."
ok "arch is x86_64"

# --- 2. python on PATH ------------------------------------------------------
# The base image ships python3 and python3.12 but no bare `python`, and
# pod/scripts/serve.sh's last line is `exec python -m vllm...`. This check is
# what catches a base bump quietly removing the symlink.
need python "pod/scripts/serve.sh ends in 'exec python -m vllm.entrypoints.openai.api_server' and would die there."
PYV=$(python -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])') || fail "'python' exists but will not run"
python -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)' \
  || fail "python is $PYV; pod/pyproject.toml declares requires-python >= 3.12"
ok "python -> $PYV ($(readlink -f "$(command -v python)"))"

# Compare the RESOLVED interpreter, not the path used to invoke it: `python` is
# this image's symlink in /usr/local/bin and `python3` is Ubuntu's alternatives
# symlink, so their sys.executable strings differ while both are
# /usr/bin/python3.12. Comparing the unresolved strings reports a failure that
# is not one.
[ "$(python  -c 'import sys,os;print(os.path.realpath(sys.executable))')" \
= "$(python3 -c 'import sys,os;print(os.path.realpath(sys.executable))')" ] \
  || fail "'python' and 'python3' resolve to different interpreters; the one with vLLM in it may not be the one serve.sh runs"
ok "python and python3 are the same interpreter"

# --- 3. vLLM ----------------------------------------------------------------
VLLM_V=$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null) \
  || fail "'import vllm' failed. Run: python -c 'import vllm' to see the traceback."
[ "$VLLM_V" = "0.27.1" ] \
  || fail "vllm is $VLLM_V, but pod/pyproject.toml pins vllm==0.27.1 (the patch wraps vllm.v1.worker.gpu_model_runner, a private module)"
ok "vllm $VLLM_V"

# The exact command serve.sh execs. On a GPU host this prints help and exits 0.
# On a GPU-LESS host (this smoke test's usual home) vLLM 0.27.1 cannot build the
# default DeviceConfig while assembling the argparse defaults, and dies with
# "Failed to infer device type". The unmodified vllm/vllm-openai:v0.27.1 image
# fails the same way on the same box, so that one error is a property of the
# host rather than of this image: tolerate it, and nothing else.
_HELP_ERR=$(mktemp)
if timeout 300 python -m vllm.entrypoints.openai.api_server --help >/dev/null 2>"$_HELP_ERR"; then
  ok "python -m vllm.entrypoints.openai.api_server --help runs"
elif grep -q 'Failed to infer device type' "$_HELP_ERR"; then
  ok "vllm api_server CLI imports (device-detection deferred: no GPU on this host)"
  SKIPPED_GPU=1
else
  echo "--- stderr ---"; tail -20 "$_HELP_ERR"
  fail "'python -m vllm.entrypoints.openai.api_server --help' failed for a reason other than missing-GPU device detection; that command is verbatim the last line of pod/scripts/serve.sh"
fi
rm -f "$_HELP_ERR"

# --- 4. torch / CUDA build --------------------------------------------------
TORCH_CUDA=$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null) \
  || fail "'import torch' failed"
TORCH_V=$(python -c 'import torch; print(torch.__version__)')
case "$TORCH_CUDA" in
  13.*) ;;
  *) fail "torch is built for CUDA $TORCH_CUDA. This image's compat handling assumes cu130 (torch 2.13.0+cu130), and vLLM's compat helper looks for /usr/local/cuda-$TORCH_CUDA/compat." ;;
esac
ok "torch $TORCH_V (cuda $TORCH_CUDA)"

# --- 5. the forward-compat libcuda is present and plausible -----------------
COMPAT=/usr/local/cuda-${TORCH_CUDA}/compat
[ -d "$COMPAT" ] || fail "$COMPAT missing. That is the exact path vllm/env_override.py derives from torch.version.cuda; without it a host on a pre-580.65.06 driver cannot run this image at all."
LIBCUDA=$(ls -1 "$COMPAT"/libcuda.so.*.* 2>/dev/null | head -1)
[ -n "$LIBCUDA" ] || fail "$COMPAT exists but has no libcuda.so.<version>"
COMPAT_V=$(basename "$LIBCUDA" | sed 's/^libcuda\.so\.//')
case "$COMPAT_V" in
  5[89]*|[6-9]*) ;;
  *) fail "compat libcuda is $COMPAT_V; CUDA 13.0 needs a 580-or-newer forward-compat driver. Anything older than the host driver is a downgrade the driver refuses." ;;
esac
ok "cuda-compat present: libcuda.so.$COMPAT_V in $COMPAT"

# --- 6. FlashInfer works without a sampler override -------------------------
python -c 'import importlib.util as u, sys; sys.exit(0 if u.find_spec("flashinfer") else 1)' \
  || fail "flashinfer-python is not installed; vLLM would fall back off the FlashInfer sampler"
FI_V=$(python -c 'import flashinfer; print(getattr(flashinfer, "__version__", "?"))' 2>/dev/null || echo "?")
ok "flashinfer-python importable (version $FI_V)"

# vllm/utils/flashinfer.py: has_flashinfer() returns False unless either the
# cubin package is installed or nvcc is on PATH. The cubin package is why this
# image does not have to JIT at engine start.
python -c 'import importlib.util as u, sys; sys.exit(0 if u.find_spec("flashinfer_cubin") else 1)' \
  || fail "flashinfer-cubin missing. Without it vLLM must JIT-compile or download cubins at engine start, which is slow at best and fails on a host with no compiler."
ok "flashinfer-cubin present (prebuilt kernels: no JIT, no artifactory fetch at start)"

python -c 'import importlib.util as u, sys; sys.exit(0 if u.find_spec("flashinfer_jit_cache") else 1)' \
  && ok "flashinfer-jit-cache present (AOT-compiled JIT modules)" \
  || echo "  -- warn   flashinfer-jit-cache not found; kernels may be compiled on first use"

python - <<'PY' || fail "vllm.utils.flashinfer.has_flashinfer() is False — vLLM will not use the FlashInfer sampler"
import sys
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cubin
print(f"  ..     vllm sees flashinfer={has_flashinfer()} cubin={has_flashinfer_cubin()}")
sys.exit(0 if has_flashinfer() else 1)
PY
ok "vLLM's own has_flashinfer() is True (so VLLM_USE_FLASHINFER_SAMPLER=0 is not needed)"

# --- 7. nvcc accepts Hopper, and ideally Blackwell --------------------------
# Checkable on a CPU box: compiling for an architecture does not require having
# one. This is what proves the JIT fallback would actually work on the pod.
need nvcc "FlashInfer's JIT fallback needs it, and vllm.utils.flashinfer checks for it."
NVCC_V=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
ok "nvcc $NVCC_V at $(command -v nvcc)"
need gcc "nvcc cannot run without a host compiler."

TD=$(mktemp -d); trap 'rm -rf "$TD"' EXIT
printf '__global__ void k() {}\n' > "$TD/t.cu"
nvcc -arch=sm_90a -cubin -o "$TD/t90a.cubin" "$TD/t.cu" 2>"$TD/err90" \
  || { sed 's/^/    /' "$TD/err90" >&2; fail "nvcc cannot compile for sm_90a (Hopper), so FlashInfer would fail to JIT on an H100/H200."; }
ok "nvcc compiles sm_90a (Hopper / H200)"

if nvcc -arch=sm_100a -cubin -o "$TD/t100a.cubin" "$TD/t.cu" 2>"$TD/err100"; then
  ok "nvcc compiles sm_100a (Blackwell)"
else
  echo "  -- warn   nvcc rejects sm_100a; this image is Hopper-ready but not Blackwell-ready for JIT kernels"
fi

# --- 8. uv ------------------------------------------------------------------
need uv "The editable installs on the pod use it."
ok "uv $(uv --version 2>&1 | head -1)"

# --- 9. sshd ----------------------------------------------------------------
[ -x /usr/sbin/sshd ] || fail "/usr/sbin/sshd missing; RunPod expects an image that listens on 22"
ssh-keygen -q -t ed25519 -N '' -f "$TD/hostkey" </dev/null >/dev/null 2>&1 \
  || fail "ssh-keygen failed"
/usr/sbin/sshd -t -h "$TD/hostkey" 2>"$TD/sshderr" \
  || { sed 's/^/    /' "$TD/sshderr" >&2; fail "sshd -t rejects the config in /etc/ssh"; }
ok "sshd config parses"

EFF=$(/usr/sbin/sshd -T -h "$TD/hostkey" 2>/dev/null)
check_eff() {
  echo "$EFF" | grep -qix "$1 $2" \
    || fail "effective sshd setting '$1' is '$(echo "$EFF" | grep -i "^$1 " || echo unset)', expected '$2'"
}
# sshd -T prints the legacy spelling "without-password" for prohibit-password;
# they are the same setting, so accept either.
echo "$EFF" | grep -qiE '^permitrootlogin (prohibit-password|without-password)$' \
  || fail "effective sshd setting 'permitrootlogin' is '$(echo "$EFF" | grep -i '^permitrootlogin ' || echo unset)', expected prohibit-password"
check_eff pubkeyauthentication yes
check_eff passwordauthentication no
check_eff allowtcpforwarding yes
echo "$EFF" | grep -qix "port 22" || fail "sshd is not configured to listen on port 22"
ok "sshd: port 22, root by key only, password auth off, TCP forwarding on"

# --- 10. HF_HOME and the /workspace defaults --------------------------------
[ "${HF_HOME:-}" = "/workspace/hf" ] \
  || fail "HF_HOME is '${HF_HOME:-unset}', expected /workspace/hf. Anything outside /workspace is wiped by a pod stop, which means re-downloading the weights every time."
ok "HF_HOME=/workspace/hf"

# --- 11. the entrypoint, dry-run --------------------------------------------
[ -x /usr/local/bin/pod-entrypoint.sh ] || fail "/usr/local/bin/pod-entrypoint.sh missing or not executable"
SMOKE_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAISMOKETESTKEY smoke@test"
PUBLIC_KEY="$SMOKE_KEY" /usr/local/bin/pod-entrypoint.sh --dry-run >"$TD/ep.log" 2>&1 \
  || { sed 's/^/    /' "$TD/ep.log" >&2; fail "entrypoint --dry-run exited non-zero"; }
ok "entrypoint dry-run completes"

[ -d /workspace/logs ] || fail "entrypoint did not create /workspace/logs; a nohup redirect into it would fail"
[ -d /workspace/hf ]   || fail "entrypoint did not create /workspace/hf"
ok "/workspace/logs and /workspace/hf created"

grep -qxF "$SMOKE_KEY" /root/.ssh/authorized_keys \
  || fail "\$PUBLIC_KEY was not appended to /root/.ssh/authorized_keys"
[ "$(stat -c '%a' /root/.ssh/authorized_keys)" = "600" ] || fail "authorized_keys is not mode 600; sshd will refuse it"
[ "$(stat -c '%a' /root/.ssh)" = "700" ] || fail "/root/.ssh is not mode 700; sshd will refuse it"
# Run it twice: the entrypoint runs again on every pod resume and must not
# accumulate a duplicate key per restart.
PUBLIC_KEY="$SMOKE_KEY" /usr/local/bin/pod-entrypoint.sh --dry-run >/dev/null 2>&1
[ "$(grep -cxF "$SMOKE_KEY" /root/.ssh/authorized_keys)" = "1" ] \
  || fail "authorized_keys gained a duplicate key on the second start"
ok "PUBLIC_KEY installed, modes correct, idempotent across restarts"

# On a CPU box there is no driver, so the entrypoint must decide NOT to load the
# forward-compat libcuda — and must say why rather than crashing.
grep -q "cuda forward-compat: off:no-nvidia-driver-visible" "$TD/ep.log" \
  || { grep "forward-compat" "$TD/ep.log" >&2; fail "entrypoint made the wrong compat decision on a GPU-less box"; }
ok "compat decision on a GPU-less box: off, with a reason"

# --- 12. env reaches ssh sessions -------------------------------------------
[ -r /etc/pod-env.sh ] || fail "/etc/pod-env.sh was not written"
[ "$(stat -c '%a' /etc/pod-env.sh)" = "600" ] || fail "/etc/pod-env.sh is not 0600 (it may carry HF_TOKEN and other injected secrets)"
grep -q '^export PATH=' /etc/pod-env.sh || fail "/etc/pod-env.sh carries no PATH"
grep -q '^export HF_HOME=' /etc/pod-env.sh || fail "/etc/pod-env.sh carries no HF_HOME"
ok "/etc/pod-env.sh written, 0600, carries PATH and HF_HOME"

grep -q '^PATH=' /etc/environment || fail "/etc/environment carries no PATH; a non-login 'ssh pod <command>' would not find python"
grep -q '^HF_HOME=' /etc/environment || fail "/etc/environment carries no HF_HOME"
grep -qi 'token\|secret\|password' /etc/environment && fail "/etc/environment (mode 0644) looks like it contains a secret; it is allowlisted for a reason"
ok "/etc/environment carries PATH and HF_HOME and no secrets (this is the pam_env hook for non-login ssh)"

# The login-shell hook, end to end.
LOGIN_PY=$(env -i /bin/bash -lc 'command -v python' 2>/dev/null)
[ -n "$LOGIN_PY" ] || fail "a login shell (bash -l) cannot find python; /etc/profile.d/10-pod-env.sh is not doing its job"
LOGIN_HF=$(env -i /bin/bash -lc 'echo $HF_HOME' 2>/dev/null)
[ "$LOGIN_HF" = "/workspace/hf" ] || fail "a login shell sees HF_HOME='$LOGIN_HF', expected /workspace/hf"
ok "login shell with an empty environment still finds python and HF_HOME"

# The interactive-shell hook.
INT_HF=$(env -i /bin/bash -ic 'echo $HF_HOME' 2>/dev/null | tail -1)
[ "$INT_HF" = "/workspace/hf" ] || fail "an interactive shell sees HF_HOME='$INT_HF'; /root/.bashrc is not sourcing /etc/pod-env.sh early enough"
ok "interactive shell finds HF_HOME"

# --- 13. tools the pod workflow needs ---------------------------------------
for t in git rsync jq curl tar ssh-keygen; do
  need "$t" "It is used to get the repository onto the pod or to check the running server."
done
ok "git, rsync, jq, curl, tar, ssh-keygen present"

echo
echo "=== all CPU-checkable requirements pass ($STEP checks) ==="
[ -n "${SKIPPED_GPU:-}" ] && echo "(vllm CLI device detection deferred to the GPU pod: no NVIDIA device here)"
echo "Still needs a GPU host to confirm: torch.cuda.is_available(), engine start,"
echo "and that a steered request round-trips."
