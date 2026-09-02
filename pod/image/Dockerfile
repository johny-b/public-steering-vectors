# syntax=docker/dockerfile:1.7
#
# Pod image for the steered vLLM server in `pod/`.
#
# It provides the environment only: Python 3.12, `vllm==0.27.1`, torch
# 2.13/cu130, FlashInfer with prebuilt cubins, the CUDA 13 toolkit and its
# forward-compatibility libcuda, sshd, and the tooling needed to get a
# repository onto the pod. It contains no repository code, no vectors and no
# model weights: those belong on `/workspace`, which is the only directory a
# RunPod stop preserves, and baking them in would mean rebuilding 31 GB to fix
# a typo.
#
# Build and check (needs ~30 GB free):
#
#   docker build --platform linux/amd64 -t <registry>/steering-vectors-pod:vllm-0.27.1 .
#   docker run --rm --entrypoint bash <image> /usr/local/bin/smoke_cpu.sh
#
# Create the pod from this image with NO container start command, so this
# image's own ENTRYPOINT runs. It starts sshd in the foreground, which is what
# keeps the container alive.

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
#
# The official vLLM release image for the exact version `pod/pyproject.toml`
# pins. It already provides:
#
#   Python 3.12, packages installed --system, no venv
#   torch 2.13.0+cu130
#   CUDA 13.0.2 with nvcc, and cuda-compat-13-0
#   flashinfer-python + flashinfer-cubin + flashinfer-jit-cache
#   uv on PATH
#
# That is a torch/vLLM/FlashInfer/NCCL combination vLLM's own release pipeline
# tested together; building the equivalent up from nvidia/cuda would mean
# re-deriving it by hand.
#
# Pinned by digest because a tag is a mutable pointer and a silently re-pushed
# base is exactly the class of surprise this image exists to remove. This is
# the multi-arch index digest; --platform selects amd64, which is what the
# wheels are built for and what RunPod runs. Moving to a newer vLLM means
# changing the tag, the digest, and the version assertions below and in
# smoke_cpu.sh together.
FROM --platform=linux/amd64 vllm/vllm-openai:v0.27.1@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967

# The base sets ENTRYPOINT ["vllm", "serve"] and WORKDIR /vllm-workspace. Both
# are reset at the bottom of this file: this image is a machine you ssh into,
# not a one-shot server container.

# The base declares NVIDIA_REQUIRE_CUDA="cuda>=13.0 ..." naming a fixed list of
# permitted driver branches. The NVIDIA container runtime enforces that at
# container *start*, before the entrypoint runs, so a host on any other branch
# refuses to run this image at all — with an error about an unsatisfied
# requirement rather than anything actionable. Such hosts exist in RunPod's
# fleet.
#
# The check is disabled because it asks the wrong question. It tests whether
# the driver natively supports CUDA 13; what actually matters is whether it
# does OR the forward-compatibility libcuda can cover the gap. That is a
# decision the entrypoint makes at start, with the driver in front of it, and
# reports in a readable line.
ENV NVIDIA_DISABLE_REQUIRE=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
#
# openssh-server            RunPod reaches a pod over ssh on port 22
# git, rsync                getting a repository onto /workspace
# jq                        reading JSON from the server's endpoints
# tmux, less                a long run outliving one ssh session
# build-essential, ninja    NOT decoration. The base installs cuda-nvcc but no
#                           host compiler, and nvcc cannot run without one. The
#                           prebuilt FlashInfer cubins mean nothing normally
#                           needs to compile; these keep the JIT fallback
#                           functional rather than merely nominal. ~200 MB.
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
      openssh-server \
      git \
      rsync \
      jq \
      tmux \
      less \
      ca-certificates \
      build-essential \
      ninja-build \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# CUDA forward compatibility
# ---------------------------------------------------------------------------
#
# The base ships cuda-compat-13-0, i.e. a forward-compatible user-mode libcuda
# at /usr/local/cuda-13.0/compat/. Record its version, so the entrypoint can
# name the libcuda it loads and a reader can tell at a glance which one a given
# image carries.
#
# Deliberately NOT re-pinned to some other version from NVIDIA's apt repo: that
# would add a network dependency to every build for no benefit, and the version
# the base ships is the one vLLM's release pipeline tested against.
#
# The directory NAME is load-bearing. vLLM's own helper
# (`vllm/env_override.py::_maybe_set_cuda_compatibility_path`) looks for
# `/usr/local/cuda-{torch.version.cuda}/compat`; torch here is 2.13.0+cu130, so
# that string is exactly `/usr/local/cuda-13.0/compat`. "Upgrading" the package
# to cuda-compat-13-3 would move the directory and silently disable that path.
RUN set -euo pipefail; \
    ls -1 /usr/local/cuda-13.0/compat/libcuda.so.*; \
    basename /usr/local/cuda-13.0/compat/libcuda.so.*.*.* \
      | sed 's/^libcuda\.so\.//' > /etc/pod-cuda-compat-version; \
    echo "baked cuda-compat libcuda: $(cat /etc/pod-cuda-compat-version)"

# ---------------------------------------------------------------------------
# `python` on PATH
# ---------------------------------------------------------------------------
#
# The base provides `python3` and `python3.12` but no bare `python`, while
# `pod/scripts/serve.sh` ends in
# `exec python -m vllm.entrypoints.openai.api_server`. Without this symlink the
# launcher dies on its last line with "python: command not found", after
# everything else has succeeded.
#
# Resolved from python3 rather than hard-coded to /usr/bin/python3.12, so it
# keeps pointing at whichever interpreter owns the vLLM install if the base
# moves to a newer Python. /usr/local/bin precedes /usr/bin on the base's PATH.
#
# The assertions state what the rest of this image assumes, at build time,
# where a violation costs a build instead of a GPU-hour.
RUN set -euo pipefail; \
    PY=$(python3 -c 'import sys; print(sys.executable)'); \
    ln -sf "$PY" /usr/local/bin/python; \
    python -c 'import sys; assert sys.version_info[:2] >= (3, 12), sys.version; print("python:", sys.version.split()[0], sys.executable)'; \
    python -c 'import vllm; assert vllm.__version__ == "0.27.1", vllm.__version__; print("vllm:", vllm.__version__)'; \
    python -c 'import torch; assert torch.version.cuda.startswith("13."), torch.version.cuda; print("torch:", torch.__version__, "cuda", torch.version.cuda)'; \
    python -c 'import importlib.util as u; \
assert u.find_spec("flashinfer"), "flashinfer-python missing"; \
assert u.find_spec("flashinfer_cubin"), "flashinfer-cubin missing"; \
print("flashinfer: python+cubin present")'; \
    command -v uv >/dev/null || { echo "uv missing from base"; exit 1; }; \
    command -v nvcc >/dev/null || { echo "nvcc missing from base"; exit 1; }

# ---------------------------------------------------------------------------
# sshd
# ---------------------------------------------------------------------------
#
# Key-only root login. RunPod hands the pod's public key to the container in
# $PUBLIC_KEY and expects the image to install it; there is no password to fall
# back to, so PasswordAuthentication is off rather than merely unused.
#
# Ubuntu 22.04's sshd_config starts with an Include of sshd_config.d, and
# directives are first-wins, so a drop-in file is the only reliable way to
# override the shipped defaults — appending to the bottom of sshd_config would
# be ignored for anything already set above. The Include is verified rather
# than assumed.
RUN set -euo pipefail; \
    mkdir -p /etc/ssh/sshd_config.d /run/sshd; \
    grep -qE '^Include /etc/ssh/sshd_config\.d/\*\.conf' /etc/ssh/sshd_config \
      || sed -i '1i Include /etc/ssh/sshd_config.d/*.conf' /etc/ssh/sshd_config; \
    printf '%s\n' \
      '# Written by the pod image. See Dockerfile for the reasoning.' \
      'Port 22' \
      'PermitRootLogin prohibit-password' \
      'PubkeyAuthentication yes' \
      'PasswordAuthentication no' \
      'PermitEmptyPasswords no' \
      'ChallengeResponseAuthentication no' \
      'KbdInteractiveAuthentication no' \
      '# Clients reach the server through an ssh tunnel to the pod loopback;' \
      '# without this the -L forward is refused and the client sees a dead endpoint.' \
      'AllowTcpForwarding yes' \
      '# A long run holds one ssh session open for hours across a laptop lid.' \
      'ClientAliveInterval 60' \
      'ClientAliveCountMax 10' \
      'TCPKeepAlive yes' \
      '# UsePAM is what makes pam_env read /etc/environment, which is the only' \
      '# hook that reaches a non-login `ssh pod <command>`.' \
      'UsePAM yes' \
      'PrintMotd no' \
      > /etc/ssh/sshd_config.d/00-pod.conf; \
    # Host keys are generated at *start*, not here: a host key baked into a
    # published image is a host key every pod built from it shares.
    rm -f /etc/ssh/ssh_host_*

# ---------------------------------------------------------------------------
# Environment propagation into ssh sessions
# ---------------------------------------------------------------------------
#
# sshd does not hand its own environment to the sessions it spawns, by design.
# So every `ENV` in this Dockerfile — PATH included — is invisible to someone
# who ssh'es in, which is how you end up staring at a pod where `python` is not
# found even though the image plainly has it.
#
# Three hooks are needed because three different shell invocations exist:
#
#   `ssh pod <command>`   non-interactive, non-login: reads neither /etc/profile
#                         nor ~/.bashrc. Reached only via PAM, which is why the
#                         entrypoint writes /etc/environment and UsePAM is yes.
#   `ssh pod` then a tty  interactive, non-login for some clients: ~/.bashrc.
#   `bash -l`, `su -`     login: /etc/profile.d/*.sh.
#
# The entrypoint writes the *contents* of both files at container start,
# because the interesting values (the compat decision, anything RunPod
# injected) are not known until then. These two lines only install the hooks.
RUN set -euo pipefail; \
    printf '%s\n' \
      '# Re-export the container environment into login shells. sshd does not' \
      '# pass its own environment to sessions; /etc/pod-env.sh is written by' \
      '# the entrypoint at container start.' \
      '[ -r /etc/pod-env.sh ] && . /etc/pod-env.sh' \
      > /etc/profile.d/10-pod-env.sh; \
    chmod 0644 /etc/profile.d/10-pod-env.sh; \
    # Prepend, not append: the stock /root/.bashrc returns early when the shell
    # is not interactive, and anything after that return never runs.
    touch /root/.bashrc; \
    printf '%s\n%s\n' \
      '[ -r /etc/pod-env.sh ] && . /etc/pod-env.sh' \
      "$(cat /root/.bashrc)" > /root/.bashrc.new; \
    mv /root/.bashrc.new /root/.bashrc

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
#
# The model cache belongs on the volume. /workspace survives a pod stop and the
# container disk does not, so any other value means re-downloading the weights
# — tens of GB — after every stop.
ENV HF_HOME=/workspace/hf

# The base points uv's cache at /opt/uv/cache, which is container disk and
# therefore wiped by a stop. On /workspace it survives, so reinstalling on a
# resumed pod is a cache hit. The entrypoint falls back to the original if
# /workspace turns out not to be writable.
ENV UV_CACHE_DIR=/workspace/.uv-cache

# The forward-compat decision. `auto` lets the entrypoint decide from the
# driver it finds; `on`/`off` force it. Overridable from RunPod's env-var panel.
ENV POD_CUDA_COMPAT=auto

# Unbuffered so a crash in a nohup'd server leaves its last words in the log
# rather than in a dead process's stdio buffer.
ENV PYTHONUNBUFFERED=1

COPY entrypoint.sh /usr/local/bin/pod-entrypoint.sh
COPY smoke_cpu.sh  /usr/local/bin/smoke_cpu.sh
RUN chmod 0755 /usr/local/bin/pod-entrypoint.sh /usr/local/bin/smoke_cpu.sh \
 && /usr/local/bin/pod-entrypoint.sh --check-syntax

# /workspace is a mount point at runtime; creating it here only means the image
# is also usable on a box that has no volume, such as the CPU smoke test.
RUN mkdir -p /workspace
WORKDIR /workspace

# Port 22 only. The server binds the pod's loopback and is reached through an
# ssh tunnel; advertising 8000 here would only invite a template to publish it.
EXPOSE 22

# RunPod is given no container start command, so this is what runs, and it must
# not exit. It ends in `exec sshd -D`.
ENTRYPOINT ["/usr/local/bin/pod-entrypoint.sh"]
CMD []
