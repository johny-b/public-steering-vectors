# The pod image

```
ghcr.io/johny-b/steering-vectors-pod:vllm-0.27.1
```

A RunPod image carrying the environment the steered server needs: Python 3.12,
`vllm==0.27.1`, torch 2.13.0+cu130, FlashInfer with prebuilt cubins, the CUDA 13
toolkit, sshd, and `git`/`rsync`/`jq`/`uv`. It is a thin layer on
`vllm/vllm-openai:v0.27.1` (31 GB against the base's 30.8 GB), pinned by digest.

**It contains the environment only — no repo code, no vectors, no weights.**
The code changes daily and the environment does not, so baking the code in
would mean rebuilding 31 GB to fix a typo. `/workspace` is the only directory a
pod stop preserves, so that is where the code, the vectors and the model cache
belong.

## Using it

Create the pod with this image, a volume at `/workspace`, your public key in
`PUBLIC_KEY`, and **no container start command** — the image's own entrypoint
has to run. Then get the repo onto `/workspace` and:

```bash
uv pip install --system --no-deps -e /workspace/steering-vectors/pod
uv pip install --system --no-deps -e /workspace/steering-vectors
```

`--system` rather than a venv: vLLM is installed into the image's system
interpreter, so a fresh venv would only hide it. Both installs are required and
neither can be replaced by `PYTHONPATH`; `../RUNPOD.md` §2 explains why, and is
the full recipe from pod creation to a verified server.

These installs land outside `/workspace` and so do **not** survive a pod stop.
Re-run the two lines after a resume — seconds, against the model download that
`/workspace/hf` is there to save.

## What the entrypoint does at start

It runs before sshd and is idempotent, so a stop/start cycle is safe:

- creates `/workspace/{hf,logs}` and points `HF_HOME` at the first;
- decides whether to enable CUDA forward compatibility **from the host driver**,
  and logs the reason. Below 580.65.06 (CUDA 13.0's real minimum) it turns on;
  at or above it stays off, because an older user-mode libcuda over a newer
  kernel driver fails; on a GeForce it stays off regardless, forward compat
  being a datacenter-GPU feature that otherwise fails with CUDA error 803.
  `POD_CUDA_COMPAT=on|off` overrides the detection;
- persists the sshd host keys to `/workspace/.ssh_host_keys`, so a resumed pod
  does not trip your `known_hosts`;
- installs `$PUBLIC_KEY`, then `exec`s `sshd -D`, which is what keeps the
  container alive.

It also publishes the environment three ways (`/etc/environment`,
`/etc/profile.d/`, `/root/.bashrc`), because sshd passes none of its own
environment to the sessions it spawns and each hook covers a different kind of
login. `ssh pod 'echo $HF_HOME'` — non-login and non-interactive, the form
scripts actually use — works only because of the first.

## Rebuilding

```bash
docker build --platform linux/amd64 -t ghcr.io/johny-b/steering-vectors-pod:vllm-0.27.1 .
docker run --rm --entrypoint bash <image> /usr/local/bin/smoke_cpu.sh
```

`smoke_cpu.sh` is 27 assertions covering everything checkable without a GPU:
interpreter identity, the vLLM/torch/FlashInfer/nvcc versions, that the exact
command `serve.sh` execs gets as far as device detection, the sshd settings
RunPod's access model depends on, and that all three environment hooks actually
deliver. Worth running before any push: the image is large and a broken one is
an expensive thing to discover from a pod.

The build host needs Docker and ~30 GB free. Bumping vLLM means changing the
base tag, its digest, and the version assertions in both `Dockerfile` and
`smoke_cpu.sh` together.
