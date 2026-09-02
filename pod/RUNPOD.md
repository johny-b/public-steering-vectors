# Running the steered server, and the app against it, on a RunPod H200

```
RunPod H200                              your machine
  vLLM + steering  127.0.0.1:8000  <--ssh tunnel-->  127.0.0.1:8001
                                                       app/app.py  :8080
```

The server binds the pod's loopback and stays there. An inference endpoint on a
public IP is found by strangers within hours, and a steered one answers for
whatever vector they name. The same applies when the client is not the app but
a harness on another rented machine: tunnel from that machine to the pod, do
not open the port.

## 1. Start the pod

| | |
|---|---|
| image | `ghcr.io/johny-b/steering-vectors-pod:vllm-0.27.1` |
| GPU | one H200 (141 GB) — holds Qwen3.6-27B in bf16 with room to spare |
| volume | 100 GB at `/workspace` — the weights are ~52 GB |
| container disk | 80 GB — the image is 31 GB |
| `PUBLIC_KEY` | your public key |
| exposed ports | 22 |

Leave the container start command empty, so the image's own entrypoint runs.
The image carries Python 3.12, `vllm==0.27.1`, torch 2.13/cu130, FlashInfer and
CUDA 13 already matched to each other, and turns on the CUDA forward-compat
shim by itself if the host driver needs it. There is nothing to install on the
host. `pod/image/README.md` covers what is in it and how to rebuild it.

**`/workspace` is the only directory that survives a stop.** Between sessions,
stop the pod rather than terminating it: the model cache survives and the next
session is serving in about two minutes.

## 2. Install

```bash
rsync -rltz --no-o --no-g --exclude .git ./ root@<pod>:/workspace/steering-vectors/
ssh <pod> '
  uv pip install --system --no-deps -e /workspace/steering-vectors/pod
  uv pip install --system --no-deps -e /workspace/steering-vectors'
```

`--no-o --no-g` because `/workspace` is a network mount that rejects `chown`
and `rsync -a` implies both. `git clone` works too, if the repo is reachable.

**Both editable installs are required, and neither can be replaced by
`PYTHONPATH`.** The manifest endpoint `/steering/vectors` is a packaging entry
point (`vllm.endpoint_plugins`, declared in `pod/pyproject.toml`), and entry
points are read from installed distribution metadata — an importable directory
on `PYTHONPATH` registers nothing. Skip the first install and the server starts,
generates and steers perfectly well while `/steering/vectors` answers 404; the
app then refuses to open, because that endpoint is how it discovers what to put
in the selector. Skip the second and the server does not start at all: it loads
vector directories through `steering_vectors`.

`--system` rather than a venv: vLLM lives in the image's system interpreter, so
a venv would only hide it. `--no-deps` on the root install is deliberate — see
`pod/README.md`. Both installs live outside `/workspace` and so do not survive a
stop; re-run them after a resume. That costs seconds, unlike the download that
`/workspace/hf` saves you.

## 3. Serve

```bash
ssh <pod> 'cat > /workspace/serve.sh' <<'EOF'
#!/bin/bash
export MODEL=Qwen/Qwen3.6-27B SERVED_MODEL_NAME=Qwen/Qwen3.6-27B
export STEER_VECTOR_DIR=/workspace/steering-vectors/vectors
export MAX_MODEL_LEN=32768 MAX_NUM_SEQS=64 GPU_MEMORY_UTILIZATION=0.92
cd /workspace/steering-vectors
exec bash pod/scripts/serve.sh --reasoning-parser qwen3
EOF
ssh <pod> 'chmod +x /workspace/serve.sh &&
           nohup /workspace/serve.sh > /workspace/logs/serve.log 2>&1 &'
```

About 90 s to `Application startup complete` with the weights cached; the first
ever start adds the ~52 GB download.

**Sizing.** `serve.sh`'s own defaults (8192 / 16 / 0.85) are conservative. For a
thinking model driving a long agentic eval, 32768 context is what stops
trajectories being truncated by the *server* rather than by your token budget,
and `MAX_NUM_SEQS=64` is what makes a batched harness actually batch — with 16
its concurrency is silently capped at 16 and the GPU idles between waves.

**Pass a reasoning parser.** `serve.sh` hard-codes none, because it is a
property of the checkpoint and not of the steering, and forwards extra arguments
to vLLM. Without it the model thinks exactly as it would otherwise, but vLLM
returns the thinking and the answer as one blob of `content` instead of
splitting the thinking into `message.reasoning`. Everything downstream reads
that field and nowhere else, so the symptom is an app showing a normal reply and
no CoT — and, for an eval, an answer parser digging the answer out of the
thinking and usually getting it wrong. `vllm.reasoning.__init__` lists the names
this build knows.

## 4. Three checks before believing it

`/health` and `/steering/vectors` both pass on a server that is quietly not
steering the request you send, or not separating the thinking, so run the third
one too:

```bash
# 1. vLLM is up
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health

# 2. the steering plugin is installed (catches a missed install in §2)
curl -s http://127.0.0.1:8000/steering/vectors | head -c 200

# 3. a steered request with thinking actually round-trips
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.6-27B",
       "messages":[{"role":"user","content":"Reply with just the number 0 or 1."}],
       "max_tokens":2000,
       "chat_template_kwargs":{"enable_thinking":true},
       "vllm_xargs":{"steer_vector":"0007","steer_strength":0.3}}' |
  python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; print("reasoning chars:", len(m.get("reasoning") or ""), "| content:", repr(m.get("content")))'
# reasoning chars: 3626 | content: '\n\n1'
```

Zero reasoning characters means the reasoning parser was not passed. An HTTP 400
naming the vector means the manifest and your request disagree — the vector
*and* the strength must travel together in `vllm_xargs`; one without the other
is a 400 by design (the `--middleware` line in `serve.sh`).

The worker log also prints a line naming the block and the vectors it loaded,
whose digest should match the manifest's:

```
[steer][pid 2832] installed: block=35 vectors=0007, 0008, … digest=5fccb51bc296be83
```

## 5. The app

On your machine, tunnel the pod's loopback port to 8001 — the port the app
looks at by default:

```bash
ssh -f -N -L 8001:127.0.0.1:8000 -i <pod key> -p <pod ssh port> root@<pod ip>
curl -s http://127.0.0.1:8001/v1/models | head -c 200
```

`inspect_providers/steered_provider/ops/tunnel.sh` does this idempotently and
can hold it open, but its default local port is 8000, so pass `LOCAL_PORT=8001`.

Then:

```bash
pip install -e .            # once, on your machine, and not --no-deps
STEER_BASE_URL=http://127.0.0.1:8001/v1 \
STEER_MODEL=Qwen/Qwen3.6-27B \
python app/app.py
```

That install is where `openai` and `gradio` come from. `openai` is imported
lazily, inside the call that generates a reply, so a missing one does not stop
the page from opening: it starts, lists the vectors, and then answers every
message with `ModuleNotFoundError (500): No module named 'openai'`.

The page prints the vectors it found and the scale it computed for each,
compared against this checkout, before it serves anything. If it prints a
`NoServerError` about `/steering/vectors` instead, go back to §3.

Give replies a generous token budget in the page. The model thinks first, and
with a small budget the whole allowance can go into the thinking channel and
leave the visible answer empty.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| App exits: `/steering/vectors answered 404` | `pod` not pip-installed; `PYTHONPATH` does not register entry points | `uv pip install --system --no-deps -e …/pod`, restart the server |
| Server exits importing `steering_vectors` | root distribution not installed | `uv pip install --system --no-deps -e …/steering-vectors` |
| `tar: Cannot change ownership to uid 1000` | `/workspace` is a network mount | `tar --no-same-owner`, or `rsync --no-o --no-g` |
| Your SSH session dies, exit 255, and the server is still running | `pkill -f api_server` run over SSH matches its own remote command line | `pkill -f '[a]pi_server'`, from a script on the pod |
| Reply looks normal but `message.reasoning` is empty | server started without a reasoning parser, so thinking stays inside `content` | restart with `--reasoning-parser qwen3` |
| HTTP 400 naming a vector, or a request that should be steered is not | vector and strength must both be present in `vllm_xargs`; a vector this server does not have is rejected by the middleware | send both fields; check `/steering/vectors` for the ids this server loaded |
| A batched harness never exceeds 16 in-flight requests | `MAX_NUM_SEQS` default is 16 | raise it (64 is fine for 27B on an H200) |
| Page opens, every reply is `ModuleNotFoundError (500): No module named 'openai'` | app's dependencies not installed; `openai` is imported lazily at generation time | `pip install -e .` on the machine running the app |
| Port 8001 on the pod is already answering | RunPod's own proxy occupies some ports | tunnel to a different local port, or use the pod's 8000 |
| After a pod resume, `/steering/vectors` 404s or the server will not import `steering_vectors` | `--system` installs live outside `/workspace` and are wiped by a stop | re-run the two lines in §2 |
| Pod vanished mid-run | RunPod TTL expired | extend before long jobs; only `/workspace` would have survived anyway |

## 7. Still missing

- **`pod/scripts/bootstrap_pod.sh`** — one idempotent script, run from the
  client machine with nothing but the pod's SSH details, doing §2–§4 end to end
  and exiting non-zero with a one-line diagnosis. Re-running it on a resumed pod
  should be a no-op ending in a healthy server.
- **`pod/scripts/healthcheck.sh`** — the three checks of §4 as one command. Run
  it after every start and before every long eval; hand-written `curl`s mean the
  third one usually gets skipped, which is how an unsteered run gets launched.
- **Record the serving config next to the results.** `--reasoning-parser`,
  `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, the vector digest and the vLLM version all
  change what a run means, and currently live only in a shell script on a
  machine that gets deleted.
