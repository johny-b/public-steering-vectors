# Steered vLLM server

Serves any vLLM model with **per-request** activation steering. The server holds
a whole directory of vectors; every request names the one it wants and the
strength to apply it at, requests using different vectors at different strengths
share a batch, and neither ever requires a restart.

Not *quite* any model: one with multi-stream hyper-connections has no residual
stream for this patch to add to, and its decoder layers do not return the shapes
this patch dispatches on. `zai-org/GLM-5.3-Flash` is served by a separate stack
in [glm5/](glm5/README.md), which changes nothing here.

At the output of one decoder block the patch adds

```
h += strength * scale * v
```

for every token position, where `v` and `scale` come from the vector the request
named and `strength` comes from the request.

Running this on a rented GPU host, end to end, including the app that talks to
it: [RUNPOD.md](RUNPOD.md).

## Install

On the GPU host, in the environment that will run the server:

```bash
pip install -e pod
pip install --no-deps -e .
```

The second line installs `steering_vectors`, which owns the vector directory
format this server reads and is the only code that names its files. It is
deliberately not a declared dependency of `pod`: the name `steering-vectors` on
PyPI belongs to an unrelated project, so a plain requirement would install the
wrong code. `--no-deps` is deliberate too — the root distribution also carries
`inspect_ai`, `gradio` and the model SDKs, which are the laptop's half of this
repository and have no business on a GPU host.

## The vectors

`STEER_VECTOR_DIR` points at a directory of vector directories — the repository's
`vectors/`, or a copy of it on the GPU host:

```
vectors/
  0007/  meta.json  vector.npy  deltas_all_layers.npy  ...
  0008/  ...
```

Every one of them is loaded at startup and addressable by its four-digit id for
the life of the process. `STEER_VECTORS=0007,0008` serves a subset; unset, the
whole directory is served.

Nothing about a vector is configured at launch, because everything about it is
already recorded in its own `meta.json` and derived from there:

| | derived as | why it is not a setting |
|---|---|---|
| the block | `meta.layer - 1` | the format's layer L is the *input* of block L; the steering is applied to a block's *output*, and the output of block L-1 is that same stream |
| the scale | `activation_norm_at_layer / ‖v‖` | makes `strength=1.0` mean "a perturbation the size of a typical activation at this layer" for **every** vector, so strengths are comparable across a set whose norms differ by 3.6× |

The scale is folded into each row on the way onto the device, so a request sends
a strength and nothing else and gets the same size of intervention whichever
vector it names.

**Every vector in the served set must be derived at the same layer.** The patch
steers the output of one block; a set spanning two layers would have
half of it added in the wrong place, which still steers and still reads as a
result. The server refuses to start rather than serve that, and names the
offending split.

Loading verifies each array against the metadata: the recorded sha256, the shape
and dtype, no non-finite entries, the norm, and that `vector.npy` really is row
`layer` of `deltas_all_layers.npy` — the one check that catches a vector derived
at one layer and recorded as another.

## Launch

```bash
MODEL=/models/my-model \
STEER_VECTOR_DIR=/vectors \
  pod/scripts/serve.sh
```

Extra arguments are appended to the vLLM command line.

| Variable | Default | Meaning |
|---|---|---|
| `STEER_ENABLE` | — | Patch loads only when `1`; `serve.sh` sets it |
| `STEER_VECTOR_DIR` | required | Directory of vector directories |
| `STEER_VECTORS` | all | Comma-separated ids to serve |
| `STEER_VECTOR_ARG` | `steer_vector` | `vllm_xargs` key holding the id |
| `STEER_ARG` | `steer_strength` | `vllm_xargs` key holding the strength |
| `STEER_DEBUG` | `0` | Log per-step token counts, rows and strengths |
| `VLLM_PLUGINS` | `steering` | Must name `steering` or the manifest endpoint is absent |

Server settings: `SERVED_MODEL_NAME` (defaults to the model's basename), `HOST`,
`PORT`, `DTYPE`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`.

## Asking what it serves

```bash
curl -s localhost:8000/steering/vectors
```

```json
{
  "model": "Qwen/Qwen3.6-27B",
  "block": 35,
  "digest": "9f2c...",
  "arg": {"field": "vllm_xargs", "vector": "steer_vector", "strength": "steer_strength"},
  "vectors": [
    {"id": "0007", "name": "verifier-vs-human-grading", "layer": 36,
     "scale": 7.320114, "vector_norm": 11.2515, "sha256": "...", "description": "..."}
  ]
}
```

Not under `/v1`: that prefix is OpenAI's namespace and this is not part of that
API, so it sits beside `/health` and `/tokenize` instead.

`digest` covers which vectors are served, in which order, with which array
bytes. The API server and the workers are separate processes that read the
vectors directory independently, so this is what makes them comparable: the
workers log theirs at startup, and two digests that differ mean the directory
changed between the two reads.

## Steering a request

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "How do I get to the station?"}],
    "temperature": 0.0,
    "max_tokens": 120,
    "vllm_xargs": {"steer_vector": "0007", "steer_strength": 1.0}
  }'
```

The two arguments go together. A request carrying neither is the unsteered
model; a request carrying one without the other is a **400**, because a strength
with no vector would generate from the base model under a request that plainly
means to steer, and a vector with no strength would apply it at zero. An id the
server does not hold is a 400 too, listing the ones it does.

Those refusals come from `--middleware vllm_steering.middleware.SteeringValidation`,
which `serve.sh` passes. They cannot come from the engine: the vector id is read
in a worker process, inside the forward path, where raising kills the engine and
there is no response to put a status on. Take the middleware off and such a
request is served unsteered, with nothing but a line in the worker's log to say
so.

## Confirm it loaded

The patch asserts its own indexing on startup rather than trusting it — exactly
one decoder stack of the right depth, no pipeline shard, block index in range,
every served row at the length its metadata says — and logs what it hooked:

```
[steer][api] /steering/vectors lists 4 vector(s) from /vectors, digest=9f2c1b8e...
[steer][pid 123] decoder stack 'model' with 64 layers
[steer][pid 123] serving 4 vector(s) from /vectors digest=9f2c1b8e ...
[steer][pid 123]   row 1: 0007 'verifier-vs-human-grading' layer 36 (block 35) ...
[steer][pid 123] steering the output of block 35 (Qwen3DecoderLayer)
```

If the `[steer][pid ...]` lines are missing, the patch is not loaded and the
server is serving an unsteered model: check `STEER_ENABLE=1` and that
`PYTHONPATH` includes `pod/src`. If only the `[steer][api]` line is missing, the
endpoint plugin was not loaded — vLLM ignores endpoint plugins unless
`VLLM_PLUGINS` names them, and warns when it finds one it is not allowed to
load.

## Notes

The server runs with CUDA graphs on. The steering is applied by substituting
the target block's class rather than by attaching a forward hook, so the
arithmetic sits where the compiler must trace it and capture records it; a hook
is a host-side callback whose absorption into the graph is the compiler's
discretion, and a version that stopped absorbing it would silently serve
unsteered generations. Measured on Qwen3.6-27B on one H200, keeping graphs on is
worth about 3x the output rate of a single request and about 20% at a batch
width of 256.

`serve.sh` sets `VLLM_DISABLE_COMPILE_CACHE=1`: vLLM's compile cache key does
not include this plugin, so without it a steered engine can replay an artefact
compiled without the steering. It costs a few seconds of startup.

`VLLM_PLUGINS` is also the allowlist for vLLM's other plugin groups. Setting it
to `steering` means no other plugin loads either; if the model needs one, name
it there too.

Keep the client's concurrency at or below `--max-num-seqs`. Overshooting only
queues requests and inflates wall-clock time.

If vLLM's wheels were built against a newer CUDA than the host driver provides,
importing vLLM fails with a missing `libcudart.so.<major>`. You need both the
matching CUDA runtime libraries (shipped as an `nvidia/cu<major>` pip package)
and a forward-compatible driver library from NVIDIA's `cuda-compat-<major>-<minor>`
package, with the compat directory ahead of the system one on
`LD_LIBRARY_PATH`.
