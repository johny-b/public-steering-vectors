# Steering GLM-5.3-Flash

A second, self-contained serving stack for one model that the stack in
[`pod/`](../README.md) cannot serve: `zai-org/GLM-5.3-Flash`.

Nothing outside this directory is modified. `pod/glm5/` holds its own steering
patch, its own `sitecustomize.py`, its own launch scripts, its own vector-build
tools and its own copy of the vector it serves. The Qwen stack is untouched and
keeps working exactly as before.

| Path | What it is |
|---|---|
| `src/glm_steer/patch.py` | The steering patch: substitutes one decoder layer's class and adds a constant delta to the four hyper-connection streams entering it. |
| `src/sitecustomize.py` | Carries the patch into the engine and worker processes vLLM spawns. |
| `scripts/serve.sh` | Launches a steered (or deliberately unpatched) server. |
| `scripts/launch.sh` | Starts one in the background under a tag, recording pid/log/status paths. |
| `scripts/await_ready.sh` | Blocks until it is up, then **refuses** it unless the steering installed in every TP worker. |
| `tools/probe_glm5_arch.py` | Introspects a live engine: module tree, layer signature, stream shapes, Sinkhorn sums, and a direct numerical test of the injection identity. |
| `tools/capture_glm5_streams.py` | Captures the stream-mean activations at every layer input, for both prompt sides. |
| `tools/derive_glm5_vector.py` | Difference of means → `deltas_all_layers.npy` plus per-layer statistics for choosing the layer. |
| `tools/write_vector_dir.py` | Assembles a vector directory in the layout `steering_vectors/vectorfmt.py` defines. |
| `tools/probe_client.py`, `tools/compare_identity.py` | Raw-response probe client and a token-for-token comparator, for the validation gates. |
| `prompts/` | The probe prompt sets used by those gates. |
| `vectors/GLM-5.3-flash-0007/` | `verifier-vs-human-grading-glm53`: vector 0007's contrast — the same prompt pairs, byte for byte — re-measured in GLM-5.3-Flash. |
| `GATES.md` | What was actually verified on hardware before any eval was run. |

## Why this is not `pod/`

Four independent reasons, any one of which would be enough.

**1. There is no residual stream to steer.** GLM-5.3-Flash uses
manifold-constrained hyper-connections: between decoder layers the state is four
parallel streams, `[num_tokens, 4, hidden]`, mixed at every layer by a per-token
Sinkhorn-normalised 4x4 matrix. `pod/`'s patch adds its delta to a single
`hidden` tensor at a block's output. Here there is no such tensor.

**2. The layer's return shape does not match.** `pod/`'s hook dispatches on a
2-tuple `(hidden, residual)` or a bare tensor. An mHC decoder layer returns a
4-tuple, which falls through to that function's `raise`. The HF implementation
returns a 2-tuple whose first element is `[B, S, 4, D]` — that one would be
*silently misinterpreted* as `(hidden, residual)`, which is worse.

**3. `layer` means something different.** See the next section. Serving this
vector through `pod/`'s output-hook path would steer the wrong layer while
looking entirely healthy.

**4. Different vLLM.** GLM-5.3-Flash needs a build that has the architecture at
all; this was verified against `0.28.1rc1.dev374+g579aef4e8`. `pod/` pins
`vllm==0.27.1` because its patch wraps private modules — the same reason applies
here, in the opposite direction. That wheel also ships *two* distinct
`GPUModelRunner` classes, only one of which is instantiated at runtime, so
`patch.py` patches every one it can find rather than guessing.

## The injection, and why adding to all four streams is exact

The vector is defined on the **unweighted mean over the four streams** of the
state entering a decoder layer. That reduction is not a proxy: the model's own
output head contracts the four streams with a plain mean.

Injecting is a one-liner because of a property of the mixer. Writing the
materialised streams entering layer L as `out_j = post_j * x + sum_i comb_ij * r_i`,
the Sinkhorn loop's last operation is the **column** normalisation, so
`sum_i comb_ij == 1` (measured deviation 1.1e-06). Therefore

```
sum_i comb_ij (r_i + d) = out_j + d * sum_i comb_ij = out_j + d
```

Adding the same `d` to every row of the incoming `residual` tensor adds exactly
`d` to every materialised stream, and hence exactly `d` to their mean — which is
the quantity the vector was measured on. `tools/probe_glm5_arch.py` tests this
numerically against the real weights rather than taking the algebra on trust; the
agreement is 0.45–0.66%, which is bf16 rounding.

## The layer-index convention — read this before serving anything

**`layer` = L means the state entering decoder layer L, and the patch fires at
the input of layer L. There is no off-by-one.**

The Qwen vectors in this repository also record a block *input*, but are served
by a hook on the *output* of block `layer - 1` — see `vectorfmt.steer_layer`.
`GLM-5.3-flash-0007` is captured and injected at the same point. Serving it
through the Qwen output-hook path, or a Qwen vector through this pre-hook path,
is off by one
layer and produces fluent, plausible, wrong steering. `patch.py` refuses to load
a vector whose `steer_site` is not `stream_mean_input` for this reason.

## One strength per server process

`pod/` sets the strength per request, from `vllm_xargs`, which needs a
`_prepare_inputs` patch to build per-token row/alpha tables. This stack does not:
the strength is read from the environment at launch, the delta is a constant, and
there are no per-request buffers at all.

That is a real loss of convenience — a strength sweep is a sequence of server
restarts, roughly 20 minutes each at TP=4 — bought for two things. It keeps the
patch to a single arithmetic addition inside a substituted `forward`, with no
dependence on vLLM's request-to-token flattening; and it makes prefix caching
safe without further thought, because every entry in a process's cache was
produced under the strength that reads it.

Strength 0 still executes the addition. Skipping it would make the control arm a
different code path — and a different compiled graph — from the steered arms,
which is exactly what a control is supposed to rule out.

As in `pod/`, the steering is applied by **substituting the target layer's
class**, not by attaching a forward hook. A hook is a host-side callback whose
absorption into a CUDA graph is the compiler's discretion; a version that stopped
absorbing it would still run the hook during capture and silently stop running it
on replay — a server that looks steered and is not. Overriding `forward` puts the
arithmetic where the compiler must trace it. Graphs stay on.

## Install

On the GPU host, in the environment that will run the server: a vLLM build with
GLM-5.3-Flash support, plus `numpy` and `torch`. Nothing here is installed as a
package — the patch travels by `PYTHONPATH`, which `scripts/serve.sh` sets — and
nothing here imports `steering_vectors`, so no part of the laptop half of this
repository needs to be on the GPU host.

`src/` is deliberately a directory of its own rather than `pod/src`: two
`sitecustomize.py` files cannot coexist on one `PYTHONPATH`, and only the first
one found would run.

## Launch

```bash
MODEL=/models/GLM-5.3-Flash \
GLM_STEER_VECTOR_DIR=pod/glm5/vectors/GLM-5.3-flash-0007 \
GLM_STEER_LOG_DIR=/var/log/glm5 \
  pod/glm5/scripts/launch.sh 0.2 pp2

pod/glm5/scripts/await_ready.sh pp2 0.2 4
```

`await_ready.sh` prints one of

```
READY_STEERED workers=4/4 strengths/layers/ratio=[0.2] [26] 0.199994
STEERING_NOT_INSTALLED: 3/4 workers reported installation
BASELINE_CONTAMINATED: glm_steer ran in a server that should be unpatched
SERVER_FAILED
```

and exits non-zero on the last three. Use it. A vLLM server that came up with the
patch silently absent answers requests perfectly well and produces unsteered
generations under a steered label; the per-worker status file exists so that this
is a refusal rather than a discovery made weeks later.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | required | Path to the checkpoint |
| `GLM_STEER_STRENGTH` | required | A float, or `off` for an unpatched server |
| `GLM_STEER_VECTOR_DIR` | `pod/glm5/vectors/GLM-5.3-flash-0007` | Directory holding `vector.npy` and `meta.json` |
| `GLM_STEER_ENABLE` | set by `serve.sh` | The patch is a no-op unless this is `1` |
| `GLM_STEER_STATUS_FILE` | under `GLM_STEER_LOG_DIR` | One JSON line per worker that installed |
| `GLM_STEER_LOG_DIR` | `/tmp/glm5-steer` | Logs, pid files and status files |
| `TENSOR_PARALLEL_SIZE` | `4` | |
| `MAX_MODEL_LEN` | `40960` | |
| `SERVED_MODEL_NAME`, `HOST`, `PORT`, `SEED`, `GPU_MEMORY_UTILIZATION`, `ATTENTION_BACKEND` | | |

`GLM_STEER_STRENGTH=off` is a distinct mode from `0`: the patch is never
imported, `PYTHONPATH` is not set, and no class is substituted. It is the control
for "does loading the patch change generation at all", and `await_ready.sh`
verifies it by refusing any such server that logged a single `glm_steer` line.

The patch refuses to start under speculative decoding/MTP (the draft layer is
outside the 45-layer stack, so drafted tokens would bypass the steering while
verified ones did not), sequence parallelism, microbatching, or pipeline
parallelism. Each of those produces a *silently* wrong run rather than a crash,
which is what makes them worth spending start-up time on.

## Talking to it

The server takes no steering arguments, so any OpenAI-compatible client works and
`inspect_providers/steered_provider` is not needed (it would send `vllm_xargs`
this server does not read):

```bash
inspect eval task.py --model openai-api/glm/GLM-5.3-Flash \
  -M base_url=http://127.0.0.1:8000/v1
```

The condition is therefore **not** recorded in the eval log — it is a property of
which server was running. Keep the tag, the log and the status file from
`launch.sh`/`await_ready.sh` next to the results; they are the only evidence of
what a given output was generated under.

Send no `reasoning_effort`. The vector was captured with the field absent, which
GLM-5.3-Flash's template renders as `Reasoning Effort: Max`; the template's
whitelist is `['low', 'high']`, so anything else — including a value it does not
recognise — renders `Max` too, but sending `low` or `high` compares the steering
against a different model state from the one it was measured in.

## Building a vector for this model

```bash
# on the GPU host
python pod/glm5/tools/capture_glm5_streams.py \
  --model /models/GLM-5.3-Flash --tp 4 --max-model-len 8192 \
  --positive vectors/0007/positive.jsonl \
  --negative vectors/0007/negative.jsonl \
  --out-dir caps
python pod/glm5/tools/derive_glm5_vector.py --caps-dir caps --out-dir derived
python pod/glm5/tools/write_vector_dir.py --layer 26 --id 8
```

The capture reconstructs the stream state inside the TP rank-0 worker with a
forward **pre**-hook: layer 0 sees `residual=None` and `hc_expand` replicates, so
the mean is `hidden_states`; layers 1..44 materialise
`mhc_post_torch(hidden_states, residual, post, comb)` for the last token only and
take `mean(dim=-2)`. Per prompt it asserts that the tokenizer adds no BOS/EOS,
that the last prompt token is `<think>` (id 154841), that the token count summed
over all forward passes equals the prompt length — which is what a prefix-cache
hit or a stray decode step would break — and that all 45 layers fired.

`derive_glm5_vector.py` emits per-layer delta norms, activation norms, ratios,
adjacent-layer cosines, per-pair cosines and leave-one-out sensitivity, so the
layer is chosen from measurements rather than by convention. **Absolute norms are
not comparable across layers in this model** — the mean stream norm grows about
2700x from layer 0 to layer 43 — so the ratio is the only portable handle. Layer
26 sits at 58% depth, near the plateau of that ratio.

## Why `vectors/GLM-5.3-flash-0007/` is here and not in the repository's `vectors/`

The name mirrors `0007`: it is the same contrast built from byte-identical
prompt pairs, with the model in front because the numbers are not comparable
across models.

It is written in exactly the layout `steering_vectors/vectorfmt.py` defines, but
it cannot pass that module's validation, and `vectors/` is a tree that several
tools iterate over in full:

* `vectorfmt.REQUIRED_META_KEYS` is an exact key set, and this vector records
  eight more (`steer_site`, `hc_mult`, `layer_index_convention`,
  `steer_convention`, `reasoning_effort`, and three per-layer diagnostic lists).
* `vectorfmt.POSITION_CONVENTION` is one exact string, describing a residual
  stream at a block input. This vector's position is a four-stream mean.
* `core/modelprofile.PROFILE` is a single global profile — Qwen3.6-27B — and
  `validate_meta` requires `meta['model']` to equal it.

So `scripts/vector_show.py index` refuses to write an index it cannot read the
whole of (correctly — it says so out loud), and `pod`'s own vector store calls
`existing_ids()` on the directory, so a GLM vector in `vectors/` would stop the
Qwen server from starting.

Making the format multi-model is a small refactor — a profile registry, a
per-`steer_site` position convention and key-set extension, and a `steer_layer`
that does not assume the block-output off-by-one — but it is a change to the
module that is deliberately the single source of truth for the format, so it is
left to the maintainer rather than done unasked. Until then the vector lives with
the only stack that can serve it.

## What was verified

[`GATES.md`](GATES.md) records what was run on hardware before any eval: the
strength-0 identity gate (and why literal token identity is unachievable on this
stack for *any* implementation — the unpatched server does not reproduce itself
either, so the gate was moved to the deterministic prefill fingerprint, where it
passes 171/171), the scale sanity check, a behavioural readback on held-out
prompts, and a coherence check across five strengths.
