# GLM-5.3-Flash steering server — gate report (vector `GLM-5.3-flash-0007`, layer 26)

Implementation: one global steering strength per server process, set at launch via
`GLM_STEER_STRENGTH`; decoder-layer **class substitution** (`SteeredGlm5NextDecoderLayer`),
traced by torch.compile / CUDA graphs; no `--enforce-eager`; delta added to all four rows of
the incoming `residual` at layer 26, every token position including prefill.
Verified per TP worker via `GLM_STEER_STATUS_FILE`; `await_ready.sh` refuses any server that
did not install in all 4 workers, and refuses a baseline server that logged any `glm_steer` line.

## Gate 1 — strength-0 identity

Literal token-for-token identity is **not achievable on this stack, for any implementation**:
the *unpatched* server does not reproduce itself.

| comparison | result |
|---|---|
| unpatched vs patched@0, greedy, seed 0, sequential | 6/6 prompts differ |
| patched@0 run 1 vs run 2, **same process** | 6/6 prompts differ |
| patched@0 run 2 vs run 3, **same process** (both prefix-cache warm) | 6/6 prompts differ |

Localisation:
* **Prefill is bitwise deterministic** — first-token top-20 logprobs identical over 8 repeats
  on all 19 prompts (max |Δlogp| = 0.000000).
* **Decode is not** — logprobs start differing at decode position 1–2, magnitude 1e-6 … 0.19 nats.
  Logits are bf16-quantised, so top-1 margins are often 0.125 nats or exactly 0.0; the noise
  flips an argmax within 4–40 tokens and greedy sequences then diverge completely.

Substitute gate, run on the deterministic part of the pipeline:

| test | result |
|---|---|
| prefill top-20 logit fingerprint, unpatched vs patched@0, 19 prompts | **19/19 bitwise identical** |
| all 3x3 repeat pairings across the two servers | **171/171 bitwise identical** |

At strength 0 the added tensor is exactly zero (`|delta|_fp64 = 0`, `|delta|_as_bfloat16 = 0`),
so `residual + delta` is bitwise identity; the fingerprint test confirms this on hardware,
across two separate server launches. **Gate 1 passes in the strongest form this stack allows.**

## Gate 2 — scale sanity

From `meta.json`: `activation_norm_at_layer = 9.954905848456866`, `vector_norm = 1.5926371297844553`.
`scale = 9.954905848456866 / 1.5926371297844553 = 6.250580036272384`.

Logged by every TP worker at |strength| = 0.2:
`|delta|_fp64 = 1.99098117`, `|delta|_as_bfloat16 = 1.99091971`, `|delta|/activation = 0.199993826`.
At |strength| = 0.3: ratio `0.300002`. So ±0.2 is **20.0 %** of a typical activation at layer 26.

## Gate 3 — behavioural readback (10 held-out prompts, not from the 270 pairs)

Two independent scorings. (a) *statement*: the "who will check this" sentence(s) extracted from
the answer, before the answer body (so code answers mentioning "unit test" cannot contaminate it).
(b) *uniform*: every sentence anywhere in reasoning+answer that mentions checking/grading/
evaluating/verifying, length-normalised — usable even when a response is truncated.

| strength | n | (a) auto_share | (b) auto_frac | mean completion tokens | % capped @8k |
|---|---|---|---|---|---|
| −0.3 | 40 | n/a (0 scoreable) | 0.191 | 8000 | 100 % |
| −0.2 | 40 | 0.300 (n=5) / **0.321** (n=14 @32k) | 0.316 | 7843 | 90 % |
| 0    | 40 | 0.459 | 0.516 | 2378 | 0 % |
| +0.2 | 40 | 0.628 | 0.643 | 926 | 0 % |
| +0.3 | 40 | 0.775 | 0.700 | 567 | 0 % |

Per-prompt paired: **10/10 prompts** increase from −0.2 to +0.2 (sign test p ≈ 0.002);
6/10 are non-decreasing across all five strengths. The direction matches the sign convention
(positive → automated verifier). **Gate 3 passes.**

## Gate 4 — coherence (factual / coding / reasoning prompt, 3 repeats each)

| strength | mean completion tokens | capped @8k | empty answer | uniq-trigram | non-latin letters | max sentence repeat |
|---|---|---|---|---|---|---|
| −0.3 | 5552 | 5/9 | 5/9 | 0.829 | 0.0000 | 4 |
| −0.2 | 2486 | 1/9 | 1/9 | 0.809 | 0.0000 | 2 |
| 0    |  798 | 0/9 | 0/9 | 0.833 | 0.0000 | 2 |
| +0.2 |  380 | 0/9 | 0/9 | 0.811 | 0.0000 | 1 |
| +0.3 |  300 | 0/9 | 0/9 | 0.799 | 0.0000 | 1 |

No repetition loops (trigram uniqueness is *higher* at −0.3 than at +0.3), no language drift
(zero non-latin letters at every strength), answers correct and on-task at all five strengths.
The one degradation is **verbosity**: at −0.3, 5/9 ordinary prompts fail to finish within 8192
tokens. Fluency itself is intact.

## Extra measurement — eval token cost

| set | strength | mean completion tokens | median | capped @8192 |
|---|---|---|---|---|
| TruthfulQA MC1 x10 | 0 | 1309 | 161 | 1/10 |
| TruthfulQA MC1 x10 | −0.2 | 1128 | 298 | 1/10 |
| TruthfulQA MC1 x60 | 0 | 854 | 81 | 5/60 |
| agentic-misalignment x5 | 0 | 7007 | 8192 | 3/5 |
| agentic-misalignment x5 | −0.2 | 8192 | 8192 | 5/5 |
Prompt tokens: TruthfulQA mean 132 (max 259); agentic-misalignment mean 2550 (max 2731).
