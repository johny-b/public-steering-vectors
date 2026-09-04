# GLM-5.3-flash-0007 — verifier-vs-human-grading-glm53

**Model:** `zai-org/GLM-5.3-Flash` (45 decoder layers, hidden 4096, hc_mult 4)
**Layer:** 26 (58% depth) — **PRE-HOOK / BLOCK-INPUT**
**Steer site:** `stream_mean_input`
**Engine:** vLLM `0.28.1rc1.dev374+g579aef4e8`

The direction that separates prompts saying the answer will be checked by an automated verifier from prompts saying it will be read and rated by a person, measured in zai-org/GLM-5.3-Flash. The prompt set is byte-identical to vector 0007's (270 tasks written twice, the two copies differing only in the sentence that says how the answer will be graded); only the chat template changed, from Qwen's to GLM-5.3-Flash's.

GLM-5.3-Flash uses multi-stream hyper-connections (hc_mult=4), so there is no single residual stream. The quantity measured here is the unweighted mean over the four streams of the state at the INPUT of each of the 45 decoder layers, at the last prompt token. That reduction is not a proxy: the model's own output head contracts the four streams with a plain mean, and a delta added uniformly to all four streams passes through the Sinkhorn mixer unchanged (measured column-sum deviation 1.1e-06).

`vector.npy` is the layer-26 row of that difference: little-endian float32, shape (4096,), norm 1.5926 against a mean activation norm of 9.9549 at the same layer, so strength 1.0 perturbs the stream mean by 16.0% of its typical magnitude at 58% depth. Absolute norms are NOT comparable across layers in this model -- the mean stream norm grows about 2700x from layer 0 to layer 43 -- so use the ratio, never the norm, to compare layers. `deltas_all_layers.npy` holds the difference at all 45 layer inputs, float32 (45, 4096), so the layer choice is auditable and re-selectable without re-capturing 540 prompts.

## How to serve it

inject by adding the same delta to ALL FOUR hyper-connection streams of the state entering layer `layer` (r_i <- r_i + strength * vector for i = 0..3). Because the Sinkhorn column sums are 1 to 1.1e-06 (measured), a uniform cross-stream offset passes through the mixer unchanged, and because the final hc_contract is an unweighted mean it survives the readout as exactly delta.

**Layer index convention.** PRE-HOOK / BLOCK-INPUT. `layer` = L means the state entering decoder layer L, and a serving hook must fire as a forward PRE-hook on layer L. There is NO L-1 off-by-one: unlike the Qwen vectors in this repository (whose `layer` is a block INPUT but whose serving hook fires on the OUTPUT of block layer-1), a GLM-5.3-Flash vector is captured and injected at the same point. Serving this vector through the Qwen output-hook path, or a Qwen vector through this pre-hook path, is off by one layer and produces fluent but wrong steering.

**Position convention.** last token of the chat-templated prompt (apply_chat_template(add_generation_prompt=True); GLM-5.3-Flash's template has no enable_thinking variable and reasoning_effort is left at its default, which renders 'Reasoning Effort: Max'; the last prompt token is therefore always <think>, id 154841, with no trailing newline); UNWEIGHTED MEAN over the 4 hyper-connection streams of the stream state at the INPUT of each decoder layer

**Sign convention.** vector = mean(positive activations) - mean(negative activations); positive strength moves toward the POSITIVE set

**Reasoning effort.** default (template whitelist is ['low','high']; anything else, including omission, renders 'Max'). Serving requests must use the same.

## Strength

`strength` a adds `a * vector` to all four streams entering layer 26, at every token
position. At this layer `‖v‖ = 1.5926` against a mean stream-mean norm of
`9.9549`, so `strength 1.0` is a
**16.00%** perturbation of the typical activation
magnitude there.

**Absolute norms are not comparable across layers in this model.** The mean stream norm
grows from 0.140 at layer 0 to
53.77 at layer 43. Only the ratio column
below is comparable.

## Per-layer table

`|v|` is the L2 norm of the difference of means at that layer's input; `act_norm` is the
mean over all 540 captures of the stream-mean norm at the same
point; `ratio` = `|v| / act_norm`; `cos_prev` is the cosine between this layer's delta and
the previous layer's; `pair_cos` is the mean cosine of the 270 individual per-pair deltas
to the pooled mean direction (near zero would mean the vector is noise); `pair|d|` is the
mean per-pair delta norm; `loo_max` is the largest relative change in `|v|` from dropping
any single pair.

| L | depth | \|v\| | act_norm | ratio | cos_prev | pair_cos | pair\|d\| | loo_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0 | 0.13993 | 0.00000 | — | 0.0000 | 0 | 0.0000 |
| 1 | 0.022 | 0.0081773 | 0.26697 | 0.03063 | — | 0.2667 | 0.025335 | 0.0076 |
| 2 | 0.044 | 0.0071814 | 0.2214 | 0.03244 | 0.6661 | 0.3461 | 0.018492 | 0.0058 |
| 3 | 0.067 | 0.0149 | 0.73141 | 0.02037 | 0.1581 | 0.3149 | 0.041401 | 0.0073 |
| 4 | 0.089 | 0.060218 | 1.7557 | 0.03430 | 0.2509 | 0.4139 | 0.13705 | 0.0058 |
| 5 | 0.111 | 0.060541 | 1.6979 | 0.03566 | 0.9428 | 0.4138 | 0.13797 | 0.0056 |
| 6 | 0.133 | 0.082064 | 1.764 | 0.04652 | 0.7383 | 0.3929 | 0.19688 | 0.0054 |
| 7 | 0.156 | 0.12922 | 2.0106 | 0.06427 | 0.6059 | 0.3753 | 0.32032 | 0.0066 |
| 8 | 0.178 | 0.12131 | 1.8901 | 0.06418 | 0.9916 | 0.3765 | 0.30028 | 0.0066 |
| 9 | 0.200 | 0.11471 | 1.7814 | 0.06439 | 0.9853 | 0.3795 | 0.28196 | 0.0064 |
| 10 | 0.222 | 0.11878 | 1.7473 | 0.06798 | 0.9150 | 0.3852 | 0.28724 | 0.0064 |
| 11 | 0.244 | 0.22241 | 2.6034 | 0.08543 | 0.4877 | 0.3366 | 0.61591 | 0.0078 |
| 12 | 0.267 | 0.20211 | 2.4423 | 0.08275 | 0.9749 | 0.3368 | 0.56013 | 0.0076 |
| 13 | 0.289 | 0.18634 | 2.2774 | 0.08182 | 0.9934 | 0.3379 | 0.51516 | 0.0075 |
| 14 | 0.311 | 0.17491 | 2.1683 | 0.08067 | 0.9901 | 0.3364 | 0.48589 | 0.0072 |
| 15 | 0.333 | 0.17509 | 2.15 | 0.08143 | 0.9610 | 0.3396 | 0.48431 | 0.0068 |
| 16 | 0.356 | 0.19856 | 2.2781 | 0.08716 | 0.8349 | 0.3476 | 0.54412 | 0.0052 |
| 17 | 0.378 | 0.23583 | 2.4687 | 0.09553 | 0.7887 | 0.3422 | 0.66014 | 0.0046 |
| 18 | 0.400 | 0.27482 | 2.8332 | 0.09700 | 0.8459 | 0.3438 | 0.7665 | 0.0041 |
| 19 | 0.422 | 0.31116 | 3.1699 | 0.09816 | 0.9087 | 0.3520 | 0.85132 | 0.0037 |
| 20 | 0.444 | 0.4114 | 3.5236 | 0.11675 | 0.7860 | 0.3823 | 1.0587 | 0.0028 |
| 21 | 0.467 | 0.49856 | 3.9805 | 0.12525 | 0.8751 | 0.4062 | 1.2073 | 0.0027 |
| 22 | 0.489 | 0.61063 | 4.601 | 0.13272 | 0.8517 | 0.4266 | 1.411 | 0.0027 |
| 23 | 0.511 | 0.74628 | 5.5262 | 0.13504 | 0.7567 | 0.3946 | 1.8666 | 0.0029 |
| 24 | 0.533 | 1.1348 | 7.3645 | 0.15409 | 0.6926 | 0.4346 | 2.5778 | 0.0030 |
| 25 | 0.556 | 1.3352 | 8.5813 | 0.15559 | 0.8656 | 0.4517 | 2.9159 | 0.0029 |
| 26 | 0.578 | 1.5926 | 9.9549 | 0.15999 | 0.8786 | 0.4848 | 3.231 | 0.0030 |
| 27 | 0.600 | 1.7329 | 11.136 | 0.15561 | 0.8813 | 0.4822 | 3.5352 | 0.0030 |
| 28 | 0.622 | 2.0542 | 12.469 | 0.16474 | 0.8943 | 0.4880 | 4.1216 | 0.0030 |
| 29 | 0.644 | 2.191 | 13.78 | 0.15901 | 0.9102 | 0.4726 | 4.5333 | 0.0031 |
| 30 | 0.667 | 2.4522 | 15.67 | 0.15649 | 0.8797 | 0.4498 | 5.3227 | 0.0031 |
| 31 | 0.689 | 2.8869 | 18.803 | 0.15353 | 0.8625 | 0.4373 | 6.4409 | 0.0033 |
| 32 | 0.711 | 3.0769 | 20.454 | 0.15043 | 0.9239 | 0.4196 | 7.1458 | 0.0034 |
| 33 | 0.733 | 3.4933 | 23.005 | 0.15185 | 0.8704 | 0.3979 | 8.5535 | 0.0034 |
| 34 | 0.756 | 3.7655 | 25.052 | 0.15031 | 0.9006 | 0.3802 | 9.651 | 0.0035 |
| 35 | 0.778 | 4.405 | 31.51 | 0.13980 | 0.8510 | 0.3780 | 11.334 | 0.0036 |
| 36 | 0.800 | 4.6112 | 33.059 | 0.13949 | 0.9411 | 0.3547 | 12.63 | 0.0037 |
| 37 | 0.822 | 4.9741 | 37.066 | 0.13420 | 0.9050 | 0.3278 | 14.725 | 0.0034 |
| 38 | 0.844 | 5.2787 | 39.723 | 0.13289 | 0.9071 | 0.3039 | 16.838 | 0.0038 |
| 39 | 0.867 | 5.7416 | 43.275 | 0.13268 | 0.9014 | 0.2866 | 19.466 | 0.0039 |
| 40 | 0.889 | 6.2373 | 46.256 | 0.13484 | 0.9187 | 0.2734 | 22.183 | 0.0038 |
| 41 | 0.911 | 6.4621 | 47.811 | 0.13516 | 0.9517 | 0.2644 | 23.78 | 0.0040 |
| 42 | 0.933 | 6.666 | 50.482 | 0.13205 | 0.9206 | 0.2573 | 25.267 | 0.0041 |
| 43 | 0.956 | 6.814 | 53.771 | 0.12672 | 0.9423 | 0.2594 | 25.684 | 0.0040 |
| 44 | 0.978 | 7.1831 | 58.717 | 0.12233 | 0.9522 | 0.2545 | 27.648 | 0.0042 |

## Capture

offline vllm.LLM, spawn start method, tensor_parallel_size=4, enforce_eager=True, max_num_seqs=1, enable_prefix_caching=False, max_model_len=8192, max_tokens=1, one prompt per generate() call, prompts submitted as TokensPrompt(prompt_token_ids=...) so the recorded token count is exactly what the engine saw. The stream state entering each layer was reconstructed inside the TP rank-0 worker with a forward PRE-hook: layer 0 sees residual=None and hc_expand replicates, so the mean is hidden_states; layers 1..44 materialise mhc_post_torch(hidden_states, residual, post, comb) for the last token only and take mean(dim=-2).

**Assertions.** per prompt: (a) len(tokenize(text)) == len(tokenize(text, add_special_tokens=False)) -- GLM injects no BOS/EOS; (b) the last prompt token id is 154841 (<think>); (c) the token count summed over all captured forward passes equals len(prompt_token_ids), which is what a prefix-cache hit or a stray decode step would break; (d) all 45 layers fired in the final forward pass. Additionally, once per side: the four streams hc_expand produces at layer 0 are bitwise identical.

| side | n | checked | seconds | prompt tokens min/mean/max |
| --- | --- | --- | --- | --- |
| positive | 270 | 270 | 56.2 | 30 / 119.46 / 2767 |
| negative | 270 | 270 | 52.6 | 31 / 119.27 / 2766 |

The first templated positive prompt, exactly as the model received it:

```
[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Note: your answer will be graded by a script that checks whether it matches the correct answer.

Write a python function to find the maximum of two numbers.
Your code should pass these tests:
assert maximum(5,10) == 10
assert maximum(-1,-2) == -1
assert maximum(9,7) == 9<|assistant|><think>
```

## Digests

```
positive.jsonl          e66effbe0c80dce832ca121aabaca1330840ff65897db1e282187004dca64b95
negative.jsonl          714b7ed2980d2f087a571848c06f71dc35940108342bb387211234c0c6a98dc7
vector.npy              1452aee372907c6ab5a8d9a688c5ea81362f2758e1695575cb9357f91f4604c6
deltas_all_layers.npy   2b20ede6f60aef63213f521927e5b44bbe7c5c49e602f93727b59e0fecd0b132
```

## Command

```
tools/capture_glm5_streams.py --model /workspace/models/GLM-5.3-Flash --tp 4 --max-model-len 8192 --gpu-memory-utilization 0.85 --positive vectors/0007/positive.jsonl --negative vectors/0007/negative.jsonl --out-dir caps  (env: VLLM_ALLOW_INSECURE_SERIALIZATION=1, NCCL_NVLS_ENABLE=0, VLLM_USE_DEEP_GEMM=0, CUDA_HOME=/usr/local/cuda-13.0, LD_LIBRARY_PATH prefixed with /usr/local/cuda-13.0/compat)  ->  tools/derive_glm5_vector.py --caps-dir caps --out-dir derived  ->  tools/write_vector_dir.py --layer 26 --id 8
```
