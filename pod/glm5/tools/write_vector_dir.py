#!/usr/bin/env python3
"""Assemble a vector directory in the layout `steering_vectors/vectorfmt.py` defines.

Runs locally (stdlib + numpy). Emits vector.npy, deltas_all_layers.npy, meta.json,
README.md and copies of the two prompt files.
"""
import argparse
import collections
import hashlib
import json
import os
import shutil

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_format(path):
    kinds, roles, n_sys = set(), collections.Counter(), 0
    n = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            obj = json.loads(line)
            msgs = obj["messages"]
            kinds.add("messages")
            pat = "+".join(m["role"] for m in msgs)
            roles[pat] += 1
            if any(m["role"] == "system" for m in msgs):
                n_sys += 1
    return {
        "kind": sorted(kinds)[0],
        "n": n,
        "n_with_system": n_sys,
        "role_patterns": dict(roles),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", required=True)
    ap.add_argument("--layer-stats", required=True)
    ap.add_argument("--capture-meta", required=True)
    ap.add_argument("--positive", required=True)
    ap.add_argument("--negative", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--created-at", required=True)
    ap.add_argument("--command", required=True)
    args = ap.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)

    deltas = np.load(args.deltas)
    assert deltas.dtype == np.dtype("<f4"), deltas.dtype
    n_layers, hidden = deltas.shape
    stats = json.load(open(args.layer_stats))
    cap = json.load(open(args.capture_meta))
    L = args.layer
    row = stats["per_layer"][L]
    assert row["layer"] == L

    vector = np.ascontiguousarray(deltas[L], dtype="<f4")
    np.save(os.path.join(out, "vector.npy"), vector)
    shutil.copyfile(args.deltas, os.path.join(out, "deltas_all_layers.npy"))
    shutil.copyfile(args.positive, os.path.join(out, "positive.jsonl"))
    shutil.copyfile(args.negative, os.path.join(out, "negative.jsonl"))

    vnorm = row["delta_norm"]
    anorm = row["activation_norm_at_layer"]

    position_convention = (
        "last token of the chat-templated prompt "
        "(apply_chat_template(add_generation_prompt=True); GLM-5.3-Flash's template "
        "has no enable_thinking variable and reasoning_effort is left at its default, "
        "which renders 'Reasoning Effort: Max'; the last prompt token is therefore "
        "always <think>, id 154841, with no trailing newline); "
        "UNWEIGHTED MEAN over the 4 hyper-connection streams of the stream state at "
        "the INPUT of each decoder layer"
    )
    layer_index_convention = (
        "PRE-HOOK / BLOCK-INPUT. `layer` = L means the state entering decoder layer L, "
        "and a serving hook must fire as a forward PRE-hook on layer L. There is NO "
        "L-1 off-by-one: unlike the Qwen vectors in this repository (whose `layer` is "
        "a block INPUT but whose serving hook fires on the OUTPUT of block layer-1), a "
        "GLM-5.3-Flash vector is captured and injected at the same point. Serving this "
        "vector through the Qwen output-hook path, or a Qwen vector through this "
        "pre-hook path, is off by one layer and produces fluent but wrong steering."
    )
    steer_convention = (
        "inject by adding the same delta to ALL FOUR hyper-connection streams of the "
        "state entering layer `layer` (r_i <- r_i + strength * vector for i = 0..3). "
        "Because the Sinkhorn column sums are 1 to 1.1e-06 (measured), a uniform "
        "cross-stream offset passes through the mixer unchanged, and because the final "
        "hc_contract is an unweighted mean it survives the readout as exactly delta."
    )

    meta = {
        "id": args.id,
        "id_str": f"{args.id:04d}",
        "name": args.name,
        "description": DESCRIPTION.format(
            layer=L,
            n_layers=n_layers,
            hidden=hidden,
            vnorm=vnorm,
            anorm=anorm,
            pct=100.0 * vnorm / anorm,
            depth=100.0 * L / n_layers,
        ),
        "model": "zai-org/GLM-5.3-Flash",
        "layer": L,
        "n_layers": n_layers,
        "hidden_size": hidden,
        "steer_site": "stream_mean_input",
        "hc_mult": int(cap["install"]["hc_mult"]),
        "layer_index_convention": layer_index_convention,
        "steer_convention": steer_convention,
        "position_convention": position_convention,
        "sign_convention": (
            "vector = mean(positive activations) - mean(negative activations); "
            "positive strength moves toward the POSITIVE set"
        ),
        "reasoning_effort": (
            "default (template whitelist is ['low','high']; anything else, including "
            "omission, renders 'Max'). Serving requests must use the same."
        ),
        "n_pos": stats["n_pos"],
        "n_neg": stats["n_neg"],
        "prompt_format": {
            "positive": prompt_format(args.positive),
            "negative": prompt_format(args.negative),
        },
        "vector_norm": vnorm,
        "activation_norm_at_layer": anorm,
        "vector_norm_over_activation_norm": vnorm / anorm,
        "per_layer_delta_norm": [r["delta_norm"] for r in stats["per_layer"]],
        "per_layer_mean_activation_norm": [
            r["activation_norm_at_layer"] for r in stats["per_layer"]
        ],
        "per_layer_norm_ratio": [r["ratio"] for r in stats["per_layer"]],
        "per_layer_pair_cos_mean": [r["pair_cos_mean"] for r in stats["per_layer"]],
        "per_layer_adjacent_delta_cosine": [
            r["cos_with_prev_layer_delta"] for r in stats["per_layer"]
        ],
        "created_at": args.created_at,
        "git_sha": None,
        "vllm_version": cap["vllm_version"],
        "templated_first_positive_prompt": cap["sides"]["positive"]["templated_first"],
        "positive_jsonl_sha256": sha256_file(args.positive),
        "negative_jsonl_sha256": sha256_file(args.negative),
        "vector_npy_sha256": sha256_file(os.path.join(out, "vector.npy")),
        "deltas_npy_sha256": sha256_file(os.path.join(out, "deltas_all_layers.npy")),
        "command": args.command,
        "capture": {
            "engine": (
                "offline vllm.LLM, spawn start method, tensor_parallel_size=4, "
                "enforce_eager=True, max_num_seqs=1, enable_prefix_caching=False, "
                "max_model_len={mml}, max_tokens=1, one prompt per generate() call, "
                "prompts submitted as TokensPrompt(prompt_token_ids=...) so the "
                "recorded token count is exactly what the engine saw. The stream "
                "state entering each layer was reconstructed inside the TP rank-0 "
                "worker with a forward PRE-hook: layer 0 sees residual=None and "
                "hc_expand replicates, so the mean is hidden_states; layers 1..44 "
                "materialise mhc_post_torch(hidden_states, residual, post, comb) "
                "for the last token only and take mean(dim=-2)."
            ).format(mml=cap["max_model_len"]),
            "positive": capture_side(cap, "positive"),
            "negative": capture_side(cap, "negative"),
            "assertions": (
                "per prompt: (a) len(tokenize(text)) == "
                "len(tokenize(text, add_special_tokens=False)) -- GLM injects no "
                "BOS/EOS; (b) the last prompt token id is 154841 (<think>); "
                "(c) the token count summed over all captured forward passes equals "
                "len(prompt_token_ids), which is what a prefix-cache hit or a stray "
                "decode step would break; (d) all 45 layers fired in the final "
                "forward pass. Additionally, once per side: the four streams "
                "hc_expand produces at layer 0 are bitwise identical."
            ),
        },
    }

    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: v for k, v in meta.items()
                      if k not in ("description", "per_layer_delta_norm",
                                   "per_layer_mean_activation_norm",
                                   "per_layer_norm_ratio",
                                   "per_layer_pair_cos_mean",
                                   "per_layer_adjacent_delta_cosine")},
                     indent=2)[:4000])
    return meta


def capture_side(cap, side):
    s = cap["sides"][side]
    pt = s["prompt_tokens"]
    return {
        "n_prompts": s["n_prompts"],
        "n_checked": s["n_checked"],
        "capture_files_from_engine_init": 0,
        "seconds": round(s["seconds"], 1),
        "max_model_len": cap["max_model_len"],
        "prompt_tokens_min": min(pt),
        "prompt_tokens_max": max(pt),
        "prompt_tokens_mean": round(sum(pt) / len(pt), 2),
    }


DESCRIPTION = (
    "The direction that separates prompts saying the answer will be checked by an "
    "automated verifier from prompts saying it will be read and rated by a person, "
    "measured in zai-org/GLM-5.3-Flash. The prompt set is byte-identical to vector "
    "0007's (270 tasks written twice, the two copies differing only in the sentence "
    "that says how the answer will be graded); only the chat template changed, from "
    "Qwen's to GLM-5.3-Flash's.\n\n"
    "GLM-5.3-Flash uses multi-stream hyper-connections (hc_mult=4), so there is no "
    "single residual stream. The quantity measured here is the unweighted mean over "
    "the four streams of the state at the INPUT of each of the {n_layers} decoder "
    "layers, at the last prompt token. That reduction is not a proxy: the model's own "
    "output head contracts the four streams with a plain mean, and a delta added "
    "uniformly to all four streams passes through the Sinkhorn mixer unchanged "
    "(measured column-sum deviation 1.1e-06).\n\n"
    "`vector.npy` is the layer-{layer} row of that difference: little-endian float32, "
    "shape ({hidden},), norm {vnorm:.4f} against a mean activation norm of "
    "{anorm:.4f} at the same layer, so strength 1.0 perturbs the stream mean by "
    "{pct:.1f}% of its typical magnitude at {depth:.0f}% depth. Absolute norms are "
    "NOT comparable across layers in this model -- the mean stream norm grows about "
    "2700x from layer 0 to layer 43 -- so use the ratio, never the norm, to compare "
    "layers. `deltas_all_layers.npy` holds the difference at all {n_layers} layer "
    "inputs, float32 ({n_layers}, {hidden}), so the layer choice is auditable and "
    "re-selectable without re-capturing 540 prompts."
)


if __name__ == "__main__":
    main()
