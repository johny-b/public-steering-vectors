#!/usr/bin/env python3
"""Capture the GLM-5.3-Flash steering-vector activations.

Site (design doc section 2.1 / 9.2): the UNWEIGHTED MEAN over the 4 hyper-connection
streams of the stream state at the INPUT of each of the 45 decoder layers, taken at
the LAST token of the chat-templated prompt.

The stream state entering layer L is not materialised by vLLM: the layer receives
the lazy 4-tuple (hidden_states, residual, post, comb) from layer L-1 and fuses the
pending hc_post with its own hc_pre.  We therefore reconstruct it in a forward
PRE-hook with

    L == 0 : residual is None, streams = hc_expand(hidden_states, 4)  (replication)
             -> mean == hidden_states
    L >= 1 : streams = mhc_post_torch(hidden_states, residual, post, comb)
             -> mean over dim=-2

which is exactly the tensor layer L consumes.  Only the last row is materialised
(mhc_post is per-token), so the hook costs nothing.

Layer index convention: PRE-HOOK / BLOCK INPUT.  Row L of the output is the state
entering decoder layer L.  There is NO L-1 off-by-one.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def load_prompts(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj["messages"])
    return rows


# --------------------------------------------------------------------------
# worker-side closures (shipped to every TP rank via collective_rpc)
# --------------------------------------------------------------------------
def _install(worker):
    import torch as _torch

    model = worker.model_runner.model
    cand = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(getattr(mod, "layers", None), _torch.nn.ModuleList)
        and hasattr(mod, "embed_tokens")
    ]
    assert len(cand) == 1, f"decoder stack heuristic matched {len(cand)}"
    stack_path, stack = cand[0]
    layers = stack.layers
    n = len(layers)

    # _active_layers holds the SAME objects (design doc 9.1 Q11) -- registering on
    # `layers` therefore covers the loop in Glm5NextModel.forward exactly once.
    act = getattr(stack, "_active_layers", None)
    same = (
        act is not None
        and len(act) == n
        and all(act[i] is layers[i] for i in range(n))
    )

    hidden = int(worker.model_runner.model_config.hf_text_config.hidden_size)
    hc_mult = int(getattr(layers[0], "n", 4))

    state = {
        "n": n,
        "hidden": hidden,
        "hc_mult": hc_mult,
        "cur": None,
        "ntok": None,
        "seen": None,
        "forwards": [],   # list of dicts: {"ntok": int, "vecs": np[45,H], "seen": int}
        "l0_streams_identical": None,
        "enabled": False,
    }
    worker._cap_state = state

    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    def make_hook(idx):
        def hook(module, args):
            st = worker._cap_state
            if not st["enabled"]:
                return None
            # Glm5NextModel.forward calls layer(positions, hidden_states, residual,
            # post, comb) positionally.
            hs = args[1]
            residual = args[2] if len(args) > 2 else None
            post = args[3] if len(args) > 3 else None
            comb = args[4] if len(args) > 4 else None

            if idx == 0:
                st["cur"] = np.zeros((st["n"], st["hidden"]), dtype=np.float32)
                st["ntok"] = int(hs.shape[0])
                st["seen"] = 0

            if st["cur"] is None:          # started mid-stack; ignore
                return None

            if residual is None or post is None or comb is None:
                # layer 0: hc_expand replicates -> all 4 streams equal hs
                m = hs[-1:].to(_torch.float32)
                if idx == 0 and st["l0_streams_identical"] is None:
                    from vllm.model_executor.layers.mhc import hc_expand
                    ex = hc_expand(hs[-1:], st["hc_mult"]).to(_torch.float32)
                    st["l0_streams_identical"] = float(
                        (ex - ex.mean(dim=-2, keepdim=True)).abs().max()
                    )
            else:
                streams = mhc_post_torch(
                    hs[-1:], residual[-1:], post[-1:], comb[-1:]
                )                                    # [1, hc_mult, H]
                m = streams.to(_torch.float32).mean(dim=-2)   # [1, H]

            st["cur"][idx] = m[0].detach().cpu().numpy()
            st["seen"] += 1

            if idx == st["n"] - 1:
                st["forwards"].append(
                    {"ntok": st["ntok"], "vecs": st["cur"], "seen": st["seen"]}
                )
                st["cur"] = None
            return None

        return hook

    handles = [layers[i].register_forward_pre_hook(make_hook(i)) for i in range(n)]
    worker._cap_handles = handles

    return {
        "rank": int(getattr(worker, "rank", 0)),
        "stack_path": stack_path,
        "n_layers": n,
        "hidden": hidden,
        "hc_mult": hc_mult,
        "active_layers_are_same_objects": same,
        "layer_class": type(layers[0]).__name__,
    }


def _enable(worker):
    worker._cap_state["enabled"] = True
    worker._cap_state["forwards"] = []
    worker._cap_state["cur"] = None
    return True


def _take(worker):
    st = worker._cap_state
    fwds = st["forwards"]
    st["forwards"] = []
    st["cur"] = None
    rank = int(getattr(worker, "rank", 0))
    if rank != 0:
        return {"rank": rank}
    return {
        "rank": rank,
        "n_forwards": len(fwds),
        "ntoks": [f["ntok"] for f in fwds],
        "seen": [f["seen"] for f in fwds],
        "last": fwds[-1]["vecs"] if fwds else None,
        "l0_identical": st["l0_streams_identical"],
    }


def _uninstall(worker):
    for h in getattr(worker, "_cap_handles", []):
        h.remove()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", required=True, help="path to the GLM-5.3-Flash checkpoint"
    )
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--positive", required=True)
    ap.add_argument("--negative", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    sides = {
        "positive": load_prompts(args.positive),
        "negative": load_prompts(args.negative),
    }
    if args.limit:
        sides = {k: v[: args.limit] for k, v in sides.items()}
    print(
        f"loaded {len(sides['positive'])} positive / {len(sides['negative'])} negative",
        flush=True,
    )

    t_load = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,   # load-bearing: see design doc C4
        max_num_seqs=1,
    )
    load_seconds = time.time() - t_load
    print(f"engine up in {load_seconds:.1f}s  vllm={vllm.__version__}", flush=True)

    rpc = llm.llm_engine.engine_core.collective_rpc
    install_info = rpc(_install)
    info0 = install_info[0] if isinstance(install_info, list) else install_info
    print("install:", json.dumps(info0, indent=2), flush=True)
    assert info0["n_layers"] == 45, info0
    assert info0["hidden"] == 4096, info0
    assert info0["hc_mult"] == 4, info0
    assert info0["active_layers_are_same_objects"] is True, info0

    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    meta = {
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "load_seconds": load_seconds,
        "install": info0,
        "max_model_len": args.max_model_len,
        "tp": args.tp,
        "sides": {},
    }

    for side, msgs_list in sides.items():
        n = len(msgs_list)
        arr_path = os.path.join(args.out_dir, f"caps_{side}.npy")
        arr = np.lib.format.open_memmap(
            arr_path, mode="w+", dtype=np.float32, shape=(n, 45, 4096)
        )
        rec = {
            "n_prompts": n,
            "prompt_tokens": [],
            "templated_first": None,
            "n_checked": 0,
            "l0_identical_max": 0.0,
            "n_forwards_hist": {},
        }
        rpc(_enable)
        t0 = time.time()
        for i, messages in enumerate(msgs_list):
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if i == 0:
                rec["templated_first"] = text
                print(f"[{side}] templated[0] = {text!r}", flush=True)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            ids_sp = tok(text)["input_ids"]
            assert len(ids) == len(ids_sp), (
                f"{side}[{i}]: add_special_tokens changes length "
                f"{len(ids)} != {len(ids_sp)}"
            )
            assert ids[-1] == 154841, (
                f"{side}[{i}]: last prompt token is {ids[-1]}, expected <think>=154841"
            )
            rec["prompt_tokens"].append(len(ids))

            llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)

            got = rpc(_take)
            cands = [x for x in (got if isinstance(got, list) else [got])
                     if isinstance(x, dict) and "last" in x]
            assert len(cands) == 1, f"{side}[{i}]: {len(cands)} rank-0 results"
            g = cands[0]
            key = str(g["n_forwards"])
            rec["n_forwards_hist"][key] = rec["n_forwards_hist"].get(key, 0) + 1
            assert g["n_forwards"] >= 1, f"{side}[{i}]: no forward captured"
            assert sum(g["ntoks"]) == len(ids), (
                f"{side}[{i}]: captured {sum(g['ntoks'])} tokens over "
                f"{g['n_forwards']} forwards but prompt has {len(ids)} "
                f"(prefix cache or extra decode step?)"
            )
            assert g["seen"][-1] == 45, f"{side}[{i}]: only {g['seen'][-1]}/45 layers"
            arr[i] = g["last"]
            if g["l0_identical"] is not None:
                rec["l0_identical_max"] = max(
                    rec["l0_identical_max"], float(g["l0_identical"])
                )
            rec["n_checked"] += 1

            if (i + 1) % 10 == 0 or i + 1 == n:
                el = time.time() - t0
                print(
                    f"[{side}] {i+1}/{n}  {el:.1f}s  "
                    f"{el/(i+1):.2f}s/prompt  eta {(n-i-1)*el/(i+1):.0f}s",
                    flush=True,
                )
                arr.flush()
        rec["seconds"] = time.time() - t0
        arr.flush()
        del arr
        meta["sides"][side] = rec
        with open(os.path.join(args.out_dir, "capture_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        print(f"[{side}] done in {rec['seconds']:.1f}s -> {arr_path}", flush=True)

    rpc(_uninstall)
    with open(os.path.join(args.out_dir, "capture_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print("CAPTURE COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
