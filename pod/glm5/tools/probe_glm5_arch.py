#!/usr/bin/env python3
"""Introspect a live GLM-5.3-Flash vLLM engine to settle the steering-design
Tier-B questions in notes/glm53_steering_design.md.

Run inside the vLLM venv on the GPU box.  Loads the model once with
enforce_eager=True (so module hooks actually fire), runs one short prompt, and
dumps:

  1. vllm/torch versions and the parallel/speculative flags that matter
  2. every nn.ModuleList in the model tree (settles decoder-stack discovery)
  3. the decoder layer forward signature
  4. the actual shapes/dtypes of the (x, residual, post, comb) 4-tuple
  5. Sinkhorn row-sum / column-sum deviations for comb  (design doc section 2.4)
  6. a direct numerical test of the uniform-delta injection identity
     (design doc section 2.4 / patch fix B1)

Everything is printed to stdout; redirect to a file.
"""

import argparse
import json
import os
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", required=True, help="path to the GLM-5.3-Flash checkpoint"
    )
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--layers", default="0,1,22,43,44",
                    help="comma-separated decoder layer indices to instrument")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    print("=" * 78)
    print("SECTION 1: versions and environment")
    print("=" * 78)
    print(f"vllm.__version__ = {vllm.__version__}")
    print(f"torch.__version__ = {torch.__version__}")
    print(f"torch.version.cuda = {torch.version.cuda}")
    print(f"cuda device count = {torch.cuda.device_count()}")
    for k in sorted(os.environ):
        if k.startswith("VLLM_"):
            print(f"env {k}={os.environ[k]}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,          # hooks must fire every step
        enable_prefix_caching=False, # capture semantics: no cross-request reuse
        max_num_seqs=1,
    )

    # ---- reach into the driver worker's model ---------------------------------
    engine = llm.llm_engine
    try:
        core = engine.engine_core
        # v1: the model lives in a worker process unless TP==1.  We instead run
        # the introspection *inside* the worker via collective_rpc.
        rpc = core.collective_rpc
    except AttributeError:
        rpc = None

    payload = {"layers": [int(x) for x in args.layers.split(",") if x != ""]}

    def _probe(worker, payload=payload):
        """Runs inside each worker process; returns a dict from rank 0 only."""
        import inspect as _inspect

        import torch as _torch

        model = worker.model_runner.model
        out = {}

        # -- 2. module tree -------------------------------------------------
        mls = []
        for name, mod in model.named_modules():
            if isinstance(mod, _torch.nn.ModuleList):
                mls.append({
                    "path": name,
                    "type": type(mod).__name__,
                    "len": len(mod),
                    "child0": type(mod[0]).__name__ if len(mod) else None,
                })
        out["module_lists"] = mls
        mds = []
        for name, mod in model.named_modules():
            if isinstance(mod, _torch.nn.ModuleDict):
                mds.append({"path": name, "len": len(mod),
                            "keys": list(mod.keys())[:4]})
        out["module_dicts"] = mds
        out["top_level_class"] = type(model).__name__
        out["top_level_module"] = type(model).__module__
        out["top_level_children"] = [n for n, _ in model.named_children()]

        # -- locate the decoder stack --------------------------------------
        cand = [
            (name, mod) for name, mod in model.named_modules()
            if isinstance(getattr(mod, "layers", None), _torch.nn.ModuleList)
            and hasattr(mod, "embed_tokens")
        ]
        out["pod_heuristic_matches"] = [
            {"path": n, "n_layers": len(m.layers)} for n, m in cand
        ]
        if not cand:
            out["error"] = "no decoder stack found"
            return out
        stack_path, stack = cand[0]
        layers = stack.layers
        out["stack_path"] = stack_path
        out["n_layers"] = len(layers)

        # -- 3. forward signature -------------------------------------------
        out["layer_class"] = type(layers[0]).__name__
        out["layer_module"] = type(layers[0]).__module__
        out["layer_forward_sig"] = str(_inspect.signature(type(layers[0]).forward))
        out["model_forward_sig"] = str(_inspect.signature(type(stack).forward))
        for attr in ("is_sequence_parallel", "mhc", "n", "num_hidden_layers",
                     "layer_kind", "is_mtp_layer", "mhc_sinkhorn_iterations",
                     "hc_eps", "mhc_post_mult_value"):
            out[f"layer0.{attr}"] = repr(getattr(layers[0], attr, "<absent>"))

        # -- config ----------------------------------------------------------
        mc = worker.model_runner.model_config
        htc = mc.hf_text_config
        out["hf_text_config.class"] = type(htc).__name__
        for attr in ("num_hidden_layers", "hidden_size",
                     "mhc_num_residual_streams", "hc_eps", "rms_norm_eps"):
            out[f"hf_text_config.{attr}"] = repr(getattr(htc, attr, "<absent>"))
        pc = worker.vllm_config.parallel_config
        out["parallel.tp"] = pc.tensor_parallel_size
        out["parallel.pp"] = pc.pipeline_parallel_size
        out["parallel.dp"] = pc.data_parallel_size
        out["parallel.enable_expert_parallel"] = pc.enable_expert_parallel
        out["parallel.use_sequence_parallel_moe"] = pc.use_sequence_parallel_moe
        out["parallel.use_ubatching"] = getattr(pc, "use_ubatching", "<absent>")
        out["speculative_config"] = repr(worker.vllm_config.speculative_config)
        out["runner.max_num_tokens"] = getattr(worker.model_runner,
                                               "max_num_tokens", "<absent>")

        # -- 4/5/6. instrument selected layers -------------------------------
        rank = getattr(worker, "rank", 0)
        recs = {}

        def make_hook(idx):
            def hook(module, inputs, output):
                if idx in recs:
                    return
                rec = {"n_out": len(output) if isinstance(output, tuple) else 1,
                       "out_types": [type(o).__name__ for o in output]
                       if isinstance(output, tuple) else [type(output).__name__]}
                if isinstance(output, tuple):
                    names = ["x", "residual", "post", "comb"]
                    for j, o in enumerate(output):
                        nm = names[j] if j < len(names) else f"out{j}"
                        if isinstance(o, _torch.Tensor):
                            rec[f"{nm}.shape"] = list(o.shape)
                            rec[f"{nm}.dtype"] = str(o.dtype)
                        else:
                            rec[f"{nm}"] = repr(o)
                    x, residual, post, comb = (list(output) + [None] * 4)[:4]
                    if isinstance(comb, _torch.Tensor):
                        c = comb.detach().float()
                        # comb[..., i, j]; out_j = sum_i comb_ij r_i
                        colsum = c.sum(dim=-2)   # over i -> per j
                        rowsum = c.sum(dim=-1)   # over j -> per i
                        rec["comb.colsum.max_abs_dev"] = float(
                            (colsum - 1).abs().max())
                        rec["comb.rowsum.max_abs_dev"] = float(
                            (rowsum - 1).abs().max())
                        rec["comb.colsum.mean"] = float(colsum.mean())
                        rec["comb.rowsum.mean"] = float(rowsum.mean())
                        rec["comb.min"] = float(c.min())
                        rec["comb.max"] = float(c.max())
                    if (isinstance(residual, _torch.Tensor)
                            and isinstance(post, _torch.Tensor)
                            and isinstance(comb, _torch.Tensor)
                            and isinstance(x, _torch.Tensor)):
                        from vllm.model_executor.kernels.mhc.torch import (
                            mhc_post_torch,
                        )
                        st = mhc_post_torch(x, residual, post, comb)
                        rec["materialised.shape"] = list(st.shape)
                        m = st.float().mean(dim=-2)
                        rec["stream_mean.norm.mean"] = float(
                            m.norm(dim=-1).mean())
                        # per-stream divergence from the mean (design Q9)
                        dev = (st.float() - m.unsqueeze(-2)).norm(dim=-1)
                        rec["stream_dev_over_mean.mean"] = float(
                            (dev / m.norm(dim=-1, keepdim=True)).mean())
                        rec["stream_dev_over_mean.max"] = float(
                            (dev / m.norm(dim=-1, keepdim=True)).max())
                        # residual streams themselves (pre-write state)
                        rr = residual.float()
                        rm = rr.mean(dim=-2)
                        rec["residual_stream_mean.norm.mean"] = float(
                            rm.norm(dim=-1).mean())

                        # ---- 6. uniform-delta injection identity ----------
                        # claim: adding d to EVERY stream of `residual` shifts
                        # mhc_post(...) by exactly d, because col sums are 1.
                        g = _torch.Generator(device=residual.device)
                        g.manual_seed(1234)
                        d = _torch.randn(residual.shape[0], residual.shape[-1],
                                         generator=g, device=residual.device,
                                         dtype=_torch.float32)
                        d = d / d.norm(dim=-1, keepdim=True) * float(
                            rm.norm(dim=-1).mean())
                        d_b = d.to(residual.dtype)
                        st2 = mhc_post_torch(
                            x, (residual + d_b.unsqueeze(-2)), post, comb)
                        actual = (st2.float() - st.float())
                        want = d.unsqueeze(-2).expand_as(actual)
                        rec["inject.delta_norm"] = float(d.norm(dim=-1).mean())
                        rec["inject.abs_err.max"] = float(
                            (actual - want).abs().max())
                        rec["inject.rel_err.mean"] = float(
                            ((actual - want).norm(dim=-1)
                             / want.norm(dim=-1).clamp_min(1e-9)).mean())
                        # and what it does to the stream mean
                        dm = st2.float().mean(dim=-2) - m
                        rec["inject.mean_shift.rel_err"] = float(
                            ((dm - d).norm(dim=-1)
                             / d.norm(dim=-1).clamp_min(1e-9)).mean())
                recs[idx] = rec
            return hook

        handles = []
        for idx in payload["layers"]:
            if 0 <= idx < len(layers):
                handles.append(layers[idx].register_forward_hook(make_hook(idx)))
        worker._probe_recs = recs
        worker._probe_handles = handles
        out["instrumented"] = [i for i in payload["layers"] if 0 <= i < len(layers)]
        out["rank"] = rank
        return out

    print()
    print("=" * 78)
    print("SECTION 2-3: module tree, signatures, config (from worker rank 0)")
    print("=" * 78)
    results = rpc(_probe) if rpc is not None else None
    r0 = results[0] if isinstance(results, list) else results
    print(json.dumps(r0, indent=2, default=str))

    # ---- run one prompt so the hooks fire ------------------------------------
    print()
    print("=" * 78)
    print("SECTION 4-6: live tensor shapes, Sinkhorn sums, injection identity")
    print("=" * 78)
    tok = llm.get_tokenizer()
    text = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2? Answer briefly."}],
        tokenize=False, add_generation_prompt=True)
    print(f"prompt repr: {text!r}")
    print(f"prompt token ids: {tok(text, add_special_tokens=False)['input_ids']}")
    out = llm.generate([text], SamplingParams(temperature=0.0, max_tokens=4))
    print(f"generated: {out[0].outputs[0].text!r}")

    def _collect(worker):
        for h in getattr(worker, "_probe_handles", []):
            h.remove()
        return getattr(worker, "_probe_recs", {})

    recs = rpc(_collect)
    r = recs[0] if isinstance(recs, list) else recs
    print(json.dumps(r, indent=2, default=str))

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"static": r0, "dynamic": r}, fh, indent=2, default=str)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
