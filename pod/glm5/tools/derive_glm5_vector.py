#!/usr/bin/env python3
"""Derive the GLM-5.3-Flash steering vector from captured stream-mean activations.

Input : caps_positive.npy / caps_negative.npy, each (n, 45, 4096) float32,
        row i of the two files being the SAME task with only the grading
        sentence changed (verified: median char-similarity 0.85, off-diagonal
        0.026).
Output: deltas_all_layers.npy (45, 4096) float32, plus a stats JSON with
        everything needed to choose the layer and to sanity-check the vector.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pos = np.load(os.path.join(args.caps_dir, "caps_positive.npy"))
    neg = np.load(os.path.join(args.caps_dir, "caps_negative.npy"))
    assert pos.shape == neg.shape, (pos.shape, neg.shape)
    n, L, H = pos.shape
    print(f"pos {pos.shape} neg {neg.shape}")
    assert np.isfinite(pos).all() and np.isfinite(neg).all(), "non-finite captures"

    # --- diversity: the classic fixed-seed / stale-buffer failure -------------
    def diversity(a, name):
        flat = a[:, L // 2, :]                       # one mid layer
        nrm = np.linalg.norm(flat, axis=1)
        # pairwise distinctness via a cheap hash of exact bytes
        uniq = len({x.tobytes() for x in a})
        # nearest-neighbour cosine among the first 60 rows
        sub = flat[:60] / np.linalg.norm(flat[:60], axis=1, keepdims=True)
        cs = sub @ sub.T
        np.fill_diagonal(cs, -2.0)
        return {
            "unique_rows": uniq,
            "n": int(a.shape[0]),
            "norm_min": float(nrm.min()),
            "norm_max": float(nrm.max()),
            "max_offdiag_cosine_first60": float(cs.max()),
            "mean_offdiag_cosine_first60": float(cs[cs > -2].mean()),
        }

    div = {"positive": diversity(pos, "pos"), "negative": diversity(neg, "neg")}
    print("diversity:", json.dumps(div, indent=2))

    # --- the vector ----------------------------------------------------------
    mean_pos = pos.mean(axis=0)                       # (L, H)
    mean_neg = neg.mean(axis=0)
    deltas = (mean_pos - mean_neg).astype(np.float32)  # (L, H)

    per_layer_delta_norm = np.linalg.norm(deltas.astype(np.float64), axis=1)
    both = np.concatenate([pos, neg], axis=0)          # (2n, L, H)
    per_layer_mean_activation_norm = np.linalg.norm(
        both.astype(np.float64), axis=2
    ).mean(axis=0)                                     # (L,)
    ratio = per_layer_delta_norm / per_layer_mean_activation_norm

    # adjacent-layer cosine of the deltas
    d64 = deltas.astype(np.float64)
    dn = np.linalg.norm(d64, axis=1)
    adj_cos = np.full(L, np.nan)
    for i in range(1, L):
        if dn[i] > 0 and dn[i - 1] > 0:
            adj_cos[i] = float(d64[i] @ d64[i - 1] / (dn[i] * dn[i - 1]))

    # --- per-pair sanity: is the mean a real direction or one outlier? --------
    pair = (pos - neg).astype(np.float64)              # (n, L, H)
    pair_norm = np.linalg.norm(pair, axis=2)           # (n, L)
    pair_unit = pair / np.clip(pair_norm[:, :, None], 1e-12, None)
    mean_unit = d64 / np.clip(dn[:, None], 1e-12, None)
    pair_cos = np.einsum("nlh,lh->nl", pair_unit, mean_unit)   # (n, L)

    # leave-one-out: how much of ||v|| does the single most influential pair own?
    loo_norm = np.zeros((n, L))
    for i in range(n):
        d_i = d64 - (pair[i] - d64) / (n - 1)
        loo_norm[i] = np.linalg.norm(d_i, axis=1)
    loo_rel = np.abs(loo_norm - dn[None, :]) / np.clip(dn[None, :], 1e-12, None)

    def q(a, ps=(0, 5, 25, 50, 75, 95, 100)):
        return {f"p{p}": float(np.percentile(a, p)) for p in ps}

    per_layer = []
    for layer in range(L):
        per_layer.append(
            {
                "layer": layer,
                "depth_frac": round(layer / L, 4),
                "delta_norm": float(per_layer_delta_norm[layer]),
                "activation_norm_at_layer": float(
                    per_layer_mean_activation_norm[layer]
                ),
                "ratio": float(ratio[layer]),
                "cos_with_prev_layer_delta": (
                    None if np.isnan(adj_cos[layer]) else float(adj_cos[layer])
                ),
                "pair_cos_mean": float(pair_cos[:, layer].mean()),
                "pair_cos_median": float(np.median(pair_cos[:, layer])),
                "pair_cos_frac_positive": float((pair_cos[:, layer] > 0).mean()),
                "pair_delta_norm": q(pair_norm[:, layer]),
                "pair_delta_norm_mean": float(pair_norm[:, layer].mean()),
                "pair_delta_norm_top1_over_median": float(
                    pair_norm[:, layer].max()
                    / max(np.median(pair_norm[:, layer]), 1e-12)
                ),
                "loo_max_rel_change": float(loo_rel[:, layer].max()),
                "snr_meannorm_over_pairnorm_mean": float(
                    per_layer_delta_norm[layer] / max(pair_norm[:, layer].mean(), 1e-12)
                ),
            }
        )

    np.save(os.path.join(args.out_dir, "deltas_all_layers.npy"),
            np.ascontiguousarray(deltas, dtype="<f4"))
    np.save(os.path.join(args.out_dir, "pair_cos.npy"), pair_cos.astype(np.float32))
    np.save(os.path.join(args.out_dir, "pair_delta_norm.npy"),
            pair_norm.astype(np.float32))

    stats = {
        "n_pos": int(n),
        "n_neg": int(n),
        "n_layers": int(L),
        "hidden_size": int(H),
        "diversity": div,
        "per_layer": per_layer,
    }
    with open(os.path.join(args.out_dir, "layer_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"{'L':>3} {'depth':>6} {'|v|':>11} {'act_norm':>11} {'ratio':>8} "
          f"{'cos_prev':>9} {'paircos':>8} {'loo_max':>8}")
    for r in per_layer:
        cp = r["cos_with_prev_layer_delta"]
        print(f"{r['layer']:>3} {r['depth_frac']:>6.3f} {r['delta_norm']:>11.5f} "
              f"{r['activation_norm_at_layer']:>11.4f} {r['ratio']:>8.5f} "
              f"{'  n/a  ' if cp is None else f'{cp:>9.4f}'} "
              f"{r['pair_cos_mean']:>8.4f} {r['loo_max_rel_change']:>8.4f}")
    print(f"\nwrote {args.out_dir}/deltas_all_layers.npy and layer_stats.json")


if __name__ == "__main__":
    main()
