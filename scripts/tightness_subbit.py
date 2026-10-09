"""The sub-bit rows of the tightness experiment (gamma < 1 bit): randomised response on a two-cell codebook.

A deterministic codebook leaks at least 1 bit at R = 1, so to reach gamma < 1 the release is randomised. A two-cell
k-means codebook is fitted on D_shadow; Q is a record's nearest codeword, and the release reports Q~ = Q with
probability p and the other cell otherwise, for p from 0.5 to 1. Then, per p:

    gamma       I(x; Q~) = H(Q~) - H_b(p), exact from the cell frequencies
    attack      rho_cb: decode Q~ to the posterior mean E[Y | Q~] (a p-weighted mix of the two cell means), on D_targets
    per_block   rho_blk, and per_cell, the k = 1 bounds (src/tightness/percell.py: cell_bounds) with the two reported
                values as cells (one seeded report per D_est record), each per-block leakage clipped to [0, gamma]

At p = 0.5, gamma = 0 and every bound is the marginal one; at p = 1 the release is the deterministic two-cell
codebook (unbalanced, so close to but not the balanced R = 1 row of tab:tightness). Checks rho* <= per-block <=
per-cell <= attack at every p. Writes results/tightness/subbit_<dataset>_<rep>.json.
"""
import argparse
import json
import math
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.bound.plugin import binary_entropy_bits  # noqa: E402
from src.bound.rho import rho_star  # noqa: E402
from src.tightness.codebook import kmeans, nearest_codeword  # noqa: E402
from src.tightness.percell import cell_bounds, h1d_cells, tightness_data, within_cell_variance  # noqa: E402

LN2 = math.log(2.0)
P_GRID = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
N_BOOT = 300                   # bootstrap resamples of the targets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--rep", default="pixel", choices=["pixel", "lpips"])
    ap.add_argument("--n-rep", type=int, default=12, help="queries per cell = n_rep * 250 * (1 + nu^2)")
    ap.add_argument("--n-coords", type=int, default=0, help="estimate the entropies on this many coordinates (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)              # every numpy draw below comes from this, in a fixed order
    ds = a.dataset

    data = tightness_data(ds, a.rep, a.data_dir, dev)
    Y = data.Y
    pairs = data.frame.bound["pairs"]
    NU2 = sorted({float(p["nu2"]) for p in pairs if int(p["k"]) == 1})
    nu2_0 = min(NU2)
    n, d = Y.shape
    N_k, sqrt_Nk = d, math.sqrt(d)                   # k = 1: one block per coordinate
    coords = (np.sort(rng.choice(d, a.n_coords, replace=False)) if (a.n_coords and 0 < a.n_coords < d)
              else np.arange(d))
    Y_sub = np.ascontiguousarray(Y[:, coords])
    s, q_target = math.sqrt(nu2_0), int(a.n_rep * 250 * (1 + nu2_0))

    # two-cell codebook on D_shadow; each record's true cell; the cell means of D_est (in Y)
    codebook = kmeans(data.F_shadow, 2, a.seed, dev)
    Q_est, Q_tgt = nearest_codeword(data.F_est, codebook), nearest_codeword(data.F_tgt, codebook)
    f = np.bincount(Q_est, minlength=2) / n
    cell_mean = np.stack([np.asarray(Y[Q_est == c], np.float64).mean(0) for c in range(2)])
    H_marg = h1d_cells(Y_sub, [np.arange(n)], s, q_target, a.seed * 977, dev, sqrt_Nk, rng=rng)[0]
    perm = rng.permutation(n)
    half1 = np.sort(perm[:n // 2])
    in_half1 = np.zeros(n, bool)
    in_half1[half1] = True

    def bounds(cells, total, gamma, seed_offset):
        """cell_bounds for a partition of D_est (or of a half) into the two reported cells."""
        H = h1d_cells(Y_sub, cells, s, q_target, a.seed * 131 + seed_offset, dev, sqrt_Nk, rng=rng)
        return cell_bounds(np.array([len(c) for c in cells]) / total, H, H_marg, within_cell_variance(Y, cells, N_k),
                           within_cell_variance(Y_sub, cells, N_k), coords, nu2_0, (0.0, float(gamma)))

    def report(Q, p, seed):
        """One randomised report per record: the true cell with probability p, the other cell otherwise."""
        out = Q.copy()
        flip = np.random.default_rng(seed).random(len(Q)) > p
        out[flip] = 1 - out[flip]
        return out

    Y_tgt64 = np.asarray(data.Y_tgt, np.float64)
    H_Q = binary_entropy_bits(f[0])
    rows = []
    print(f"[subbit {ds}/{a.rep}] n_est={n} d={d} H(Q)={H_Q:.4f} f={f.round(3)} nu2_0={nu2_0} coords={len(coords)}",
          flush=True)
    print(f"{'p':>5} {'gamma':>6} {'rho*':>6} {'p-blk':>6} {'p-cell':>6} {'attack':>6} {'SE':>5} {'cap':>5} {'ok':>3}",
          flush=True)
    for p in P_GRID:
        gamma = binary_entropy_bits(f[0] * p + f[1] * (1 - p)) - binary_entropy_bits(p)
        reported = report(Q_est, p, a.seed * 131 + int(round(p * 100)))
        cells = [np.where(reported == c)[0] for c in range(2)]
        b = bounds(cells, n, gamma, 101)
        rho_cell, rho_blk = math.sqrt(max(0.0, b["rho2_cell"])), math.sqrt(max(0.0, b["rho2_blk"]))
        r1 = math.sqrt(max(0.0, bounds([c[in_half1[c]] for c in cells], len(half1), gamma, 202)["rho2_cell"]))
        r2 = math.sqrt(max(0.0, bounds([c[~in_half1[c]] for c in cells], n - len(half1), gamma, 303)["rho2_cell"]))
        # attack: decode a report to E[Y | Q~], the mix of the two cell means with weights P(Q = c | Q~)
        W = np.zeros((2, 2))                                          # W[c, c~] = P(Q = c | Q~ = c~)
        for ct in range(2):
            for c in range(2):
                W[c, ct] = (p if c == ct else 1 - p) * f[c]
            W[:, ct] /= W[:, ct].sum()
        decoded = np.stack([W[0, ct] * cell_mean[0] + W[1, ct] * cell_mean[1] for ct in range(2)])
        err = ((Y_tgt64 - decoded[report(Q_tgt, p, a.seed * 263 + int(round(p * 100)))]) ** 2).sum(1)
        attack = math.sqrt(float(err.mean()))
        attack_se = float(np.std([math.sqrt(err[rng.integers(0, len(err), len(err))].mean()) for _ in range(N_BOOT)]))
        bound = rho_star(gamma * LN2, pairs, "h_lb_bits")["rho_star"]
        ok = (bound <= rho_blk + 1e-3) and (rho_blk <= rho_cell + 1e-3) and (rho_cell <= attack + 2 * attack_se + 1e-3)
        rows.append(dict(p=p, gamma=gamma, Hq=H_Q, rho_star=bound, pt=rho_star(gamma * LN2, pairs, "h_hat_bits")["rho_star"],
                         per_block=rho_blk, per_cell=rho_cell, band=abs(r1 - r2), attack=attack, attack_se=attack_se,
                         cap_floor=math.sqrt(max(0.0, b["cap2"])), negent=b["negent_bits"], ordering_ok=bool(ok)))
        print(f"{p:>5.2f} {gamma:>6.3f} {bound:>6.3f} {rho_blk:>6.3f} {rho_cell:>6.3f} {attack:>6.3f} {attack_se:>5.3f} "
              f"{rows[-1]['cap_floor']:>5.3f} {'OK' if ok else 'VIO':>3}", flush=True)

    os.makedirs("results/tightness", exist_ok=True)
    path = f"results/tightness/subbit_{ds}_{a.rep}.json"
    with open(path, "w") as f_out:
        json.dump(dict(dataset=ds, rep=a.rep, d=d, n_est=n, HQ=H_Q, NU2=NU2, rows=rows), f_out, indent=1)
    print(f"wrote {path} (ordering OK at every p: {all(r['ordering_ok'] for r in rows)})", flush=True)


if __name__ == "__main__":
    main()
