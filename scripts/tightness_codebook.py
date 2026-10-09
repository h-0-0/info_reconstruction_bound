"""The codebook rows of the tightness experiment (fig:tightness, tab:tightness) at k = 1 and nu^2 = the bound's smallest.

For each rate R, a codebook of 2^R codewords is fitted on D_shadow (k-means, then refitted with equal-size cells on
D_est), and the release Q is each record's nearest codeword. The paper reports, per R (keys in the output rows):

    rho_blk_N0       the per-block bound rho_blk (prop:block_leakage) for this release
    bal_dec_target   rho_cb, the cell-mean decoder's error on the held-out D_targets (+ bal_dec_se)
    Hhat_bal         the release's entropy H(Q)

and, per nu^2 (per_nu), the per-cell bound rho2_full with its estimates on the two halves of D_est (rho2_H1, rho2_H2),
the figure's band (src/tightness/percell.py: cell_bounds). Replacing the per-cell bound's max-entropy terms by their
Gaussian values gives the cell-mean decoder's error exactly, so the gap between the two is the in-cell negentropy.
With --n-coords the entropies are estimated on a coordinate subset.

Self-checks: J1, at R = 0 (one cell) the per-cell bound equals the power-mean bound fed the same entropies (an
identity); J2, the per-cell bound is at most the decoder's error + 2 SE. Writes
results/tightness/percell_k1<suffix>_<dataset>_<rep>.json (the paper's runs: --targets-decoder --min-nu2-only
--out-suffix bal), rewritten after every rate.
"""
import argparse
import json
import math
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation at high rates
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.bound.plugin import plugin_entropy_all_blocks  # noqa: E402
from src.bound.rho import rho_star  # noqa: E402
from src.tightness.codebook import balanced_labels, entropy_bits, kmeans, nearest_codeword  # noqa: E402
from src.tightness.percell import (CELL_CAP, cell_bounds, h1d_cells, maxent_var, tightness_data,  # noqa: E402
                                   within_cell_variance)

LN2 = math.log(2.0)
TWO_PI_E = 2.0 * math.pi * math.e
Q_TOTAL = 8000                 # Monte-Carlo queries per coordinate in J1 (dithers per centre = Q_TOTAL / m)
N_BOOT = 300                   # bootstrap resamples of the targets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--rep", default="pixel", choices=["pixel", "lpips"])
    ap.add_argument("--rates", default="1,2,3,4,5,6,7,8,9,10,11,12", help="codebook rates R (bits)")
    ap.add_argument("--n-min", type=int, default=20, help="stop at the first rate with a cell smaller than this")
    ap.add_argument("--n-mc", type=int, default=32, help="recorded in the output (not used at k = 1)")
    ap.add_argument("--n-coords", type=int, default=0, help="estimate the entropies on this many coordinates (0: all)")
    ap.add_argument("--n-rep", type=int, default=12, help="queries per cell = n_rep * 250 * (1 + nu^2)")
    ap.add_argument("--out-suffix", default="", help="output: percell_k1<suffix>_<dataset>_<rep>.json")
    ap.add_argument("--targets-decoder", action="store_true", help="also the decoders' errors on D_targets")
    ap.add_argument("--min-nu2-only", action="store_true", help="only the smallest nu^2 (the reported one)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)              # every numpy draw below comes from this, in a fixed order
    ds = a.dataset
    out_path = f"results/tightness/percell_k1{a.out_suffix}_{ds}_{a.rep}.json"
    n_min_half = max(8, a.n_min // 2)

    data = tightness_data(ds, a.rep, a.data_dir, dev, targets=a.targets_decoder)
    Y, F_est = data.Y, data.F_est
    k1_pairs = [p for p in data.frame.bound["pairs"] if int(p["k"]) == 1]
    NU2 = sorted({float(p["nu2"]) for p in k1_pairs})
    if a.min_nu2_only:
        NU2 = [min(NU2)]
    n, d = Y.shape
    N_k, sqrt_Nk = d, math.sqrt(d)                   # k = 1: one block per coordinate
    # the cell variances (the decoder's error) use every coordinate; the entropies only the subset
    if a.n_coords and 0 < a.n_coords < d:
        coords = np.sort(rng.choice(d, a.n_coords, replace=False))
        print(f"[codebook] entropies on {a.n_coords}/{d} coordinates; cell variances on all", flush=True)
    else:
        coords = np.arange(d)
    Y_sub = np.ascontiguousarray(Y[:, coords])
    perm = rng.permutation(n)
    half1 = np.sort(perm[:n // 2])
    in_half1 = np.zeros(n, bool)
    in_half1[half1] = True
    n_half1, n_half2 = len(half1), n - len(half1)

    def decode_targets(est_labels, M, codebook):
        """rho of the cell-mean decoder on D_targets: each target goes to its nearest codeword and is decoded to the
        mean of that cell's D_est records (in Y). Returns (rho, bootstrap SE)."""
        cell_mean = np.zeros((M, d), np.float32)
        for c in range(M):
            members = np.where(est_labels == c)[0]
            if members.size:
                cell_mean[c] = np.asarray(Y[members], np.float32).mean(0)
        decoded = torch.as_tensor(cell_mean[nearest_codeword(data.F_tgt, codebook)], device=dev)
        e = ((torch.as_tensor(np.asarray(data.Y_tgt, np.float32), device=dev) - decoded) ** 2).sum(1)
        e = e.clamp_min(0).double().cpu().numpy()
        boots = [math.sqrt(e[rng.integers(0, len(e), len(e))].mean()) for _ in range(N_BOOT)]
        return math.sqrt(float(e.mean())), float(np.std(boots))

    print(f"[codebook {ds}/{a.rep}] n_est={n} d={d} n_min={a.n_min} NU2={NU2}", flush=True)

    # J1: at R = 0 the per-cell bound and the power-mean bound coincide
    j1 = {}
    for nu2 in NU2:
        centres = rng.choice(n, CELL_CAP, replace=False) if n > CELL_CAP else np.arange(n)
        X = np.asarray(Y_sub[centres], np.float32) * np.float32(sqrt_Nk)
        h_b = plugin_entropy_all_blocks(X, 1, math.sqrt(nu2), -(-Q_TOTAL // len(centres)), a.seed + 1,
                                        device=dev)[0] / LN2
        p2 = float(maxent_var(h_b).mean()) - nu2
        h_a2 = 0.5 * math.log2(float(np.mean(np.exp2(2.0 * h_b / 1))))          # power-mean entropy
        floor_a2 = (1 / TWO_PI_E) * (2.0 ** (2.0 * h_a2 / 1)) - nu2
        j1[nu2] = dict(P2=math.sqrt(max(0, p2)), floor_a2=math.sqrt(max(0, floor_a2)), identity_abs=abs(p2 - floor_a2),
                       rho_pt_floor=rho_star(0.0, [p for p in k1_pairs if p["nu2"] == nu2], "h_hat_bits")["rho_star"])
    j1_ok = all(v["identity_abs"] < 1e-6 for v in j1.values())
    print(f"J1 (R=0 power-mean identity): {'PASS' if j1_ok else 'FAIL'}  "
          f"max|P2-floor_a2|={max(v['identity_abs'] for v in j1.values()):.2e}", flush=True)

    nu2_0 = min(NU2)
    # marginal entropies h(z~_b) over all of D_est (rate-independent), for the per-block leakage gamma_b
    H_marg = {nu2: h1d_cells(Y_sub, [np.arange(n)], math.sqrt(nu2), int(a.n_rep * 250 * (1.0 + nu2)),
                             a.seed * 977 + int(nu2 * 100), dev, sqrt_Nk, rng=rng)[0] for nu2 in NU2}

    rows, R_max_full, R_max_half = [], None, None
    for R in [int(x) for x in a.rates.split(",")]:
        M = 2 ** R
        codebook_shadow = kmeans(data.F_shadow, M, a.seed, dev)      # ordinary codebook: k-means on D_shadow
        labels_ord = nearest_codeword(F_est, codebook_shadow)
        # balanced codebook: codewords refitted as the means of equal-size cells of D_est; the release is still
        # each record's nearest codeword
        labels_fit = balanced_labels(F_est, codebook_shadow, dev)
        codebook = np.stack([F_est[labels_fit == c].mean(0) if (labels_fit == c).any() else codebook_shadow[c]
                             for c in range(M)])
        labels = nearest_codeword(F_est, codebook)
        counts = np.bincount(labels, minlength=M)
        if counts.min() < a.n_min:
            print(f"R={R}: smallest cell {counts.min()} < n_min={a.n_min}; stop (R_max={R_max_full})", flush=True)
            break
        R_max_full = R
        if a.targets_decoder:
            bal_dec, bal_se = decode_targets(labels, M, codebook)
            ord_dec, ord_se = decode_targets(labels_ord, M, codebook_shadow)
            H_bal, H_ord = entropy_bits(labels), entropy_bits(labels_ord)
        else:
            bal_dec = bal_se = ord_dec = ord_se = H_bal = H_ord = None
        cells = [np.where(labels == c)[0] for c in range(M) if counts[c] > 0]
        cells1, cells2 = [c[in_half1[c]] for c in cells], [c[~in_half1[c]] for c in cells]
        do_half = all(c1.size >= n_min_half and c2.size >= n_min_half for c1, c2 in zip(cells1, cells2))
        if do_half:
            R_max_half = R
        # (cells, their probabilities, within-cell variances on all coordinates and on the subset) for D_est and halves
        parts = [(cs, np.array([c.size for c in cs]) / total, within_cell_variance(Y, cs, N_k),
                  within_cell_variance(Y_sub, cs, N_k))
                 for cs, total in ([(cells, n)] + ([(cells1, n_half1), (cells2, n_half2)] if do_half else []))]
        per_nu = {}
        for i, nu2 in enumerate(NU2):
            s, q_target = math.sqrt(nu2), int(a.n_rep * 250 * (1.0 + nu2))
            seeds = [a.seed * 131 + R * 17 + i * 7919, a.seed * 191 + R * 13 + i * 5003, a.seed * 251 + R * 29 + i * 6011]
            b = [cell_bounds(P, h1d_cells(Y_sub, cs, s, q_target, seed, dev, sqrt_Nk, rng=rng), H_marg[nu2], V_all,
                             V_sub, coords, nu2, (-np.inf, float(R)))     # gamma_b <= R: I(z_b; Q) <= H(Q) <= R
                 for (cs, P, V_all, V_sub), seed in zip(parts, seeds)]
            per_nu[nu2] = dict(rho2_full=b[0]["rho2_cell"], rho_full=math.sqrt(max(0, b[0]["rho2_cell"])),
                               cap_floor=math.sqrt(max(0, b[0]["cap2"])), rho2_blk=b[0]["rho2_blk"],
                               rho_blk=math.sqrt(max(0, b[0]["rho2_blk"])), gamma_b_mean=b[0]["gamma_b_mean"],
                               coord_mc_se_rho2=b[0]["mc_se"], n_coords_used=len(coords),
                               negent_bits=b[0]["negent_bits"], rho2_H1=b[1]["rho2_cell"] if do_half else None,
                               rho2_H2=b[2]["rho2_cell"] if do_half else None, q_target=q_target)
        row0 = per_nu[nu2_0]
        band = (abs(math.sqrt(max(0, row0["rho2_H1"])) - math.sqrt(max(0, row0["rho2_H2"]))) if do_half
                else float("nan"))
        # J2: the per-cell bound is at most the decoder's error (+ 2 SE)
        j2_ok = row0["rho_full"] <= (bal_dec + 2 * bal_se if a.targets_decoder else row0["cap_floor"]) + 1e-9
        rows.append(dict(R=R, nu2_N0=nu2_0, rho_N0=row0["rho_full"], rho_blk_N0=row0["rho_blk"],
                         gamma_b_mean_N0=row0["gamma_b_mean"], band_N0=band, negent_bits=row0["negent_bits"],
                         cap_floor=row0["cap_floor"], bal_dec_target=bal_dec, bal_dec_se=bal_se,
                         ord_dec_target=ord_dec, ord_dec_se=ord_se, Hhat_bal=H_bal, Hhat_ord=H_ord, do_half=do_half,
                         per_nu={str(nu): per_nu[nu] for nu in NU2}, J2_ok=bool(j2_ok)))
        print(f"R={R:>2} blk={row0['rho_blk']:.3f} cell={row0['rho_full']:>5.3f} band={band:>5.3f} "
              f"cap={row0['cap_floor']:>5.3f} neg={row0['negent_bits']:>6.3f} "
              f"balDec={'n/a' if bal_dec is None else f'{bal_dec:.3f}'} {'OK' if j2_ok else 'VIOL':>4}", flush=True)
        os.makedirs("results/tightness", exist_ok=True)
        with open(out_path, "w") as f:                               # after every rate: long runs keep their rows
            json.dump(dict(dataset=ds, rep=a.rep, k=1, n_min=a.n_min, n_mc=a.n_mc, n_est=n, d=d, R_max_full=R_max_full,
                           R_max_half=R_max_half, NU2=NU2, J1=dict(pass_=j1_ok, per_nu={str(nu): j1[nu] for nu in NU2}),
                           rows=rows), f, indent=1)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
