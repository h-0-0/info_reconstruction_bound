"""The confidence bound on the block entropy over the grid G of (k, nu^2) pairs (prop:coverage, app:entropy).

For each pair: rotate the estimation sample, cut it into N_k = d/k blocks Pi_b = sqrt(N_k) P_b U, estimate each block's
dithered entropy with the plug-in (:mod:`src.bound.plugin`), average over blocks, and subtract the two margins:

    h_LB(k, nu^2) = h_hat_{n,MC} - t_MC - t_MD  <=  h_bar_U(k),   for every pair at once with probability >= 1 - alpha.

The plug-in's own bias is downward (Goldfeld eq. 70) and needs no margin. Everything is computed on the unit-trace
scale: the data are centred on the D_perp mean and divided by sqrt(tr Sigma_hat(D_perp)), so s = sqrt(nu^2 / k) and
h_LB plugs straight into rho* (:mod:`src.bound.rho`). The centring mean, the scale and the eigenbasis come from D_perp,
never from the estimation sample, so that one record moves only one plug-in centre (the bounded-differences argument
behind t_MD). This standardisation is a fixed affine map, so rho is unchanged by it; the margin t_MC is evaluated in
it (c_MC, eq:c_mc_def, is not scale-invariant).
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
from scipy.special import logsumexp

from src.bound.blocks import (
    apply_rotation,
    eig_rotation,
    hadamard_factors,
    next_hadamard_dim,
    sign_rotation,
)
from src.bound.plugin import (
    c_mc_constant,
    plugin_entropy_all_blocks,
    t_mc_bits,
    t_md_bits,
)

LN2 = math.log(2.0)
LOG2_2PIE = math.log2(2.0 * math.pi * math.e)

DEFAULT_NU2 = (0.2, 0.4, 0.8, 1.6, 3.2, 6.4)


def divisor_grid(d: int, k_max: int = 32) -> list[int]:
    """Block sizes of the grid: the divisors of d (cor:rp needs k | d) up to k_max.

    Args:
        d: dimension of the rotated representation.
        k_max: largest block size.

    Returns:
        Sorted list of block sizes k.
    """
    return [k for k in range(1, k_max + 1) if d % k == 0]


def centre_and_scale(X_scale) -> tuple[np.ndarray, float, float]:
    """Centring mean and unit-trace scale of a sample, streamed in float64 chunks.

    Args:
        X_scale: (n, d) sample (array, memmap, or anything with ``.shape`` and row slicing).

    Returns:
        mu: (d,) mean.
        tr_cov: trace of the sample covariance (divisor n - 1).
        scale: sqrt(tr_cov).
    """
    n_sc, d_raw = X_scale.shape
    rows = min(8192, max(256, (1 << 27) // d_raw))   # 8192 for d_raw <= 16384; ~1 GB chunks above
    mu = np.zeros(d_raw, dtype=np.float64)
    for s0 in range(0, n_sc, rows):
        mu += np.asarray(X_scale[s0:s0 + rows], dtype=np.float64).sum(axis=0)
    mu /= n_sc
    sq = 0.0
    for s0 in range(0, n_sc, rows):
        xb = np.asarray(X_scale[s0:s0 + rows], dtype=np.float64) - mu[None, :]
        sq += float((xb ** 2).sum())
    tr_cov = sq / (n_sc - 1)
    return mu, tr_cov, math.sqrt(tr_cov)


def bound_frame(X_v, rotation: str, seed: int) -> dict:
    """The bound's rotation and standardisation, fixed by D_perp alone.

    Args:
        X_v: (n_v, d_raw) D_perp, raw scale (array, memmap, or anything :func:`centre_and_scale` accepts).
        rotation: "eig" (U = d^{-1/2} H V^T, d = d_raw) or "signs" (U = d^{-1/2} H D, d_raw zero-padded to the next
            Hadamard order).
        seed: run seed (fixes the sign flip of "signs").

    Returns:
        {"d", "factors", "V", "signs", "mu", "tr_cov", "scale_c"}: the rotated dimension, the arguments of
        :func:`src.bound.blocks.apply_rotation`, and the centring mean, trace and unit-trace scale of D_perp.
    """
    d_raw = X_v.shape[1]
    if rotation == "eig":
        d, V, signs = d_raw, eig_rotation(X_v, d_raw), None
    elif rotation == "signs":
        d, V, signs = next_hadamard_dim(d_raw), None, sign_rotation(d_raw, seed=seed + 90_210)
    else:
        raise ValueError(f"unknown rotation {rotation!r} (use 'eig' or 'signs')")
    mu, tr_cov, scale_c = centre_and_scale(X_v)
    return {"d": d, "factors": hadamard_factors(d), "V": V, "signs": signs, "mu": mu, "tr_cov": tr_cov,
            "scale_c": scale_c}


def _pair_seed(seed: int, k: int, nu2_idx: int, b: int) -> int:
    """Seed of the dither draws of one (pair, block); deterministic, so an interrupted grid resumes identically.

    Args:
        seed: run seed.
        k: block size.
        nu2_idx: index of nu^2 in the grid.
        b: block index.

    Returns:
        A 63-bit integer seed.
    """
    ss = np.random.SeedSequence([int(seed), int(k), int(nu2_idx), int(b)])
    return int(ss.generate_state(1, dtype=np.uint64)[0] & 0x7FFFFFFFFFFFFFFF)


def _rotate(X_est, factors, V, signs, mu, scale_c, d, rotated_cache, rotated_order, verbose):
    """The rotated, standardised estimation sample, computed once and reused by every pair of the grid.

    Args:
        X_est: (n, d_raw) estimation sample, raw scale.
        factors, V, signs: the rotation (see :func:`src.bound.blocks.apply_rotation`).
        mu, scale_c: centring mean and unit-trace scale (from D_perp).
        d: rotated dimension.
        rotated_cache: optional path of a float32 .npy memmap holding the result. A complete cache (marked by a
            ``.ok`` file next to it) of the right shape is reused read-only; anything else is (re)written.
        rotated_order: "C" or "F" memory layout of the result; storage only, values identical.
        verbose: print progress.

    Returns:
        (n, d) float32 array or memmap.
    """
    n = X_est.shape[0]
    chunk = min(4096, max(256, (1 << 28) // d))       # rows per rotation chunk: bounded memory at large d
    if rotated_cache is not None and os.path.exists(rotated_cache) and os.path.exists(rotated_cache + ".ok"):
        Y = np.load(rotated_cache, mmap_mode="r")
        if Y.shape == (n, d):
            if verbose:
                print(f"[grid] reusing rotated cache {rotated_cache}", flush=True)
            return Y
        if verbose:                                    # stale cache from an earlier data generation
            print(f"[grid] rotated cache shape {Y.shape} != {(n, d)}; rebuilding", flush=True)
        del Y
        os.remove(rotated_cache)
        os.remove(rotated_cache + ".ok")
    if rotated_cache is not None:
        Y = np.lib.format.open_memmap(rotated_cache, mode="w+", dtype=np.float32, shape=(n, d),
                                      fortran_order=(rotated_order == "F"))
    else:
        Y = np.empty((n, d), dtype=np.float32, order=rotated_order)
    apply_rotation(X_est, factors, V=V, signs=signs, shift=mu, scale=scale_c, out=Y, chunk=chunk)
    if rotated_cache is not None:
        Y.flush()
        with open(rotated_cache + ".ok", "w") as f:
            f.write("ok")
    return Y


def _pair_record(Y, k, nu2, nu2_idx, n_mc, seed, device, G_size, alpha_mc, t_md):
    """Plug-in estimate, margins and diagnostics of one grid pair.

    Args:
        Y: (n, d) rotated, standardised estimation sample.
        k: block size.
        nu2: dither level nu^2 (unit-trace scale, so s = sqrt(nu^2 / k)).
        nu2_idx: index of nu2 in the grid (seeds the dither).
        n_mc: reference replicate count (see :func:`entropy_bound_grid`).
        seed: run seed.
        device: plug-in device.
        G_size: |G|, for the Bonferroni split.
        alpha_mc: confidence budget of t_MC.
        t_md: the McDiarmid margin (bits), the same for every pair.

    Returns:
        The pair's record (keys listed in :func:`entropy_bound_grid`).
    """
    n, d = Y.shape
    N_k = d // k
    s = math.sqrt(nu2 / k)
    # Monte Carlo replicates per centre: 2 at k = 1 (the d blocks average its error away), max(k, 4) above at the
    # default n_mc = 16, so the cost (d/k) * n_mc_k stays flat in k; the reference 16 is fixed so that extending the
    # k grid changes no existing pair
    n_mc_k = max(2 if k == 1 else 4, int(round(n_mc * k / 16)))
    h_b, m_b = plugin_entropy_all_blocks(Y, k, s, n_mc_k, _pair_seed(seed, k, nu2_idx, 0), device=device)
    c_mc_bar = float(np.mean(c_mc_constant(k, s, np.asarray(m_b, dtype=np.float64))))
    h_hat_bits = float(h_b.mean()) / LN2
    t_mc = t_mc_bits(c_mc_bar, N_k, n, n_mc_k, G_size, alpha_mc)
    # diagnostic only (no margins, not used by rho*): the power mean (k/2) log2(mean_b 2^{2 h_b/k}) that the bound's
    # block average lower-bounds by Jensen; equal to h_hat_bits when the blocks are balanced
    log2_powmean = logsumexp(2.0 * h_b / k, b=1.0 / N_k) / LN2       # h_b in nats: log2 mean_b 2^{2 h_b[bits] / k}
    return {
        "k": int(k),
        "nu2": float(nu2),
        "s_unit": float(s),
        "n_mc": int(n_mc_k),
        "h_hat_bits": float(h_hat_bits),
        "h_lb_bits": float(h_hat_bits - t_mc - t_md),
        "t_mc_bits": float(t_mc),
        "t_md_bits": float(t_md),
        "c_mc_bar": float(c_mc_bar),
        "h_block_bits_min": float(float(h_b.min()) / LN2),
        "h_block_bits_max": float(float(h_b.max()) / LN2),
        "h_a2_bits": float(0.5 * k * log2_powmean),
        "h_block_bits_std": float(np.std(h_b / LN2)),
        "h_gauss_iso_bits": float(0.5 * k * (LOG2_2PIE + math.log2((1.0 + nu2) / k))),
        "balance_max_dev": float(np.abs(m_b * n / (n - 1) - 1.0).max()),
    }


def entropy_bound_grid(
    X_est: np.ndarray,
    X_v: np.ndarray | None = None,
    *,
    K: list[int] | None = None,
    NU2: tuple[float, ...] = DEFAULT_NU2,
    alpha: float = 0.05,
    n_mc: int = 16,
    seed: int = 0,
    rotation: str = "eig",
    device: str | None = None,
    rotated_cache: str | None = None,
    checkpoint_path: str | None = None,
    pair_subset: list[tuple[int, float]] | None = None,
    verbose: bool = True,
    rotated_order: str = "C",
) -> dict:
    """The confidence bound h_LB(k, nu^2) at every pair of the grid G = K x NU2.

    Args:
        X_est: (n, d_raw) estimation sample D_est, raw scale (grey levels or LPIPS units); read only, may be a memmap.
        X_v: (n_v, d_raw) D_perp, disjoint from X_est: fixes the centring mean, the unit-trace scale and (for "eig")
            the eigenbasis. Required for "eig"; without it ("signs" only) the mean and scale fall back to X_est and
            t_MD is no longer strictly valid.
        K: block sizes (each must divide d); default: the divisors of d up to 32.
        NU2: dither levels nu^2 = k s^2 / tr Sigma.
        alpha: total confidence budget, split evenly between t_MC and t_MD.
        n_mc: reference replicate count; pair (k, .) draws max(2 if k == 1 else 4, round(n_mc k / 16)) dithers per
            centre, which at the default 16 is the paper's 2 at k = 1 and max(k, 4) above.
        seed: run seed (dither draws, and the sign flip of "signs").
        rotation: "eig" (U = d^{-1/2} H V^T, d = d_raw) or "signs" (U = d^{-1/2} H D, d_raw zero-padded to the next
            Hadamard order).
        device: plug-in device ("cuda" or "cpu"; default: CUDA if available).
        rotated_cache: optional path to keep the rotated sample as a float32 memmap (shared by array jobs).
        checkpoint_path: optional per-pair JSON checkpoint; a rerun resumes from it (the draws are deterministic).
        pair_subset: compute only these (k, nu^2) pairs (|G| still counts the whole grid).
        verbose: print progress.
        rotated_order: "C" or "F" layout of the rotated memmap; "F" makes each block one contiguous read once n*d
            exceeds the page cache. Values are identical.

    Returns:
        The bound record (written to entropy_bound.json):
            version, estimator, rotation, rotated_order, seed, elapsed_s;
            d, d_raw, n: rotated dimension, raw dimension, n_est;
            n_v: records that fixed the eigenbasis (0 for "signs");
            alpha, alpha_mc, alpha_md, n_mc, G_size, grid_k, grid_nu2;
            tr_Sigma_raw, scale_c, prior_rmse_perdim: tr Sigma_hat(D_perp), its square root, sqrt(tr / d_raw);
            t_md_bits: the McDiarmid margin;
            balance: per k, the largest relative deviation of a block's mean squared norm from 1;
            pairs: one record per pair with k, nu2, s_unit, n_mc, h_hat_bits, h_lb_bits, t_mc_bits, t_md_bits,
                c_mc_bar, h_block_bits_{min,max,std}, h_a2_bits, h_gauss_iso_bits (the balanced-Gaussian ceiling),
                balance_max_dev.
    """
    t_start = time.time()
    n, d_raw = X_est.shape

    if rotation == "eig" and X_v is None:
        raise ValueError("eig rotation requires the disjoint V sample X_v")
    if X_v is None:
        print("  [warn] no D_perp; centring/scale taken from X_est (McDiarmid margin not strictly valid)", flush=True)
    frame = bound_frame(X_est if X_v is None else X_v, rotation, seed)
    d, factors, V, signs = frame["d"], frame["factors"], frame["V"], frame["signs"]
    mu, tr_cov, scale_c = frame["mu"], frame["tr_cov"], frame["scale_c"]
    n_v = int(X_v.shape[0]) if rotation == "eig" else 0

    K = sorted(int(k) for k in (divisor_grid(d) if K is None else K))
    for k in K:
        if d % k != 0:
            raise ValueError(f"k={k} does not divide d={d}")
    NU2 = tuple(float(v) for v in NU2)
    G_size = len(K) * len(NU2)
    alpha_mc = alpha_md = alpha / 2.0

    if verbose:
        import torch
        dev_desc = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else "cpu"
        print(f"[grid] n={n} d_raw={d_raw} d={d} rotation={rotation} factors={[F.shape[0] for F in factors]} "
              f"K={K} NU2={list(NU2)} |G|={G_size} n_mc={n_mc} device={dev_desc}", flush=True)
    Y = _rotate(X_est, factors, V, signs, mu, scale_c, d, rotated_cache, rotated_order, verbose)
    t_md = t_md_bits(n, G_size, alpha_md)

    # resume: reuse the pairs of a checkpoint written with the same settings
    done: dict[str, dict] = {}
    ck_meta = {"n": n, "d": d, "G_size": G_size, "n_mc": n_mc, "seed": seed,
               "rotation": rotation, "n_mc_rule": "prop-k-min4", "eval": "logit-form-v4"}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ck = json.load(f)
        if ck.get("meta") == ck_meta:
            done = ck.get("pairs", {})
            if verbose and done:
                print(f"[resume] {len(done)}/{G_size} pairs from {checkpoint_path}", flush=True)
        elif verbose:
            print("[resume] checkpoint meta mismatch; recomputing all pairs", flush=True)

    # k descending: large-k pairs are the cheapest and the ones rho* usually selects, so a run that hits its wall
    # time has checkpointed the useful pairs first. The result does not depend on the order.
    pairs, balance = [], {}
    for k in sorted(K, reverse=True):
        bal_dev = None
        for j, nu2 in enumerate(NU2):
            if pair_subset is not None and (k, float(nu2)) not in pair_subset:
                continue
            key = f"k{k}_nu{nu2:g}"
            if key in done:
                rec = done[key]
                bal_dev = rec.get("balance_max_dev") if bal_dev is None else bal_dev
            else:
                t0 = time.time()
                rec = _pair_record(Y, k, nu2, j, n_mc, seed, device, G_size, alpha_mc, t_md)
                done[key] = rec
                bal_dev = rec["balance_max_dev"]
                if checkpoint_path:
                    with open(checkpoint_path + ".tmp", "w") as f:
                        json.dump({"meta": ck_meta, "pairs": done}, f)
                    os.replace(checkpoint_path + ".tmp", checkpoint_path)
                if verbose:
                    print(f"[k={k:>3} nu2={nu2:<4g}] h_hat={rec['h_hat_bits']:8.3f} bits "
                          f"(iso {rec['h_gauss_iso_bits']:8.3f})  t_MC={rec['t_mc_bits']:.4f}  t_MD={t_md:.4f}  "
                          f"bal_dev={bal_dev:.3f}  [{time.time() - t0:.1f}s]", flush=True)
            pairs.append(rec)
        if bal_dev is not None:
            balance[str(k)] = float(bal_dev)

    return {
        "version": "plugin-v1",
        "estimator": "goldfeld2020-plugin",
        "rotation": rotation,
        "rotated_order": rotated_order,
        "d": int(d),
        "d_raw": int(d_raw),
        "n": int(n),
        "n_v": n_v,
        "alpha": float(alpha),
        "alpha_mc": float(alpha_mc),
        "alpha_md": float(alpha_md),
        "n_mc": int(n_mc),
        "G_size": int(G_size),
        "grid_k": K,
        "grid_nu2": list(NU2),
        "tr_Sigma_raw": float(tr_cov),
        "prior_rmse_perdim": float(math.sqrt(tr_cov / d_raw)),
        "scale_c": float(scale_c),
        "t_md_bits": float(t_md),
        "balance": balance,
        "pairs": pairs,
        "seed": int(seed),
        "elapsed_s": float(time.time() - t_start),
    }
