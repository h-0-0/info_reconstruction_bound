"""Finite-sample confidence bounds on the reference scale tr Sigma from D_perp (tab:trsigma).

Pairing the i.i.d. D_perp rows by a fixed random permutation gives i.i.d. pair terms Y_k = ||x_a - x_b||^2 / 2 with
E[Y_k] = tr Sigma, each in [0, B/2], where B bounds ||x - x'||^2 a priori (pixels: 255^2 d; LPIPS: lpips_range()).
The empirical Bernstein bound (Maurer & Pontil 2009, Thm. 4) then bounds tr Sigma from either side.
"""
from __future__ import annotations

import math

import numpy as np


def pair_terms(X, chunk: int = 1000, seed: int = 0) -> np.ndarray:
    """Y_k = ||x_a - x_b||^2 / 2 over disjoint row pairs drawn by a fixed random permutation (independent of the data,
    so the Y_k are i.i.d. with mean tr Sigma whatever the order of the rows).

    Args:
        X: (n, d) sample (array or memmap; read in chunks of pairs, rows in increasing order).
        chunk: pairs per chunk.
        seed: permutation seed.

    Returns:
        (n // 2,) pair terms.
    """
    pairs = np.random.default_rng(seed).permutation(len(X))[:(len(X) // 2) * 2].reshape(-1, 2)
    out = []
    for c in range(0, len(pairs), chunk):
        idx = pairs[c:c + chunk].reshape(-1)
        order = np.argsort(idx)
        rows = np.empty((len(idx), X.shape[1]))
        rows[order] = np.asarray(X[idx[order]], np.float64)          # sorted reads: fast on a memmap
        out.append(0.5 * ((rows[0::2] - rows[1::2]) ** 2).sum(1))
    return np.concatenate(out)


def empirical_bernstein(Y: np.ndarray, B: float, delta: float = 0.05) -> tuple[float, float, int]:
    """One-sided empirical Bernstein bounds on E[Y] (Maurer & Pontil 2009, Thm. 4), each at level 1 - delta:

        E[Y] <= Ybar + sqrt(2 V ln(2 / delta) / m) + 7 (B / 2) ln(2 / delta) / (3 (m - 1))

    and the mirror-image lower bound, with V the unbiased sample variance of Y / (B / 2).

    Args:
        Y: m i.i.d. terms in [0, B / 2].
        B: a-priori bound on ||x - x'||^2.
        delta: failure probability of each bound.

    Returns:
        Lower bound, upper bound, m.
    """
    b = B / 2.0
    Z = np.asarray(Y) / b
    m = len(Z)
    assert Z.min() >= 0 and Z.max() <= 1 + 1e-9, f"pair term outside the a-priori range: max {Z.max():.4g} > 1"
    L = math.log(2.0 / delta)
    rad = math.sqrt(2 * float(np.var(Z, ddof=1)) * L / m) + 7 * L / (3 * (m - 1))
    return b * (Z.mean() - rad), b * (Z.mean() + rad), m


def lpips_range() -> float:
    """B = 2 sum_l max_c w_lc, at least every ||Phi(x) - Phi(x')||^2: ||Phi(x) - Phi(x')||^2 =
    sum_l (1 / H_l W_l) sum_hw sum_c w_lc (a_c - b_c)^2 with a, b unit-norm and nonnegative (post-ReLU features), and
    sum_c w_c (a_c - b_c)^2 <= 2 max_c w_c for such a, b (LPIPS v0.1 AlexNet weights).
    """
    from src.bound.lpips_phi import PhiExtractor
    return 2.0 * sum(float(sw.max()) ** 2 for sw in PhiExtractor(device="cpu").sqrt_w)
