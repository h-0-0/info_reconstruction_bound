"""(epsilon, delta) accounting of DP-SGD, under add/remove adjacency with sensitivity C (Opacus's convention).

  compute_epsilon_prv     the paper's epsilon: the PRV accountant (Gopi et al. 2021); closed form for a single step
  compute_delta           delta = n^{-1.1}
"""

from __future__ import annotations

import math

from opacus.accountants import PRVAccountant
from scipy.optimize import brentq
from scipy.stats import norm


def compute_delta(n: int) -> float:
    """delta = n^{-1.1}.

    Args:
        n: training-set size.

    Returns:
        delta.
    """
    return n ** (-1.1)


def _analytic_gaussian_eps(sigma: float, delta: float, sens: float = 1.0) -> float:
    """Exact epsilon of one Gaussian mechanism at delta (the analytic Gaussian mechanism, Balle & Wang 2018)."""
    mu = sens / sigma

    def delta_of_eps(eps):
        return norm.cdf(mu / 2 - eps / mu) - math.exp(min(eps, 700.0)) * norm.cdf(-mu / 2 - eps / mu) - delta

    hi = 5.0
    while delta_of_eps(hi) > 0 and hi < 1e6:
        hi *= 2
    return float(brentq(delta_of_eps, 1e-12, hi))


def compute_epsilon_prv(sigma_m: float, delta: float, q: float, T: int) -> float:
    """The paper's epsilon of a DP-SGD schedule, from the PRV accountant.

    A single full-batch step (q = 1, T = 1, the attack's release) is one Gaussian mechanism, whose epsilon has a
    closed form; there Opacus's numerical accountant, which works on a grid, would overstate it by ~0.5%. Every other
    schedule uses Opacus's PRV accountant.

    Args:
        sigma_m: noise multiplier.
        delta: target delta.
        q: sampling rate.
        T: number of steps.

    Returns:
        epsilon.
    """
    if q >= 1.0 and int(T) == 1:
        return _analytic_gaussian_eps(sigma_m, delta)
    acc = PRVAccountant()
    acc.history = [(float(sigma_m), float(q), int(T))]
    return float(acc.get_epsilon(delta=delta))
