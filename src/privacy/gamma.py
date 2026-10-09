"""The leakage gamma of a DP-SGD schedule (thm:gc_canonical), in nats (bits = nats / ln 2).

Under add/remove adjacency with sensitivity C, each step of the Poisson-subsampled Gaussian mechanism is dominated by
the pair P = (1 - q) N(0, sigma^2) + q N(1, sigma^2) (record present with probability q) and Q = N(0, sigma^2)
(record absent); gamma is their KL divergence, the mean of the privacy-loss random variable, summed over the steps.
"""

from __future__ import annotations

import math

import numpy as np


def gamma_prv_mean(T: int, q: float, sigma: float, n_quad: int = 256) -> float:
    """gamma = T * KL(P || Q) for T steps at a constant sampling rate and noise multiplier.

    The one-dimensional KL integral is evaluated by n_quad-node Gauss-Hermite quadrature, with
    log(P/Q)(x) = log((1 - q) + q exp((2x - 1) / (2 sigma^2))) computed via logaddexp (no overflow at small sigma).
    q = 1 is the closed form T / (2 sigma^2).

    Args:
        T: number of steps.
        q: Poisson sampling rate.
        sigma: noise multiplier (noise std / C).
        n_quad: quadrature nodes.

    Returns:
        gamma in nats (0 if q <= 0).
    """
    if q <= 0.0:
        return 0.0
    s2 = sigma ** 2
    if q >= 1.0:
        return T * 0.5 / s2

    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    sqrt2, sqrtpi = math.sqrt(2.0), math.sqrt(math.pi)
    log1mq, logq = math.log(1.0 - q), math.log(q)

    def log_ratio(x: np.ndarray) -> np.ndarray:                 # log(P/Q)(x)
        return np.logaddexp(log1mq, logq + (2.0 * x - 1.0) / (2.0 * s2))

    def gauss_expect(mu: float) -> float:                       # E_{N(mu, sigma^2)}[log(P/Q)]
        xs = mu + sqrt2 * sigma * nodes
        return float(np.sum(weights * log_ratio(xs)) / sqrtpi)

    kl = (1.0 - q) * gauss_expect(0.0) + q * gauss_expect(1.0)
    return T * kl
