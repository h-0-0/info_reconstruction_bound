"""Orthogonal rotations for the block bound: U = d^{-1/2} H V^T or U = d^{-1/2} H D (rem:choice_of_U).

H is a Hadamard matrix of order d, kept as a list of small Kronecker factors (Sylvester doubling for powers of two,
the Paley constructions for orders 12, 20, 28, ...) so it is never formed at full size. V is the eigenbasis of the
D_perp covariance ("eig" mode); D is a fixed random sign flip ("signs" mode), used where the d x d eigendecomposition
is unaffordable, after zero-padding d to the next constructible order. Both are fixed independently of the
estimation sample. Every factor's orthogonality is asserted when it is built.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import hadamard

__all__ = [
    "hadamard_factors",
    "apply_hadamard",
    "next_hadamard_dim",
    "eig_rotation",
    "sign_rotation",
    "apply_rotation",
    "block_view",
]


# ---------------------------------------------------------------------------
# Direct constructions
# ---------------------------------------------------------------------------

def _is_prime(q: int) -> bool:
    """Whether q is prime, by trial division (q is at most a few hundred here).

    Args:
        q: integer to test.

    Returns:
        True if q is prime.
    """
    return q >= 2 and all(q % f for f in range(2, math.isqrt(q) + 1))


def _is_pow2(order: int) -> bool:
    """Whether order is a power of two.

    Args:
        order: positive integer.

    Returns:
        True if order = 2^a for some a >= 0.
    """
    return order >= 1 and order & (order - 1) == 0


def _jacobsthal(q: int) -> np.ndarray:
    """Jacobsthal matrix of F_q: Q[i, j] = chi(i - j), chi the quadratic character mod q.

    Args:
        q: an odd prime.

    Returns:
        (q, q) int8 matrix with entries 0 and +-1.
    """
    chi = np.full(q, -1, dtype=np.int8)
    chi[0] = 0
    chi[np.unique((np.arange(1, q) ** 2) % q)] = 1     # quadratic residues
    idx = (np.arange(q)[:, None] - np.arange(q)[None, :]) % q
    return chi[idx]


def _paley1(q: int) -> np.ndarray:
    """Paley I Hadamard matrix H = I + S, with S = [[0, 1^T], [-1, Q]] skew-symmetric (Q: Jacobsthal).

    Args:
        q: prime with q = 3 (mod 4).

    Returns:
        (q+1, q+1) int8 matrix with H H^T = (q+1) I.
    """
    if not (_is_prime(q) and q % 4 == 3):
        raise ValueError(f"Paley I needs prime q = 3 mod 4, got {q}")
    n = q + 1
    S = np.zeros((n, n), dtype=np.int8)
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = _jacobsthal(q)
    H = (np.eye(n, dtype=np.int8) + S).astype(np.int8)
    return H


def _paley2(q: int) -> np.ndarray:
    """Paley II Hadamard matrix H = kron(S, [[1,1],[1,-1]]) + kron(I, [[1,-1],[-1,-1]]), S = [[0, 1^T], [1, Q]].

    Args:
        q: prime with q = 1 (mod 4).

    Returns:
        (2(q+1), 2(q+1)) int8 matrix with H H^T = 2(q+1) I.
    """
    if not (_is_prime(q) and q % 4 == 1):
        raise ValueError(f"Paley II needs prime q = 1 mod 4, got {q}")
    n = q + 1
    S = np.zeros((n, n), dtype=np.int8)
    S[0, 1:] = 1
    S[1:, 0] = 1
    S[1:, 1:] = _jacobsthal(q)
    A = np.array([[1, 1], [1, -1]], dtype=np.int8)
    B = np.array([[1, -1], [-1, -1]], dtype=np.int8)
    H = np.kron(S, A) + np.kron(np.eye(n, dtype=np.int8), B)
    return H.astype(np.int8)


def _paley(order: int) -> tuple[int, int] | None:
    """Which Paley construction, if any, gives a Hadamard matrix of this order.

    Args:
        order: matrix order.

    Returns:
        (1, q) for Paley I, (2, q) for Paley II, or None.
    """
    if order >= 4 and order % 4 == 0:
        if _is_prime(order - 1) and (order - 1) % 4 == 3:
            return 1, order - 1
        q = order // 2 - 1
        if _is_prime(q) and q % 4 == 1:
            return 2, q
    return None


def _direct(order: int) -> np.ndarray | None:
    """A Hadamard matrix of this order built in one step (Sylvester doubling or Paley).

    Args:
        order: matrix order.

    Returns:
        (order, order) int8 matrix, or None if neither construction applies.
    """
    if _is_pow2(order):
        return hadamard(order, dtype=np.int8)           # Sylvester doubling
    p = _paley(order)
    if p is None:
        return None
    return _paley1(p[1]) if p[0] == 1 else _paley2(p[1])


# ---------------------------------------------------------------------------
# Composite construction as a Kronecker factor list
# ---------------------------------------------------------------------------

_SYL_CHUNK = 64   # split Sylvester parts into factors <= 64 so apply is n*d*64


def _order_decomposition(d: int) -> list[int] | None:
    """Orders of directly constructible factors whose product is d: powers of two split into factors of at most
    _SYL_CHUNK, the rest absorbed by one Paley factor (the smallest that works).

    Args:
        d: target order.

    Returns:
        List of factor orders, or None if d has no such decomposition.
    """
    if d == 1:
        return []
    if d & (d - 1) == 0:                                # pure power of two
        out = []
        while d > 1:
            f = min(d, _SYL_CHUNK)
            out.append(f)
            d //= f
        return out
    # find a Paley-capable divisor t (smallest first keeps factors small),
    # then the cofactor must itself decompose (recursively).
    divs = sorted(t for t in range(4, d + 1) if d % t == 0)
    for t in divs:
        if t & (t - 1) == 0:
            continue                                    # pure 2-power: handled by cofactor path
        if _paley(t) is not None:
            rest = _order_decomposition(d // t)
            if rest is not None:
                return [t] + rest
    return None


def hadamard_factors(d: int) -> list[np.ndarray]:
    """Hadamard matrix of order d as Kronecker factors, H_d = F_1 (x) F_2 (x) ...

    Args:
        d: matrix order.

    Returns:
        List of int8 +-1 matrices F_i with F_i F_i^T = order_i I (asserted).

    Raises:
        ValueError: if d has no construction; zero-pad to next_hadamard_dim(d) instead.
    """
    orders = _order_decomposition(d)
    if orders is None:
        raise ValueError(
            f"no Hadamard construction found for order {d} "
            f"(supported: products of 2-powers and Paley orders); "
            f"use next_hadamard_dim({d}) and zero-pad")
    factors = []
    for o in orders:
        F = _direct(o)
        G = F.astype(np.int64)
        assert (G @ G.T == o * np.eye(o, dtype=np.int64)).all(), \
            f"factor of order {o} is not Hadamard"
        factors.append(F)
    assert math.prod(F.shape[0] for F in factors) == d
    return factors


def apply_hadamard(X: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
    """Multiply every row by H without forming it: y_i = H x_i (H is not symmetric in general).

    Args:
        X: (n, d) rows.
        factors: Kronecker factors of H (from hadamard_factors).

    Returns:
        (n, d) array, same dtype as X.
    """
    n, d = X.shape
    shape = [F.shape[0] for F in factors]
    Y = X.reshape([n] + shape)
    # contract axis j (of size f_j) with F_j:  y[.., i_j, ..] = sum_l F[i_j, l] x[.., l, ..]
    for j, F in enumerate(factors):
        Y = np.moveaxis(
            np.tensordot(Y, F.astype(X.dtype), axes=([j + 1], [1])), -1, j + 1)
    return Y.reshape(n, d)


def next_hadamard_dim(d_min: int) -> int:
    """Smallest constructible Hadamard order >= d_min, of the form m * 2^a with m in {1, 3, 5, 7, 49}.

    Args:
        d_min: minimum order (e.g. the raw feature dimension).

    Returns:
        The order to zero-pad to.
    """
    cands = []
    for m in (1, 3, 5, 7, 49):
        c = m
        while c < d_min:
            c *= 2
        # ensure the composite actually decomposes (needs enough 2-power for Paley)
        while _order_decomposition(c) is None:
            c *= 2
        cands.append(c)
    return min(cands)


# ---------------------------------------------------------------------------
# The two rotation modes
# ---------------------------------------------------------------------------

def eig_rotation(X_v: np.ndarray, d: int) -> np.ndarray:
    """Eigenbasis of the D_perp sample covariance, for the "eig" mode U = d^{-1/2} H V^T.

    Args:
        X_v: (n_v, d) D_perp sample (centred internally; if n_v < d the null space is completed arbitrarily).
        d: dimension.

    Returns:
        (d, d) orthogonal V, columns = eigenvectors in decreasing eigenvalue order.
    """
    Xc = np.array(X_v, dtype=np.float64)                # one float64 copy, centred in place (X_v is not modified)
    Xc -= Xc.mean(axis=0)
    C = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    w, V = np.linalg.eigh(C)                            # ascending; order irrelevant
    return V[:, ::-1].copy()                            # descending, conventional


def sign_rotation(d: int, seed: int) -> np.ndarray:
    """Random sign flip D for the "signs" mode U = d^{-1/2} H D; fixed by the seed, so independent of the data.

    Args:
        d: dimension.
        seed: random seed.

    Returns:
        (d,) array of +-1.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0]), size=d)


def apply_rotation(
    X: np.ndarray,
    factors: list[np.ndarray],
    *,
    V: np.ndarray | None = None,
    signs: np.ndarray | None = None,
    shift: np.ndarray | None = None,
    scale: float = 1.0,
    chunk: int = 4096,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate rows to the block representation: y = U (x - shift) / scale.

    U = d^{-1/2} H V^T if V is given ("eig"), or d^{-1/2} H D if signs is given ("signs"; rows narrower than d are
    zero-padded). Processed in chunks of rows.

    Args:
        X: (n, d_raw) rows; may be a memmap.
        factors: Kronecker factors of H, of total order d.
        V: (d, d) eigenbasis, for the "eig" mode.
        signs: (d_raw,) +-1 sign flip, for the "signs" mode.
        shift: (d_raw,) centring mean.
        scale: divisor (sqrt of the trace, for unit trace).
        chunk: rows per chunk.
        out: optional (n, d) array (e.g. a float32 memmap) to write into.

    Returns:
        (n, d) rotated rows (``out`` if given, else float32).
    """
    if (V is None) == (signs is None):
        raise ValueError("provide exactly one of V (eig mode) or signs (signs mode)")
    n, d_raw = X.shape
    d = math.prod(F.shape[0] for F in factors)
    if out is None:
        out = np.empty((n, d), dtype=np.float32)
    inv_sqrt_d = 1.0 / math.sqrt(d)
    for s in range(0, n, chunk):
        xb = np.array(X[s:s + chunk], dtype=np.float64)  # a copy, so the centring below can be in place
        if shift is not None:
            xb -= shift[None, :]
        if scale != 1.0:
            xb /= scale
        if V is not None:
            if d_raw != d:
                raise ValueError("eig mode requires unpadded d")
            zb = xb @ V                                  # rows: (V^T x)^T
        else:
            zb = xb * signs[None, :d_raw]
            if d_raw != d:                               # zero-pad to the Hadamard order
                zb = np.concatenate(
                    [zb, np.zeros((zb.shape[0], d - d_raw), dtype=zb.dtype)], axis=1)
        yb = apply_hadamard(zb, factors)
        yb *= inv_sqrt_d
        out[s:s + chunk] = yb                            # cast to out's dtype on assignment
    return out


def block_view(Y: np.ndarray, k: int, b: int) -> np.ndarray:
    """Block b of the rotated data, scaled as Pi_b = sqrt(N_k) P_b U.

    Args:
        Y: (n, d) rotated data.
        k: block size (divides d).
        b: block index.

    Returns:
        (n, k) float64 copy.
    """
    d = Y.shape[1]
    if d % k != 0:
        raise ValueError(f"k={k} does not divide d={d}")
    N_k = d // k
    return np.asarray(Y[:, b * k:(b + 1) * k], dtype=np.float64) * math.sqrt(N_k)
