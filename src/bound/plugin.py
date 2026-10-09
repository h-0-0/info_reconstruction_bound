"""The plug-in entropy estimator of Goldfeld et al. (IEEE-IT 2020) and the two margins of the confidence bound.

For a block z = Pi_b x with dither W ~ N(0, s^2 I_k), the plug-in is the entropy of the empirical mixture
f_hat_n(y) = (1/n) sum_j phi_s(y - z_j) (eq:plugin_def), integrated by Monte Carlo: every centre is visited with
n_mc fresh dithers, and

    h_hat_{n,MC} = -(1/(n n_mc)) sum_i sum_j ln f_hat_n(z_i + W_ij)     [nats; the paper's eq:mc_def is this / ln 2].

The mixture density is evaluated exactly (all n centres, log-sum-exp); truncating it would overstate the entropy.
Given the sample, the integrator is unbiased for h_hat_n with variance <= c_MC / (n n_mc) (Goldfeld Thm. 5(ii);
eq:c_mc), which gives t_MC; one record moves h_hat_n by at most H_b(1/n) bits (lem:bdd_diff), which gives t_MD; and
the plug-in's own bias is downward (Goldfeld eq. 70), the safe direction, so it needs no margin.
"""

from __future__ import annotations

import math

import numpy as np

LN2 = math.log(2.0)

__all__ = [
    "plugin_entropy_all_blocks",
    "c_mc_constant",
    "binary_entropy_bits",
    "t_md_bits",
    "t_mc_bits",
]


# ---------------------------------------------------------------------------
# The margins (both in bits)
# ---------------------------------------------------------------------------

def c_mc_constant(k: int, s: float, m_hat):
    """Monte Carlo variance constant c_MC = [9 k s^2 + 8 (2 + s sqrt k) m_hat + 3 (11 s sqrt k + 1) sqrt(m_hat)] / s^2
    (eq:c_mc_def); Var[h_hat_MC | sample] <= c_MC / (n n_mc) nats^2.

    c_MC depends on the scale of the data; src.bound.grid evaluates it on the standardised blocks (centred, unit
    trace).

    Args:
        k: block size.
        s: dither standard deviation.
        m_hat: mean squared norm of the block's centres, (1/n) sum_j ||z_j||^2; a float or an array (one per block).

    Returns:
        c_MC, a float or an array like m_hat.
    """
    sk = s * math.sqrt(k)
    c = (9.0 * k * s * s
         + 8.0 * (2.0 + sk) * m_hat
         + 3.0 * (11.0 * sk + 1.0) * np.sqrt(np.maximum(m_hat, 0.0))) / (s * s)
    return float(c) if np.ndim(c) == 0 else c


def binary_entropy_bits(p: float) -> float:
    """Binary entropy H_b(p) in bits; at p = 1/n, the bounded-differences constant of the plug-in (lem:bdd_diff).

    Args:
        p: probability.

    Returns:
        H_b(p) (0 at p <= 0 or p >= 1).
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def t_md_bits(n: int, G_size: int, alpha_md: float) -> float:
    """McDiarmid margin t_MD = H_b(1/n) sqrt((n/2) ln(|G| / alpha_md)) (eq:t_md), Bonferroni over the grid.

    Args:
        n: estimation sample size.
        G_size: number of grid pairs |G|.
        alpha_md: confidence budget of this margin.

    Returns:
        t_MD in bits.
    """
    return binary_entropy_bits(1.0 / n) * math.sqrt(0.5 * n * math.log(G_size / alpha_md))


def t_mc_bits(c_mc_bar: float, N_k: int, n: int, n_mc: int, G_size: int, alpha_mc: float) -> float:
    """Cantelli margin t_MC = (1/ln 2) sqrt(c_bar / (N_k n n_mc) * (|G| / alpha_mc - 1)) on the block-averaged
    integrator (eq:t_mc); the 1/N_k comes from the blocks' independent dither draws (lem:mc_moments (iii)).

    Args:
        c_mc_bar: c_MC averaged over the blocks.
        N_k: number of blocks.
        n: estimation sample size.
        n_mc: dithers per centre.
        G_size: number of grid pairs |G|.
        alpha_mc: confidence budget of this margin.

    Returns:
        t_MC in bits.
    """
    var_nats2 = c_mc_bar / (N_k * n * n_mc)
    return math.sqrt(var_nats2 * (G_size / alpha_mc - 1.0)) / LN2


# ---------------------------------------------------------------------------
# The Monte Carlo integrator (eq:mc_def)
# ---------------------------------------------------------------------------

def _resolve_device(device: str | None):
    """Torch device for ``device`` ("cuda" or "cpu"; None: CUDA if available)."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def plugin_entropy_all_blocks(
    Y: np.ndarray,
    k: int,
    s: float,
    n_mc: int,
    seed: int,
    *,
    device: str | None = None,
    z_budget_bytes: int = 1 << 30,
    dist_budget_bytes: int = 2 << 30,
    block_batch_cap: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """The plug-in estimate of every block of one (k, nu^2) pair: the exact mixture, every centre visited with n_mc
    dithers, batched across blocks with an on-device float64 accumulator.

    The memory budgets are fixed constants, not read off the device, so the batching, and with it the dither stream,
    is the same on every device and on resume. For a single block, pass Y = the block and k = its width.

    Args:
        Y: (n, d) rotated, standardised sample (array or memmap); block b is sqrt(N_k) * Y[:, b*k:(b+1)*k].
        k: block size (divides d).
        s: dither standard deviation, on the unit-trace scale.
        n_mc: dithers per centre.
        seed: seed of the dither draws.
        device: "cuda" or "cpu" (default: CUDA if available).
        z_budget_bytes: memory for the blocks' centres held at once.
        dist_budget_bytes: memory for one batch of query-to-centre products.
        block_batch_cap: most blocks per batch.

    Returns:
        h: (N_k,) estimates, in nats.
        m: (N_k,) mean squared centre norms.
    """
    n, d = Y.shape
    if d % k != 0:
        raise ValueError(f"k={k} does not divide d={d}")
    N_k = d // k
    sqrt_Nk = math.sqrt(N_k)
    log_norm = -math.log(n) - 0.5 * k * math.log(2.0 * math.pi * s * s)

    dev = _resolve_device(device)
    import torch
    # blocks per batch: capped so each batched matrix product keeps many rows (much faster on the GPU)
    bb_z = max(1, z_budget_bytes // (n * k * 4))
    bb_dist = max(1, dist_budget_bytes // (n_mc * n * 4))
    Bb = int(min(N_k, bb_z, bb_dist, block_batch_cap))
    c_tile = int(max(1, dist_budget_bytes // (Bb * n_mc * n * 4)))     # centres per batch

    # ln f_hat(q) = log_norm - ||q||^2/(2s^2) + lse_l(<q, z_l>/s^2 - ||z_l||^2/(2s^2)), without forming distances
    inv_s2 = 1.0 / (s * s)
    h = np.empty(N_k)
    m = np.empty(N_k)
    for si, b0 in enumerate(range(0, N_k, Bb)):
        b1 = min(b0 + Bb, N_k)
        nb = b1 - b0
        cols = np.ascontiguousarray(Y[:, b0 * k:b1 * k], dtype=np.float32)
        Z = torch.as_tensor(cols, device=dev)
        Z = (Z * sqrt_Nk).view(n, nb, k).permute(1, 0, 2).contiguous()     # (nb, n, k)
        bl = (Z * Z).sum(dim=2).mul_(-0.5 * inv_s2)                         # (nb, n)
        m_slice = (Z.double() ** 2).sum(dim=(1, 2)) / n                     # (nb,)
        gen = torch.Generator(device=dev)
        gen.manual_seed((int(seed) * 1_000_003 + si) & 0x7FFFFFFFFFFFFFFF)
        acc_lse = torch.zeros(nb, dtype=torch.float64, device=dev)
        acc_qn = torch.zeros(nb, dtype=torch.float64, device=dev)
        for c0 in range(0, n, c_tile):
            c1 = min(c0 + c_tile, n)
            Q = Z[:, c0:c1, :].repeat_interleave(n_mc, dim=1)              # (nb, c * n_mc, k)
            Q = Q + s * torch.randn(Q.shape, generator=gen, dtype=torch.float32, device=dev)
            acc_qn += (Q * Q).sum(dim=2).double().sum(dim=1)
            G = torch.bmm(Q * inv_s2, Z.transpose(1, 2))                    # (nb, cq, n)
            G += bl[:, None, :]
            acc_lse += torch.logsumexp(G, dim=2).double().sum(dim=1)
        nq = float(n * n_mc)
        h[b0:b1] = -log_norm + 0.5 * inv_s2 * acc_qn.cpu().numpy() / nq - acc_lse.cpu().numpy() / nq
        m[b0:b1] = m_slice.cpu().numpy()
    return h, m
