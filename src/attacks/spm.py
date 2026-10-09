"""Shadow posterior-mean (SPM) attack: full-dimensional release, posterior mean over a shadow-set candidate prior
(the per-step mixture likelihood of Hayes, Balle & Mahloujifar, NeurIPS 2023, App. G).

The release is generated in the full input space: G_t = 1_t a(u) u + sigma C xi_t, xi_t ~ N(0, I_d), where u is the
centred target, a(u) = min(1, C / ||u||) its clipping factor and 1_t its participation in step t. The likelihood is
evaluated in full d against the clipped candidates c_j = a(z_j) z_j of the shadow set D_shadow (58k MNIST, 44k
CelebA). To keep that tractable the attack batches over targets and streams candidate chunks (candidate matrix on
the CPU, one chunk on the GPU at a time): scores in one pass, posterior mean in a second."""
import math

import torch

from src.privacy.gamma import gamma_prv_mean

LN2 = math.log(2.0)


def gen_release_single(clipped_target_grads, noise_std, seed, gamma_bits, device):
    """Single full-batch release (q = 1, T = 1) for every target.

    Args:
        clipped_target_grads: (n_targets, d) each target's clipped gradient a(u) u (= the clipped, centred target).
        noise_std: noise standard deviation sigma * C.
        seed: random seed.
        gamma_bits: leakage the release is calibrated to; with seed, fixes the noise.
        device: torch device.

    Returns:
        (n_targets, d) releases, one noisy gradient per target.
    """
    n, d = clipped_target_grads.shape
    g = torch.Generator(device=device).manual_seed(seed*7 + int(gamma_bits*1000))
    return clipped_target_grads + torch.randn(n, d, generator=g, device=device) * noise_std


def gen_release_sub(clipped_target_grad, noise_std, seed, gamma_bits, target_idx, T, q, device):
    """T-step Poisson-subsampled release for ONE target, participating in each step with probability q.

    Args:
        clipped_target_grad: (d,) the target's clipped gradient a(u) u (= the clipped, centred target).
        noise_std: noise standard deviation sigma * C.
        seed: random seed.
        gamma_bits: leakage the release is calibrated to; with seed and target_idx, fixes participation and noise.
        target_idx: index of the target.
        T: number of steps.
        q: sampling rate.
        device: torch device.

    Returns:
        release: (T, d) one noisy step per row.
        participation: (T,) 0/1 participation in each step; the attacker does not see it.
    """
    d = clipped_target_grad.numel()
    g = torch.Generator(device=device).manual_seed(seed*19 + int(gamma_bits*100) + target_idx)
    participation = (torch.rand(T, generator=g, device=device) < q).float()
    noise = torch.randn(T, d, generator=g, device=device) * noise_std
    release = participation[:, None] * clipped_target_grad[None, :] + noise
    return release, participation


def attack_batched(releases, cand_clipped_grads, cand_norm2, cand_clip_factor, noise_std, q, device, chunk=8192):
    """Posterior over the candidates given each release, and its mean.

    Step t scores candidate j by the log likelihood ratio L_tj = (2 <G_t, c_j> - ||c_j||^2) / (2 noise_std^2), mixed
    over the unseen participation as log((1 - q) + q exp(L_tj)) (just L_tj at q = 1) and summed over steps; the
    weights are the softmax of the scores and the reconstruction is sum_j w_j z_j, with z_j = c_j / a_j.

    Args:
        releases: (n_targets, T, d) each target's release, on ``device``.
        cand_clipped_grads: (n_cand, d) clipped candidate gradients c_j, on the CPU.
        cand_norm2: (n_cand,) squared norms ||c_j||^2, on ``device``.
        cand_clip_factor: (n_cand,) clipping factors a_j, on ``device``.
        noise_std: noise standard deviation sigma * C.
        q: sampling rate (1.0 for the single full-batch step).
        device: torch device.
        chunk: candidates moved to ``device`` at a time.

    Returns:
        u_mean: (n_targets, d) posterior mean of the centred target.
        u_map: (n_targets, d) the most probable candidate z_j.
        max_score: (n_targets,) the largest score.
        entropy: (n_targets,) entropy of the posterior weights, in nats.
        w_map: (n_targets,) weight of the most probable candidate.
    """
    n_targets, _, d = releases.shape; n_cand = cand_clipped_grads.shape[0]
    scores = torch.empty(n_targets, n_cand, dtype=torch.float64, device=device)
    for i in range(0, n_cand, chunk):                                              # pass 1: log-evidence
        cand = cand_clipped_grads[i:i+chunk].to(device, non_blocking=True).float()       # (ch,d)
        dots = torch.einsum("ntd,cd->ntc", releases, cand).double()                # (n_targets,T,ch)  <G_t,c_j>
        llr = (2.0*dots - cand_norm2[i:i+chunk][None, None, :].double()) / (2.0*noise_std*noise_std)
        if q >= 1.0:
            scores[:, i:i+chunk] = llr.sum(1)
        else:
            log1mq = math.log(1.0-q); logq = math.log(q)
            scores[:, i:i+chunk] = torch.logaddexp(torch.full_like(llr, log1mq), logq + llr).sum(1)
    weights = torch.softmax(scores, 1)                                             # (n_targets,n_cand) float64
    entropy = -(weights * torch.log(weights + 1e-300)).sum(1)                      # (n_targets,)
    j_map = torch.argmax(scores, 1)                                                # (n_targets,)
    w_map = weights[torch.arange(n_targets, device=device), j_map]
    u_mean = torch.zeros(n_targets, d, dtype=torch.float32, device=device)
    u_map = torch.zeros(n_targets, d, dtype=torch.float32, device=device)
    unclip = (1.0 / cand_clip_factor)                                              # z_j = c_j / a_j
    for i in range(0, n_cand, chunk):  # pass 2: posterior mean (+ gather MAP)
        cand = cand_clipped_grads[i:i+chunk].to(device, non_blocking=True).float()       # (ch,d)
        w_unclip = (weights[:, i:i+chunk] * unclip[i:i+chunk][None]).float()  # (n_targets,ch) weights on c_j -> z_j
        u_mean += w_unclip @ cand
        hit = (j_map >= i) & (j_map < i+chunk)  # targets whose MAP lands in this chunk
        if hit.any():
            idx = (j_map[hit] - i)
            u_map[hit] = cand[idx] * unclip[i:i+chunk][idx][:, None]
    return u_mean, u_map, scores.max(1).values, entropy, w_map


def nm_of_gamma(gamma_bits):
    """Noise multiplier of the single full-batch step that leaks a given leakage: gamma = 1 / (2 ln2 sigma^2).

    Args:
        gamma_bits: leakage in bits.

    Returns:
        The noise multiplier sigma.
    """
    return math.sqrt(0.5/(gamma_bits*LN2))


def solve_sigma(T, q, gamma_bits):
    """Noise multiplier at which T Poisson-subsampled steps at rate q leak a given leakage (bisection).

    Args:
        T: number of steps.
        q: sampling rate.
        gamma_bits: leakage in bits.

    Returns:
        The noise multiplier sigma.
    """
    lo, hi = 1e-3, 1e4
    for _ in range(100):
        mid = math.sqrt(lo*hi); lo, hi = (mid, hi) if gamma_prv_mean(T, q, mid)/LN2 > gamma_bits else (lo, mid)
    return math.sqrt(lo*hi)
