"""The codebook release of the tightness experiment (app:exp_compressor): k-means codewords fitted on D_shadow, a
size-capped (balanced) assignment, the nearest-codeword release Q, and its entropy H(Q).
"""
import math

import numpy as np
import torch
from scipy.stats import entropy


def _nearest(X: torch.Tensor, C: torch.Tensor, chunk: int) -> tuple[torch.Tensor, float]:
    """Nearest centroid of each row of X, by squared Euclidean distance, in chunks of rows.

    Args:
        X: (n, d) points.
        C: (M, d) centroids.
        chunk: rows per chunk.

    Returns:
        (n,) index of the nearest centroid.
        Total squared distance to the nearest centroids (the k-means distortion).
    """
    labels = torch.empty(len(X), dtype=torch.long, device=X.device)
    c_norm2 = (C * C).sum(1)
    distortion = 0.0
    for s in range(0, len(X), chunk):
        xb = X[s:s + chunk]
        d2 = (xb * xb).sum(1, keepdim=True) + c_norm2[None, :] - 2.0 * (xb @ C.T)
        d2_min, labels[s:s + chunk] = d2.min(1)
        distortion += float(d2_min.sum())
    return labels, distortion


def kmeans(X: np.ndarray, M: int, seed: int, device, iters: int = 30, restarts: int = 3,
           chunk: int = 16384) -> np.ndarray:
    """Lloyd's k-means in torch: random-point initialisation, empty clusters reseeded to random points, the lowest
    distortion over restarts kept.

    Args:
        X: (n, d) points.
        M: number of centroids.
        seed: random seed.
        device: torch device.
        iters: Lloyd iterations per restart.
        restarts: number of restarts.
        chunk: rows per distance chunk.

    Returns:
        (M, d) float64 centroids.
    """
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    n = len(Xt)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    best_C, best_distortion = None, math.inf
    for _ in range(restarts):
        C = Xt[torch.randperm(n, generator=g, device=device)[:M]].clone()
        for _ in range(iters):
            labels, _ = _nearest(Xt, C, chunk)
            sums = torch.zeros_like(C).index_add_(0, labels, Xt)          # CUDA atomics: not bit-reproducible
            counts = torch.zeros(M, device=device).index_add_(0, labels, torch.ones(n, device=device))
            empty = counts == 0
            C = sums / counts.clamp_min(1.0)[:, None]
            if empty.any():
                C[empty] = Xt[torch.randperm(n, generator=g, device=device)[:int(empty.sum())]]
        _, distortion = _nearest(Xt, C, chunk)
        if distortion < best_distortion:
            best_distortion, best_C = distortion, C.clone()
    return best_C.double().cpu().numpy()


def nearest_codeword(X: np.ndarray, codebook: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """The release rule: the index of each point's nearest codeword.

    Args:
        X: (n, d) points.
        codebook: (M, d) codewords.
        chunk: rows per distance chunk.

    Returns:
        (n,) int64 codeword indices.
    """
    c_norm2 = (codebook * codebook).sum(1)
    out = np.empty(len(X), dtype=np.int64)
    for s in range(0, len(X), chunk):
        xb = X[s:s + chunk]
        d2 = (xb * xb).sum(1, keepdims=True) + c_norm2[None, :] - 2.0 * (xb @ codebook.T)
        out[s:s + chunk] = d2.argmin(1)
    return out


def entropy_bits(labels: np.ndarray) -> float:
    """Empirical entropy of a labelling.

    Args:
        labels: (n,) non-negative integer labels.

    Returns:
        The entropy in bits.
    """
    return float(entropy(np.bincount(labels), base=2))


def balanced_labels(X: np.ndarray, codebook: np.ndarray, device, n_candidates: int = 96) -> np.ndarray:
    """Size-capped assignment to the codebook: each of the M cells takes at most ceil(n / M) points, so every cell
    keeps enough points at high rates. Points are placed greedily, those with the largest gap between their nearest
    and second-nearest codeword first, each into its nearest cell that still has room.

    Args:
        X: (n, d) points.
        codebook: (M, d) codewords.
        device: torch device for the distances.
        n_candidates: nearest cells tried per point; a point whose candidates are all full goes to the emptiest cell.

    Returns:
        (n,) int cell labels.
    """
    n, M = len(X), len(codebook)
    cap = -(-n // M)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Ct = torch.as_tensor(codebook, dtype=torch.float32, device=device)
    K = min(M, n_candidates)
    dist = torch.empty(n, K)
    cand = torch.empty(n, K, dtype=torch.long)
    for s in range(0, n, 8192):
        dist[s:s + 8192], cand[s:s + 8192] = torch.cdist(Xt[s:s + 8192], Ct).topk(K, dim=1, largest=False)
    order = np.argsort(-(dist[:, 1] - dist[:, 0]).numpy())
    cand = cand.numpy()
    labels = -np.ones(n, int)
    counts = np.zeros(M, int)
    leftovers = []
    for i in order:
        for c in cand[i]:
            if counts[c] < cap:
                labels[i] = c
                counts[c] += 1
                break
        else:
            leftovers.append(i)
    for i in leftovers:
        c = int(np.argmin(counts))
        labels[i] = c
        counts[c] += 1
    return labels
