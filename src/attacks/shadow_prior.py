"""The mean of the shadow set D_shadow, which centres the attack's pixel-space input (u = x - mu).

It is stored as a Gaussian prior N(mu, Sigma) fitted on D_shadow and cached under results/shadow_prior/. 
Only ``mu`` is used: the attack's prior over records is the candidate set D_shadow itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

PRIOR_CACHE_DIR = "results/shadow_prior"


@dataclass
class GaussianPrior:
    mu: np.ndarray         # (d,)
    eigvals: np.ndarray    # (K,) top-K λ, descending (matched with eigvecs; K=d for the full prior)
    eigvecs: np.ndarray    # (d, K) columns = leading eigenvectors V (Σ ≈ V diag(λ) Vᵀ)
    d: int
    xnorm_hi: float = 0.0  # upper edge of the ‖x‖ grid for gain marginalisation
    eigvals_full: "Optional[np.ndarray]" = None  # ALL nonzero λ (for exact MMSE/trace when eigvecs are truncated)


def fit_gaussian_prior(X: np.ndarray) -> GaussianPrior:
    """Fit N(μ, Σ) to shadow images ``X`` (n, d) in [0,1]; eigendecompose Σ once."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Sigma = np.cov(X - mu, rowvar=False)
    lam, V = np.linalg.eigh(Sigma)            # ascending
    lam = lam[::-1].copy()
    V = V[:, ::-1].copy()
    lam = np.clip(lam, 0.0, None)
    xnorm_hi = 1.1 * float(np.sqrt((X ** 2).sum(1)).max())
    return GaussianPrior(mu=mu, eigvals=lam, eigvecs=V, d=X.shape[1], xnorm_hi=xnorm_hi, eigvals_full=lam)


def prior_cache_path(dataset: str) -> str:
    return os.path.join(PRIOR_CACHE_DIR, f"prior_{dataset}.npz")


def load_or_fit_prior(dataset: str, X_shadow: np.ndarray) -> GaussianPrior:
    path = prior_cache_path(dataset)
    if os.path.exists(path):
        d = np.load(path)
        ev_full = d["eigvals_full"] if "eigvals_full" in d.files else None
        xh = float(d["xnorm_hi"]) if "xnorm_hi" in d.files and float(d["xnorm_hi"]) > 0 else None
        return GaussianPrior(mu=d["mu"], eigvals=d["eigvals"], eigvecs=d["eigvecs"],
                             d=int(d["mu"].shape[0]), eigvals_full=ev_full, xnorm_hi=xh)
    print(f"  fitting Gaussian prior on D_shadow {X_shadow.shape} (eigendecomp)...",
          flush=True)
    prior = fit_gaussian_prior(X_shadow)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mu=prior.mu, eigvals=prior.eigvals, eigvecs=prior.eigvecs,
             eigvals_full=(prior.eigvals_full if prior.eigvals_full is not None else prior.eigvals),
             xnorm_hi=float(getattr(prior, "xnorm_hi", 0.0) or 0.0))
    return prior
