"""The confidence bound on the block entropy in LPIPS feature space: the pixel bound of :mod:`src.bound.grid` with the
representation Psi_LPIPS (``Phi`` in the code, :mod:`src.bound.lpips_phi`) in place of the pixels.

Two things differ from pixels. The sample is the feature matrix Phi(x_j), cached as float32 (the bound is about the
mixture whose centres are the stored values; float16 rounding would move it by ~1e-3, comparable to the margins).
And the rotation is the "signs" mode U = d^{-1/2} H D: the m = 31,872 features admit no Hadamard order and no
affordable m x m eigenbasis, so they are zero-padded to d = next_hadamard_dim(m) and mixed by a fixed random sign
flip (rem:choice_of_U; padding is safe, since an attack on Phi is an attack on the padded features with the same
error). ||Phi(x) - Phi(x')||^2 is exactly the LPIPS distance (checked when the cache is built), so
E[LPIPS] >= tr Sigma_Phi * rho*^2.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from src.bound.blocks import next_hadamard_dim
from src.bound.grid import DEFAULT_NU2, divisor_grid, entropy_bound_grid

__all__ = ["build_phi_cache", "lpips_entropy_bound_grid", "phi_cache_path"]


def phi_cache_path(out_dir: str | Path, n: int) -> Path:
    """Path of the cached features of an n-record sample.

    Args:
        out_dir: the bound's directory (results/bounds/lpips_<dataset>).
        n: number of records (distinguishes the D_est and D_perp caches).

    Returns:
        Path of the float32 .npy file.
    """
    return Path(out_dir) / f"phi_cache_f32_n{n}.npy"


def build_phi_cache(
    dataset: str,
    images: np.ndarray,
    out_dir: str | Path,
    *,
    layout: str,
    batch_size: int = 256,
    verify_pairs: int = 1000,
    seed: int = 42,
    device: str | None = None,
) -> Path:
    """Extract Phi of every image into a float32 .npy memmap, or reuse an existing cache of the right shape.

    Before extracting, checks ||Phi(x) - Phi(x')||^2 = LPIPS(x, x') on random pairs and stops if it fails (a wrong
    layout or scale would silently bound the wrong representation); the check is saved as phi_verification.json.

    Args:
        dataset: dataset name (sets the input transform).
        images: (n, d_pixels) images on [0, 1], flattened in ``layout`` order.
        out_dir: directory of the cache.
        layout: "chw" or "hwc", how the rows were flattened.
        batch_size: images per forward pass.
        verify_pairs: number of random pairs for the LPIPS check.
        seed: seed of the check's pairs.
        device: torch device (default: CUDA if available).

    Returns:
        Path of the cache.
    """
    import torch
    from src.bound.lpips_phi import PhiExtractor, verify_phi

    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    n = images.shape[0]
    cp = phi_cache_path(out_dir, n)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = PhiExtractor(device=device)

    if cp.exists():
        mm = np.load(cp, mmap_mode="r")
        if mm.shape == (n, ext.m):
            print(f"[phi] reusing cache {cp} {mm.shape}", flush=True)
            return cp
        print(f"[phi] cache shape {mm.shape} != {(n, ext.m)}; rebuilding", flush=True)

    ver = verify_phi(ext, images, n_pairs=verify_pairs, seed=seed, dataset=dataset, layout=layout)
    with open(out_dir / "phi_verification.json", "w") as f:
        json.dump(ver, f, indent=2)
    print(f"[phi] verify: max|err|={ver['max_abs_err']:.3e} (lpips mean {ver['lpips_mean']:.4f})", flush=True)
    if ver["max_abs_err"] > 1e-4:
        raise SystemExit(f"GATE 1 FAILED: ||Phi(x)-Phi(x')||^2 vs lpips max err {ver['max_abs_err']:.3e} > 1e-4")

    mm = np.lib.format.open_memmap(cp, mode="w+", dtype=np.float32, shape=(n, ext.m))
    for s in range(0, n, batch_size):
        mm[s:s + batch_size] = ext.phi(images[s:s + batch_size], batch_size=batch_size, dataset=dataset, layout=layout)
        if (s // batch_size) % 20 == 0:
            print(f"[phi] cached {min(s + batch_size, n)}/{n}", flush=True)
    mm.flush()
    print(f"[phi] cache -> {cp} ({cp.stat().st_size / 1e9:.2f} GB)", flush=True)
    return cp


def lpips_entropy_bound_grid(
    phi_est: np.ndarray,
    *,
    phi_v: np.ndarray | None = None,
    K: list[int] | None = None,
    k_max: int = 32,
    NU2: tuple[float, ...] | None = None,
    alpha: float = 0.05,
    n_mc: int = 16,
    seed: int = 0,
    device: str | None = None,
    rotated_cache: str | None = None,
    checkpoint_path: str | None = None,
    pair_subset: list[tuple[int, float]] | None = None,
    verbose: bool = True,
    rotated_order: str = "C",
) -> dict:
    """The bound record of :func:`src.bound.grid.entropy_bound_grid` for the LPIPS features ("signs" rotation).

    Args:
        phi_est: (n, m) features of D_est (the float32 memmap of :func:`build_phi_cache`).
        phi_v: (n_v, m) features of D_perp; fix the centring mean and the unit-trace scale.
        K: block sizes; default: the divisors of the padded dimension up to k_max.
        k_max: largest default block size.
        NU2: dither levels nu^2; default src.bound.grid.DEFAULT_NU2.
        alpha, n_mc, seed, device, rotated_cache, checkpoint_path, pair_subset, verbose, rotated_order:
            as in :func:`src.bound.grid.entropy_bound_grid`. ``rotated_cache`` should be on scratch disk: the
            rotated sample is n x 32,768 float32 (7.7 GB at n = 59,000).

    Returns:
        The bound record, plus space ("phi"), source ("lpips_alexnet_phi") and prior_sqrt_lpips = sqrt(tr Sigma_Phi),
        the RMSE of the prior-mean guess in feature space.
    """
    if K is None:
        K = divisor_grid(next_hadamard_dim(phi_est.shape[1]), k_max)
    bound = entropy_bound_grid(
        phi_est, phi_v, K=K, NU2=DEFAULT_NU2 if NU2 is None else NU2, alpha=alpha, n_mc=n_mc, seed=seed,
        rotation="signs", device=device, rotated_cache=rotated_cache, checkpoint_path=checkpoint_path,
        pair_subset=pair_subset, verbose=verbose, rotated_order=rotated_order)
    bound["space"] = "phi"
    bound["source"] = "lpips_alexnet_phi"
    bound["prior_sqrt_lpips"] = math.sqrt(bound["tr_Sigma_raw"])
    return bound
