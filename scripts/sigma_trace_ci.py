"""The reference scale tr Sigma on D_perp: point estimate trS_hat = mean_i ||x_i - xbar||^2 and one-sided 95%
confidence bounds trS_lcb, trS_ucb (src/bound/reference_scale.py), per dataset and space (tab:trsigma).

Pixels are on the grey-level scale (as the bound's tr_Sigma_raw); LPIPS reads the bound's cached Phi(D_perp) and checks
the features are nonnegative, which the a-priori range B assumes. Writes results/sigma_trace_ci.json
{"<space>|<dataset>": row}.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.bound.lpips_grid import phi_cache_path  # noqa: E402
from src.bound.reference_scale import empirical_bernstein, lpips_range, pair_terms  # noqa: E402
from src.bound.rho import LPIPS_BOUND_DIR  # noqa: E402
from src.data.partition import GREY, load_est_and_v  # noqa: E402

DATASETS = ["mnist", "cifar10", "celeba", "imagenette"]
DELTA = 0.05                 # each one-sided bound holds with probability >= 1 - DELTA
METHOD = "empirical Bernstein (Maurer-Pontil) on pair terms"
OUT = "results/sigma_trace_ci.json"


def trace_row(space: str, ds: str, sq_dev: np.ndarray, Y: np.ndarray, B: float, d: int) -> dict:
    """One row of the output.

    Args:
        space: "pixel" or "lpips".
        ds: dataset name.
        sq_dev: (n,) ||x_i - xbar||^2.
        Y: pair terms.
        B: a-priori bound on ||x - x'||^2.
        d: dimension.

    Returns:
        trS_hat, trS_lcb, trS_ucb, n, n_pairs, range_B, delta, d, rmse_prior = sqrt(trS_hat / d),
        rel_ucb = trS_ucb / trS_hat - 1, and the method.
    """
    tr = float(np.mean(sq_dev))
    lo, hi, n_pairs = empirical_bernstein(Y, B, DELTA)
    print(f"{space} {ds:11}: trS={tr:.4g}  95% LCB/UCB=[{lo:.4g},{hi:.4g}]  n={len(sq_dev)}  pairs={n_pairs}  "
          f"rel_ucb={(hi / tr - 1) * 100:.3f}%", flush=True)
    return dict(space=space, dataset=ds, trS_hat=tr, trS_lcb=lo, trS_ucb=hi, n=len(sq_dev), n_pairs=n_pairs,
                range_B=B, delta=DELTA, d=d, rmse_prior=math.sqrt(tr / d), rel_ucb=hi / tr - 1, method=METHOD)


def pixel_row(ds: str, X_v: np.ndarray) -> dict:
    """The pixel row: D_perp on the grey-level scale."""
    X = np.asarray(X_v, dtype=np.float64) * GREY
    sq_dev = ((X - X.mean(0)) ** 2).sum(1)
    return trace_row("pixel", ds, sq_dev, pair_terms(X), float(GREY ** 2 * X.shape[1]), X.shape[1])


def lpips_row(ds: str, n_v: int, chunk: int = 4000) -> dict:
    """The LPIPS row: Phi(D_perp) from the bound's cache, streamed (two passes: mean, then deviations)."""
    path = phi_cache_path(LPIPS_BOUND_DIR[ds], n_v)            # keyed by size: the D_perp cache, not D_est's
    if not path.exists():
        raise SystemExit(f"{path} missing: run the LPIPS bound's --prep for {ds} first")
    F = np.load(path, mmap_mode="r")
    total, f_min = np.zeros(F.shape[1]), np.inf
    for i in range(0, len(F), chunk):
        blk = np.asarray(F[i:i + chunk], np.float64)
        total += blk.sum(0)
        f_min = min(f_min, float(blk.min()))
    assert f_min >= 0, f"Phi cache has negative entries ({f_min}); the range B assumes post-ReLU features"
    mu = total / len(F)
    sq_dev = np.concatenate([((np.asarray(F[i:i + chunk], np.float64) - mu) ** 2).sum(1)
                             for i in range(0, len(F), chunk)])
    return trace_row("lpips", ds, sq_dev, pair_terms(F), lpips_range(), F.shape[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS,
                    help="recompute only these; the other rows of the output are kept")
    a = ap.parse_args()
    out = json.load(open(OUT)) if (a.datasets != DATASETS and os.path.exists(OUT)) else {}
    for ds in a.datasets:
        _, X_v = load_est_and_v(ds, "data")
        out[f"pixel|{ds}"] = pixel_row(ds, X_v)
        if ds in LPIPS_BOUND_DIR:                                   # no LPIPS bound on ImageNette
            out[f"lpips|{ds}"] = lpips_row(ds, len(X_v))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT} ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    main()
