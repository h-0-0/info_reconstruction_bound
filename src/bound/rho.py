"""Final step of the bound: the lower bound rho* on the reconstruction ratio (eq:rho_certified).

  rho_star(gamma, pairs)            rho* from the leakage gamma and a bound's (k, nu^2) entropy grid
  rho_curve(pairs)                  rho*(gamma) on a log grid of leakages (the tightness figures' curve)
  load_bound(dataset, space)        the shipped grid (results/bounds/<space>_<dataset>/entropy_bound.json)
  rho_at_schedule(dataset, nm, ...) rho* for a DP-SGD schedule (q, nm, T), also in grey levels or LPIPS units

The grid is computed once per dataset (scripts/run_entropy_bound.py, src/bound/grid.py); rho* for any schedule is then
a few microseconds of arithmetic. The mechanism enters only through gamma.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.privacy.gamma import gamma_prv_mean

LN2 = math.log(2.0)

BOUNDS_DIR = "results/bounds"
PIXEL_BOUND_DIR = {ds: f"{BOUNDS_DIR}/pixel_{ds}" for ds in ("mnist", "cifar10", "celeba", "imagenette")}
# ImageNette has no LPIPS bound: Phi's AlexNet backbone was trained on ImageNet, which contains every ImageNette
# image, so Phi is not admissible there.
LPIPS_BOUND_DIR = {ds: f"{BOUNDS_DIR}/lpips_{ds}" for ds in ("mnist", "cifar10", "celeba")}


def rho_star(gamma_nats: float, pairs: list[dict], field: str = "h_lb_bits") -> dict:
    """Lower bound rho* on the reconstruction ratio, on the unit-trace scale (eq:rho_certified):

        rho*^2 = max over (k, nu^2) of ( k/(2 pi e) * 2^{2 (h_LB(k) - gamma)/k} - nu^2 )_+

    Each pair is a valid bound (cor:rp), and the margins in h_LB hold for all pairs at once, so the max is too. The
    -nu^2 removes the dither exactly. A value above 1 (possible only with field="h_hat_bits" or a point-estimate
    normaliser, rem:rho_le_one) is capped at 1 and flagged.

    Args:
        gamma_nats: the per-record leakage gamma, in nats.
        pairs: the bound's per-pair records (entropy_bound.json["pairs"]), each with k, nu2 and ``field``.
        field: "h_lb_bits" for the bound; "h_hat_bits" for the point estimate without margins (a reference, not a
            bound).

    Returns:
        Dict with rho_star; k_star and nu2_star, the maximising pair (None if every pair gives 0); per_pair, each
        pair's rho^2 by "k{k}_nu{nu2}"; and capped_at_one.
    """
    gamma_bits = gamma_nats / LN2
    two_pi_e = 2.0 * math.pi * math.e
    best_sq, best_k, best_nu2 = 0.0, None, None
    per_pair = {}
    for p in pairs:
        k, nu2, h_lb = int(p["k"]), float(p["nu2"]), float(p[field])
        expo = 2.0 * (h_lb - gamma_bits) / k
        try:
            val = (k / two_pi_e) * (2.0 ** expo) - nu2           # 2**expo underflows to 0 at huge gamma: fine
        except OverflowError:
            val = float("inf")
        rho_sq = max(0.0, val)
        per_pair[f"k{k}_nu{nu2:g}"] = rho_sq
        if rho_sq > best_sq:
            best_sq, best_k, best_nu2 = rho_sq, k, nu2
    rho = math.sqrt(best_sq)
    capped = rho > 1.0
    if capped:
        rho = 1.0
    return {
        "rho_star": float(rho),
        "k_star": best_k,
        "nu2_star": best_nu2,
        "per_pair": per_pair,
        "capped_at_one": bool(capped),
    }


def rho_curve(pairs: list[dict], gamma_min_bits: float = 0.05, n: int = 200) -> dict:
    """rho* on a log grid of leakages, with the bound (margins in) and the point estimate (margins out).

    The grid runs from gamma_min_bits to where the point estimate reaches 0 (searched in steps of x1.3 from 0.5 bits,
    plus 10%), and to at least 220 bits (epsilon ~ 100).

    Args:
        pairs: the bound's per-pair records.
        gamma_min_bits: smallest leakage (bits).
        n: grid points.

    Returns:
        {"gamma_bits", "rho_star", "rho_pt"}: (n,) arrays.
    """
    g = 0.5
    while rho_star(g * LN2, pairs, "h_hat_bits")["rho_star"] > 1e-4 and g < 1e5:
        g *= 1.3
    gammas = np.logspace(math.log10(gamma_min_bits), math.log10(max(g * 1.1, 220.0)), n)
    return {"gamma_bits": gammas,
            "rho_star": np.array([rho_star(gb * LN2, pairs, "h_lb_bits")["rho_star"] for gb in gammas]),
            "rho_pt": np.array([rho_star(gb * LN2, pairs, "h_hat_bits")["rho_star"] for gb in gammas])}


def read_bound(path: str | Path) -> dict:
    """Read a bound record, checking it was written by the plug-in pipeline.

    Args:
        path: path of an entropy_bound.json (written by scripts/run_entropy_bound.py).

    Returns:
        The full record (see src.bound.grid.entropy_bound_grid).
    """
    with open(path) as f:
        bound = json.load(f)
    if bound.get("version") != "plugin-v1":
        raise ValueError(f"{path}: not a plugin-v1 bound record (version={bound.get('version')!r})")
    return bound


def load_bound(dataset: str, space: str = "pixel") -> dict:
    """What rho* and its unit conversions need from a shipped bound.

    Args:
        dataset: dataset name.
        space: "pixel" or "lpips".

    Returns:
        Dict with pairs (the per-pair records) and tr_sigma (tr Sigma_hat on D_perp, raw units), plus
        prior_rmse_per_pixel (RMSE of the prior-mean guess per pixel, grey levels) for "pixel", or prior_sqrt_lpips
        (sqrt(tr Sigma_Phi), the prior-mean guess's sqrt(E[LPIPS])) for "lpips".
    """
    if space == "pixel":
        bound = read_bound(Path(PIXEL_BOUND_DIR[dataset]) / "entropy_bound.json")
        return {"pairs": bound["pairs"], "tr_sigma": float(bound["tr_Sigma_raw"]),
                "prior_rmse_per_pixel": float(bound["prior_rmse_perdim"])}
    if space == "lpips":
        bound = read_bound(Path(LPIPS_BOUND_DIR[dataset]) / "entropy_bound.json")
        tr_sigma = float(bound["tr_Sigma_raw"])
        return {"pairs": bound["pairs"], "tr_sigma": tr_sigma, "prior_sqrt_lpips": math.sqrt(tr_sigma)}
    raise ValueError(f"unknown space {space!r} (use 'pixel' or 'lpips')")


def rho_at_schedule(dataset: str, noise_multiplier: float, q: float = 1.0, T: int = 1, metric: str = "mse",
                    bound: dict | None = None, entropy_field: str = "h_lb_bits") -> dict:
    """rho* for a DP-SGD schedule, with gamma = gamma_prv_mean(T, q, noise_multiplier) (thm:gc_canonical).

    Args:
        dataset: dataset name.
        noise_multiplier: sigma; 0 means no noise (infinite leakage, rho* = 0).
        q: sampling rate (default 1: the single full-batch step).
        T: number of steps.
        metric: "mse" (the pixel bound) or "lpips" (the LPIPS bound).
        bound: a preloaded :func:`load_bound` record (default: load the shipped one).
        entropy_field: as ``field`` in :func:`rho_star`.

    Returns:
        Dict with gamma_nats, gamma_bits, rho (= rho*), k_star and nu2_star, plus, in the metric's own units:
            "mse":   prior_rmse_per_pixel and rmse_bound_per_pixel = rho * prior_rmse_per_pixel (grey levels);
            "lpips": prior_sqrt_lpips and sqrt_lpips_bound = rho * prior_sqrt_lpips (sqrt(E[LPIPS])).
    """
    space = {"mse": "pixel", "lpips": "lpips"}.get(metric)
    if space is None:
        raise ValueError(f"unknown metric {metric!r} (use 'mse' or 'lpips')")
    if bound is None:
        bound = load_bound(dataset, space)
    if noise_multiplier <= 0.0:
        out = {"gamma_nats": float("inf"), "gamma_bits": float("inf"), "rho": 0.0, "k_star": None, "nu2_star": None}
    else:
        gamma_nats = float(gamma_prv_mean(T, q, noise_multiplier))
        res = rho_star(gamma_nats, bound["pairs"], field=entropy_field)
        out = {"gamma_nats": gamma_nats, "gamma_bits": gamma_nats / LN2, "rho": res["rho_star"],
               "k_star": res["k_star"], "nu2_star": res["nu2_star"]}
    if space == "pixel":
        out["prior_rmse_per_pixel"] = bound["prior_rmse_per_pixel"]
        out["rmse_bound_per_pixel"] = bound["prior_rmse_per_pixel"] * out["rho"]
    else:
        out["prior_sqrt_lpips"] = bound["prior_sqrt_lpips"]
        out["sqrt_lpips_bound"] = bound["prior_sqrt_lpips"] * out["rho"]
    return out
