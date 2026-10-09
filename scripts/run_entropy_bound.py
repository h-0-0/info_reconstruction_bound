"""Compute the entropy confidence bound h_LB(k, nu^2) of one dataset, in pixel or LPIPS-feature space.

Data (app:data_handling): the estimation sample D_est (the attack targets excluded) feeds the estimator; the disjoint,
target-free D_perp fixes the eigenbasis (pixel "eig" rotation) and the centring and unit-trace scale, so the scale
does not move with D_est (app:bdd_diff). See :func:`src.data.partition.load_est_and_v`. Pixels are on the grey-level
scale [0, 255]; LPIPS features (Phi) use the "signs" rotation.

A full bound is one job per grid pair (reproduce/run_all.sh):
    --prep                  LPIPS: build the Phi caches and the shared rotated sample; CelebA pixel: cache the
                            decoded extra images (CPU)
    --pair-index i          compute the i-th (k, nu^2) pair -> results/bounds/<space>_<ds>/pairs/pair_<i>_*.json
    --assemble              merge the pairs into results/bounds/<space>_<ds>/entropy_bound.json
Without these flags the whole grid is computed in one process.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.bound.blocks import next_hadamard_dim                    # noqa: E402
from src.bound.grid import divisor_grid, entropy_bound_grid       # noqa: E402
from src.bound.lpips_grid import build_phi_cache, lpips_entropy_bound_grid   # noqa: E402
from src.bound.rho import LPIPS_BOUND_DIR, PIXEL_BOUND_DIR, rho_star  # noqa: E402
from src.data.partition import GREY, load_est_and_v, split_policy  # noqa: E402
from src.data.pool import SPLIT_SEED                              # noqa: E402

# keys every pair file of one bound must share
_SHARED_KEYS = ["n", "d", "d_raw", "seed", "alpha", "n_mc", "G_size", "grid_k", "grid_nu2", "rotation", "dataset"]


def _write_json(path: Path, obj: dict) -> None:
    """Write JSON atomically (array jobs and readers never see a partial file)."""
    with open(f"{path}.tmp", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(f"{path}.tmp", path)


def _print_rho(pairs: list[dict]) -> None:
    """Print rho* at a few leakage levels (a sanity read-out)."""
    for g_bits in (0.0, 0.1, 1.0, 5.0, 20.0):
        r = rho_star(g_bits * np.log(2.0), pairs)
        print(f"[floor] gamma={g_bits:>5.1f} bits -> rho*={r['rho_star']:.4f} "
              f"(k*={r['k_star']}, nu2*={r['nu2_star']})", flush=True)


def _select_pair(d: int, k_max: int, NU2: tuple, index: int) -> list | None:
    """The grid pair an array task computes.

    Args:
        d: rotated dimension.
        k_max: largest block size of the grid.
        NU2: dither levels.
        index: array-task index.

    Returns:
        [(k, nu2)], or None if the index is past the end of the grid.
    """
    grid = [(k, nu2) for k in sorted(divisor_grid(d, k_max)) for nu2 in NU2]
    if index >= len(grid):
        print(f"[pair] index {index} >= |G|={len(grid)}; nothing to do", flush=True)
        return None
    print(f"[pair] computing {grid[index]}", flush=True)
    return [grid[index]]


def assemble(out: Path) -> None:
    """Merge the pair files of a bound into entropy_bound.json, checking they share one setup and cover the grid.

    Args:
        out: the bound's output directory.
    """
    files = sorted(glob.glob(str(out / "pairs" / "pair_*.json")))
    if not files:
        raise SystemExit(f"no pair files under {out / 'pairs'}")
    bounds = [json.load(open(f)) for f in files]
    base = bounds[0]
    for c, f in zip(bounds, files):
        for key in _SHARED_KEYS:
            if c.get(key) != base.get(key):
                raise SystemExit(f"{f}: {key}={c.get(key)!r} != {base.get(key)!r}")
    pairs, seen, balance = [], set(), {}
    for c in bounds:
        for p in c["pairs"]:
            if (p["k"], p["nu2"]) in seen:
                raise SystemExit(f"duplicate pair {(p['k'], p['nu2'])}")
            seen.add((p["k"], p["nu2"]))
            pairs.append(p)
        balance.update(c.get("balance", {}))
    missing = {(k, nu2) for k in base["grid_k"] for nu2 in base["grid_nu2"]} - seen
    if missing:
        raise SystemExit(f"grid incomplete: missing {sorted(missing)}")
    bound = dict(base)
    bound["pairs"] = sorted(pairs, key=lambda p: (p["k"], p["nu2"]))
    bound["balance"] = balance
    bound["elapsed_s"] = sum(c.get("elapsed_s", 0.0) for c in bounds)
    bound["assembled_from"] = len(files)
    _write_json(out / "entropy_bound.json", bound)
    print(f"[assemble] {len(pairs)} pairs -> {out / 'entropy_bound.json'}", flush=True)
    _print_rho(bound["pairs"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["mnist", "cifar10", "celeba", "imagenette"])
    ap.add_argument("--space", default="pixel", choices=["pixel", "phi"])
    ap.add_argument("--data-dir", default=str(_ROOT / "data"))
    ap.add_argument("--nu2", default="0.2,0.4,0.8,1.6,3.2,6.4", help="comma-separated nu^2 grid")
    ap.add_argument("--k-max", type=int, default=32, help="block sizes: the divisors of d up to this")
    ap.add_argument("--n-mc", type=int, default=16,
                    help="reference dither count: pair (k, .) draws max(2 if k == 1 else 4, round(n_mc k / 16)) "
                         "dithers per centre")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--batch-size", type=int, default=256, help="Phi extraction batch")
    ap.add_argument("--pair-index", type=int, default=None, help="compute only the i-th grid pair")
    ap.add_argument("--assemble", action="store_true", help="merge the pair files into entropy_bound.json")
    ap.add_argument("--prep", action="store_true", help="prepare the caches the pair jobs share; no pairs")
    args = ap.parse_args()

    NU2 = tuple(float(v) for v in args.nu2.split(","))
    if args.space == "phi" and args.dataset == "imagenette":
        ap.error("no LPIPS bound on ImageNette: its images are in the LPIPS backbone's training set")
    out = _ROOT / (PIXEL_BOUND_DIR if args.space == "pixel" else LPIPS_BOUND_DIR)[args.dataset]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    if args.assemble:
        assemble(out)
        return

    X_est01, X_v01 = load_est_and_v(args.dataset, args.data_dir)     # (CelebA: decodes and caches the extra images)
    n_est, n_v = len(X_est01), len(X_v01)
    print(f"[data] D_est n={n_est}, D_perp n={n_v}, d={X_est01.shape[1]} (disjoint)", flush=True)
    if args.prep and args.space == "pixel":
        print("[prep] pixel data ready", flush=True)
        return

    ckpt = None if (args.pair_index is not None or args.prep) else str(out / "checkpoint.json")
    pair_subset = None
    if args.space == "pixel":
        X_est = X_est01 * np.float32(GREY)                  # exact grey levels; float32 halves memory
        X_v = X_v01 * np.float32(GREY)
        del X_est01, X_v01
        if args.pair_index is not None:
            pair_subset = _select_pair(X_est.shape[1], args.k_max, NU2, args.pair_index)
            if pair_subset is None:
                return
        bound = entropy_bound_grid(
            X_est, X_v, K=divisor_grid(X_est.shape[1], args.k_max), NU2=NU2, alpha=args.alpha, n_mc=args.n_mc,
            seed=args.seed, rotation="signs" if args.dataset == "imagenette" else "eig", device=args.device,
            checkpoint_path=ckpt, pair_subset=pair_subset)
        bound["space"] = "pixel"
        bound["grey_scale"] = GREY
    else:
        phi_est = np.load(build_phi_cache(args.dataset, np.asarray(X_est01, dtype=np.float32), out, layout="chw",
                                          batch_size=args.batch_size, seed=SPLIT_SEED), mmap_mode="r")
        phi_v = np.load(build_phi_cache(args.dataset, np.asarray(X_v01, dtype=np.float32), out, layout="chw",
                                        batch_size=args.batch_size, seed=SPLIT_SEED), mmap_mode="r")
        del X_est01, X_v01
        rot_cache = str(out / f"rotated_f32_{args.dataset}.npy")   # written once by --prep, read by every pair job
        if args.pair_index is not None:
            if not os.path.exists(rot_cache + ".ok"):
                raise SystemExit("shared rotated sample missing: run --prep first")
            pair_subset = _select_pair(next_hadamard_dim(phi_est.shape[1]), args.k_max, NU2, args.pair_index)
            if pair_subset is None:
                return
        elif args.prep:
            pair_subset = []
        bound = lpips_entropy_bound_grid(
            phi_est, phi_v=phi_v, k_max=args.k_max, NU2=NU2, alpha=args.alpha, n_mc=args.n_mc, seed=args.seed,
            device=args.device, rotated_cache=rot_cache, checkpoint_path=ckpt, pair_subset=pair_subset)
        if args.prep:
            print(f"[prep] rotated sample ready: {rot_cache} ({time.time() - t0:.0f}s)", flush=True)
            return

    bound["dataset"] = args.dataset
    bound["split_policy"] = split_policy(args.dataset, n_est, n_v)
    if args.pair_index is not None:
        k, nu2 = pair_subset[0]
        os.makedirs(out / "pairs", exist_ok=True)
        path = out / "pairs" / f"pair_{args.pair_index:02d}_k{k}_nu{nu2:g}.json"
        _write_json(path, bound)
        print(f"[pair] wrote {path} ({time.time() - t0:.0f}s)", flush=True)
        return
    _write_json(out / "entropy_bound.json", bound)
    print(f"[done] {out / 'entropy_bound.json'} ({time.time() - t0:.0f}s)", flush=True)
    _print_rho(bound["pairs"])


if __name__ == "__main__":
    main()
