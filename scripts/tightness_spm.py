"""The shadow posterior-mean (SPM) attack on DP-SGD (app:exp_attack), for one dataset and representation.

The candidates are D_shadow; the targets are the first --n-targets of D_targets (asserted disjoint from them). For each
leakage gamma, the attack sees a single full-batch step (q = 1, T = 1) and a subsampled schedule (q, T) with the same
gamma, and reconstructs each target by the posterior mean over the candidates (and by the arg-max). In LPIPS space the
attack works on centred, scaled features beta (Phi - mean Phi); the subsampled feature estimate is also inverted to an
image (rho_img). Modes:

    default                 results/tightness_spm/spm_<ds>_<rep>_base.json (rho per gamma, bootstrap SE, paired
                            subsampled - single difference)
    --exemplar-mode N       single step only; the first N targets' reconstructions -> <outdir>/exemplars_<ds>_<rep>.npz
    --single-img-only       LPIPS: image output of the single-step attack -> <outdir>/spm_<ds>_lpips_single_img.json
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.attacks.shadow_prior import load_or_fit_prior  # noqa: E402
from src.attacks.spm import attack_batched, gen_release_single, gen_release_sub, nm_of_gamma, solve_sigma  # noqa: E402
from src.bound.rho import LPIPS_BOUND_DIR, PIXEL_BOUND_DIR, read_bound  # noqa: E402
from src.data.partition import GREY  # noqa: E402
from src.data.pool import load_splits  # noqa: E402

N_BOOT = 300                 # bootstrap resamples of the targets for the standard errors


def phi_cached(phi, images: np.ndarray, tag: str) -> np.ndarray:
    """Phi of many images, cached in float16 under data/spm_phi_cache/<tag>.npy.

    Args:
        phi: images -> (B, m) features.
        images: (N, d) images.
        tag: cache name.

    Returns:
        (N, m) float16 features.
    """
    path = Path("data/spm_phi_cache") / f"{tag}.npy"
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.empty((len(images), phi(images[:1]).shape[1]), np.float16)
    for i in range(0, len(images), 1024):
        out[i:i + 1024] = phi(images[i:i + 1024]).astype(np.float16)
        if i % 20480 == 0:
            print(f"  [phi {tag}] {i}/{len(images)}", flush=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, out)
    os.replace(tmp, path)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep", default="pixel", choices=["pixel", "lpips"])
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.2, 0.4, 0.72, 2.89, 8.01, 72.13],
                    help="leakages (bits)")
    ap.add_argument("--T", type=int, default=100, help="steps of the subsampled schedule")
    ap.add_argument("--q", type=float, default=0.02, help="sampling rate of the subsampled schedule")
    ap.add_argument("--C", type=float, default=0.3, help="clip norm")
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-prior", type=int, default=2000, help="LPIPS: D_shadow images averaged for the feature mean")
    ap.add_argument("--phi-scale", type=float, default=3.0, help="LPIPS: feature scale beta")
    ap.add_argument("--n-steps", type=int, default=500, help="LPIPS: pre-image descent steps")
    ap.add_argument("--n-restart", type=int, default=12, help="LPIPS: pre-image starting images")
    ap.add_argument("--chunk", type=int, default=8192, help="candidates per chunk")
    ap.add_argument("--outdir", default="results/tightness_spm")
    ap.add_argument("--exemplar-mode", type=int, default=0, metavar="N",
                    help="single step only; save the reconstructions of the first N targets per gamma")
    ap.add_argument("--single-img-only", action="store_true",
                    help="LPIPS: image output of the single-step attack, checked against the default run's rho")
    ap.add_argument("--rho-tol", type=float, default=1e-6,
                    help="--single-img-only: largest accepted |rho - rho of the default run| (GPU differences)")
    a = ap.parse_args()
    ds, rep, C, n = a.dataset, a.rep, a.C, a.n_targets
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    splits = load_splits(ds)
    shadow = np.asarray(splits["D_shadow"]["images"], np.float32)
    targets = np.asarray(splits["D_targets"]["images"][:n], np.float32)
    target_bytes = {x.tobytes() for x in targets}
    collisions = sum(x.tobytes() in target_bytes for x in shadow)
    assert collisions == 0, f"{collisions} candidates equal a target"

    # the attack's input space: z = beta (x - centre), with x the image (pixel) or its features Phi (LPIPS)
    if rep == "pixel":
        beta, ext = 1.0, None
        centre = load_or_fit_prior(ds, splits["D_shadow"]["images"]).mu.astype(np.float64)
        cand_src, target_x = shadow, targets
    else:
        from src.attacks.phi_preimage import descend_preimage, phi_of_images
        from src.bound.lpips_phi import PhiExtractor
        beta, ext = a.phi_scale, PhiExtractor(device=str(dev))

        def phi(images):
            return phi_of_images(ext, torch.as_tensor(images, dtype=torch.float32, device=dev), ds,
                                 "chw").detach().cpu().numpy().astype(np.float32)

        centre = phi(shadow[:a.n_prior]).astype(np.float64).mean(0)
        cand_src = phi_cached(phi, shadow, f"{ds}_sh")                   # float16
        target_x = phi(targets).astype(np.float64)
        init = shadow[:a.n_restart]

        def image_error(Phi_hat: np.ndarray) -> np.ndarray:
            """Invert each feature estimate to an image; squared feature distance of that image to its target."""
            e = np.zeros(len(Phi_hat))
            for ti in range(len(Phi_hat)):
                image = descend_preimage(ext, Phi_hat[ti], init, ds, "chw", n_steps=a.n_steps, tv_weight=1e-4,
                                         seed=1000 + ti)["image"]
                e[ti] = float(((phi(np.asarray(image, np.float32)[None])[0] - target_x[ti]) ** 2).sum())
            return e
    U_targets = (beta * (target_x - centre[None])).astype(np.float32)
    d, n_cand = U_targets.shape[1], len(shadow)

    # clipped candidates c_j = a(z_j) z_j: float16 on the CPU, built in chunks; norms and clip factors on the device
    cand = torch.empty(n_cand, d, dtype=torch.float16)
    clip_np, norm2_np = np.empty(n_cand), np.empty(n_cand)
    for lo in range(0, n_cand, a.chunk):
        z = beta * (cand_src[lo:lo + a.chunk].astype(np.float64) - centre[None])
        z_norm2 = (z * z).sum(1)
        factor = np.minimum(1.0, C / np.sqrt(np.maximum(z_norm2, 1e-12)))
        cand[lo:lo + a.chunk] = torch.as_tensor((factor[:, None] * z).astype(np.float16))
        clip_np[lo:lo + a.chunk], norm2_np[lo:lo + a.chunk] = factor, factor * factor * z_norm2
    cand_norm2 = torch.as_tensor(norm2_np, dtype=torch.float32, device=dev)
    cand_clip = torch.as_tensor(clip_np, dtype=torch.float32, device=dev)
    clip_frac = float((norm2_np >= C * C - 1e-6).mean())                  # share of candidates at the clip norm
    log_n = math.log(n_cand)

    U = torch.as_tensor(U_targets, dtype=torch.float32, device=dev)
    U_norm2 = (U * U).sum(1)
    clipped_targets = torch.minimum(torch.ones(n, device=dev), C / torch.sqrt(U_norm2))[:, None] * U

    # error normalisers: sqrt(tr Sigma) of the bound (per grey level in pixel space), and the targets' own spread
    tr_sigma = float(read_bound(Path((LPIPS_BOUND_DIR if rep == "lpips" else PIXEL_BOUND_DIR)[ds])
                                / "entropy_bound.json")["tr_Sigma_raw"])
    denom_a = math.sqrt(tr_sigma) / GREY if rep == "pixel" else math.sqrt(tr_sigma)
    denom_b = math.sqrt(float(U_norm2.mean())) / beta
    print(f"[spm {ds} {rep} base] d={d} ncand={n_cand} beta={beta} denomA={denom_a:.3f} clip={clip_frac:.4f} "
          f"log_n={log_n:.3f} coll={collisions} dev={dev}", flush=True)

    target_t = torch.as_tensor(target_x, dtype=torch.float32, device=dev)
    centre_t = torch.as_tensor(centre, dtype=torch.float32, device=dev)

    def estimate(u_mean):
        """(n, d) posterior means in the attack's space -> estimates in image (pixel, clamped) or feature space."""
        return (centre_t[None] + u_mean).clamp(0, 1) if rep == "pixel" else centre_t[None] + u_mean / beta

    def sq_error(u_mean) -> np.ndarray:
        return ((estimate(u_mean) - target_t) ** 2).sum(1).detach().cpu().numpy()

    rng_boot = np.random.default_rng(a.seed)

    def summarise(e, gamma, ent=None, w_map=None, other=None):
        """rho = sqrt(mean error) / denom_a with a bootstrap SE; optionally the posterior's entropy and arg-max weight,
        and the paired difference to another series on the same targets."""
        boots = [rng_boot.integers(0, n, n) for _ in range(N_BOOT)]
        row = dict(gamma=gamma, rho=math.sqrt(e.mean()) / denom_a,
                   rho_se=float(np.std([math.sqrt(e[ix].mean()) / denom_a for ix in boots])),
                   per_err=[float(v) for v in e])
        if ent is not None:
            row.update(mean_entropy=float(np.mean(ent)), log_n=log_n, frac_map_gt_half=float(np.mean(w_map > 0.5)))
        if other is not None:
            row.update(paired_diff=(math.sqrt(e.mean()) - math.sqrt(other.mean())) / denom_a,
                       paired_se=float(np.std([(math.sqrt(e[ix].mean()) - math.sqrt(other[ix].mean())) / denom_a
                                               for ix in boots])))
        return row

    def attack(gamma, schedule):
        """Release and attack every target: (posterior means, arg-max estimates, entropies, arg-max weights)."""
        if schedule == "single":
            noise_std = nm_of_gamma(gamma) * C
            G = gen_release_single(clipped_targets, noise_std, a.seed, gamma, dev)[:, None, :]          # (n, 1, d)
        else:
            noise_std = solve_sigma(a.T, a.q, gamma) * C
            G = torch.stack([gen_release_sub(clipped_targets[ti], noise_std, a.seed, gamma, ti, a.T, a.q, dev)[0]
                             for ti in range(n)], 0)                                                      # (n, T, d)
        u_mean, u_map, _, ent, w_map = attack_batched(G, cand, cand_norm2, cand_clip, noise_std,
                                                      1.0 if schedule == "single" else a.q, dev, a.chunk)
        return u_mean, u_map, ent.detach().cpu().numpy(), w_map.detach().cpu().numpy()

    if a.exemplar_mode:                                                   # fig:exemplars
        N, recs, rhos = a.exemplar_mode, [], []
        for gamma in a.gammas:
            u_mean = attack(gamma, "single")[0]
            rhos.append(math.sqrt(sq_error(u_mean).mean()) / denom_a)
            est = estimate(u_mean[:N]).detach().cpu().numpy()
            if rep == "lpips":                                            # feature estimate -> image
                est = np.stack([np.asarray(descend_preimage(ext, est[ti].astype(np.float64), init, ds, "chw",
                                                            n_steps=a.n_steps, tv_weight=1e-4, seed=1000 + ti)
                                           ["image"], np.float32).reshape(-1) for ti in range(N)])
            recs.append(est)
            print(f"  [exemplar {ds} {rep}] gamma={gamma:g} rho={rhos[-1]:.3f}", flush=True)
        os.makedirs(a.outdir, exist_ok=True)
        path = f"{a.outdir}/exemplars_{ds}_{rep}.npz"
        np.savez_compressed(path, gammas=np.asarray(a.gammas), rho=np.asarray(rhos), targets=targets[:N],
                            recon=np.stack(recs), ncand=n_cand, C=C, n_targets=n)
        print(f"wrote {path}", flush=True)
        return

    if a.single_img_only:                                                 # fig:tightness, single-step image output
        assert rep == "lpips", "--single-img-only is for the LPIPS runs"
        ref = {round(r["gamma"], 6): r["rho"] for r in json.load(open(f"{a.outdir}/spm_{ds}_{rep}_base.json"))["single"]}
        rows = []
        for gamma in a.gammas:
            u_mean = attack(gamma, "single")[0]
            rho = math.sqrt(sq_error(u_mean).mean()) / denom_a
            rho_ref = ref[round(gamma, 6)]
            assert abs(rho - rho_ref) < a.rho_tol, f"single rho {rho} != default run {rho_ref} at gamma={gamma}"
            e = image_error(estimate(u_mean).detach().cpu().numpy().astype(np.float64))
            rows.append(dict(summarise(e, gamma), rho_feature=rho, rho_feature_paper=rho_ref,
                             rho_feature_diff=rho - rho_ref))
            print(f"  [single img {ds}] gamma={gamma:g} rho(feature)={rho:.3f} rho_img={rows[-1]['rho']:.3f}", flush=True)
        path = f"{a.outdir}/spm_{ds}_{rep}_single_img.json"
        with open(path, "w") as f:
            json.dump(dict(dataset=ds, rep=rep, cand_set="base", attack="spm_posterior_mean_fulld",
                           config="single (q=1,T=1)", output="image (feature posterior mean inverted)",
                           n_steps=a.n_steps, n_restart=a.n_restart, denom_a=denom_a, n_targets=n, rows=rows),
                      f, indent=1)
        print(f"wrote {path}", flush=True)
        return

    out = dict(dataset=ds, rep=rep, cand_set="base", attack="spm_posterior_mean_fulld", T=a.T, q=a.q, C=C, beta=beta,
               n_targets=n, ncand=n_cand, log_n=log_n, denom_a=denom_a, denom_b=denom_b, clip_cand=clip_frac,
               single=[], subsampled=[], single_map=[], subsampled_map=[])
    for gamma in a.gammas:
        um_s, umap_s, ent_s, w_s = attack(gamma, "single")
        um_b, umap_b, ent_b, w_b = attack(gamma, "sub")
        e_s, e_s_map, e_b, e_b_map = sq_error(um_s), sq_error(umap_s), sq_error(um_b), sq_error(umap_b)
        out["single"].append(summarise(e_s, gamma, ent=ent_s, w_map=w_s))
        out["single_map"].append(summarise(e_s_map, gamma))
        out["subsampled"].append(summarise(e_b, gamma, ent=ent_b, w_map=w_b, other=e_s))   # paired: sub - single
        out["subsampled_map"].append(summarise(e_b_map, gamma))
        if rep == "lpips":                                                # image output of the subsampled attack
            e = image_error(estimate(um_b).detach().cpu().numpy().astype(np.float64))
            out["subsampled"][-1]["rho_img"] = math.sqrt(e.mean()) / denom_a

    os.makedirs(a.outdir, exist_ok=True)
    path = f"{a.outdir}/spm_{ds}_{rep}_base.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n{'gamma':>7} {'single':>7} {'sub':>7} {'sub-single':>10} {'ent':>7} {'fMAP>.5':>7}", flush=True)
    for s, b in zip(out["single"], out["subsampled"]):
        print(f"{s['gamma']:>7.2f} {s['rho']:>7.3f} {b['rho']:>7.3f} {b['paired_diff']:>+10.3f} "
              f"{b['mean_entropy']:>7.2f} {b['frac_map_gt_half']:>7.2f}", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
