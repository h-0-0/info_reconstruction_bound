"""The realizable LPIPS prior baseline: the grey line of the LPIPS tightness panels (paper_tightness.ipynb).

The bound's prior error is set by the feature variance, which no actual image attains. The realizable baseline is the
best an adversary can do with the prior alone: invert the mean feature E[Phi] over D_shadow to an image x_c (by
pre-image descent) and report its root-mean LPIPS to the targets. The unattainable distance of the mean feature itself
is reported alongside:

    realizable_prior_sqrt_lpips  = sqrt(mean_t ||Phi(target_t) - Phi(x_c)||^2)
    feature_centroid_sqrt_lpips  = sqrt(mean_t ||Phi(target_t) - E[Phi]||^2)

Writes results/tightness/realizable_prior_<dataset>.json.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.attacks.phi_preimage import descend_preimage, phi_of_images  # noqa: E402
from src.bound.lpips_phi import PhiExtractor  # noqa: E402
from src.data.pool import load_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["mnist", "celeba"])
    ap.add_argument("--n_targets", type=int, default=100)
    ap.add_argument("--n_centroid", type=int, default=2000, help="D_shadow images averaged for E[Phi]")
    ap.add_argument("--n_steps", type=int, default=400, help="pre-image descent steps")
    ap.add_argument("--tv_weight", type=float, default=1e-4, help="total-variation weight of the descent")
    a = ap.parse_args()
    ds = a.dataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = PhiExtractor(device=str(dev))
    splits = load_splits(ds)
    shadow = splits["D_shadow"]["images"].astype(np.float32)
    targets = splits["D_targets"]["images"][:a.n_targets].astype(np.float32)

    def phi(images):
        return phi_of_images(ext, torch.as_tensor(images, device=dev), ds, "chw").detach().cpu().numpy()

    phi_mean = phi(shadow[:a.n_centroid]).mean(0)
    x_c = np.asarray(descend_preimage(ext, phi_mean, shadow[:16], ds, "chw", n_steps=a.n_steps,
                                      tv_weight=a.tv_weight)["image"], np.float32)
    phi_t = phi(targets)
    out = {"dataset": ds, "n_targets": int(a.n_targets),
           "realizable_prior_sqrt_lpips": math.sqrt(float(((phi_t - phi(x_c[None])) ** 2).sum(1).mean())),
           "feature_centroid_sqrt_lpips": math.sqrt(float(((phi_t - phi_mean) ** 2).sum(1).mean()))}
    os.makedirs("results/tightness", exist_ok=True)
    path = f"results/tightness/realizable_prior_{ds}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{ds}] realizable {out['realizable_prior_sqrt_lpips']:.4f}, mean feature "
          f"{out['feature_centroid_sqrt_lpips']:.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
