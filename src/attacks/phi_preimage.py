"""Inverting the LPIPS feature map Φ: an image whose features are as close as possible to a target vector.

Used twice in the paper (app:exp_phi, app:exp_attack):
  * the data-blind LPIPS baseline: the image closest in Φ to the feature mean (scripts/compute_realizable_prior.py);
  * turning the SPM attack's feature-space estimate into an image (scripts/tightness_spm.py), for the dashed
    image-space lines and the LPIPS exemplars.

The optimisation variable is the image in its **native pixel space** (mnist 1×28×28, cifar10 3×32×32, celeba
3×64×64), pushed through the same input transform as Φ (:func:`src.bound.lpips_phi.image_to_lpips_input`); pixels
stay in [0,1] via x = ½(tanh u + 1), and restarts are optimised as one batch so each Adam step is a single Φ
forward/backward.
"""

from __future__ import annotations

import numpy as np

from src.bound.lpips_phi import DATASET_SHAPES, image_to_lpips_input


# ---------------------------------------------------------------------------
# Image priors and the differentiable Phi of an image (torch).
# ---------------------------------------------------------------------------

def _to_native(x_flat, dataset: str, layout: str):
    """(B, d) flat rows -> (B, C, H, W) in the dataset's native resolution."""
    C, H, W = DATASET_SHAPES[dataset]
    B = x_flat.shape[0]
    if layout == "chw":
        return x_flat.view(B, C, H, W)
    return x_flat.view(B, H, W, C).permute(0, 3, 1, 2)


def total_variation(x_flat, dataset: str, layout: str):
    """Anisotropic TV per image in native pixel space -> (B,)."""
    img = _to_native(x_flat, dataset, layout)
    dh = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    dw = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    return dh + dw


def phi_of_images(ext, x_flat, dataset: str, layout: str):
    """Grad-enabled Φ of (B, d) native image rows in [0,1] -> (B, m) tensor."""
    t = image_to_lpips_input(x_flat, dataset, layout).to(ext.device)
    return ext.phi_from_input(t)


# ---------------------------------------------------------------------------
# Pre-image optimisers.  Variable is the native image in [0,1]; restarts and
# chains are stacked into one batch so each step is a single Φ forward/backward.
# ---------------------------------------------------------------------------

def _mu_tensor(mu, ext):
    import torch
    return torch.as_tensor(np.asarray(mu, dtype=np.float32), device=ext.device)


def _atanh_box(x, torch):
    """Inverse of x = 0.5(tanh(u)+1): map [0,1] image to unconstrained u."""
    y = (2.0 * x - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
    return 0.5 * torch.log((1 + y) / (1 - y))


def descend_preimage(ext, mu, init_images: np.ndarray, dataset: str,
                     layout: str, *, tv_weight: float = 0.0,
                     n_steps: int = 1500, lr: float = 0.05,
                     seed: int = 0) -> dict:
    """Adam descent to the argmin pre-image of μ, box-constrained via tanh.

    ``init_images`` is (B, d) in [0,1] — each row seeds one restart; they are
    optimised as one batch and the lowest-‖Φ(x)−μ‖² restart is returned.

    Returns ``image`` (best native row, [0,1]), ``dist2`` (its feature
    distance), ``all_dist2`` (per-restart final), ``trace`` (min feature
    distance over the batch per step).
    """
    import torch

    torch.manual_seed(seed)
    mu_t = _mu_tensor(mu, ext)
    x0 = torch.as_tensor(np.asarray(init_images, dtype=np.float32),
                         device=ext.device)
    u = _atanh_box(x0, torch).clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)

    trace = np.empty(n_steps, dtype=np.float64)
    for step in range(n_steps):
        opt.zero_grad()
        x = 0.5 * (torch.tanh(u) + 1.0)
        phi = phi_of_images(ext, x, dataset, layout)
        fd2 = ((phi - mu_t) ** 2).sum(dim=1)               # (B,) per restart
        loss = fd2.sum()
        if tv_weight > 0:
            loss = loss + tv_weight * total_variation(x, dataset, layout).sum()
        loss.backward()
        opt.step()
        trace[step] = float(fd2.min().detach())

    with torch.no_grad():
        x = 0.5 * (torch.tanh(u) + 1.0)
        fd2 = ((phi_of_images(ext, x, dataset, layout) - mu_t) ** 2).sum(dim=1)
        j = int(fd2.argmin())
        return {
            "image": x[j].detach().cpu().numpy().astype(np.float32),
            "dist2": float(fd2[j]),
            "all_dist2": fd2.detach().cpu().numpy().astype(np.float64),
            "trace": trace,
        }
