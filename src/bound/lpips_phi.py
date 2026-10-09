"""The LPIPS feature map Phi (the paper's Psi_LPIPS), built so that ||Phi(x) - Phi(x')||^2 = LPIPS(x, x') exactly.

From the frozen ``lpips`` AlexNet (v0.1 calibration weights), per layer l (the 5 ReLU taps):

    yhat  = y / (||y||_2 over channels + 1e-10)          (lpips.normalize_tensor)
    Phi_l = yhat * sqrt(w_l) / sqrt(H_l W_l)             (w_l >= 0: the calibration weights)

stacked over (l, c, h, w), m = 31,872 coordinates. LPIPS is sum_l mean_hw sum_c w_lc (yhat - yhat')^2, which is why
Phi carries sqrt(w) and the 1/sqrt(HW). Images arrive as flat rows on [0, 1] (CHW, or HWC with ``layout``); the
input transform (:func:`image_to_lpips_input`) replicates grey to 3 channels, resizes to 64 x 64 and maps to
[-1, 1]. The same transform scores reconstructions with the packaged model (:meth:`PhiExtractor.lpips_pairs`), and
:func:`verify_phi` checks the identity numerically.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

LPIPS_IMG_SIZE = 64  # every dataset is resized to 64 x 64 before the AlexNet trunk

# native (C, H, W) per dataset
DATASET_SHAPES = {
    "mnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
    "celeba": (3, 64, 64),
    "imagenette": (3, 128, 128),
}


def image_to_lpips_input(x: torch.Tensor, dataset: str, layout: str = "chw") -> torch.Tensor:
    """The input transform of Phi, on a tensor (differentiable, so the image inversion can use it).

    Args:
        x: (n, d) flat images on [0, 1].
        dataset: dataset name (sets the native shape).
        layout: "chw" or "hwc", how the rows were flattened.

    Returns:
        (n, 3, 64, 64) tensor on [-1, 1]: grey replicated to 3 channels, bilinear resize (no antialiasing).
    """
    C, H, W = DATASET_SHAPES[dataset]
    n = x.shape[0]
    if layout == "chw":
        t = x.view(n, C, H, W)
    elif layout == "hwc":
        t = x.view(n, H, W, C).permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"unknown layout {layout!r} (use 'chw' or 'hwc')")
    if C == 1:
        t = t.repeat(1, 3, 1, 1)
    if (H, W) != (LPIPS_IMG_SIZE, LPIPS_IMG_SIZE):
        t = F.interpolate(t, size=(LPIPS_IMG_SIZE, LPIPS_IMG_SIZE), mode="bilinear", align_corners=False)
    return t * 2.0 - 1.0


def dataset_to_lpips_input(X: np.ndarray, dataset: str, layout: str = "chw", device: str = "cpu") -> torch.Tensor:
    """:func:`image_to_lpips_input` for a numpy array, after checking it is on [0, 1].

    Args:
        X: (n, d) flat images on [0, 1].
        dataset: dataset name.
        layout: "chw" or "hwc".
        device: torch device of the result.

    Returns:
        (n, 3, 64, 64) tensor on [-1, 1].
    """
    X = np.asarray(X, dtype=np.float32)
    if X.min() < -1e-6 or X.max() > 1.0 + 1e-6:
        raise ValueError(f"dataset_to_lpips_input expects [0,1] input, got range [{X.min():.4g}, {X.max():.4g}]")
    return image_to_lpips_input(torch.from_numpy(X).to(device), dataset, layout)


class PhiExtractor:
    """The frozen feature map Phi: flat images -> R^m (backbone and calibration weights in eval mode, no gradients
    to the weights).

    Attributes:
        model: the packaged lpips.LPIPS model.
        layer_shapes: (C, H, W) of each tapped layer.
        m: feature dimension (31,872).
    """

    def __init__(self, device: str = "cpu", verbose: bool = False):
        """Load the model and read the feature dimension off one forward pass.

        Args:
            device: torch device.
            verbose: let the lpips package print its loading messages.
        """
        import lpips as lpips_pkg

        self.device = device
        self.model = lpips_pkg.LPIPS(net="alex", version="0.1", verbose=verbose)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._normalize_tensor = lpips_pkg.normalize_tensor

        # sqrt(w) per layer, (1, C_l, 1, 1); the v0.1 weights are non-negative by construction, and Phi needs them so
        self.sqrt_w = []
        for lin in self.model.lins:
            w = lin.model[1].weight.detach()
            if float(w.min()) < 0.0:
                raise ValueError("LPIPS lin weights have negative entries; Phi with sqrt(w) is invalid.")
            self.sqrt_w.append(torch.sqrt(w).to(device))

        with torch.no_grad():
            probe = torch.zeros(1, 3, LPIPS_IMG_SIZE, LPIPS_IMG_SIZE, device=device)
            outs = self.model.net.forward(self.model.scaling_layer(probe))
        self.layer_shapes = [tuple(o.shape[1:]) for o in outs]
        self.layer_dims = [c * h * w for (c, h, w) in self.layer_shapes]
        self.m = int(sum(self.layer_dims))

    def phi_from_input(self, t: torch.Tensor) -> torch.Tensor:
        """Phi of transformed images; gradients flow to ``t`` if it requires them (used by the image inversion).

        Args:
            t: (n, 3, 64, 64) tensor on [-1, 1] (from :func:`image_to_lpips_input`).

        Returns:
            (n, m) tensor.
        """
        outs = self.model.net.forward(self.model.scaling_layer(t))
        feats = []
        for ell in range(self.model.L):
            yhat = self._normalize_tensor(outs[ell])              # unit norm over channels at each (h, w)
            C, H, W = self.layer_shapes[ell]
            scaled = yhat * self.sqrt_w[ell] / float(np.sqrt(H * W))
            feats.append(scaled.reshape(scaled.shape[0], -1))
        return torch.cat(feats, dim=1)

    @torch.no_grad()
    def phi(self, X: np.ndarray, *, dataset: str, batch_size: int = 256, layout: str = "chw") -> np.ndarray:
        """Phi of a set of images, in batches.

        Args:
            X: (n, d) flat images on [0, 1].
            dataset: dataset name.
            batch_size: images per forward pass.
            layout: "chw" or "hwc".

        Returns:
            (n, m) float32 array.
        """
        out = np.empty((X.shape[0], self.m), dtype=np.float32)
        for s in range(0, X.shape[0], batch_size):
            t = dataset_to_lpips_input(X[s:s + batch_size], dataset, layout, device=self.device)
            out[s:s + batch_size] = self.phi_from_input(t).cpu().numpy()
        return out

    @torch.no_grad()
    def lpips_pairs(self, X0: np.ndarray, X1: np.ndarray, *, dataset: str, batch_size: int = 256,
                    layout: str = "chw") -> np.ndarray:
        """The packaged LPIPS distance of each pair, through the same input transform as Phi.

        Args:
            X0, X1: (n, d) flat images on [0, 1], paired row by row.
            dataset: dataset name.
            batch_size: pairs per forward pass.
            layout: "chw" or "hwc".

        Returns:
            (n,) LPIPS distances.
        """
        vals = []
        for s in range(0, X0.shape[0], batch_size):
            t0 = dataset_to_lpips_input(X0[s:s + batch_size], dataset, layout, device=self.device)
            t1 = dataset_to_lpips_input(X1[s:s + batch_size], dataset, layout, device=self.device)
            vals.append(self.model(t0, t1, normalize=False).reshape(-1).cpu().numpy())
        return np.concatenate(vals)


def verify_phi(extractor: PhiExtractor, X: np.ndarray, *, dataset: str, n_pairs: int = 1000, seed: int = 0,
               batch_size: int = 256, layout: str = "chw") -> dict:
    """Check ||Phi(x) - Phi(x')||^2 = LPIPS(x, x') on random pairs (distances in float64; ~1e-5 expected).

    Args:
        extractor: the feature map.
        X: (n, d) flat images on [0, 1] to draw pairs from.
        dataset: dataset name.
        n_pairs: number of random pairs.
        seed: seed of the pairs.
        batch_size: images per forward pass.
        layout: "chw" or "hwc".

    Returns:
        Dict with n_pairs, max_abs_err, mean_abs_err, max_rel_err, lpips_mean, lpips_max and m.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    i0 = rng.integers(0, n, size=n_pairs)
    i1 = rng.integers(0, n, size=n_pairs)

    kw = dict(batch_size=batch_size, dataset=dataset, layout=layout)
    lp = extractor.lpips_pairs(X[i0], X[i1], **kw).astype(np.float64)
    ph0 = extractor.phi(X[i0], **kw).astype(np.float64)
    ph1 = extractor.phi(X[i1], **kw).astype(np.float64)
    err = np.abs(((ph0 - ph1) ** 2).sum(axis=1) - lp)
    return {
        "n_pairs": int(n_pairs),
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "max_rel_err": float((err / np.maximum(lp, 1e-12)).max()),
        "lpips_mean": float(lp.mean()),
        "lpips_max": float(lp.max()),
        "m": int(extractor.m),
    }
