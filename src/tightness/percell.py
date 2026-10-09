"""Per-cell and per-block bounds of the tightness experiment's codebook releases (prop:block_leakage, k = 1).

  BoundFrame          the shipped bound's coordinates Y for any images (rotation, centring, scale from D_perp)
  tightness_data      codebook features and Y for D_shadow, D_est and D_targets
  h1d_cells           per-coordinate one-dimensional dithered plug-in entropies of many cells at once
  cell_bounds         the per-cell and per-block bounds from the cells' entropies and variances
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.bound.blocks import apply_rotation
from src.bound.grid import bound_frame
from src.bound.rho import LPIPS_BOUND_DIR, PIXEL_BOUND_DIR, read_bound
from src.data.partition import GREY, load_est_and_v
from src.data.pool import load_pool, split_indices

LN2 = math.log(2.0)
TWO_PI_E = 2.0 * math.pi * math.e
CELL_CAP = 1200                # centres per cell in the entropy estimate (P(c) uses the true count)
QCELL_BUDGET = 6_000_000       # per-cell queries x centres: big cells get fewer queries, small cells more


class BoundFrame:
    """A shipped bound's representation of its dataset: the features (grey levels, or LPIPS features Phi) and their
    rotated, unit-trace coordinates Y, in exactly the frame the bound used (rotation, centring and scale from D_perp).
    The unit-trace scale is checked against the one the bound recorded.

    Args:
        dataset: dataset name.
        space: "pixel" or "lpips".
        X_v: (n_v, d_raw) D_perp images in [0, 1] (from :func:`src.data.partition.load_est_and_v`).
        device: torch device for Phi.
    """

    def __init__(self, dataset: str, space: str, X_v: np.ndarray, device=None):
        self.bound_dir = (LPIPS_BOUND_DIR if space == "lpips" else PIXEL_BOUND_DIR)[dataset]
        self.dataset, self.space = dataset, space
        self.bound = read_bound(f"{self.bound_dir}/entropy_bound.json")
        tol = 1e-9
        if space == "lpips":
            from src.bound.lpips_grid import phi_cache_path
            from src.bound.lpips_phi import PhiExtractor
            self.extractor = PhiExtractor(device=device)
            cache = phi_cache_path(self.bound_dir, len(X_v))
            if cache.exists():
                F_v = np.load(cache, mmap_mode="r")
                tol = 1e-6
            else:
                F_v = self.features(X_v)
                tol = 1e-4                                # a recomputed Phi may differ in the last float32 bits
        else:
            F_v = self.features(X_v)
        self.frame = bound_frame(F_v, self.bound["rotation"], int(self.bound["seed"]))
        self.d = self.frame["d"]
        assert self.d == int(self.bound["d"]), (self.d, self.bound["d"])
        assert math.isclose(self.frame["scale_c"], float(self.bound["scale_c"]), rel_tol=tol), \
            (self.frame["scale_c"], self.bound["scale_c"])

    def features(self, images: np.ndarray) -> np.ndarray:
        """The bound's raw features of images.

        Args:
            images: (n, d_raw) images in [0, 1].

        Returns:
            (n, d_raw) float32 grey levels (as the bound scales them), or (n, m) float32 Phi.
        """
        images = np.asarray(images, dtype=np.float32)
        if self.space == "lpips":
            return np.ascontiguousarray(self.extractor.phi(images, dataset=self.dataset), dtype=np.float32)
        return images * np.float32(GREY)

    def rotate(self, F: np.ndarray) -> np.ndarray:
        """Rotate and standardise features into the bound's frame.

        Args:
            F: (n, d_raw) output of :meth:`features`.

        Returns:
            (n, d) float32 Y.
        """
        f = self.frame
        return apply_rotation(F, f["factors"], V=f["V"], signs=f["signs"], shift=f["mu"], scale=f["scale_c"])

    def check_rotated_cache(self, Y: np.ndarray, n_rows: int = 200, atol: float = 5e-3) -> None:
        """Check the rotated D_est against the bound's own rotated copy, when the bound kept one, on a few rows (a
        recomputed Phi may differ slightly between GPUs, hence the tolerance).

        Args:
            Y: (n_est, d) output of :meth:`rotate` on D_est.
            n_rows: rows compared.
            atol: largest allowed absolute difference.
        """
        path = Path(self.bound_dir) / f"rotated_f32_{self.dataset}.npy"
        if not path.exists():
            return
        cache = np.load(path, mmap_mode="r")
        assert cache.shape == Y.shape, (cache.shape, Y.shape)
        rows = np.sort(np.random.default_rng(0).choice(len(Y), n_rows, replace=False))
        diff = float(np.max(np.abs(Y[rows].astype(np.float64) - np.asarray(cache[rows], np.float64))))
        assert diff < atol, f"rotated D_est differs from the bound's ({path}): max |diff| = {diff:.2e}"


def h1d_cells(Y, cells, s, q_target, seed, device, sqrt_Nk, cap=CELL_CAP, mem_bytes=1 << 30, rng=None):
    """Per-coordinate dithered plug-in entropies (eq:mc_def at k = 1) of many cells at once.

    Each cell's estimate is the exact mixture over its (at most ``cap``) centres, integrated by Monte Carlo with
    queries drawn as a uniformly random centre plus N(0, s^2) dither; the query count min(q_target,
    max(3000, QCELL_BUDGET / cell size)) keeps large cells cheap and gives small cells many queries. Cells are sorted
    by size and packed into padded, masked batches, so the many small cells at high rates share GPU launches.

    Args:
        Y: (n, d) rotated, standardised sample.
        cells: list of index arrays into Y, one per cell.
        s: dither standard deviation.
        q_target: largest query count per cell.
        seed: random seed (queries and dither).
        device: torch device.
        sqrt_Nk: scale of each coordinate (sqrt(d) at k = 1, so a coordinate has unit mean square).
        cap: largest number of centres per cell (larger cells are subsampled).
        mem_bytes: memory budget of one distance tile.
        rng: numpy generator for the subsampling (default: seeded from ``seed``).

    Returns:
        (n_cells, d) entropies in bits.
    """
    import torch
    n_cells, d = len(cells), Y.shape[1]
    H = np.empty((n_cells, d))
    rng = rng or np.random.default_rng(seed)
    inv_2s2 = 1.0 / (2.0 * s * s)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    centres = [c if c.size <= cap else c[rng.choice(c.size, cap, replace=False)] for c in cells]
    order = sorted(range(n_cells), key=lambda i: centres[i].size)
    i = 0
    while i < n_cells:
        # the next batch: cells i..j (sorted by size), padded to the largest, within the memory budget
        j, m = i, centres[order[i]].size
        while j + 1 < n_cells:
            m_next, batch_next = centres[order[j + 1]].size, j + 2 - i
            if batch_next * m_next * m_next * 4 > mem_bytes or batch_next > 4096:
                break
            j, m = j + 1, m_next
        batch = order[i:j + 1]
        nb = len(batch)
        Zp = np.zeros((nb, m, d), np.float32)
        valid = np.zeros((nb, m), bool)
        sizes = np.empty(nb, np.int64)
        for r, ci in enumerate(batch):
            c = centres[ci]
            Zp[r, :c.size] = np.asarray(Y[c], np.float32) * sqrt_Nk
            valid[r, :c.size] = True
            sizes[r] = c.size
        Zt = torch.as_tensor(Zp, device=device)
        sizes_t = torch.as_tensor(sizes, device=device)
        log_norm = -np.log(sizes) - 0.5 * math.log(2.0 * math.pi * s * s)                  # (nb,)
        n_q = int(min(q_target, max(3000, QCELL_BUDGET // m)))                              # queries per cell
        q_centre = (torch.rand(nb, n_q, generator=gen, device=device) * sizes_t[:, None]).long()
        pad_mask = torch.where(torch.as_tensor(valid, device=device), 0.0, -1e30)[:, None, :, None]
        dc = max(1, int(mem_bytes // (nb * max(1, min(n_q, 4096)) * m * 4)))                # coordinates per launch
        for c0 in range(0, d, dc):
            Z = Zt[:, :, c0:c0 + dc]                                                        # (nb, m, dcc)
            dcc = Z.shape[2]
            Q = torch.gather(Z, 1, q_centre[:, :, None].expand(nb, n_q, dcc))
            Q = Q + s * torch.randn(nb, n_q, dcc, generator=gen, dtype=torch.float32, device=device)
            acc = torch.zeros(nb, dcc, dtype=torch.float64, device=device)
            qt = max(1, int(mem_bytes // (nb * m * dcc * 4)))
            for q0 in range(0, n_q, qt):
                d2 = (Q[:, q0:q0 + qt, None, :] - Z[:, None, :, :]) ** 2                    # (nb, qc, m, dcc)
                acc += torch.logsumexp(-d2 * inv_2s2 + pad_mask, dim=2).double().sum(dim=1)
            h_nats = -(acc.cpu().numpy() / n_q + log_norm[:, None])                         # (nb, dcc)
            for r, ci in enumerate(batch):
                H[ci, c0:c0 + dc] = h_nats[r] / LN2
        i = j + 1
    return H


@dataclass
class TightnessData:
    """Codebook features F (grey-level images as [0, 1] floats, or Phi) and the bound's coordinates Y."""
    frame: BoundFrame
    F_shadow: np.ndarray
    F_est: np.ndarray
    F_tgt: np.ndarray | None
    Y: np.ndarray
    Y_tgt: np.ndarray | None


def tightness_data(dataset: str, space: str, data_dir: str, device, targets: bool = True) -> TightnessData:
    """The tightness experiment's data: D_shadow (fits the codebook), D_est (the bound's estimation sample, cut into
    cells) and D_targets (held out, scores the decoder), with D_est checked row for row against the bound's.

    Args:
        dataset: dataset name.
        space: "pixel" or "lpips".
        data_dir: dataset root.
        device: torch device.
        targets: also D_targets.

    Returns:
        The features (pixel: the images, float64; LPIPS: Phi, float64 for D_shadow, float32 otherwise) and Y (float32).
    """
    images, _ = load_pool(dataset, data_dir)
    idx = split_indices(dataset, data_dir)
    shadow, tgt = np.sort(idx["idx_shadow"]), np.sort(idx["idx_targets"])
    X_est, X_v = load_est_and_v(dataset, data_dir)
    frame = BoundFrame(dataset, space, X_v, device)
    if space == "pixel":
        F_shadow, F_est = images[shadow].astype(np.float64), X_est.astype(np.float64)
        F_tgt = images[tgt].astype(np.float64) if targets else None
        Y = frame.rotate(frame.features(X_est))
        Y_tgt = frame.rotate(frame.features(images[tgt])) if targets else None
    else:
        F_shadow, F_est = frame.features(images[shadow]).astype(np.float64), frame.features(X_est)
        F_tgt = frame.features(images[tgt]) if targets else None
        Y = frame.rotate(F_est)
        Y_tgt = frame.rotate(F_tgt) if targets else None
    frame.check_rotated_cache(Y)
    return TightnessData(frame, F_shadow, F_est, F_tgt, np.ascontiguousarray(Y), Y_tgt)


def maxent_var(h_bits):
    """The max-entropy variance (1 / 2 pi e) 2^{2 h} of a one-dimensional entropy h (bits)."""
    return (1 / TWO_PI_E) * np.exp2(2.0 * h_bits / 1)


def within_cell_variance(Y: np.ndarray, cells: list[np.ndarray], scale: float) -> np.ndarray:
    """Each cell's per-coordinate variance (unbiased) of scale * Y.

    Args:
        Y: (n, d) coordinates.
        cells: index arrays into Y.
        scale: the coordinate scale squared (N_k: unit mean square per coordinate).

    Returns:
        (n_cells, d) variances.
    """
    return scale * np.stack([np.asarray(Y[c], np.float64).var(axis=0, ddof=1) for c in cells])


def cell_bounds(P: np.ndarray, H: np.ndarray, H_marg: np.ndarray, V_all: np.ndarray, V_sub: np.ndarray,
                coords: np.ndarray, nu2: float, gamma_b_range: tuple[float, float]) -> dict:
    """The per-cell and per-block bounds (squared, unit-trace scale) of a release with cells c of probability P(c).

    Per coordinate b, with h(z~_b | c) the cells' dithered entropies on the coordinate subset:
        per-cell   rho^2 >= mean_b sum_c P(c) (1 / 2 pi e) 2^{2 h(z~_b | c)} - nu^2
        per-block  rho^2 >= mean_b (1 / 2 pi e) 2^{2 (h(z~_b) - gamma_b)} - nu^2,
                   gamma_b = h(z~_b) - sum_c P(c) h(z~_b | c), clipped to gamma_b_range
    Both are evaluated on the subset and carried to all coordinates through the cell-mean decoder's error (their
    Gaussian counterpart), which is exact on every coordinate: rho^2 = cap^2_all - (cap^2_subset - bound_subset).

    Args:
        P: (n_cells,) cell probabilities.
        H: (n_cells, |coords|) conditional entropies (bits).
        H_marg: (|coords|,) marginal entropies (bits).
        V_all: (n_cells, d) within-cell variances on every coordinate.
        V_sub: (n_cells, |coords|) within-cell variances on the subset.
        coords: the subset's coordinate indices.
        nu2: dither level.
        gamma_b_range: (low, high) clip of the per-block leakage.

    Returns:
        rho2_cell, rho2_blk, cap2 (the decoder's squared error), negent_bits (mean in-cell negentropy),
        gamma_b_mean, and mc_se (the coordinate-subsampling standard error of rho2_cell).
    """
    var_all = (P[:, None] * V_all).sum(0)
    cap2_full, cap2_sub = float(var_all.mean()), float(var_all[coords].mean())
    term = (P[:, None] * maxent_var(H)).sum(0)
    gamma_b = np.clip(H_marg - (P[:, None] * H).sum(0), *gamma_b_range)
    blk_sub = (1.0 / TWO_PI_E) * float(np.exp2(2.0 * (H_marg - gamma_b)).mean()) - nu2
    negent = (P[:, None] * (0.5 * np.log2(TWO_PI_E * np.maximum(V_sub + nu2, 1e-12)) - H)).sum(0)
    return dict(rho2_cell=cap2_full - (cap2_sub - (float(term.mean()) - nu2)),
                rho2_blk=cap2_full - (cap2_sub - blk_sub), cap2=cap2_full, negent_bits=float(negent.mean()),
                gamma_b_mean=float(gamma_b.mean()),
                mc_se=float(np.std(var_all[coords] - term)) / math.sqrt(len(coords)))
