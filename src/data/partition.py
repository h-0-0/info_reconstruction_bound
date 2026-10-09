"""The data behind the entropy bound (app:data_handling): the estimation sample D_est and the disjoint D_perp, built
from the attack split of :mod:`src.data.pool`; both exclude the attack targets.

    dataset              D_est (plug-in estimate, t_MD)     D_perp (centring mean, unit-trace scale, eigenbasis)
    MNIST, CIFAR-10      D_rest + D_shadow                  the rest of the pool
    CelebA               D_shadow                           D_rest + the 112,000 extra images after the pool
    ImageNette           D_shadow                           D_rest (no eigenbasis: "signs" rotation)

D_perp never overlaps D_est: the mean, the scale and the eigenbasis must not depend on the estimation sample, or
one record would move more than one plug-in centre (app:bdd_diff).
"""

from __future__ import annotations

import numpy as np

from src.data.pool import SPLIT_SEED, load_celeba_extra, load_pool, split_indices

GREY = 255.0            # [0, 1] pool scale -> grey levels (Delta = 1)


def load_est_and_v(dataset: str, data_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
    """D_est and D_perp of a dataset, as in the table above.

    Args:
        dataset: "mnist", "cifar10", "celeba" or "imagenette".
        data_dir: data directory.

    Returns:
        X_est: (n_est, d) float32 images on [0, 1], in pool order.
        X_v: (n_perp, d) float32 images on [0, 1] (D_perp).
    """
    idx = split_indices(dataset, data_dir)
    images = np.asarray(load_pool(dataset, data_dir)[0], dtype=np.float32)
    rest, shadow, targets = (np.sort(idx[k]) for k in ("idx_rest", "idx_shadow", "idx_targets"))
    if dataset in ("celeba", "imagenette"):
        est, perp = shadow, rest
    else:
        est = np.sort(np.concatenate([rest, shadow]))
        perp = np.setdiff1d(np.arange(images.shape[0]), np.concatenate([est, targets]))
    if perp.size == 0:
        raise ValueError(f"{dataset}: no pool records left for D_perp")
    for a, b, what in ((est, perp, "D_est and D_perp"), (est, targets, "D_est and the targets"),
                       (perp, targets, "D_perp and the targets")):
        assert np.intersect1d(a, b).size == 0, f"{dataset}: {what} overlap"
    X_est, X_v = images[est], images[perp]
    if dataset == "celeba":                            # the extra images follow the pool, so are disjoint from it
        X_v = np.concatenate([X_v, load_celeba_extra(data_dir)])
    return X_est, X_v


def split_policy(dataset: str, n_est: int, n_v: int) -> dict:
    """The data-policy record stored in every bound file.

    Args:
        dataset: dataset name.
        n_est: size of D_est.
        n_v: size of D_perp.

    Returns:
        Dict with est and v (which splits form D_est and D_perp), source, split_seed, n_est and n_v.
    """
    est = {"celeba": "D_shadow (targets excluded)",
           "imagenette": "D_shadow (signs rotation; targets excluded)"}.get(
               dataset, "D_rest+D_shadow (targets excluded)")
    v = {"celeba": "D_rest + the 112k extra shard images",
         "imagenette": "D_rest (scale only)"}.get(dataset, "pool leftover")
    return {"est": est, "v": v, "source": "repo", "split_seed": SPLIT_SEED, "n_est": int(n_est), "n_v": int(n_v)}
