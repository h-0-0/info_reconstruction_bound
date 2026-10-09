"""Train / validation / test loaders for the privacy-spectrum DP-SGD runs (app:exp_spectrum).

This train/val/test split is separate from the attack split of :mod:`src.data.pool`:
    MNIST       official train set split 50,000 / 10,000 (seeded), official test set (10,000)
    CIFAR-10    official train set split 45,000 / 5,000 (seeded), official test set (10,000); inputs standardised
                with fixed (ImageNet) statistics
    CelebA      the 50,000-image pool permuted (seeded) into 45,000 / 4,000 / 1,000
    ImageNette  the 13,394-image pool permuted (seeded) into 9,469 / 2,000 / 1,925
MNIST stays on [0, 1]; CelebA and ImageNette are mapped to [-1, 1] by fixed constants (no dataset statistics).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from src.data.pool import load_pool

# CIFAR-10 is standardised with the ImageNet per-channel statistics: fixed constants that do not depend on CIFAR-10
# (ImageNet does not contain it), so no record's preprocessing depends on another record (asm:iid). The usual CIFAR-10
# statistics would not do: they were computed on the official training set, which contains D_train.
CIFAR10_MEAN = (0.485, 0.456, 0.406)
CIFAR10_STD = (0.229, 0.224, 0.225)

# pool datasets: (image side, n_train, n_val); the rest of the pool is the test set
POOL_SPLITS = {"celeba": (64, 45_000, 4_000), "imagenette": (128, 9_469, 2_000)}


def make_dataloaders(dataset: str, batch_size: int = 1024,
                     data_dir: str = "data") -> tuple[DataLoader, DataLoader, DataLoader]:
    """The train, validation and test loaders of a dataset (the split is fixed by seed 0).

    Args:
        dataset: "mnist", "cifar10", "celeba" or "imagenette".
        batch_size: batch size of all three loaders (the train loader shuffles).
        data_dir: data directory.

    Returns:
        train_loader, val_loader, test_loader, yielding (image, label) batches; images are (C, H, W) tensors.
    """
    if dataset in ("mnist", "cifar10"):
        if dataset == "mnist":
            source, tf, n_val = datasets.MNIST, transforms.ToTensor(), 10_000
        else:
            source, n_val = datasets.CIFAR10, 5_000
            tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
        full_train = source(data_dir, train=True, download=True, transform=tf)
        test_ds = source(data_dir, train=False, download=True, transform=tf)
        gen = torch.Generator().manual_seed(0)
        train_idx, val_idx = torch.utils.data.random_split(
            range(len(full_train)), [len(full_train) - n_val, n_val], generator=gen)
        train_ds, val_ds = Subset(full_train, train_idx.indices), Subset(full_train, val_idx.indices)
    elif dataset in POOL_SPLITS:
        side, n_train, n_val = POOL_SPLITS[dataset]
        X, y = load_pool(dataset, data_dir)
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).view(-1, 3, side, side)
        X.sub_(0.5).div_(0.5)                                  # [0, 1] -> [-1, 1], in place on the fresh pool array
        y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        assert X.shape[0] > n_train + n_val, f"{dataset}: pool {X.shape[0]} too small for the split"
        perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0))
        tr, va, te = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
        train_ds, val_ds, test_ds = (TensorDataset(X[i], y[i]) for i in (tr, va, te))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False))
