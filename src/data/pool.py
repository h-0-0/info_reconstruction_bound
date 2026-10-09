"""The record pool of each dataset and its fixed split into three disjoint parts (app:data_handling):
  D_rest     — records outside the shadow set (part of D_perp, or of D_est on MNIST/CIFAR-10)
  D_shadow   — the attack's shadow sample (and all or most of the entropy estimation sample D_est)
  D_targets  — a held-out slice of 1,000 records: the attack's targets are its first 100 (the paper's D_targets);
               the codebook decoder of the tightness chain is scored on all 1,000

Images are on [0, 1], flat (N, d) float32 in CHW order. The split indices ship in results/attack/splits/; decoded
CelebA and ImageNette pools are cached as .npz under data/.
"""

import io
import os
import shutil
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms

SPLITS_DIR = "results/attack/splits"
SPLIT_SEED = 42

SPLIT_SIZES = {
    "mnist": {"D_rest": 1_000, "D_shadow": 58_000, "D_targets": 1_000},
    "cifar10": {"D_rest": 5_000, "D_shadow": 44_000, "D_targets": 1_000},
    "celeba": {"D_rest": 5_000, "D_shadow": 44_000, "D_targets": 1_000},
    "imagenette": {"D_rest": 1_000, "D_shadow": 10_000, "D_targets": 1_000},
}

# CelebA: aligned faces centre-cropped to 178x178 and resized to 64x64, label = the Male attribute. Read from the
# HuggingFace parquet mirror (torchvision's Google-Drive download is rate-limited). The pool is the first 50k images
# of the shard stream; the next 112k are the "extra" images that join D_perp.
CELEBA_IMG_SIZE = 64
CELEBA_ATTR = "Male"
CELEBA_POOL = 50_000
CELEBA_EXTRA_N = 112_000
CELEBA_HF_URL = "https://huggingface.co/datasets/tpremoli/CelebA-attrs/resolve/main/{path}"
CELEBA_HF_SHARDS = [f"data/train-0000{i}-of-00003.parquet" for i in range(3)]

# ImageNette: imagenette2-320 extracted under data/, official train+val pooled (13,394 images), resized and
# centre-cropped to 128x128 (d = 49,152); labels = the 10 synsets in sorted order.
IMAGENETTE_IMG_SIZE = 128
IMAGENETTE_DIR = "imagenette/imagenette2-320"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_SYNSETS = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
                      "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"]


def _save_npz(path: str, **arrays) -> None:
    """Write arrays to a compressed .npz through a temporary file, so an interrupted write leaves no corrupt cache.

    Args:
        path: destination .npz path.
        **arrays: the arrays to store, by name.
    """
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ── raw pools ─────────────────────────────────────────────────────────────────

def _download_celeba_shard(data_dir: str, shard: str) -> str:
    """One CelebA parquet shard under data_dir/celeba/, downloaded from the HuggingFace mirror if absent.

    Args:
        data_dir: data directory.
        shard: shard path within the mirror (an entry of CELEBA_HF_SHARDS).

    Returns:
        Local path of the shard.
    """
    dst = os.path.join(data_dir, "celeba", os.path.basename(shard))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = CELEBA_HF_URL.format(path=shard)
    print(f"  [celeba] downloading {url}")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst), suffix=".parquet.tmp")
    os.close(fd)
    try:
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dst


def _decode_celeba(data_dir: str, skip: int, n: int):
    """Decode images [skip, skip + n) of the CelebA shard stream (centre crop 178, resize to 64).

    Args:
        data_dir: data directory.
        skip: number of images to skip first.
        n: number of images to decode.

    Returns:
        images: (n, 12288) float32 on [0, 1], CHW.
        labels: (n,) int64, the Male attribute as 0/1.
    """
    tf = transforms.Compose([transforms.CenterCrop(178), transforms.Resize(CELEBA_IMG_SIZE), transforms.ToTensor()])
    images = np.empty((n, 3 * CELEBA_IMG_SIZE ** 2), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    seen = filled = 0
    for shard in CELEBA_HF_SHARDS:
        if filled >= n:
            break
        df = pd.read_parquet(_download_celeba_shard(data_dir, shard), columns=["image", CELEBA_ATTR])
        for cell, attr in zip(df["image"], df[CELEBA_ATTR]):
            if seen < skip:
                seen += 1
                continue
            if filled >= n:
                break
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            images[filled] = tf(Image.open(io.BytesIO(raw)).convert("RGB")).numpy().reshape(-1)
            labels[filled] = 1 if int(attr) > 0 else 0          # attributes are stored as -1/1
            filled += 1
            seen += 1
        print(f"  [celeba] decoded {filled}/{n} images", flush=True)
    return images[:filled], labels[:filled]


def _load_celeba(data_dir: str):
    """The CelebA pool (first 50,000 images of the shard stream), decoded once and cached as .npz."""
    cache = os.path.join(data_dir, f"celeba_{CELEBA_IMG_SIZE}_{CELEBA_ATTR}_pool{CELEBA_POOL}.npz")
    if not os.path.exists(cache):
        images, labels = _decode_celeba(data_dir, 0, CELEBA_POOL)
        _save_npz(cache, images=images, labels=labels)
    with np.load(cache) as d:
        return d["images"], d["labels"]


def load_celeba_extra(data_dir: str) -> np.ndarray:
    """The 112,000 CelebA images that follow the pool in the shard stream (disjoint from it by position).

    Args:
        data_dir: data directory.

    Returns:
        (112000, 12288) float32 images on [0, 1], CHW.
    """
    cache = os.path.join(data_dir, f"celeba_extra_{CELEBA_IMG_SIZE}_skip{CELEBA_POOL}_n{CELEBA_EXTRA_N}.npz")
    if not os.path.exists(cache):
        _save_npz(cache, images=_decode_celeba(data_dir, CELEBA_POOL, CELEBA_EXTRA_N)[0])
    with np.load(cache) as d:
        return d["images"]


def _load_imagenette(data_dir: str):
    """The ImageNette pool (official train + val, 13,394 images at 128 x 128), decoded once and cached as .npz."""
    cache = os.path.join(data_dir, f"imagenette_{IMAGENETTE_IMG_SIZE}_pool.npz")
    if not os.path.exists(cache):
        root = os.path.join(data_dir, IMAGENETTE_DIR)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"ImageNette not found at {root}: download and extract {IMAGENETTE_URL} there.")
        tf = transforms.Compose([transforms.Resize(IMAGENETTE_IMG_SIZE), transforms.CenterCrop(IMAGENETTE_IMG_SIZE),
                                 transforms.ToTensor()])
        imgs, labs = [], []
        for split in ("train", "val"):
            for label, synset in enumerate(IMAGENETTE_SYNSETS):
                cls_dir = os.path.join(root, split, synset)
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                        pil = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
                        imgs.append(tf(pil).numpy().reshape(-1))
                        labs.append(label)
            print(f"  [imagenette] decoded {len(imgs)} images after {split}", flush=True)
        _save_npz(cache, images=np.asarray(imgs, dtype=np.float32), labels=np.asarray(labs, dtype=np.int64))
    with np.load(cache) as d:
        return d["images"], d["labels"]


def load_pool(dataset: str, data_dir: str = "data"):
    """The whole record pool of a dataset.

    MNIST and CIFAR-10: the official train and test sets, concatenated. CelebA and ImageNette: see above.

    Args:
        dataset: "mnist", "cifar10", "celeba" or "imagenette".
        data_dir: data directory.

    Returns:
        images: (N, d) float32 on [0, 1], flattened in CHW order.
        labels: (N,) int64.
    """
    if dataset == "celeba":
        return _load_celeba(data_dir)
    if dataset == "imagenette":
        return _load_imagenette(data_dir)
    sets = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}
    if dataset not in sets:
        raise ValueError(f"Unsupported dataset: {dataset}")
    images, labels = [], []
    for train in (True, False):
        ds = sets[dataset](data_dir, train=train, download=True)
        raw = np.asarray(ds.data)                                   # uint8: MNIST (N, 28, 28), CIFAR-10 (N, 32, 32, 3)
        if raw.ndim == 4:
            raw = raw.transpose(0, 3, 1, 2)                         # HWC -> CHW
        images.append(raw.reshape(len(raw), -1).astype(np.float32) / np.float32(255.0))   # = ToTensor, bit for bit
        labels.append(np.asarray(ds.targets, dtype=np.int64))
    return np.concatenate(images), np.concatenate(labels)


# ── the attack split ──────────────────────────────────────────────────────────

def split_indices(dataset: str, data_dir: str = "data") -> dict:
    """The fixed attack split (seed SPLIT_SEED), created on first use and then read from results/attack/splits/.

    Args:
        dataset: dataset name.
        data_dir: data directory (only needed to create the split).

    Returns:
        Dict of pool-index arrays idx_rest, idx_shadow and idx_targets.
    """
    path = os.path.join(SPLITS_DIR, f"{dataset}_seed{SPLIT_SEED}.npz")
    if not os.path.exists(path):
        sizes = SPLIT_SIZES[dataset]
        n_pool = len(load_pool(dataset, data_dir)[0])
        if n_pool < sum(sizes.values()):
            raise ValueError(f"{dataset} has {n_pool} samples but needs {sum(sizes.values())}")
        idx = np.random.default_rng(SPLIT_SEED).permutation(n_pool)
        a, b, c = np.cumsum([sizes["D_rest"], sizes["D_shadow"], sizes["D_targets"]])
        os.makedirs(SPLITS_DIR, exist_ok=True)
        _save_npz(path, idx_rest=idx[:a], idx_shadow=idx[a:b], idx_targets=idx[b:c])
    with np.load(path) as f:
        return {k: f[k] for k in ("idx_rest", "idx_shadow", "idx_targets")}


def load_splits(dataset: str, data_dir: str = "data") -> dict:
    """The three parts of the attack split, with their images and labels.

    Args:
        dataset: dataset name.
        data_dir: data directory.

    Returns:
        Dict mapping "D_rest", "D_shadow" and "D_targets" to {"images": (N, d) float32, "labels": (N,) int64}.
    """
    images, labels = load_pool(dataset, data_dir)
    idx = split_indices(dataset, data_dir)
    return {name: {"images": images[idx[key]], "labels": labels[idx[key]]}
            for name, key in (("D_rest", "idx_rest"), ("D_shadow", "idx_shadow"), ("D_targets", "idx_targets"))}
