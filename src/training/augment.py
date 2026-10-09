"""Data-independent augmentation for DP-SGD training (augmentation multiplicity, De et al. 2022).

Each view is a random crop after reflect-padding and/or a horizontal flip, drawn per sample: no statistic of other
records is involved, so augmentation adds no coupling between records (asm:iid, app:data_handling). torchvision's
RandomCrop / RandomHorizontalFlip draw one view per call (the same for a whole batch), hence the batched version here.
"""
import torch
import torch.nn.functional as F

# per-dataset policy: reflect-pad by `pad` then crop back at a random offset; flip horizontally at random
AUG_POLICY = {
    "mnist": dict(pad=0, hflip=False),        # none
    "cifar10": dict(pad=4, hflip=True),
    "celeba": dict(pad=0, hflip=True),        # faces: flip only
    "imagenette": dict(pad=8, hflip=True),
}


def batched_augment(x, pad: int = 0, hflip: bool = False, generator=None):
    """One random view per sample: crop after reflect-padding by ``pad``, then flip with probability 1/2.

    Args:
        x: (B, C, H, W) batch.
        pad: reflect-padding before the random crop back to H x W (0: no crop).
        hflip: flip each sample horizontally with probability 1/2.
        generator: optional torch generator for the random offsets and flips.

    Returns:
        (B, C, H, W) augmented batch (``x`` itself, with no random draws, when pad = 0 and hflip is False).
    """
    B, C, H, W = x.shape
    if pad > 0:
        xp = F.pad(x, (pad,) * 4, mode="reflect")
        dx = torch.randint(0, 2 * pad + 1, (B,), generator=generator, device=x.device)
        dy = torch.randint(0, 2 * pad + 1, (B,), generator=generator, device=x.device)
        idx_h = (dy[:, None] + torch.arange(H, device=x.device)[None, :])
        idx_w = (dx[:, None] + torch.arange(W, device=x.device)[None, :])
        xp = xp.gather(2, idx_h[:, None, :, None].expand(B, C, H, xp.shape[3]))
        x = xp.gather(3, idx_w[:, None, None, :].expand(B, C, H, W))
    if hflip:
        flip = torch.rand(B, generator=generator, device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    return x
