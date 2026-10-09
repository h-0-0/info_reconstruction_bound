"""DP-SGD training with Opacus, as in the privacy-spectrum experiment (app:exp_spectrum).

Poisson-subsampled DP-SGD in noise-multiplier mode with SGD momentum 0.9, augmentation multiplicity K (K views per
record, gradients averaged before clipping), physical micro-batches (same gradients, less memory), and an exponential
moving average (EMA) of the weights for evaluation. Momentum and the EMA only post-process the clipped, noised
updates, so the accountant and the leakage gamma are unchanged.

The EMA is updated once per physical micro-batch with decay 0.999, while the weights change only on the last
micro-batch of each step; its decay per optimiser step is therefore 0.999^(micro-batches per step).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from src.privacy.accounting import compute_delta
from src.training.augment import AUG_POLICY, batched_augment

MOMENTUM = 0.9
EMA_DECAY = 0.999


@dataclass
class DPSGDConfig:
    """One DP-SGD run.

    Args:
        dataset: dataset name (selects the augmentation policy).
        noise_multiplier: sigma (0: clipping only, no privacy; the non-private baseline).
        max_grad_norm: clipping norm C.
        lr: learning rate.
        epochs: number of epochs.
        augmult_k: augmentation multiplicity K.
        max_physical_batch: largest micro-batch.
    """

    dataset: str
    noise_multiplier: float
    max_grad_norm: float
    lr: float
    epochs: int
    augmult_k: int = 1
    max_physical_batch: int = 256


def train_model_dpsgd(model: nn.Module, train_loader: DataLoader, config: DPSGDConfig, device) -> dict:
    """Train a model with Opacus DP-SGD in noise-multiplier mode.

    Args:
        model: the network (must already be Opacus-compatible).
        train_loader: training loader; its batch size is the expected logical batch (Opacus resamples it with
            Poisson sampling).
        config: the run's settings.
        device: torch device.

    Returns:
        model: the final weights.
        ema_model: the EMA of the weights (the evaluated model).
        train_acc, train_loss: on the last epoch's augmented views.
        noise_multiplier, total_steps, sampling_rate, n_train, target_delta: the privacy schedule (delta = n^{-1.1}).
        augmult_k, momentum, aug_policy, ema_decay: the recipe.
    """
    if config.noise_multiplier < 0:
        raise ValueError(f"noise_multiplier must be >= 0 (got {config.noise_multiplier})")
    ModuleValidator.validate(model, strict=True)
    model.to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=MOMENTUM)
    privacy_engine = PrivacyEngine()
    private_model, optimizer, train_loader = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=train_loader,
        noise_multiplier=config.noise_multiplier, max_grad_norm=config.max_grad_norm)
    n_train = len(train_loader.dataset)
    delta = compute_delta(n_train)
    aug_policy = AUG_POLICY[config.dataset]
    K = config.augmult_k

    with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=config.max_physical_batch,
                            optimizer=optimizer) as loader:
        for epoch in range(config.epochs):
            private_model.train()
            total_loss, correct, total = 0.0, 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                # one backward over the mean loss of K views: Opacus clips the view-averaged per-sample gradient,
                # so the sensitivity stays C (De et al. 2022)
                loss = 0.0
                for _ in range(K):
                    logits = private_model(batched_augment(x, **aug_policy))
                    loss = loss + F.cross_entropy(logits.float(), y) / K
                loss.backward()
                optimizer.step()
                ema.update_parameters(model)

                total_loss += loss.item() * len(x)
                correct += (logits.argmax(dim=-1) == y).sum().item()
                total += len(x)

            if epoch == 0 or (epoch + 1) % 10 == 0:
                eps = privacy_engine.get_epsilon(delta=delta) if config.noise_multiplier > 0 else math.inf
                print(f"  Epoch {epoch + 1}/{config.epochs} loss={total_loss / total:.4f} "
                      f"acc={correct / total:.4f} eps={eps:.2f}")

    history = privacy_engine.accountant.history          # entries (sigma, q, steps)
    return {
        "model": model,
        "ema_model": ema.module,
        "train_acc": correct / total,
        "train_loss": total_loss / total,
        "noise_multiplier": config.noise_multiplier,
        "total_steps": sum(steps for _, _, steps in history),
        "sampling_rate": float(history[0][1]),
        "n_train": n_train,
        "target_delta": delta,
        "augmult_k": K,
        "momentum": MOMENTUM,
        "aug_policy": dict(aug_policy),
        "ema_decay": EMA_DECAY,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> dict:
    """Top-1 accuracy and mean cross-entropy of a model.

    Args:
        model: the network.
        loader: evaluation loader.
        device: torch device.

    Returns:
        {"accuracy": ..., "loss": ...}.
    """
    model.eval()
    model.to(device)
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y).item() * len(x)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += len(x)
    return {"accuracy": correct / total, "loss": total_loss / total}
