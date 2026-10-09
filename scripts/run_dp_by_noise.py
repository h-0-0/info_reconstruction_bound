"""One DP-SGD training run at a given noise multiplier (the privacy-spectrum fleet, app:exp_spectrum).

With ``--lr`` the learning rate is fixed (every fleet run); with ``--lr_candidates`` the script trains once per
candidate and keeps the one with the best validation accuracy (the non-private baseline's lr search, run at noise
multiplier 0 with a huge clip norm). Writes one JSON record per run to ``--runs_dir`` (accuracies, the leakage gamma,
the PRV epsilon at delta = n^{-1.1}, and the recipe), and skips a run whose record already exists.

    python scripts/run_dp_by_noise.py --model lenet5 --dataset mnist --noise_multiplier 1.11 \
        --max_grad_norm 1 --lr 0.5 --seed 42 --epochs 100 --batch_size 1024

The augmentation multiplicity and the physical batch come from the environment (AUGMULT_K, MAX_PHYSICAL_BATCH), as
reproduce/slurm_shared_fleet.sh sets them.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.datasets import make_dataloaders                        # noqa: E402
from src.models import get_model                                      # noqa: E402
from src.privacy.accounting import compute_epsilon_prv                # noqa: E402
from src.privacy.gamma import gamma_prv_mean                          # noqa: E402
from src.training.train_dpsgd import DPSGDConfig, evaluate, train_model_dpsgd   # noqa: E402
from src.utils import get_completed_run_ids, get_device, set_seed, write_run_result   # noqa: E402

DEFAULT_LR_CANDIDATES = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]


def experiment_id(args) -> str:
    """The run's id (and record file name): dataset, model, noise multiplier, C and seed."""
    return f"dpsgd_nm_{args.dataset}_{args.model}_nm{args.noise_multiplier}_C{args.max_grad_norm}_s{args.seed}"


def train_and_evaluate(args, lr: float, device) -> dict:
    """Train one model at one learning rate and evaluate its EMA weights.

    Args:
        args: parsed command line.
        lr: learning rate.
        device: torch device.

    Returns:
        The training result of :func:`train_model_dpsgd` plus val_acc, val_loss, test_acc, test_loss and num_params.
    """
    set_seed(args.seed)
    train_loader, val_loader, test_loader = make_dataloaders(args.dataset, args.batch_size, args.data_dir)
    model = get_model(args.dataset, args.model)
    config = DPSGDConfig(
        dataset=args.dataset, noise_multiplier=args.noise_multiplier, max_grad_norm=args.max_grad_norm, lr=lr,
        epochs=args.epochs, augmult_k=int(os.environ.get("AUGMULT_K", 1)),
        max_physical_batch=int(os.environ.get("MAX_PHYSICAL_BATCH", 0) or 256))
    result = train_model_dpsgd(model, train_loader, config, device)
    val = evaluate(result["ema_model"], val_loader, device)           # the EMA weights are the evaluated model
    test = evaluate(result["ema_model"], test_loader, device)
    return {**result, "val_acc": val["accuracy"], "val_loss": val["loss"], "test_acc": test["accuracy"],
            "test_loss": test["loss"], "num_params": sum(p.numel() for p in model.parameters())}


def run_record(args, lr: float, m: dict) -> dict:
    """The run's JSON record.

    Args:
        args: parsed command line.
        lr: the learning rate used (the search winner).
        m: output of :func:`train_and_evaluate` at that learning rate.

    Returns:
        The record (accuracies, privacy schedule, gamma, epsilon, recipe).
    """
    T, q, sigma = m["total_steps"], m["sampling_rate"], m["noise_multiplier"]
    if sigma > 0:
        gamma_nats = gamma_prv_mean(T, q, sigma)
        epsilon = compute_epsilon_prv(sigma, m["target_delta"], q, T)
    else:                                                               # the non-private baseline: no guarantee
        gamma_nats = epsilon = math.inf
    return {
        "experiment_id": experiment_id(args),
        "model_name": args.model,
        "dataset": args.dataset,
        "max_grad_norm": args.max_grad_norm,
        "noise_multiplier": f"{sigma:.6f}",
        "best_lr": lr,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_acc": f"{m['train_acc']:.6f}",
        "val_acc": f"{m['val_acc']:.6f}",
        "test_acc": f"{m['test_acc']:.6f}",
        "test_loss": f"{m['test_loss']:.6f}",
        "epsilon": f"{epsilon:.6f}",
        "gamma_prv_nats": f"{gamma_nats:.8f}",
        "gamma_prv_bits": f"{gamma_nats / math.log(2):.8f}",
        "total_steps": T,
        "sampling_rate": f"{q:.8f}",
        "num_params": m["num_params"],
        "n_train": m["n_train"],
        "target_delta": m["target_delta"],
        **{k: m[k] for k in ("augmult_k", "momentum", "aug_policy", "ema_decay")},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["lenet5", "wrn16_4_gn", "resnet18gn"])
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--noise_multiplier", type=float, required=True)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=None, help="fixed learning rate (no search)")
    ap.add_argument("--lr_candidates", nargs="+", type=float, default=DEFAULT_LR_CANDIDATES,
                    help="learning rates to search when --lr is not given")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1024, help="expected (Poisson) batch size")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--runs_dir", default=os.path.join("results", "utility", "runs"), help="where the record goes")
    args = ap.parse_args()

    exp_id = experiment_id(args)
    if exp_id in get_completed_run_ids(args.runs_dir):
        print(f"[skip] {exp_id} already completed")
        return
    print(f"Running {exp_id}")
    device = get_device()
    best_lr, best = None, None
    for lr in [args.lr] if args.lr is not None else args.lr_candidates:
        m = train_and_evaluate(args, lr, device)
        print(f"  lr={lr}  val_acc={m['val_acc']:.4f}  test_acc={m['test_acc']:.4f}")
        if best is None or m["val_acc"] > best["val_acc"]:
            best_lr, best = lr, m
    print(f"Best LR: {best_lr}  (val_acc={best['val_acc']:.4f})")
    record = run_record(args, best_lr, best)
    write_run_result(args.runs_dir, exp_id, record)
    print(f"Done: test_acc={record['test_acc']}  best_lr={best_lr}  gamma_prv={record['gamma_prv_bits']} bits")


if __name__ == "__main__":
    main()
