"""Build the DP-SGD fleet's run table: results/utility/shared_eps_fleet_params.tsv, one run per row
(dataset, model, C, noise multiplier, lr, seed), run by reproduce/slurm_shared_fleet.sh.

Every dataset is trained at the same target epsilons: 0.1 to 32 (EPS), then 64 and 128 (HIGH_EPS), each with
seeds 42, 43, 44, so 33 runs per dataset. For each (dataset, epsilon) the noise multiplier is solved with the PRV
accountant at the dataset's sampling rate q and step count T (read from any 100-epoch run of its model; the
non-private baseline runs suffice) and delta = n^{-1.1}. C = 1 everywhere (neither gamma nor the bound depends on it)
and the learning rate is fixed per dataset (FLEET_LR), not tuned per privacy level.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scipy.optimize import brentq  # noqa: E402

from src.privacy.accounting import compute_epsilon_prv  # noqa: E402
from src.utils import load_run_records  # noqa: E402

EPS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
HIGH_EPS = [64.0, 128.0]
SEEDS = [42, 43, 44]
MODEL = {"mnist": "lenet5", "cifar10": "wrn16_4_gn", "celeba": "resnet18gn", "imagenette": "resnet18gn"}
FLEET_C = 1.0
FLEET_LR = {"mnist": 0.5, "cifar10": 1.0, "celeba": 0.25, "imagenette": 0.25}
EPOCHS = 100
BATCH = 1024


def schedules() -> dict[str, tuple[float, int]]:
    """The sampling rate q and step count T of each dataset's 100-epoch runs.

    Returns:
        {dataset: (q, T)}, from the first matching run record.
    """
    out = {}
    for r in load_run_records(os.path.join(ROOT, "results/utility/runs")):
        if MODEL.get(r["dataset"]) == r["model_name"] and int(r["epochs"]) == EPOCHS:
            out.setdefault(r["dataset"], (float(r["sampling_rate"]), int(r["total_steps"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/utility/shared_eps_fleet_params.tsv"))
    ap.add_argument("--datasets", nargs="+", default=list(MODEL))
    a = ap.parse_args()
    sched = schedules()
    missing = [ds for ds in a.datasets if ds not in sched]
    if missing:
        raise SystemExit(f"no {EPOCHS}-epoch run for {missing} in results/utility/runs/ (run the baseline first)")
    rows = []
    for eps_grid in (EPS, HIGH_EPS):                     # all datasets' main grid first, then the high block
        for ds in a.datasets:
            q, T = sched[ds]
            delta = round(BATCH / q) ** -1.1             # n = b / q (Opacus sets q = 1 / ceil(n / b))
            print(f"{ds}: q={q:.5f} T={T} delta={delta:.2e} C={FLEET_C:g} lr={FLEET_LR[ds]:g}", flush=True)
            for eps in eps_grid:
                nm = brentq(lambda s: compute_epsilon_prv(s, delta, q, T) - eps, 0.30, 300.0, xtol=1e-4)
                print(f"    eps={eps:<6g} -> nm={nm:.6f}", flush=True)
                rows += [(ds, MODEL[ds], f"{FLEET_C:g}", f"{nm:.6f}", f"{FLEET_LR[ds]:g}", str(s)) for s in SEEDS]
    with open(a.out + ".tmp", "w") as f:
        f.writelines("\t".join(r) + "\n" for r in rows)
    os.replace(a.out + ".tmp", a.out)
    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
