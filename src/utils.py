import glob
import json
import os
import random
import tempfile
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_run_filename(experiment_id: str) -> str:
    """experiment_ids in this repo are already filesystem-safe; guard regardless."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(experiment_id))


def write_run_result(runs_dir: str, experiment_id: str, result: Dict) -> str:
    """Atomically write one run's result dict to ``{runs_dir}/{experiment_id}.json``.

    Returns the path written.  The dict is copied and ``experiment_id`` is ensured
    to be present (merge/dedup keys on it).
    """
    os.makedirs(runs_dir, exist_ok=True)
    payload = dict(result)
    payload.setdefault("experiment_id", experiment_id)
    path = os.path.join(runs_dir, _safe_run_filename(experiment_id) + ".json")
    fd, tmp = tempfile.mkstemp(dir=runs_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, path) 
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def get_completed_run_ids(runs_dir: str) -> set:
    """Set of experiment_ids that already have a per-run JSON file (for resume)."""
    if not os.path.isdir(runs_dir):
        return set()
    return {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(runs_dir, "*.json"))
    }


def load_run_records(runs_dir: str) -> list[dict]:
    """Every run record in a directory.

    Args:
        runs_dir: directory of per-run JSON files (written by :func:`write_run_result`).

    Returns:
        The records, in file-name order.
    """
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(runs_dir, "*.json")))]
