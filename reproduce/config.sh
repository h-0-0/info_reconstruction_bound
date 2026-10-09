#!/usr/bin/env bash
# reproduce/config.sh — edit this for your system, then `bash reproduce/run_all.sh`.
# Every value is overridable from the environment (export before calling).

# ── Backend ──────────────────────────────────────────────────────────────────
# slurm : submit sbatch jobs, run in parallel, wired by --dependency (a cluster).
# local : run every step now, sequentially, on this machine (no SLURM needed).
export LAUNCHER="${LAUNCHER:-slurm}"

# ── Paths ────────────────────────────────────────────────────────────────────
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PY="${PY:-$REPO/.venv/bin/python}"
export PYTHONPATH="$REPO"

# ── Which datasets ───────────────────────────────────────────────────────────
export DATASETS="${DATASETS:-mnist cifar10 celeba imagenette}"

# ── SLURM knobs (ignored when LAUNCHER=local) ────────────────────────────────
export GPU_PART="${GPU_PART:-gpu}"                 # your cluster's GPU partition(s), comma-separated
export CPU_PART="${CPU_PART:-cpu}"                 # your cluster's CPU partition(s)
export GPU_MEM="${GPU_MEM:-24G}"
export MAX_CONC="${MAX_CONC:-16}"                  # array %concurrency throttle
# If your scheduler needs an account/qos, put the extra sbatch flags here:
export SBATCH_EXTRA="${SBATCH_EXTRA:-}"            # e.g. "--account=proj --qos=gpu"

source "$REPO/reproduce/lib.sh"
