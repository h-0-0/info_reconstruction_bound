#!/bin/bash
# Phi prep: build the LPIPS feature cache + the SHARED rotated memmap once,
# before the pair array runs (the array tasks reuse both read-only).
#   sbatch --export=ALL,DATASET=mnist reproduce/slurm_bound_prep_phi.sh
# Partition: pass --partition=<GPU partition> to sbatch (reproduce/run_all.sh does, from reproduce/config.sh).
#SBATCH --job-name=plugin_prep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/plugin_prep_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"          # submit from the repo root
export PYTHONPATH="$PWD"
.venv/bin/python scripts/run_entropy_bound.py \
    --dataset "${DATASET}" --space phi --prep
