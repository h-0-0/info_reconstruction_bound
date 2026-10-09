#!/bin/bash
# Merge array-task pair files into the final entropy_bound.json (CPU, seconds).
#   sbatch --dependency=afterok:<arrayjob> --export=ALL,DATASET=...,SPACE=... \
#          reproduce/slurm_bound_assemble.sh
# Partition: the cluster default, or pass --partition=... to sbatch (reproduce/run_all.sh does).
#SBATCH --job-name=plugin_asm
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/plugin_asm_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"          # submit from the repo root
export PYTHONPATH="$PWD"
.venv/bin/python scripts/run_entropy_bound.py \
    --dataset "${DATASET}" --space "${SPACE}" --assemble
