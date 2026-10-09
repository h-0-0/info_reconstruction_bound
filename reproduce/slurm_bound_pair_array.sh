#!/bin/bash
# ============================================================
# slurm_bound_pair_array.sh — ONE grid pair per array task.
#
# Submit (array range = grid size, %N throttles concurrency):
#   sbatch --array=0-27%16 --export=ALL,DATASET=mnist,SPACE=pixel \
#          reproduce/slurm_bound_pair_array.sh
# Tasks with index >= |G| exit 0 immediately (safe upper bound for phi,
# whose grid size depends on the padded feature dimension).
# Follow with slurm_bound_assemble.sh (afterok on the array).
# ============================================================
# Partition: pass --partition=<GPU partition> to sbatch (reproduce/run_all.sh does, from reproduce/config.sh).
#SBATCH --job-name=plugin_pair
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/plugin_pair_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"          # submit from the repo root
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET="${DATASET:-mnist}"
SPACE="${SPACE:-pixel}"
echo "== plugin pair ${SLURM_ARRAY_TASK_ID}: ${DATASET}/${SPACE} (job ${SLURM_JOB_ID}) =="
.venv/bin/python scripts/run_entropy_bound.py \
    --dataset "${DATASET}" --space "${SPACE}" \
    --pair-index "${SLURM_ARRAY_TASK_ID}" ${EXTRA:-}
