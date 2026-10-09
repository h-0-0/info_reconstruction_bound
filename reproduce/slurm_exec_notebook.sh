#!/bin/bash
# Execute a notebook in place on a compute node (login-node nbconvert is banned).
#   sbatch --export=ALL,NB=notebooks/foo.ipynb reproduce/slurm_exec_notebook.sh
# Partition: the cluster default, or pass --partition=... to sbatch (reproduce/run_all.sh does).
#SBATCH --job-name=nb_exec
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/nb_exec_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"          # submit from the repo root
export PYTHONPATH="$PWD"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"   # fixed PDF timestamps: figures are byte-reproducible
KERNEL="${KERNEL:-python3}"
# the venv's own nbconvert (not `python -m jupyter nbconvert`, which can dispatch to another install on PATH)
.venv/bin/python -m nbconvert --to notebook --execute --inplace "${NB}" \
    --ExecutePreprocessor.kernel_name="${KERNEL}" \
    --ExecutePreprocessor.timeout=1800 || { echo "FAILED ${NB}"; exit 1; }
echo "executed ${NB}"
