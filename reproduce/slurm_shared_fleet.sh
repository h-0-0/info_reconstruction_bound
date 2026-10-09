#!/bin/bash
# Shared target-epsilon momentum fleet: one array task per row of
# results/utility/shared_eps_fleet_params.tsv  (ds  model  C  nm  lr  seed).
# Momentum 0.9 recipe (matches the archived fleet), 100 epochs. Dedicated GPU.
#   sbatch --array=0-<N-1>%16 reproduce/slurm_shared_fleet.sh      (RUNS_DIR=<dir> writes the records elsewhere)
# Partition: pass --partition=<GPU partition> to sbatch (reproduce/run_all.sh does, from reproduce/config.sh).
#SBATCH --job-name=shared_fleet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/shared_fleet_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"          # submit from the repo root
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PARAMS=results/utility/shared_eps_fleet_params.tsv
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAMS")
[ -z "$LINE" ] && { echo "no row for idx ${SLURM_ARRAY_TASK_ID}"; exit 0; }
IFS=$'\t' read -r DS MODEL C NM LR SEED <<< "$LINE"

case "$DS" in
  mnist)      K=1; PHYS=256 ;;
  imagenette) K=4; PHYS=16  ;;
  *)          K=4; PHYS=16  ;;
esac
echo "== shared fleet idx ${SLURM_ARRAY_TASK_ID}: ${DS}/${MODEL} C=${C} nm=${NM} lr=${LR} seed=${SEED} (momentum 0.9) =="
export AUGMULT_K=$K
export MAX_PHYSICAL_BATCH=$PHYS
.venv/bin/python scripts/run_dp_by_noise.py \
    --model "$MODEL" --dataset "$DS" \
    --noise_multiplier "$NM" --max_grad_norm "$C" \
    --lr "$LR" --seed "$SEED" --runs_dir "${RUNS_DIR:-results/utility/runs}" \
    --epochs 100 --batch_size 1024
