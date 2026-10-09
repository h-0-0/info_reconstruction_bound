#!/usr/bin/env bash
# reproduce/run_all.sh — reproduce every paper result from a fresh checkout.
#
#   bash reproduce/run_all.sh              # the whole DAG, in dependency order
#   bash reproduce/run_all.sh bounds       # one stage (setup|bounds|tightness|dpsgd|notebooks)
#   LAUNCHER=local bash reproduce/run_all.sh   # no SLURM: run sequentially here
#
# Edit reproduce/config.sh first (backend, partitions, datasets). The DAG:
#   setup ─► bounds ─┬─► tightness ─► notebooks   (tightness, bound details)
#                    └─► dpsgd ─────► notebooks   (utility curves)
# Every stage writes under results/; the notebooks turn results/ into figures/ and tables/. The shipped results/
# bundle already holds every stage's output, so the notebooks alone reproduce the paper's figures and tables.
# See README.md for the full picture. SLURM mode submits jobs and returns; run
# `squeue` to watch, or append `--wait` (below) to block.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
cd "$REPO"

declare -A MODEL=( [mnist]=lenet5 [cifar10]=wrn16_4_gn [celeba]=resnet18gn [imagenette]=resnet18gn )
declare -A KAUG=(  [mnist]=1 [cifar10]=4 [celeba]=4 [imagenette]=4 )
declare -A PHYS=(  [mnist]=256 [cifar10]=32 [celeba]=32 [imagenette]=16 )
declare -A PXRANGE=( [mnist]=0-47 )                    # pixel array range; others default 0-59
GPU="--partition=$GPU_PART --gres=gpu:1 --mem=$GPU_MEM --cpus-per-task=4 --time=24:00:00"
CPU="--partition=$CPU_PART --mem=8G --cpus-per-task=8 --time=02:00:00"

# ── STAGE: setup — check env + prime data caches (mnist/cifar/celeba self-download) ──
stage_setup() {
    echo "== setup =="
    [ -x "$PY" ] || { echo "!! no venv at $PY — create it: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
    "$PY" - <<'PY' || exit 1
import importlib.util as u, sys
miss=[m for m in ("torch","opacus","lpips","numpy","scipy","sklearn","pandas","pyarrow","matplotlib","nbclient") if u.find_spec(m) is None]
print("!! missing:",miss) or sys.exit(1) if miss else print("deps OK")
PY
    echo "   imagenette is the only MANUAL dataset: put imagenette2-320 under data/imagenette/"
}

# ── STAGE 1: bound computation (the ROOT) — pixel + phi per dataset ──
stage_bounds() {
    echo "== bound computation (Stream 1) =="
    for ds in $DATASETS; do
        local px="${PXRANGE[$ds]:-0-59}"
        # pixel: [celeba prep] -> array -> assemble
        local pxdep=""
        if [ "$ds" = celeba ]; then
            sub "cprep_$ds" "" $CPU --wrap "$PY scripts/run_entropy_bound.py --dataset $ds --space pixel --prep"
            pxdep="cprep_$ds"
        fi
        sub "cpx_arr_$ds" "$pxdep" $GPU --array "${px}%${MAX_CONC}" --export "ALL,DATASET=$ds,SPACE=pixel" --script reproduce/slurm_bound_pair_array.sh
        sub "cpx_$ds" "cpx_arr_$ds" $CPU --export "ALL,DATASET=$ds,SPACE=pixel" --script reproduce/slurm_bound_assemble.sh
        # phi (LPIPS): prep (mandatory) -> array 0-35 -> assemble. Not on ImageNette: its images are in ImageNet,
        # which the LPIPS backbone was trained on, so Phi is not admissible there.
        [ "$ds" = imagenette ] && continue
        sub "cphi_prep_$ds" "" $GPU --export "ALL,DATASET=$ds" --script reproduce/slurm_bound_prep_phi.sh
        sub "cphi_arr_$ds" "cphi_prep_$ds" $GPU --array "0-35%${MAX_CONC}" --export "ALL,DATASET=$ds,SPACE=phi" --script reproduce/slurm_bound_pair_array.sh
        sub "cphi_$ds" "cphi_arr_$ds" $CPU --export "ALL,DATASET=$ds,SPACE=phi" --script reproduce/slurm_bound_assemble.sh
    done
}

# ── STAGE 2: DP-SGD utility fleet (Stream 2) — tune -> nm-grid -> 132 trainings (+ 4 non-private baselines) ──
stage_dpsgd() {
    echo "== dpsgd utility fleet (Stream 2): momentum-0.9 shared-epsilon fleet =="
    # Recipe of the plotted fleet: momentum 0.9, AugMult K (1 on mnist), 100 epochs, batch 1024, C = 1.
    #   1 non-private baseline: sigma=0 LR search (the paper reports its lr; the utility notebook reads it from these
    #     runs). The fleet itself uses the locked lr in scripts/build_shared_eps_fleet.py.
    #   2 fleet params: 33 rows per dataset (eps 0.1..32 and 64/128, seeds 42-44).
    #   3 the fleet: one reproduce/slurm_shared_fleet.sh array task per TSV row.
    local lrc="0.025 0.05 0.1 0.25 0.5" base_names="" nds=0
    for ds in $DATASETS; do
        local m=${MODEL[$ds]} env="AUGMULT_K=${KAUG[$ds]},MAX_PHYSICAL_BATCH=${PHYS[$ds]}"
        sub "base_$ds" "" $GPU --export "ALL,$env" --wrap \
            "$PY scripts/run_dp_by_noise.py --model $m --dataset $ds --noise_multiplier 0 --max_grad_norm 1e9 --lr_candidates $lrc --seed 42 --epochs 100 --batch_size 1024"
        base_names+=" base_$ds"; nds=$((nds + 1))
    done
    sub "fleetparams" "$base_names" $CPU --wrap "$PY scripts/build_shared_eps_fleet.py --datasets $DATASETS"
    sub "fleet" "fleetparams" $GPU --array "0-$((33 * nds - 1))%${MAX_CONC}" --script reproduce/slurm_shared_fleet.sh
}

# ── STAGE 3: the tightness experiment (fig:tightness*, tab:tightness, tab:attack_results, exemplars) ──
# MNIST and CelebA, pixel and LPIPS, with the exact commands behind the shipped results. Needs the bounds of those
# datasets (stage bounds). The GPU steps reproduce the shipped numbers to the digit only on the GPU model they were
# made with (RTX 2080 Ti): CUDA random streams and convolution precision differ across models (README, Determinism).
stage_tightness() {
    echo "== tightness experiment =="
    local TGPU="--partition=$GPU_PART --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=06:00:00"
    local TCPU="--partition=$CPU_PART --cpus-per-task=4 --mem=32G --time=02:00:00"
    local nb_names="" ds rep bdeps nc rates
    for ds in mnist celeba; do
        bdeps="cpx_$ds cphi_$ds"
        rates="1,2,3,4,5,6,7,8,9,10,11"; [ "$ds" = celeba ] && rates="1,2,3,4,5,6,7,8,9,10"
        for rep in pixel lpips; do
            nc=""; [ "$rep" = lpips ] && nc="--n-coords 4000"
            sub "codebook_${ds}_${rep}" "$bdeps" $TGPU --wrap \
                "$PY scripts/tightness_codebook.py --dataset $ds --rep $rep --targets-decoder --min-nu2-only --out-suffix bal --rates $rates $nc"
            sub "subbit_${ds}_${rep}" "$bdeps" $TGPU --wrap \
                "$PY scripts/tightness_subbit.py --dataset $ds --rep $rep $nc"
            sub "spm_${ds}_${rep}" "" $TGPU --wrap \
                "$PY scripts/tightness_spm.py --dataset $ds --rep $rep --gammas 0.2 0.4 0.72 2.89 8.01 72.13 --outdir results/tightness_spm"
            sub "exemplars_${ds}_${rep}" "" $TGPU --wrap \
                "$PY scripts/tightness_spm.py --dataset $ds --rep $rep --gammas 0.2 0.72 2.89 8.01 30 72.13 500 --exemplar-mode 6 --outdir results/tightness_spm/exemplars"
            nb_names+=" codebook_${ds}_${rep} subbit_${ds}_${rep} spm_${ds}_${rep} exemplars_${ds}_${rep}"
        done
        sub "spm_img_$ds" "spm_${ds}_lpips" $TGPU --wrap \
            "$PY scripts/tightness_spm.py --dataset $ds --rep lpips --single-img-only --rho-tol 5e-3"
        sub "realizable_$ds" "" $TGPU --wrap "$PY scripts/compute_realizable_prior.py --dataset $ds"
        nb_names+=" spm_img_$ds realizable_$ds"
    done
    local tr_deps=""; for ds in $DATASETS; do tr_deps+=" cpx_$ds cphi_$ds"; done
    sub "trace_ci" "$tr_deps" $TCPU --wrap "$PY scripts/sigma_trace_ci.py --datasets $DATASETS"
    TIGHTNESS_DONE="trace_ci$nb_names"
}

# ── STAGE 4: paper notebooks ──
stage_notebooks() {
    echo "== notebooks =="
    # every notebook reads the bounds; tightness and bound details also the tightness stage; utility also the fleet
    local adep="${TIGHTNESS_DONE:-}" udep="fleet"
    for ds in $DATASETS; do adep+=" cpx_$ds cphi_$ds"; udep+=" cpx_$ds cphi_$ds base_$ds"; done
    sub "nb_tightness" "$adep" $CPU --export "ALL,NB=notebooks/paper_tightness.ipynb"      --script reproduce/slurm_exec_notebook.sh
    sub "nb_source"    "$adep" $CPU --export "ALL,NB=notebooks/paper_bound_details.ipynb" --script reproduce/slurm_exec_notebook.sh
    sub "nb_utility" "$udep"  $CPU --export "ALL,NB=notebooks/paper_utility_curves.ipynb" --script reproduce/slurm_exec_notebook.sh
}

main() {
    local stages="${1:-all}"
    if [ "$stages" = all ]; then
        stage_setup; stage_bounds; stage_tightness; stage_dpsgd; stage_notebooks
    else
        for s in "$@"; do "stage_$s"; done
    fi
    wait_all
    echo "== done ($LAUNCHER). Outputs: results/ (every stage), figures/*.pdf and tables/*.tex (the notebooks) =="
}
main "$@"
