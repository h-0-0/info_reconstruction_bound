#!/usr/bin/env bash
# reproduce/lib.sh — launcher abstraction so the SAME recipe runs on SLURM or a
# single machine. Everything downstream calls sub()/pdep(); nothing calls sbatch
# directly. Switch backends with LAUNCHER=slurm|local (see config.sh).
#
#   LAUNCHER=slurm : sub() submits an sbatch job, records its id under a logical
#                    NAME, and wires --dependency=afterok from named deps. Jobs
#                    run in parallel, ordered only by their dependency edges.
#   LAUNCHER=local : sub() runs the command NOW, synchronously, in dependency
#                    order (deps are already satisfied because we ran them). GPU
#                    arrays become a sequential for-loop. No SLURM needed.
set -uo pipefail
declare -A _JID          # logical name -> slurm job id (slurm mode only)

_deps() {                # echo "--dependency=afterok:ID:ID" for logical dep names
    local ids=() n
    for n in "$@"; do [ -n "${_JID[$n]:-}" ] && ids+=("${_JID[$n]}"); done
    ((${#ids[@]})) && printf -- "--dependency=afterany:%s" "$(IFS=:; echo "${ids[*]}")"
}

# sub NAME "DEP DEP" [extra sbatch opts...] (--script FILE | --wrap "CMD") \
#         [--array LO-HI[%N]] [--export ALL,K=V,...]
sub() {
    local name="$1" deps="$2"; shift 2
    local script="" wrap="" array="" export_kv="ALL" extra=()
    while (($#)); do
        case "$1" in
            --script) script="$2";    shift 2;;
            --wrap)   wrap="$2";       shift 2;;
            --array)  array="$2";      shift 2;;
            --export) export_kv="$2";  shift 2;;
            *)        extra+=("$1");   shift;;
        esac
    done
    if [ "${LAUNCHER:-slurm}" = local ]; then
        _run_local "$name" "$script" "$wrap" "$array" "$export_kv"; return
    fi
    local d a="" id
    d=$(_deps $deps)
    [ -n "$array" ] && a="--array=$array"
    if [ -n "$script" ]; then
        id=$(sbatch --parsable ${SBATCH_EXTRA:-} $d $a --job-name="$name" --export="$export_kv" "${extra[@]}" "$script")
    else
        id=$(sbatch --parsable ${SBATCH_EXTRA:-} $d $a --job-name="$name" --export="$export_kv" "${extra[@]}" --wrap="$wrap")
    fi
    _JID[$name]=$id
    printf '  %-26s %s  deps=[%s]\n' "$name" "$id" "$deps"
}

_run_local() {           # sequential fallback: apply K=V, loop the array range, run
    local name="$1" script="$2" wrap="$3" array="$4" export_kv="$5"
    local kv="${export_kv#ALL}"; kv="${kv#,}"
    local old_ifs="$IFS"; IFS=','
    local pair; for pair in $kv; do [ -n "$pair" ] && export "${pair?}"; done
    IFS="$old_ifs"
    local lo=0 hi=0
    if [ -n "$array" ]; then lo=${array%%-*}; hi=${array#*-}; hi=${hi%%%*}; fi
    local i
    for i in $(seq "$lo" "$hi"); do
        export SLURM_ARRAY_TASK_ID=$i
        printf '  >> %-24s local task %s\n' "$name" "$i"
        if [ -n "$script" ]; then bash "$script" || return 1
        else bash -c "$wrap" || return 1; fi
    done
}

wait_all() {             # SLURM: block until every submitted job leaves the queue
    [ "${LAUNCHER:-slurm}" = local ] && return 0
    local ids="${_JID[*]:-}"; [ -z "$ids" ] && return 0
    echo ">> waiting on: ${ids// /,}"
    while squeue -j "${ids// /,}" -h 2>/dev/null | grep -q .; do sleep 30; done
    echo ">> all jobs left the queue"
}
