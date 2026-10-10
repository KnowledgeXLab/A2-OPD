#!/bin/bash
# Convert verl/FSDP actor checkpoints to HuggingFace format.
#
# Usage:
#   bash scripts/merge_checkpoints.sh
#   CKPT_TAG=best_valid ONLY=S_Qwen3-4B bash scripts/merge_checkpoints.sh
#   CKPT_ROOT=/path/to/A2OPD@16-math bash scripts/merge_checkpoints.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export TZ='UTC-8'
cst_date() { TZ='UTC-8' date "$@"; }

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VERL_ROOT="${REPO_ROOT}/verl"
# shellcheck source=opd_paths.sh
source "${REPO_ROOT}/scripts/opd_paths.sh"

export CKPT_ROOT="${CKPT_ROOT:-${A2OPD_CKPT_ROOT}/A2OPD@16-math}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_PREFIX:-${HOME}/miniconda3/envs/a2opd}}"

activate_conda_env() {
    if [[ ! -f "${CONDA_SH}" ]]; then
        echo "ERROR: conda not found at ${CONDA_SH}" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV_PATH}"
}

if ! python3 -c "import ray" >/dev/null 2>&1; then
    activate_conda_env
fi
PYTHON="${PYTHON:-python3}"
if ! "${PYTHON}" -c "import ray" >/dev/null 2>&1; then
    echo "ERROR: ray is not available in ${PYTHON} ($(command -v "${PYTHON}"))" >&2
    exit 1
fi

CKPT_TAG="${CKPT_TAG:-all}"          # all | best_valid | last
ONLY="${ONLY:-}"                     # optional substring filter on experiment name
FORCE="${FORCE:-0}"                  # 1 = re-merge even if merged_hf exists
DELETE_FSDP="${DELETE_FSDP:-1}"      # 1 = delete actor/ FSDP shards after successful merge
USE_CPU_INIT="${USE_CPU_INIT:-1}"    # 1 = pass --use_cpu_initialization to merger

if [[ -z "${MERGE_EXPERIMENTS:-}" ]]; then
    if [[ ! -d "${CKPT_ROOT}" ]]; then
        echo "ERROR: CKPT_ROOT not found: ${CKPT_ROOT}" >&2
        exit 1
    fi

    shopt -s nullglob
    experiments=()
    for exp_dir in "${CKPT_ROOT}"/*/; do
        experiments+=("$(basename "${exp_dir}")")
    done

    if [[ ${#experiments[@]} -eq 0 ]]; then
        echo "ERROR: no experiment directories found under ${CKPT_ROOT}" >&2
        exit 1
    fi

    mapfile -t experiments < <(printf '%s\n' "${experiments[@]}" | sort)
    MERGE_EXPERIMENTS="$(printf '%s\n' "${experiments[@]}")"
fi

run_name="${run_name:-merge_ckpt_$(cst_date +%Y%m%d_%H%M%S)}"
log_dir="${REPO_ROOT}/logs/merge-ckpt"
mkdir -p "${log_dir}"
LOG_FILE="${log_dir}/${run_name}.log"

log_info() {
    echo "[$(cst_date '+%Y-%m-%d %H:%M:%S')] [${run_name}] $*"
}

_on_exit() {
    local ec=$?
    log_info "========== run finished (exit_code=${ec}) =========="
    log_info "log file: ${LOG_FILE}"
    exit "${ec}"
}
trap _on_exit EXIT

exec > >(tee "${LOG_FILE}") 2>&1

cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

log() {
    log_info "$@"
}

has_hf_weights() {
    local dir="$1"
    compgen -G "${dir}"/*.safetensors >/dev/null 2>&1 \
        || compgen -G "${dir}"/*.bin >/dev/null 2>&1 \
        || [[ -f "${dir}/model.safetensors.index.json" ]]
}

delete_fsdp_actor() {
    local exp_name="$1"
    local ckpt_tag="$2"
    local fsdp_dir="$3"
    local target_dir="$4"

    if [[ "${DELETE_FSDP}" != "1" ]]; then
        return 0
    fi
    if ! has_hf_weights "${target_dir}"; then
        log "SKIP delete ${exp_name}/${ckpt_tag}: merged_hf not ready at ${target_dir}"
        return 0
    fi
    if [[ ! -d "${fsdp_dir}" ]]; then
        return 0
    fi

    local freed_size
    freed_size="$(du -sh "${fsdp_dir}" | awk '{print $1}')"
    log "DELETE ${exp_name}/${ckpt_tag}: removing FSDP actor dir ${fsdp_dir} (freed ~${freed_size})"
    rm -rf "${fsdp_dir}"
}

merge_one() {
    local exp_name="$1"
    local ckpt_tag="$2"
    local exp_dir="${CKPT_ROOT}/${exp_name}"
    local fsdp_dir="${exp_dir}/${ckpt_tag}/actor"
    local target_dir="${exp_dir}/${ckpt_tag}/merged_hf"

    if [[ ! -d "${fsdp_dir}" ]]; then
        log "SKIP ${exp_name}/${ckpt_tag}: missing ${fsdp_dir}"
        return 0
    fi
    if [[ ! -f "${fsdp_dir}/fsdp_config.json" ]]; then
        log "SKIP ${exp_name}/${ckpt_tag}: not an FSDP actor checkpoint (${fsdp_dir})"
        return 0
    fi

    if [[ "${FORCE}" != "1" ]] && has_hf_weights "${target_dir}"; then
        log "SKIP ${exp_name}/${ckpt_tag}: merged_hf already exists at ${target_dir}"
        delete_fsdp_actor "${exp_name}" "${ckpt_tag}" "${fsdp_dir}" "${target_dir}"
        return 0
    fi

    log "MERGE ${exp_name}/${ckpt_tag}"
    log "  source: ${fsdp_dir}"
    log "  target: ${target_dir}"

    local cpu_args=()
    if [[ "${USE_CPU_INIT}" == "1" ]]; then
        cpu_args=(--use_cpu_initialization)
    fi

    mkdir -p "${target_dir}"
    "${PYTHON}" -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${fsdp_dir}" \
        --target_dir "${target_dir}" \
        "${cpu_args[@]}"

    if has_hf_weights "${target_dir}"; then
        local fsdp_size merged_size
        fsdp_size="$(du -sh "${fsdp_dir}" | awk '{print $1}')"
        merged_size="$(du -sh "${target_dir}" | awk '{print $1}')"
        log "DONE ${exp_name}/${ckpt_tag}: fsdp=${fsdp_size} -> merged_hf=${merged_size}"
        delete_fsdp_actor "${exp_name}" "${ckpt_tag}" "${fsdp_dir}" "${target_dir}"
    else
        log "ERROR ${exp_name}/${ckpt_tag}: merge finished but no weight files found in ${target_dir}"
        return 1
    fi
}

mapfile -t EXPERIMENTS <<< "${MERGE_EXPERIMENTS}"

CKPT_TAGS=()
case "${CKPT_TAG}" in
    all)
        CKPT_TAGS=(best_valid last)
        ;;
    best_valid|last)
        CKPT_TAGS=("${CKPT_TAG}")
        ;;
    *)
        log "ERROR: CKPT_TAG must be one of: all, best_valid, last (got: ${CKPT_TAG})"
        exit 1
        ;;
esac

log "========== merge checkpoints start =========="
log "python=$(${PYTHON} --version 2>&1) path=$(command -v "${PYTHON}")"
log "CKPT_ROOT=${CKPT_ROOT} CKPT_TAG=${CKPT_TAG} ONLY=${ONLY:-<all>} FORCE=${FORCE} DELETE_FSDP=${DELETE_FSDP}"
log "host=$(hostname) user=${USER:-unknown} pid=$$"

failed=0
for exp_name in "${EXPERIMENTS[@]}"; do
    exp_name="$(echo "${exp_name}" | xargs)"
    [[ -z "${exp_name}" || "${exp_name}" == \#* ]] && continue
    if [[ -n "${ONLY}" && "${exp_name}" != *"${ONLY}"* ]]; then
        continue
    fi
    for ckpt_tag in "${CKPT_TAGS[@]}"; do
        if ! merge_one "${exp_name}" "${ckpt_tag}"; then
            failed=$((failed + 1))
        fi
    done
done

if [[ "${failed}" -gt 0 ]]; then
    log "Finished with ${failed} failure(s)."
    exit 1
fi

log "All merges completed."
