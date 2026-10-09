#!/bin/bash
# Offline distribution KL evaluation for a trained A²OPD run.
# Usage: exp_path=/path/to/run bash scripts/a2opd_distribution_kl_eval.sh
# Batch: eval_jobs=/path/to/jobs.csv bash scripts/a2opd_distribution_kl_eval.sh
# CSV columns: exp_path,student_base,teacher_expert,teacher_base[,ckpt_slot]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VERL_ROOT="${REPO_ROOT}/verl"
# shellcheck source=../../../scripts/opd_paths.sh
source "${REPO_ROOT}/scripts/opd_paths.sh"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export RAY_DEDUP_LOGS=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export TZ='UTC-8'
export RAY_memory_usage_threshold=0.85

cst_date() { TZ='UTC-8' date "$@"; }

prompt_files="${prompt_files:-${OPD_PUBLIC_DATA_ROOT}/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6-sample500_seed42.parquet}"
distribution_kl_max_samples="${distribution_kl_max_samples:--1}"
distribution_kl_batch_size="${distribution_kl_batch_size:-128}"
distribution_kl_do_sample="${distribution_kl_do_sample:-true}"
distribution_kl_temperature="${distribution_kl_temperature:-1.0}"
distribution_kl_top_p="${distribution_kl_top_p:-1.0}"
ckpt_slot="${ckpt_slot:-last}"
opd_top_k="${opd_top_k:-16}"
distribution_kl_skip_existing="${distribution_kl_skip_existing:-true}"
opd_parallel_student_base="${opd_parallel_student_base:-false}"
actor_param_offload="${actor_param_offload:-true}"
n_node="${n_node:-1}"
n_gpu="${n_gpu:-8}"
nccl_timeout="${nccl_timeout:-7200}"
log_dir="${REPO_ROOT}/logs/a2_opd/distribution_kl"
today_sec=$(cst_date +%Y%m%d_%H%M%S)
summary_csv="${summary_csv:-${OPD_DISTRIBUTION_KL_RESULT_ROOT}/distribution_kl_summary_${today_sec}.csv}"

mkdir -p "${log_dir}" "${OPD_DISTRIBUTION_KL_RESULT_ROOT}" "$(dirname "${summary_csv}")"

distribution_kl_batch_mode=false

log_info() {
    echo "[$(cst_date '+%Y-%m-%d %H:%M:%S')] [distribution_kl] $*"
}

log_job() {
    local log_file="$1"
    shift
    echo "[$(cst_date '+%Y-%m-%d %H:%M:%S')] [distribution_kl] $*" >> "${log_file}"
}

init_distribution_kl_batch_log() {
    local jobs_file="$1"
    local batch_log_file="${distribution_kl_batch_log:-${log_dir}/distribution_kl_batch_${today_sec}.log}"

    : > "${batch_log_file}"
    ln -sfn "${batch_log_file}" "${log_dir}/distribution_kl_batch_latest.log" 2>/dev/null || true
    exec > >(tee -a "${batch_log_file}") 2>&1

    log_info "batch_log_file=${batch_log_file}"
    log_info "batch workspace_latest=${log_dir}/distribution_kl_batch_latest.log"
    log_info "batch mode: eval_jobs=${jobs_file} skip_existing=${distribution_kl_skip_existing}"
}

has_hf_weights_dir() {
    local dir="$1"
    compgen -G "${dir}"/*.safetensors >/dev/null 2>&1 \
        || compgen -G "${dir}"/*.bin >/dev/null 2>&1 \
        || [[ -f "${dir}/model.safetensors.index.json" ]]
}

resolve_student_ckpt_from_exp_path() {
    local exp_path="$1"
    local slot="${2:-${ckpt_slot}}"
    local resolved

    exp_path="$(readlink -f "${exp_path}")"
    case "${slot}" in
        last|best_valid) ;;
        *)
            echo "ERROR: ckpt_slot must be 'last' or 'best_valid', got: ${slot}" >&2
            return 1
            ;;
    esac

    resolved="${exp_path}/${slot}"
    if [[ -d "${resolved}/actor" ]] && [[ -f "${resolved}/actor/fsdp_config.json" ]]; then
        DISTRIBUTION_KL_STUDENT_CKPT_KIND=fsdp
        echo "${resolved}"
        return 0
    fi
    if has_hf_weights_dir "${resolved}/merged_hf"; then
        DISTRIBUTION_KL_STUDENT_CKPT_KIND=merged_hf
        echo "${resolved}"
        return 0
    fi

    echo "ERROR: checkpoint not found at ${resolved}/actor or ${resolved}/merged_hf" >&2
    echo "       exp_path=${exp_path}, ckpt_slot=${slot}" >&2
    return 1
}

resolve_distribution_kl_result_dir() {
    local exp_path="$1"
    local job_ckpt_slot="${2:-${ckpt_slot}}"
    local exp_label
    if [[ "${job_ckpt_slot}" == "base" ]]; then
        exp_label="${exp_path}"
        echo "${OPD_DISTRIBUTION_KL_RESULT_ROOT}/fixed_rollout/${exp_label}/base"
        return 0
    fi
    exp_label="$(basename "${exp_path}")"
    echo "${OPD_DISTRIBUTION_KL_RESULT_ROOT}/${exp_label}/${job_ckpt_slot}"
}

resolve_distribution_kl_job_paths() {
    local exp_path="$1"
    local job_ckpt_slot="${2:-${ckpt_slot}}"
    local exp_label

    if [[ "${job_ckpt_slot}" == "base" ]]; then
        exp_label="${exp_path}"
    else
        exp_label="$(basename "${exp_path}")"
    fi
    DISTRIBUTION_KL_RESULT_DIR="$(resolve_distribution_kl_result_dir "${exp_path}" "${job_ckpt_slot}")"
    DISTRIBUTION_KL_LOG_LATEST="${log_dir}/distribution_kl_${exp_label}_${job_ckpt_slot}_latest.log"
    export DISTRIBUTION_KL_RESULT_DIR DISTRIBUTION_KL_LOG_LATEST
}

is_valid_distribution_kl_metrics_json() {
    local json_path="$1"
    python3 - "${json_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)
metrics = payload.get("metrics")
if not isinstance(metrics, dict):
    sys.exit(1)
required = (
    "a2opd/proxy_kl",
    "a2opd/proxy_base_target_base_kl",
    "a2opd/target_transfer_kl",
    "a2opd/anchor_kl",
    "a2opd/transfer_gap",
)
for key in required:
    if key not in metrics:
        sys.exit(1)
sys.exit(0)
PY
}

append_distribution_kl_summary_row() {
    local output_json="$1"
    local exp_path="$2"
    local job_ckpt_slot="$3"
    local student_ckpt="$4"

    python3 - "${output_json}" "${exp_path}" "${job_ckpt_slot}" "${student_ckpt}" "${summary_csv}" <<'PY'
import json
import sys
from pathlib import Path

json_path, exp_path, ckpt_slot, student_ckpt, csv_path = sys.argv[1:6]
metrics = json.loads(Path(json_path).read_text(encoding="utf-8"))["metrics"]

def fmt(key):
    value = metrics.get(key)
    return "" if value is None else f"{value:.6f}"

row = [
    exp_path,
    ckpt_slot,
    student_ckpt,
    fmt("a2opd/proxy_kl"),
    fmt("a2opd/proxy_base_target_base_kl"),
    fmt("a2opd/target_transfer_kl"),
    fmt("a2opd/anchor_kl"),
    fmt("a2opd/transfer_gap"),
    fmt("a2opd/topk_overlap_target_vs_teacher_ratio"),
    fmt("a2opd/topk_overlap_target_vs_teacher_base_ratio"),
    fmt("a2opd/topk_overlap_teacher_vs_teacher_base_ratio"),
    json_path,
]
header = "exp_path,ckpt_slot,student_ckpt,proxy_kl,proxy_base_target_base_kl,target_transfer_kl,anchor_kl,transfer_gap,topk_overlap_target_vs_teacher,topk_overlap_target_vs_teacher_base,topk_overlap_teacher_vs_teacher_base,metrics_json"
path = Path(csv_path)
if not path.exists() or path.stat().st_size == 0:
    path.write_text(header + "\n", encoding="utf-8")
with path.open("a", encoding="utf-8") as handle:
    handle.write(",".join(row) + "\n")
print(f"appended summary row for {exp_path} ({ckpt_slot})")
PY
}

run_one_distribution_kl_eval() {
    local exp_path="$1"
    local student_base_path="$2"
    local teacher_expert_path="$3"
    local teacher_base_path="$4"
    local job_ckpt_slot="${5:-${ckpt_slot}}"

    local student_ckpt="" student_ckpt_kind="" use_base_target=false
    local rollout_cache_path="" rollout_cache_save=false
    if [[ "${job_ckpt_slot}" == "base" ]]; then
        use_base_target=true
        if [[ -n "${distribution_kl_rollout_cache:-}" ]]; then
            rollout_cache_path="${distribution_kl_rollout_cache}"
            if [[ -f "${rollout_cache_path}" ]]; then
                rollout_cache_save=false
            else
                rollout_cache_save=true
            fi
        fi
    else
        student_ckpt="$(resolve_student_ckpt_from_exp_path "${exp_path}" "${job_ckpt_slot}")"
        student_ckpt_kind="${DISTRIBUTION_KL_STUDENT_CKPT_KIND:-fsdp}"
    fi

    local exp_label
    if [[ "${job_ckpt_slot}" == "base" ]]; then
        exp_label="${exp_path}"
    else
        exp_label="$(basename "${exp_path}")"
    fi
    local result_dir
    result_dir="$(resolve_distribution_kl_result_dir "${exp_path}" "${job_ckpt_slot}")"
    mkdir -p "${result_dir}"

    local output_json="${result_dir}/metrics.json"
    local rollout_parquet="${result_dir}/rollouts.parquet"
    local job_sec
    job_sec=$(cst_date +%Y%m%d_%H%M%S)
    local log_file="${result_dir}/eval_${job_sec}.log"

    # skip_existing only recognizes the canonical result root above (not exp_path/distribution_kl/).
    if [[ "${distribution_kl_skip_existing}" == "true" ]] && [[ -f "${output_json}" ]] && is_valid_distribution_kl_metrics_json "${output_json}"; then
        append_distribution_kl_summary_row "${output_json}" "${exp_path}" "${job_ckpt_slot}" "${student_ckpt}" >/dev/null
        DISTRIBUTION_KL_LAST_JOB_SKIPPED=1
        if [[ "${distribution_kl_batch_mode}" == "true" ]]; then
            log_info "[job log] skipped metrics=${output_json}"
        fi
        return 0
    fi
    DISTRIBUTION_KL_LAST_JOB_SKIPPED=0

    local ray_tmpdir
    ray_tmpdir="$(mktemp -d /tmp/r_kl_XXXXXX)"
    mkdir -p "${ray_tmpdir}"

    : > "${log_file}"
    ln -sfn "${log_file}" "${log_dir}/distribution_kl_${exp_label}_${job_ckpt_slot}_${job_sec}.log" 2>/dev/null || true
    ln -sfn "${log_file}" "${log_dir}/distribution_kl_${exp_label}_${job_ckpt_slot}_latest.log" 2>/dev/null || true

    if [[ "${distribution_kl_batch_mode}" == "true" ]]; then
        log_info "[job log] running eval_log=${log_file} log_latest=${log_dir}/distribution_kl_${exp_label}_${job_ckpt_slot}_latest.log"
    fi

    local -a job_hydra_extra=()
    if [[ -n "${student_ckpt}" ]]; then
        job_hydra_extra+=(+trainer.distribution_kl_student_ckpt="${student_ckpt}")
        job_hydra_extra+=(+trainer.distribution_kl_student_ckpt_kind="${student_ckpt_kind}")
    fi
    if [[ -n "${rollout_cache_path}" ]]; then
        job_hydra_extra+=(+trainer.distribution_kl_rollout_cache_path="${rollout_cache_path}")
        job_hydra_extra+=(+trainer.distribution_kl_rollout_cache_save="${rollout_cache_save}")
    fi

    run_distribution_kl_job_body() {
        set -x
        log_job "${log_file}" "========== exp=${exp_label} ckpt_slot=${job_ckpt_slot} =========="
        log_job "${log_file}" "exp_path=${exp_path}"
        log_job "${log_file}" "use_base_target=${use_base_target}"
        log_job "${log_file}" "student_ckpt=${student_ckpt:-<base-model>}"
        log_job "${log_file}" "student_ckpt_kind=${student_ckpt_kind:-n/a}"
        log_job "${log_file}" "rollout_cache_path=${rollout_cache_path:-<none>}"
        log_job "${log_file}" "rollout_cache_save=${rollout_cache_save}"
        log_job "${log_file}" "student_base=${student_base_path}"
        log_job "${log_file}" "teacher_expert=${teacher_expert_path}"
        log_job "${log_file}" "teacher_base=${teacher_base_path}"
        log_job "${log_file}" "prompt_files=${prompt_files}"
        log_job "${log_file}" "result_dir=${result_dir}"
        log_job "${log_file}" "ray_tmpdir=${ray_tmpdir}"
        log_job "${log_file}" "log_file=${log_file}"

        (
            export RAY_TMPDIR="${ray_tmpdir}"
            cd "${VERL_ROOT}"
            export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

            unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
            unset RAY_ADDRESS RAY_NAMESPACE RAY_DASHBOARD_ADDRESS
            unset RAY_RUNTIME_ENV RAY_JOB_ID RAY_HEAD_IP RAY_PORT

            python3 -m verl.trainer.main_a2opd_distribution_kl_eval \
                +algorithm.train_mode=heterogeneous_distill \
                +algorithm.hetero_distill.use_opd=true \
                +algorithm.hetero_distill.opd_top_k="${opd_top_k}" \
                +algorithm.hetero_distill.opd_parallel_student_base="${opd_parallel_student_base}" \
                algorithm.adv_estimator=grpo \
                algorithm.use_kl_in_reward=false \
                algorithm.a2opd.enable=true \
                algorithm.a2opd.budget_adaptive_enable=false \
                algorithm.a2opd.lambda_dual_enable=false \
                +data.prompt_files="${prompt_files}" \
                data.val_files="${prompt_files}" \
                data.train_files="${prompt_files}" \
                +data.distribution_kl_max_samples="${distribution_kl_max_samples}" \
                +data.distribution_kl_batch_size="${distribution_kl_batch_size}" \
                data.train_batch_size="${distribution_kl_batch_size}" \
                data.max_prompt_length=1024 \
                data.max_response_length=8192 \
                data.filter_overlong_prompts=True \
                data.truncation=error \
                data.shuffle=false \
                data.return_raw_chat=true \
                +data.apply_chat_template_kwargs.enable_thinking=false \
                actor_rollout_ref.model.path="${student_base_path}" \
                +actor_rollout_ref.model.base_model_path="${student_base_path}" \
                +actor_rollout_ref.ref.model.path="${teacher_expert_path}" \
                +actor_rollout_ref.ref.model.base_model_path="${teacher_base_path}" \
                actor_rollout_ref.model.use_remove_padding=true \
                actor_rollout_ref.actor.ppo_mini_batch_size="${distribution_kl_batch_size}" \
                actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
                actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
                actor_rollout_ref.actor.fsdp_config.param_offload="${actor_param_offload}" \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
                actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
                actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
                actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
                actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.enforce_eager=true \
                actor_rollout_ref.rollout.free_cache_engine=true \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
                actor_rollout_ref.rollout.calculate_log_probs=false \
                actor_rollout_ref.rollout.val_kwargs.do_sample="${distribution_kl_do_sample}" \
                actor_rollout_ref.rollout.val_kwargs.temperature="${distribution_kl_temperature}" \
                actor_rollout_ref.rollout.val_kwargs.top_p="${distribution_kl_top_p}" \
                actor_rollout_ref.rollout.val_kwargs.n=1 \
                actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
                actor_rollout_ref.ref.fsdp_config.param_offload=true \
                actor_rollout_ref.nccl_timeout="${nccl_timeout}" \
                teacher_rollout.nccl_timeout="${nccl_timeout}" \
                critic.enable=false \
                reward_model.reward_manager=naive \
                +trainer.distribution_kl_eval_only=true \
                +trainer.distribution_kl_do_sample="${distribution_kl_do_sample}" \
                +trainer.distribution_kl_use_base_target="${use_base_target}" \
                +trainer.distribution_kl_output_path="${output_json}" \
                +trainer.distribution_kl_rollout_output_path="${rollout_parquet}" \
                +trainer.distribution_kl_exp_path="${exp_path}" \
                +trainer.distribution_kl_ckpt_slot="${job_ckpt_slot}" \
                "${job_hydra_extra[@]}" \
                trainer.logger='["console"]' \
                trainer.n_gpus_per_node="${n_gpu}" \
                trainer.nnodes="${n_node}" \
                trainer.val_before_train=false \
                trainer.total_epochs=1 \
                "${extra_hydra_args[@]}"
        )
        local eval_status=$?
        set +x
        if [[ "${eval_status}" -ne 0 ]]; then
            log_job "${log_file}" "failed (exit=${eval_status}) -> ${result_dir}"
            return "${eval_status}"
        fi

        append_distribution_kl_summary_row "${output_json}" "${exp_path}" "${job_ckpt_slot}" "${student_ckpt}"
        log_job "${log_file}" "done -> ${result_dir}"
    }

    if [[ "${distribution_kl_batch_mode}" == "true" ]]; then
        run_distribution_kl_job_body >> "${log_file}" 2>&1
    else
        run_distribution_kl_job_body 2>&1 | tee -a "${log_file}"
    fi
}

parse_jobs_line() {
    local line="$1"
    local exp_path student_base teacher_expert teacher_base job_ckpt_slot

    IFS=$'\t' read -r exp_path student_base teacher_expert teacher_base job_ckpt_slot <<< "${line}"
    if [[ -z "${teacher_expert:-}" ]]; then
        IFS=',' read -r exp_path student_base teacher_expert teacher_base job_ckpt_slot <<< "${line}"
    fi
    [[ -z "${job_ckpt_slot:-}" ]] && job_ckpt_slot="${ckpt_slot}"
    printf '%s\n' "${exp_path}" "${student_base}" "${teacher_expert}" "${teacher_base}" "${job_ckpt_slot}"
}

count_eval_jobs() {
    local jobs_file="$1"
    local count=0
    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ -z "${line}" ]] && continue
        [[ "${line}" =~ ^# ]] && continue
        [[ "${line}" == exp_path,* ]] && continue
        count=$((count + 1))
    done < "${jobs_file}"
    echo "${count}"
}

extra_hydra_args=("$@")

if [[ -n "${eval_jobs:-}" ]]; then
    if [[ ! -f "${eval_jobs}" ]]; then
        echo "ERROR: eval_jobs file not found: ${eval_jobs}" >&2
        exit 1
    fi
    if [[ "${eval_jobs}" == *ref_overlap* ]]; then
        distribution_kl_rollout_cache="${OPD_DISTRIBUTION_KL_RESULT_ROOT}/fixed_rollout/Qwen3-8B_base/rollout_cache.pkl"
    fi
    distribution_kl_batch_mode=true
    init_distribution_kl_batch_log "${eval_jobs}"
    if [[ -n "${distribution_kl_rollout_cache:-}" ]] && [[ "${eval_jobs}" == *ref_overlap* ]]; then
        log_info "rollout cache for ref_overlap jobs: ${distribution_kl_rollout_cache}"
    fi
    total_jobs="$(count_eval_jobs "${eval_jobs}")"
    log_info "batch total jobs=${total_jobs}"
    job_count=0
    skipped_count=0
    failed_count=0
    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ -z "${line}" ]] && continue
        [[ "${line}" =~ ^# ]] && continue
        [[ "${line}" == exp_path,* ]] && continue
        job_count=$((job_count + 1))
        mapfile -t fields < <(parse_jobs_line "${line}")
        resolve_distribution_kl_job_paths "${fields[0]}" "${fields[4]}"
        log_info "[job ${job_count}/${total_jobs}] start exp=$(basename "${fields[0]}") ckpt_slot=${fields[4]} result_dir=${DISTRIBUTION_KL_RESULT_DIR} log_latest=${DISTRIBUTION_KL_LOG_LATEST}"
        if run_one_distribution_kl_eval "${fields[0]}" "${fields[1]}" "${fields[2]}" "${fields[3]}" "${fields[4]}"; then
            if [[ "${DISTRIBUTION_KL_LAST_JOB_SKIPPED:-0}" -eq 1 ]]; then
                log_info "[job ${job_count}/${total_jobs}] skipped (existing metrics)"
                skipped_count=$((skipped_count + 1))
            else
                log_info "[job ${job_count}/${total_jobs}] done"
            fi
        else
            failed_count=$((failed_count + 1))
            log_info "[job ${job_count}/${total_jobs}] failed"
        fi
    done < "${eval_jobs}"
    if [[ "${job_count}" -eq 0 ]]; then
        echo "ERROR: no jobs parsed from ${eval_jobs}" >&2
        exit 1
    fi
    log_info "batch complete: ${job_count} jobs (${skipped_count} skipped, ${failed_count} failed), summary_csv=${summary_csv}"
    if [[ "${failed_count}" -gt 0 ]]; then
        exit 1
    fi
else
    if [[ -z "${exp_path:-}" ]]; then
        echo "ERROR: set exp_path to the training run directory (trainer.default_local_dir)." >&2
        echo "       Example: exp_path=${OPD_A2OPD_MODEL_ROOT}/A2OPD@16-math/S_..._A2OPD_20260816_1954" >&2
        exit 1
    fi

    student_model_name="${student_model_name:-Qwen3-8B}"
    teacher_model_name="${teacher_model_name:-Qwen3-4B}"
    teacher_base_model_name="${teacher_base_model_name:-Qwen3-1.7B}"

    student_base_path="${student_base_path:-${OPD_PUBLIC_MODEL_ROOT}/${student_model_name}}"
    teacher_expert_path="${teacher_expert_path:-${OPD_PUBLIC_MODEL_ROOT}/${teacher_model_name}}"
    teacher_base_path="${teacher_base_path:-${OPD_PUBLIC_MODEL_ROOT}/${teacher_base_model_name}}"

    run_one_distribution_kl_eval \
        "${exp_path}" \
        "${student_base_path}" \
        "${teacher_expert_path}" \
        "${teacher_base_path}" \
        "${ckpt_slot}"
    log_info "single job complete, summary_csv=${summary_csv}"
fi
