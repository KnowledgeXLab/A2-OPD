#!/bin/bash

set -x
set -o pipefail
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export RAY_DEDUP_LOGS=1
export RAY_BACKEND_LOG_LEVEL="${RAY_BACKEND_LOG_LEVEL:-info}"
export CUDA_LAUNCH_BLOCKING=1
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
# vLLM CuMemAllocator (used by rollout sleep/wake when free_cache_engine=true)
# asserts expandable_segments is disabled.
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export TZ='UTC-8'
cst_date() { TZ='UTC-8' date "$@"; }

export RAY_memory_usage_threshold=0.85

# student 8B, teacher 4B (proxy expert + base)
student_model_name="${student_model_name:-Qwen3-8B}"
teacher_model_name="${teacher_model_name:-Qwen3-4B-Non-Thinking-RL-Math-Step500}" # proxy expert
teacher_base_model_name="${teacher_base_model_name:-Qwen3-4B}"           # proxy base
student_tag="${student_tag:-Qwen3-8B}"
teacher_tag="${teacher_tag:-4B_Math_RL_Step500}"
ability="${ability:-Math}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${REPO_ROOT}/verl"
source "${REPO_ROOT}/scripts/opd_paths.sh"

# Auto-start math verify HTTP server (Ray workers call it remotely; do not run math_verify in-process).
source "${REPO_ROOT}/math_eval_server/start_math_verify_server.sh"
start_math_verify_server "${REPO_ROOT}" || exit 1
trap stop_math_verify_server EXIT

cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export MATH_VERIFY_SERVER_URL

test_files="${A2OPD_MATH_TEST_FILE}"
train_files="${A2OPD_MATH_TRAIN_FILE}"

student_model_path="${A2OPD_MODEL_ROOT}/${student_model_name}"
teacher_model_path="${A2OPD_MODEL_ROOT}/${teacher_model_name}"
teacher_base_model_path="${A2OPD_MODEL_ROOT}/${teacher_base_model_name}"

today=$(cst_date +%Y%m%d_%H%M)
today_sec=$(cst_date +%Y%m%d_%H%M%S)

use_opd=true
opd_top_k=${opd_top_k:-16}   # required for A²OPD token_sampling (micro layer)
lambda_vals=1.0
student_rollout_n=1

# actor param offload: set false when GPU memory allows (skips actor CPU<->GPU each step)
actor_param_offload=true

# include student primary-base in parallel OPD prep (teacher proxies always parallel)
opd_parallel_student_base=false

update_mode=both

# validation config
val_n=16   # periodically evaluate on data.val_files (test.parquet)
validation_rollout_batch_size=1024
test_val_metric_group=math_avg
test_val_metric_sources=AIME2024,AIME2025,AIME2026,SMT2025,CMIMC2025,HMMT2025FEB,HMMT2025NOV,HMMT2026FEB
test_val_metric_sources_hydra="[${test_val_metric_sources}]"

# base learning rate
lr=1e-6
n_node=1
n_gpu=8
nccl_timeout=7200

# training length: total_epochs by default; set total_training_steps to cap max steps (overrides epoch count)
total_epochs=${total_epochs:-3}
total_training_steps=${total_training_steps:-80}


project_name=A2OPD@${val_n}-math

resume_path=${resume_path:-""}


extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume_path)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --resume_path requires a path argument" >&2
                exit 1
            fi
            resume_path="$2"
            shift 2
            ;;
        --resume_path=*)
            resume_path="${1#--resume_path=}"
            shift
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done
set -- "${extra_args[@]}"

output_model_name="S_${student_tag}-T_${teacher_tag}-TopK${opd_top_k}"

resolve_run_from_resume_path() {
    local rp resolved project_dir rel
    rp="$(readlink -f "$1")"
    project_dir="$(readlink -f "${A2OPD_CKPT_ROOT}/${project_name}")"
    if [[ "$rp" != "${project_dir}/"* ]]; then
        echo "ERROR: resume_path must be under ${project_dir}, got: $1" >&2
        return 1
    fi
    rel="${rp#"${project_dir}/"}"
    run_name="${rel%%/*}"
    output_path="${project_dir}/${run_name}"
}

resume_args=()
if [[ -n "$resume_path" ]]; then
    resolve_run_from_resume_path "$resume_path"
    experiment_name="${run_name}_resume_${today_sec}"
    resume_args=(
        trainer.resume_mode=resume_path
        trainer.resume_from_path="$resume_path"
    )
else
    experiment_name="${output_model_name}_A2OPD_${today}"
    output_path="${A2OPD_CKPT_ROOT}/${project_name}/${experiment_name}"
fi

# --- logging (timestamp + experiment name) ---
log_dir="${REPO_ROOT}/logs/math"
mkdir -p "${log_dir}"
LOG_FILE="${log_dir}/${experiment_name}.log"
# Ray AF_UNIX socket path must stay <=107 bytes; use short /tmp dir, link from log_dir
RAY_TMPDIR="/tmp/r_a2opd_${today}_$$"
mkdir -p "${RAY_TMPDIR}"
ln -sfn "${RAY_TMPDIR}" "${log_dir}/ray_${experiment_name}" 2>/dev/null || true
export RAY_TMPDIR

log_info() {
    echo "[$(cst_date '+%Y-%m-%d %H:%M:%S')] [${experiment_name}] $*"
}

_on_exit() {
    local ec=$?
    log_info "========== run finished (exit_code=${ec}) =========="
    log_info "log file: ${LOG_FILE}"
    log_info "ray logs: ${RAY_TMPDIR}"
    stop_math_verify_server
    exit "${ec}"
}
trap _on_exit EXIT

exec > >(tee "${LOG_FILE}") 2>&1

log_info "========== A²OPD training start =========="
log_info "project=${project_name}"
log_info "experiment=${experiment_name}"
log_info "output_path=${output_path}"
log_info "student=${student_model_path}"
log_info "teacher=${teacher_model_path} (base=${teacher_base_model_path})"
log_info "use_opd=${use_opd} opd_top_k=${opd_top_k} resume_path=${resume_path:-<none>}"
log_info "total_epochs=${total_epochs} total_training_steps=${total_training_steps:-<auto>}"
log_info "math_verify_server=${MATH_VERIFY_SERVER_URL:-<unset>}"
log_info "ray_tmpdir=${RAY_TMPDIR}"
log_info "host=$(hostname) user=${USER:-unknown} pid=$$"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

unset RAY_ADDRESS
unset RAY_NAMESPACE
unset RAY_DASHBOARD_ADDRESS
unset RAY_RUNTIME_ENV
unset RAY_JOB_ID
unset RAY_HEAD_IP
unset RAY_PORT


python3 -m verl.trainer.main_ppo \
    +algorithm.train_mode=heterogeneous_distill \
    +algorithm.hetero_distill.student_rollout_n=$student_rollout_n \
    +algorithm.hetero_distill.use_opd=$use_opd \
    +algorithm.hetero_distill.update_mode=$update_mode \
    +algorithm.hetero_distill.opd_top_k=$opd_top_k \
    +algorithm.hetero_distill.opd_parallel_student_base=$opd_parallel_student_base \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=3412 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path=$student_model_path \
    +actor_rollout_ref.model.base_model_path=$student_model_path \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    +actor_rollout_ref.ref.model.base_model_path=$teacher_base_model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.nccl_timeout=$nccl_timeout \
    teacher_rollout.nccl_timeout=$nccl_timeout \
    actor_rollout_ref.actor.fsdp_config.param_offload=$actor_param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$val_n \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    actor_rollout_ref.actor.policy_loss.lambda_vals=$lambda_vals \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=false \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=0 \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$n_gpu \
    trainer.nnodes=$n_node \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.save_freq=10 \
    +trainer.val_aggregate_group=$test_val_metric_group \
    +trainer.val_aggregate_sources="${test_val_metric_sources_hydra}" \
    +trainer.validation_rollout_batch_size=$validation_rollout_batch_size \
    +trainer.best_metric_name="val-core/${test_val_metric_group}/reward/mean@${val_n}" \
    +trainer.best_metric_mode="max" \
    trainer.default_local_dir="$output_path" \
    "${resume_args[@]}" \
    trainer.test_freq=10 \
    trainer.total_epochs=$total_epochs \
    ${total_training_steps:+trainer.total_training_steps=$total_training_steps} \
    "$@"
