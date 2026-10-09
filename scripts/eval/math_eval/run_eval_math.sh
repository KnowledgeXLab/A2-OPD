#!/bin/bash


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
# shellcheck source=../../opd_paths.sh
source "$REPO_ROOT/scripts/opd_paths.sh"
# shellcheck source=../gpu_alloc.sh
source "$REPO_ROOT/scripts/eval/gpu_alloc.sh"

is_true() {
    [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "True" ]]
}

bench_output_dir() {
    echo "$1" | tr '[:upper:]' '[:lower:]'
}

has_model_weights() {
    local path="$1"
    compgen -G "${path}/*.safetensors" >/dev/null || compgen -G "${path}/*.bin" >/dev/null
}

resolve_model_name() {
    basename "$1"
}

sanitize_log_tag() {
    local tag="$1"
    tag="${tag//\//_}"
    tag="${tag// /_}"
    echo "$tag"
}

# MODEL_NAME is only for output directory naming; MODEL_PATH is the HF weights dir.
normalize_model_identity() {
    if [[ "${MODEL_NAME:-}" == */* ]]; then
        echo "Warning: MODEL_NAME looks like a path (${MODEL_NAME}); using basename for outputs." >&2
        if [[ -z "${MODEL_PATH:-}" || ! -d "${MODEL_PATH:-}" ]]; then
            MODEL_PATH=$(cd "${MODEL_NAME}" && pwd)
            echo "Warning: MODEL_PATH unset; inferred MODEL_PATH=${MODEL_PATH} from MODEL_NAME." >&2
        fi
        MODEL_NAME=$(basename "$MODEL_NAME")
    fi
    if [[ -n "${MODEL_PATH:-}" ]]; then
        MODEL_NAME="${MODEL_NAME:-$(resolve_model_name "$MODEL_PATH")}"
    fi
    MODEL_NAME=$(basename "$MODEL_NAME")
}

resolve_math_eval_model() {
    if [[ -n "${A2OPD_EXP_PATH:-}" ]]; then
        # shellcheck source=../a2opd_model_resolve.sh
        source "$REPO_ROOT/scripts/eval/a2opd_model_resolve.sh"
        resolve_a2opd_eval_model || exit 1
        normalize_model_identity
        export MODEL_PATH MODEL_NAME
        return 0
    fi

    if [[ -n "${MODEL_PATH:-}" ]]; then
        MODEL_PATH=$(cd "$MODEL_PATH" && pwd)
        MODEL_NAME=${MODEL_NAME:-$(resolve_model_name "$MODEL_PATH")}
    else
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-naive_qwen3-4b_mopd}
        CKPT_ROOT=${CKPT_ROOT:-$OPD_MULTI_OPD_MODEL_ROOT/${EXPERIMENT_NAME}}
        CKPT_STEP=${CKPT_STEP:-20}
        for candidate in \
            "${CKPT_ROOT}/global_step_${CKPT_STEP}/actor/merged_hf" \
            "${CKPT_ROOT}/global_step_${CKPT_STEP}/actor/huggingface" \
            "${CKPT_ROOT}/global_step_${CKPT_STEP}/hf" \
            "${CKPT_ROOT}/global_step_${CKPT_STEP}"; do
            if [[ -d "$candidate" ]] && has_model_weights "$candidate"; then
                MODEL_PATH="$candidate"
                break
            fi
        done
        MODEL_NAME=${MODEL_NAME:-${EXPERIMENT_NAME}_step${CKPT_STEP}}
    fi

    normalize_model_identity

    if [[ -z "${MODEL_PATH:-}" || ! -d "$MODEL_PATH" ]] || ! has_model_weights "$MODEL_PATH"; then
        echo "Cannot resolve MODEL_PATH." >&2
        echo "Option A: set MODEL_PATH to a HuggingFace directory containing *.safetensors or *.bin." >&2
        echo "Option B: leave MODEL_PATH unset and provide EXPERIMENT_NAME (or CKPT_ROOT) + CKPT_STEP." >&2
        if [[ -z "${MODEL_PATH:-}" ]]; then
            echo "Tried CKPT_ROOT=${CKPT_ROOT:-unset}, CKPT_STEP=${CKPT_STEP:-unset}" >&2
        fi
        exit 1
    fi

    export MODEL_PATH MODEL_NAME
    if [[ -n "${EXPERIMENT_NAME:-}" ]]; then
        export EXPERIMENT_NAME
    fi
    if [[ -n "${CKPT_STEP:-}" ]]; then
        export CKPT_STEP
    fi
}

# Usage (pick one):
#   1) Direct HF model path (no CKPT_STEP needed):
#        MODEL_PATH=/path/to/merged_hf bash scripts/eval/math_eval/run_eval_math.sh
#   2) Resolve from verl checkpoint:
#        EXPERIMENT_NAME=my_exp CKPT_STEP=20 bash scripts/eval/math_eval/run_eval_math.sh
#   EVAL_KIND=math EVAL_MODELS_FILE=scripts/eval/eval_models.conf bash scripts/eval/run_eval_batch.sh
#   3) Optional output name override:
#        MODEL_PATH=... MODEL_NAME=my_model bash scripts/eval/math_eval/run_eval_math.sh
#   4) Optional test parquet override:
#        TEST_FILES=/path/to/test.parquet bash scripts/eval/math_eval/run_eval_math.sh
#   5) Toggle benches (default: subset enabled below):
#        RUN_AIME2026=false RUN_HMMT2025NOV=false bash scripts/eval/math_eval/run_eval_math.sh
#   6) vLLM CUDA graph (default false for speed; set true if unstable):
#        VLLM_ENFORCE_EAGER=true bash scripts/eval/math_eval/run_eval_math.sh
#
# Logs: logs/math_eval/math_eval_<MODEL_NAME>_<timestamp>.log
#       plus symlink math_eval_<MODEL_NAME>_latest.log

# MODEL_PATH=$OPD_PUBLIC_MODEL_ROOT/Qwen2.5-Math-7B-Instruct
# MODEL_PATH=$OPD_PUBLIC_MODEL_ROOT/Qwen3-4B
# MODEL_PATH=$OPD_TEACHER_MODEL_ROOT/Qwen3-4B-Non-Thinking-RL-Math-Step500

# EXPERIMENT_NAME=spin_qwen3-4b_mopd_sparse_exclusive_all_layers
# EXPERIMENT_NAME=spin_qwen3-4b_mopd_sparse_exclusive_stride_4
# EXPERIMENT_NAME=spin_qwen3-4b_mopd_math_code_if_sparse_exclusive_all_layers
# EXPERIMENT_NAME=naive_qwen3-4b_mopd_math_code_if
# CKPT_STEP=70

if [[ -z "${MATH_EVAL_LOGGING_ACTIVE:-}" ]]; then
    resolve_math_eval_model
    export MATH_EVAL_MODEL_RESOLVED=1
    MATH_EVAL_LOG_DIR=${MATH_EVAL_LOG_DIR:-$REPO_ROOT/logs/math_eval}
    mkdir -p "$MATH_EVAL_LOG_DIR"
    opd_refresh_log_datetime
    MATH_EVAL_LOG_STAMP=${MATH_EVAL_LOG_STAMP:-$(opd_log_date +%Y%m%d_%H%M%S)}
    MATH_EVAL_LOG_TAG=${MATH_EVAL_LOG_TAG:-$(sanitize_log_tag "$MODEL_NAME")}
    MATH_EVAL_LOG_FILE=${MATH_EVAL_LOG_FILE:-$MATH_EVAL_LOG_DIR/math_eval_${MATH_EVAL_LOG_TAG}_${MATH_EVAL_LOG_STAMP}.log}
    ln -sfn "$MATH_EVAL_LOG_FILE" "${MATH_EVAL_LOG_DIR}/math_eval_${MATH_EVAL_LOG_TAG}_latest.log"
    export MATH_EVAL_LOGGING_ACTIVE=1
    export MATH_EVAL_LOG_FILE MATH_EVAL_LOG_TAG
    exec > >(tee -a "$MATH_EVAL_LOG_FILE") 2>&1
fi

set -x
nvidia-smi || true

cd "$SCRIPT_DIR"
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$OPD_EXP_RESULT_ROOT/math_eval_outputs}
if [[ -z "${TEST_FILES:-}" ]]; then
    if [[ -f "$OPD_DATA_ROOT/training/MathTestTotal/test.parquet" ]]; then
        TEST_FILES="$OPD_DATA_ROOT/training/MathTestTotal/test.parquet"
    else
        TEST_FILES="$OPD_DATA_ROOT/training/MathTestTotal/test-sampled_10pct.parquet"
        echo "Warning: full MathTestTotal/test.parquet is unavailable; using 10% sample: $TEST_FILES" >&2
    fi
fi

if [[ ! -f "$TEST_FILES" ]]; then
    echo "Missing math test parquet: ${TEST_FILES}" >&2
    exit 1
fi

RUN_AIME2024=${RUN_AIME2024:-true}
RUN_AIME2025=${RUN_AIME2025:-true}
RUN_AIME2026=${RUN_AIME2026:-true}
RUN_SMT2025=${RUN_SMT2025:-true}
RUN_CMIMC2025=${RUN_CMIMC2025:-true}
RUN_HMMT2025FEB=${RUN_HMMT2025FEB:-true}
RUN_HMMT2025NOV=${RUN_HMMT2025NOV:-true}
RUN_HMMT2026FEB=${RUN_HMMT2026FEB:-true}

if [[ -z "${MATH_EVAL_MODEL_RESOLVED:-}" ]]; then
    resolve_math_eval_model
    export MATH_EVAL_MODEL_RESOLVED=1
fi

MAX_TOKENS=${MAX_TOKENS:-16384}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-1.0}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
BATCH_SIZE=${BATCH_SIZE:-32}
N=${N:-32}
SEED=${SEED:-42}
ENABLE_THINKING=${ENABLE_THINKING:-False}
FORCE_RERUN=${FORCE_RERUN:-false}
# Parallel bench workers each spawn vLLM+torch.compile; eager avoids triton cache races.
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-true}
export VLLM_ENFORCE_EAGER

COMMON_ARGS=(
    --model_path "$MODEL_PATH"
    --max_tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --max_num_seqs "$MAX_NUM_SEQS"
    --batch_size "$BATCH_SIZE"
    --n "$N"
    --begin_idx -1
    --end_idx -1
    --seed "$SEED"
)

if [[ "$ENABLE_THINKING" == "True" || "$ENABLE_THINKING" == "true" || "$ENABLE_THINKING" == "1" ]]; then
    COMMON_ARGS+=(--enable_thinking)
fi

declare -A BENCH_LABELS=(
    [AIME2024]="AIME2024"
    [AIME2025]="AIME2025"
    [AIME2026]="AIME2026"
    [SMT2025]="SMT2025"
    [CMIMC2025]="CMIMC2025"
    [HMMT2025FEB]="HMMT2025(Feb)"
    [HMMT2025NOV]="HMMT2025(Nov)"
    [HMMT2026FEB]="HMMT2026(Feb)"
)

DATASETS=()
is_true "$RUN_AIME2024" && DATASETS+=("AIME2024")
is_true "$RUN_AIME2025" && DATASETS+=("AIME2025")
is_true "$RUN_AIME2026" && DATASETS+=("AIME2026")
is_true "$RUN_SMT2025" && DATASETS+=("SMT2025")
is_true "$RUN_CMIMC2025" && DATASETS+=("CMIMC2025")
is_true "$RUN_HMMT2025FEB" && DATASETS+=("HMMT2025FEB")
is_true "$RUN_HMMT2025NOV" && DATASETS+=("HMMT2025NOV")
is_true "$RUN_HMMT2026FEB" && DATASETS+=("HMMT2026FEB")

NUM_TASKS=${#DATASETS[@]}
if [[ "$NUM_TASKS" -eq 0 ]]; then
    echo "No math benches enabled. Set RUN_AIME2024=true (etc.) to enable at least one bench." >&2
    exit 1
fi

MODEL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}/${MODEL_NAME}"
for dataset in "${DATASETS[@]}"; do
    mkdir -p "${MODEL_OUTPUT_ROOT}/$(bench_output_dir "$dataset")"
done

ALLOCATED_GPU_CSV="$(resolve_allocated_gpu_csv)"
export CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU_CSV"
IFS=',' read -r -a ALL_GPUS <<< "$ALLOCATED_GPU_CSV"
NUM_GPUS=${#ALL_GPUS[@]}

echo "MODEL_PATH=${MODEL_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
echo "TEST_FILES=${TEST_FILES}"
echo "MATH_EVAL_LOG_TAG=${MATH_EVAL_LOG_TAG:-}"
echo "MATH_EVAL_LOG_FILE=${MATH_EVAL_LOG_FILE:-}"
echo "ALLOCATED_GPU_CSV=${ALLOCATED_GPU_CSV}"
echo "ENABLED_BENCHES=${DATASETS[*]}"
echo "RUN_AIME2024=${RUN_AIME2024} RUN_AIME2025=${RUN_AIME2025} RUN_AIME2026=${RUN_AIME2026} RUN_SMT2025=${RUN_SMT2025}"
echo "RUN_CMIMC2025=${RUN_CMIMC2025} RUN_HMMT2025FEB=${RUN_HMMT2025FEB} RUN_HMMT2025NOV=${RUN_HMMT2025NOV} RUN_HMMT2026FEB=${RUN_HMMT2026FEB}"
echo "BATCH_SIZE=${BATCH_SIZE} FORCE_RERUN=${FORCE_RERUN}"
echo "VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER}"

ENABLED_BENCHES_CSV=$(IFS=,; echo "${DATASETS[*]}")

update_summary() {
    opd_refresh_log_datetime
    python3 math_eval_summary.py \
        --output_root "$MODEL_OUTPUT_ROOT" \
        --model_name "$MODEL_NAME" \
        --benches "$ENABLED_BENCHES_CSV" \
        --n_samples "$N" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS"
}

bench_result_file() {
    local dataset="$1"
    echo "${MODEL_OUTPUT_ROOT}/$(bench_output_dir "$dataset")/${MODEL_NAME}.jsonl"
}

should_skip_bench() {
    local dataset="$1"
    local output_file="$2"
    if is_true "$FORCE_RERUN"; then
        return 1
    fi
    python3 - "$dataset" "$output_file" "$TEST_FILES" <<'PY'
import sys
from math_eval_utils import count_completed_by_id, count_expected_samples, is_result_complete

dataset, output_file, test_files = sys.argv[1:4]
expected = count_expected_samples(test_files, dataset)
completed = count_completed_by_id(output_file, test_files, dataset)
if is_result_complete(output_file, test_files, dataset):
    print(f"[math] skip {dataset}: complete ({completed}/{expected} ids) -> {output_file}")
    raise SystemExit(0)
if completed > 0:
    print(f"[math] resume {dataset}: partial ({completed}/{expected} ids) -> {output_file}")
raise SystemExit(1)
PY
}

run_one() {
    local devices="$1"
    local dataset="$2"
    local output_file
    output_file="$(bench_result_file "$dataset")"
    local output_dir
    output_dir="$(bench_output_dir "$dataset")"

    export VLLM_CACHE_ROOT="${MODEL_OUTPUT_ROOT}/.vllm_cache/${output_dir}/worker_${$}"
    mkdir -p "$VLLM_CACHE_ROOT"
    export TORCHINDUCTOR_CACHE_DIR="${VLLM_CACHE_ROOT}/inductor"
    export TRITON_CACHE_DIR="${VLLM_CACHE_ROOT}/triton"
    if is_true "$VLLM_ENFORCE_EAGER"; then
        export TORCHDYNAMO_DISABLE=1
    fi

    echo "[math] dataset=${dataset} devices=${devices} input=${TEST_FILES}"
    CUDA_VISIBLE_DEVICES="$devices" python3 eval_math.py \
        --input_file "$TEST_FILES" \
        --val_metric_source "$dataset" \
        --output_file "$output_file" \
        "${COMMON_ARGS[@]}"
}

run_bench() {
    local devices="$1"
    local dataset="$2"
    if should_skip_bench "$dataset" "$(bench_result_file "$dataset")"; then
        update_summary
        return 0
    fi
    if ! run_one "$devices" "$dataset"; then
        return 1
    fi
    update_summary
    return 0
}

bench_needs_gpu() {
    local dataset="$1"
    if should_skip_bench "$dataset" "$(bench_result_file "$dataset")"; then
        return 1
    fi
    return 0
}

launch_math_eval_bench_worker() {
    local gpu_str="$1"
    local dataset="$2"
    (
        export MATH_EVAL_SINGLE_DATASET="$dataset"
        export MATH_EVAL_SINGLE_GPU="$gpu_str"
        export MATH_EVAL_LOGGING_ACTIVE=1
        cd "$SCRIPT_DIR"
        exec "$BASH" "${BASH_SOURCE[0]}"
    ) &
}

schedule_pending_math_eval_benches() {
    local failed=0
    local -a pids=()
    local i dataset start_idx task_gpus gpu_str pid
    local offset wave_num wave_size remaining GPUS_PER_TASK

    if [[ "$NUM_PENDING" -le "$NUM_GPUS" ]]; then
        GPUS_PER_TASK=$(( NUM_GPUS / NUM_PENDING ))
        echo "[math] scheduling ${NUM_PENDING} pending bench(es) in parallel: ${GPUS_PER_TASK} GPUs each, ${NUM_GPUS} total"
        for i in "${!PENDING_DATASETS[@]}"; do
            dataset="${PENDING_DATASETS[$i]}"
            start_idx=$(( i * GPUS_PER_TASK ))
            task_gpus=("${ALL_GPUS[@]:start_idx:GPUS_PER_TASK}")
            gpu_str=$(IFS=,; echo "${task_gpus[*]}")
            echo "[math] scheduling ${dataset} on GPUs: ${gpu_str}"
            launch_math_eval_bench_worker "$gpu_str" "$dataset"
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done
    else
        echo "[math] scheduling ${NUM_PENDING} pending bench(es) in waves of up to ${NUM_GPUS}, ${NUM_GPUS} GPUs total"
        offset=0
        wave_num=1
        while [[ "$offset" -lt "$NUM_PENDING" ]]; do
            wave_size=$NUM_GPUS
            remaining=$(( NUM_PENDING - offset ))
            if [[ "$remaining" -lt "$wave_size" ]]; then
                wave_size=$remaining
            fi
            GPUS_PER_TASK=$(( NUM_GPUS / wave_size ))
            pids=()
            for ((i=0; i<wave_size; i++)); do
                dataset="${PENDING_DATASETS[$((offset + i))]}"
                start_idx=$(( i * GPUS_PER_TASK ))
                task_gpus=("${ALL_GPUS[@]:start_idx:GPUS_PER_TASK}")
                gpu_str=$(IFS=,; echo "${task_gpus[*]}")
                echo "[math] scheduling ${dataset} on GPUs: ${gpu_str} wave ${wave_num}"
                launch_math_eval_bench_worker "$gpu_str" "$dataset"
                pids+=("$!")
            done
            offset=$(( offset + wave_size ))
            wave_num=$(( wave_num + 1 ))
            for pid in "${pids[@]}"; do
                if ! wait "$pid"; then
                    failed=1
                fi
            done
        done
    fi

    return "$failed"
}

# Single-bench worker: isolated process avoids background function fall-through.
if [[ -n "${MATH_EVAL_SINGLE_DATASET:-}" ]]; then
    echo "[math] worker dataset=${MATH_EVAL_SINGLE_DATASET} devices=${MATH_EVAL_SINGLE_GPU}"
    if ! run_bench "${MATH_EVAL_SINGLE_GPU}" "${MATH_EVAL_SINGLE_DATASET}"; then
        exit 1
    fi
    exit 0
fi

PENDING_DATASETS=()
SKIPPED_DATASETS=()
for dataset in "${DATASETS[@]}"; do
    if bench_needs_gpu "$dataset"; then
        PENDING_DATASETS+=("$dataset")
    else
        SKIPPED_DATASETS+=("$dataset")
    fi
done

NUM_PENDING=${#PENDING_DATASETS[@]}
NUM_SKIPPED=${#SKIPPED_DATASETS[@]}
echo "PENDING_BENCHES (${NUM_PENDING}): ${PENDING_DATASETS[*]:-(none)}"
echo "SKIPPED_BENCHES (${NUM_SKIPPED}): ${SKIPPED_DATASETS[*]:-(none)}"

if [[ "$NUM_PENDING" -eq 0 ]]; then
    echo "[math] all ${#DATASETS[@]} enabled bench(es) already complete; skipping GPU eval."
    update_summary
    exit 0
fi

failed=0
if ! schedule_pending_math_eval_benches; then
    failed=1
fi

update_summary

if [[ -d "${MODEL_OUTPUT_ROOT}/.vllm_cache" ]]; then
    rm -rf "${MODEL_OUTPUT_ROOT}/.vllm_cache"
    echo "Removed vLLM cache: ${MODEL_OUTPUT_ROOT}/.vllm_cache"
fi

if [[ "$failed" -ne 0 ]]; then
    echo "One or more math eval subprocesses failed." >&2
    exit 1
fi
echo "Model ${MODEL_NAME} math eval done!"
exit 0
