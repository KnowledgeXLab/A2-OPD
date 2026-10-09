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

model_identifier() {
    echo "$1" | sed 's#^\./##' | tr '/' '-'
}

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

resolve_code_eval_model() {
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
            echo "If you only have a verl/FSDP checkpoint, merge it first, for example:" >&2
            echo "python3 -m verl.model_merger merge --backend fsdp \\" >&2
            echo "  --local_dir ${CKPT_ROOT}/global_step_${CKPT_STEP}/actor \\" >&2
            echo "  --target_dir ${CKPT_ROOT}/global_step_${CKPT_STEP}/actor/merged_hf" >&2
        else
            echo "MODEL_PATH=${MODEL_PATH} is missing weight files." >&2
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
#   1) Direct HF model path:
#        MODEL_PATH=/path/to/merged_hf bash scripts/eval/code_eval/run_eval_code.sh
#   2) Resolve from verl checkpoint:
#        EXPERIMENT_NAME=my_exp CKPT_STEP=20 bash scripts/eval/code_eval/run_eval_code.sh
#   EVAL_KIND=code EVAL_MODELS_FILE=scripts/eval/eval_models.conf bash scripts/eval/run_eval_batch.sh
#   3) Benchmark switches (default: LCB on, humaneval/mbpp off):
#        RUN_HUMANEVAL=true  RUN_MBPP=true  RUN_LCB=false
#        RUN_HUMANEVAL=false RUN_MBPP=false RUN_LCB=true   # LCB only
#        RUN_HUMANEVAL=true  RUN_MBPP=false RUN_LCB=false  # HumanEval+ only
#      GPU layout (default two-phase):
#        phase 1: HumanEval + MBPP parallel, GPUs split evenly (e.g. 8 GPU -> 4+4)
#        phase 2: LCB alone on all GPUs (e.g. tp=8)
#   4) Optional inference overrides, shared by EvalPlus and LCB:
#        MODEL_NAME=my_model N_SAMPLES=4 TEMPERATURE=1.0 TOP_P=1.0 MAX_TOKENS=16384 BATCH_SIZE=32
#        VLLM_ENFORCE_EAGER=false VLLM_MAX_MODEL_LEN=16384 bash ...
#   5) Parallelism controls (8B models often need serial on 4 GPU):
#        MIN_GPUS_PER_BENCH=2 bash scripts/eval/code_eval/run_eval_code.sh
#        FORCE_SERIAL=true bash scripts/eval/code_eval/run_eval_code.sh
#        FORCE_PARALLEL=true bash scripts/eval/code_eval/run_eval_code.sh
#   6) Skip completed benches unless forced:
#        FORCE_RERUN=true bash scripts/eval/code_eval/run_eval_code.sh
#      LCB generation/eval/cache live under code_eval_outputs/{MODEL}/lcb/
#      (same tree as HumanEval+/MBPP+ samples under code_eval_outputs/{MODEL}/).
#   7) Optional benchmark data root override:
#        CODE_BENCHMARK_ROOT=/path/to/benchmarks/code bash scripts/eval/code_eval/run_eval_code.sh
#
# Logs: logs/code_eval/code_eval_<MODEL_NAME>_<timestamp>.log
#       plus symlink code_eval_<MODEL_NAME>_latest.log

# MODEL_PATH=$OPD_PUBLIC_MODEL_ROOT/Qwen3-4B-Instruct
# MODEL_PATH=$OPD_TEACHER_MODEL_ROOT/Qwen3-4B-Non-Thinking-RL-Code-Step300
# EXPERIMENT_NAME=naive_qwen3-4b_mopd
# EXPERIMENT_NAME=spin_qwen3-4b_mopd_sparse_exclusive_all_layers
# EXPERIMENT_NAME=spin_qwen3-4b_mopd_sparse_exclusive_stride_4
# EXPERIMENT_NAME=spin_qwen3-4b_mopd_math_code_if_sparse_exclusive_all_layers
# EXPERIMENT_NAME=naive_qwen3-4b_mopd_math_code_if
# CKPT_STEP=70

if [[ -z "${CODE_EVAL_LOGGING_ACTIVE:-}" ]]; then
    resolve_code_eval_model
    export CODE_EVAL_MODEL_RESOLVED=1
    CODE_EVAL_LOG_DIR=${CODE_EVAL_LOG_DIR:-$REPO_ROOT/logs/code_eval}
    mkdir -p "$CODE_EVAL_LOG_DIR"
    opd_refresh_log_datetime
    CODE_EVAL_LOG_STAMP=${CODE_EVAL_LOG_STAMP:-$(opd_log_date +%Y%m%d_%H%M%S)}
    CODE_EVAL_LOG_TAG=${CODE_EVAL_LOG_TAG:-$(sanitize_log_tag "$MODEL_NAME")}
    CODE_EVAL_LOG_FILE=${CODE_EVAL_LOG_FILE:-$CODE_EVAL_LOG_DIR/code_eval_${CODE_EVAL_LOG_TAG}_${CODE_EVAL_LOG_STAMP}.log}
    ln -sfn "$CODE_EVAL_LOG_FILE" "${CODE_EVAL_LOG_DIR}/code_eval_${CODE_EVAL_LOG_TAG}_latest.log"
    export CODE_EVAL_LOGGING_ACTIVE=1
    export CODE_EVAL_LOG_FILE CODE_EVAL_LOG_TAG
    exec > >(tee -a "$CODE_EVAL_LOG_FILE") 2>&1
fi

describe_exit_code() {
    local code="$1"
    case "$code" in
        137) echo "SIGKILL (often OOM / process leak)" ;;
        139) echo "SIGSEGV" ;;
        141) echo "SIGPIPE (tee/log pipe broken)" ;;
        143) echo "SIGTERM" ;;
        130) echo "SIGINT (interrupted)" ;;
        0) echo "ok" ;;
        *) echo "exit ${code}" ;;
    esac
}

# Write directly to the log file so messages survive even if tee is dead
# (SIGKILL still cannot be trapped; that case needs the Manager leak fix).
_code_eval_log_direct() {
    local msg="$1"
    if [[ -n "${CODE_EVAL_LOG_FILE:-}" ]]; then
        printf '%s\n' "$msg" >>"$CODE_EVAL_LOG_FILE" 2>/dev/null || true
    fi
    printf '%s\n' "$msg" || true
}

_on_code_eval_exit() {
    local ec=$?
    if [[ "$ec" -ne 0 ]]; then
        _code_eval_log_direct "[code_eval] finished with code=${ec} ($(describe_exit_code "$ec")) at $(opd_log_ts)"
    fi
}
trap _on_code_eval_exit EXIT
trap 'echo "[code_eval] caught SIGTERM"; exit 143' TERM
trap 'echo "[code_eval] caught SIGINT"; exit 130' INT

export PYTHONUNBUFFERED=1
# Silence multiprocess ResourceTracker __del__ noise on Python 3.12 (RLock API change).
export PYTHONSTARTUP="${CODE_EVAL_ROOT:-$SCRIPT_DIR}/multiprocess_py312_compat.py"
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
# EvalPlus ProcessPool: spawn avoids fork + vLLM/multiprocess ResourceTracker noise on Py3.12.
export EVALPLUS_MP_START_METHOD=${EVALPLUS_MP_START_METHOD:-spawn}
# EvalPlus-style parallelism: cpu//2 capped at 16 (override with LCB_NUM_PROCESS_EVALUATE).
if [[ -z "${LCB_NUM_PROCESS_EVALUATE:-}" ]]; then
    LCB_NUM_PROCESS_EVALUATE="$(python3 -c 'import os; print(max(4, min(16, (os.cpu_count() or 8)//2)))')"
fi
export LCB_NUM_PROCESS_EVALUATE

set -x
nvidia-smi || true

CODE_EVAL_ROOT=${CODE_EVAL_ROOT:-$SCRIPT_DIR}
cd "$CODE_EVAL_ROOT"

EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$OPD_EXP_RESULT_ROOT/code_eval_outputs}

export PYTHONPATH="${CODE_EVAL_ROOT}/coding/evalplus:${PYTHONPATH:-}"
# Model-generated code often uses invalid escapes (e.g. "\w", "\+"); Py3.12+ emits
# SyntaxWarning on every compile and can flood eval logs (millions of lines).
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::SyntaxWarning}"

CODE_BENCHMARK_ROOT=${CODE_BENCHMARK_ROOT:-$OPD_DATA_ROOT/benchmarks/code}
EVALPLUS_DATA_ROOT=${EVALPLUS_DATA_ROOT:-$CODE_BENCHMARK_ROOT}
DEFAULT_HUMANEVAL_PATH="${EVALPLUS_DATA_ROOT}/Humaneval_Plus/HumanEvalPlus.jsonl"
DEFAULT_MBPP_PATH="${EVALPLUS_DATA_ROOT}/Mbpp_Plus/MbppPlus.jsonl"
LCB_DATA_ROOT=${LCB_DATA_ROOT:-$CODE_BENCHMARK_ROOT/LivecodeBench}

resolve_evalplus_dataset_path() {
    local name="$1"
    local override="${2:-}"
    local default_path="$3"
    if [[ -n "$override" && -f "$override" ]]; then
        echo "$override"
        return
    fi
    if [[ -n "$override" ]]; then
        echo "Ignoring invalid ${name}=${override}; using ${default_path}" >&2
    fi
    echo "$default_path"
}

_he_override="${HUMANEVAL_OVERRIDE_PATH:-}"
_mbpp_override="${MBPP_OVERRIDE_PATH:-}"
HUMANEVAL_OVERRIDE_PATH="$(resolve_evalplus_dataset_path HUMANEVAL_OVERRIDE_PATH "$_he_override" "$DEFAULT_HUMANEVAL_PATH")"
MBPP_OVERRIDE_PATH="$(resolve_evalplus_dataset_path MBPP_OVERRIDE_PATH "$_mbpp_override" "$DEFAULT_MBPP_PATH")"
export HUMANEVAL_OVERRIDE_PATH MBPP_OVERRIDE_PATH

if [[ -z "${CODE_EVAL_MODEL_RESOLVED:-}" ]]; then
    resolve_code_eval_model
    export CODE_EVAL_MODEL_RESOLVED=1
fi

# G-OPD paper code eval defaults and scripts/run_lcb_gen.sh paper settings:
# n=4, temperature=1.0, top_p=1.0, max_tokens=16384.
N_SAMPLES=${N_SAMPLES:-8}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-1.0}
MAX_TOKENS=${MAX_TOKENS:-16384}
# Total vLLM sequences per generate call (problems/prompts × n_samples).
BATCH_SIZE=${BATCH_SIZE:-32}
GREEDY=${GREEDY:-0}
FORCE_RERUN=${FORCE_RERUN:-false}
MIN_GPUS_PER_BENCH=${MIN_GPUS_PER_BENCH:-2}
FORCE_SERIAL=${FORCE_SERIAL:-false}
FORCE_PARALLEL=${FORCE_PARALLEL:-false}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-true}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-$MAX_TOKENS}
RUN_HUMANEVAL=${RUN_HUMANEVAL:-true}
RUN_MBPP=${RUN_MBPP:-true}
RUN_LCB=${RUN_LCB:-true}
LCB_ROOT=${LCB_ROOT:-$REPO_ROOT/scripts/eval/code_eval/coding/LiveCodeBench}
LCB_MODEL=${LCB_MODEL:-Qwen3-4B-NonThinking}
LCB_RELEASE_VERSION=${LCB_RELEASE_VERSION:-v6}
# LCB tensor parallel size; empty = use all GPUs allocated to this job.
LCB_TP=${LCB_TP:-}

MODEL_ID=$(model_identifier "$MODEL_PATH")
if is_true "$GREEDY"; then
    SAMPLES_ID="samples_n1_temp0.0"
else
    SAMPLES_ID="samples_n${N_SAMPLES}_temp${TEMPERATURE}"
fi

codegen_prompt_batch_size() {
    local n="${N_SAMPLES:-1}"
    [[ "$n" -lt 1 ]] && n=1
    local prompt_batch=$(( BATCH_SIZE / n ))
    [[ "$prompt_batch" -lt 1 ]] && prompt_batch=1
    echo "$prompt_batch"
}
CODEGEN_PROMPT_BATCH_SIZE="$(codegen_prompt_batch_size)"

if ! is_true "$RUN_HUMANEVAL" && ! is_true "$RUN_MBPP" && ! is_true "$RUN_LCB"; then
    echo "At least one of RUN_HUMANEVAL, RUN_MBPP, RUN_LCB must be enabled." >&2
    exit 1
fi

if is_true "$RUN_HUMANEVAL" && [[ ! -f "$HUMANEVAL_OVERRIDE_PATH" ]]; then
    echo "Missing EvalPlus dataset: ${HUMANEVAL_OVERRIDE_PATH}" >&2
    exit 1
fi
if is_true "$RUN_MBPP" && [[ ! -f "$MBPP_OVERRIDE_PATH" ]]; then
    echo "Missing EvalPlus dataset: ${MBPP_OVERRIDE_PATH}" >&2
    exit 1
fi
if is_true "$RUN_LCB" && [[ ! -f "${LCB_DATA_ROOT}/test.jsonl" ]]; then
    echo "Missing LiveCodeBench dataset: ${LCB_DATA_ROOT}/test.jsonl" >&2
    exit 1
fi

MODEL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}/${MODEL_NAME}"
LCB_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}/lcb"
output_subdirs=()
is_true "$RUN_HUMANEVAL" && output_subdirs+=(humaneval)
is_true "$RUN_MBPP" && output_subdirs+=(mbpp)
is_true "$RUN_LCB" && output_subdirs+=(lcb)
mkdir -p "$REPO_ROOT/logs/code_eval" "$MODEL_OUTPUT_ROOT"
for subdir in "${output_subdirs[@]}"; do
    mkdir -p "${MODEL_OUTPUT_ROOT}/${subdir}"
done

echo "MODEL_PATH=${MODEL_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT}"
echo "LCB_OUTPUT_ROOT=${LCB_OUTPUT_ROOT}"
echo "SAMPLES_ID=${SAMPLES_ID}"
echo "CODE_EVAL_LOG_TAG=${CODE_EVAL_LOG_TAG:-}"
echo "CODE_EVAL_LOG_FILE=${CODE_EVAL_LOG_FILE:-}"
echo "N_SAMPLES=${N_SAMPLES} TEMPERATURE=${TEMPERATURE} TOP_P=${TOP_P} BATCH_SIZE=${BATCH_SIZE} CODEGEN_PROMPT_BATCH_SIZE=${CODEGEN_PROMPT_BATCH_SIZE}"
echo "CODE_BENCHMARK_ROOT=${CODE_BENCHMARK_ROOT}"
echo "HUMANEVAL_OVERRIDE_PATH=${HUMANEVAL_OVERRIDE_PATH}"
echo "MBPP_OVERRIDE_PATH=${MBPP_OVERRIDE_PATH}"
echo "LCB_DATA_ROOT=${LCB_DATA_ROOT}"
echo "RUN_HUMANEVAL=${RUN_HUMANEVAL} RUN_MBPP=${RUN_MBPP} RUN_LCB=${RUN_LCB}"
echo "FORCE_RERUN=${FORCE_RERUN}"
echo "MIN_GPUS_PER_BENCH=${MIN_GPUS_PER_BENCH}"
echo "FORCE_SERIAL=${FORCE_SERIAL} FORCE_PARALLEL=${FORCE_PARALLEL}"
echo "VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER} VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"

ALLOCATED_GPU_CSV="$(resolve_allocated_gpu_csv)"
export CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU_CSV"
IFS=',' read -r -a ALL_GPUS <<< "$ALLOCATED_GPU_CSV"
NUM_GPUS=${#ALL_GPUS[@]}
echo "ALLOCATED_GPU_CSV=${ALLOCATED_GPU_CSV}"

evalplus_dataset_path() {
    local dataset="$1"
    if [[ "$dataset" == "humaneval" ]]; then
        echo "$HUMANEVAL_OVERRIDE_PATH"
    else
        echo "$MBPP_OVERRIDE_PATH"
    fi
}

evalplus_samples_file() {
    local dataset="$1"
    echo "${MODEL_OUTPUT_ROOT}/${dataset}/${SAMPLES_ID}.jsonl"
}

evalplus_result_file() {
    local dataset="$1"
    echo "${MODEL_OUTPUT_ROOT}/${dataset}/eval_results.json"
}

should_skip_evalplus() {
    local dataset="$1"
    if is_true "$FORCE_RERUN"; then
        return 1
    fi
    python3 - "$dataset" "$(evalplus_samples_file "$dataset")" "$(evalplus_result_file "$dataset")" "$(evalplus_dataset_path "$dataset")" "$N_SAMPLES" "$GREEDY" <<'PY'
import json
import sys
from pathlib import Path

dataset, samples_file, result_file, dataset_path, n_samples, greedy = sys.argv[1:7]
greedy = greedy.lower() in ("1", "true")
n_expected = 1 if greedy else int(n_samples)

expected_ids = set()
with open(dataset_path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            expected_ids.add(json.loads(line)["task_id"])
if not expected_ids:
    raise SystemExit(1)

samples_path = Path(samples_file)
if not samples_path.is_file():
    raise SystemExit(1)

counts = {}
with samples_path.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            task_id = json.loads(line)["task_id"]
            counts[task_id] = counts.get(task_id, 0) + 1

if not all(counts.get(task_id, 0) >= n_expected for task_id in expected_ids):
    completed = sum(1 for task_id in expected_ids if counts.get(task_id, 0) >= n_expected)
    print(
        f"[evalplus] resume {dataset}: partial ({completed}/{len(expected_ids)} tasks) "
        f"-> {samples_file}"
    )
    raise SystemExit(1)

result_path = Path(result_file)
if not result_path.is_file():
    print(f"[evalplus] samples complete but missing {result_file}; will run evaluate")
    raise SystemExit(1)

try:
    data = json.loads(result_path.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    raise SystemExit(1)

pass_at_k = data.get("pass_at_k", {})
plus = pass_at_k.get("plus", {})
base = pass_at_k.get("base", {})
if plus.get("pass@1") is None and base.get("pass@1") is None:
    raise SystemExit(1)

print(f"[evalplus] skip {dataset}: complete -> {result_file}")
raise SystemExit(0)
PY
}

should_skip_lcb() {
    if is_true "$FORCE_RERUN"; then
        return 1
    fi
    python3 - "$MODEL_OUTPUT_ROOT" "$N_SAMPLES" "$TEMPERATURE" <<'PY'
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
n_samples = int(sys.argv[2])
temperature = float(sys.argv[3])
lcb_dir = output_root / "lcb"
patterns = [
    f"Scenario.codegeneration_{n_samples}_{temperature}_eval_all.json",
    f"codegeneration_{n_samples}_{temperature}_eval_all.json",
]
for pattern in patterns:
    candidate = lcb_dir / pattern
    if candidate.is_file():
        print(f"[lcb] skip: complete -> {candidate}")
        raise SystemExit(0)
raise SystemExit(1)
PY
}

run_evalplus_one() {
    local dataset="$1"
    local devices="$2"
    local tp="$3"

    local samples_file
    samples_file="$(evalplus_samples_file "$dataset")"
    local result_file
    result_file="$(evalplus_result_file "$dataset")"

    export VLLM_CACHE_ROOT="${MODEL_OUTPUT_ROOT}/.vllm_cache/${dataset}"
    mkdir -p "$VLLM_CACHE_ROOT"
    export TORCHINDUCTOR_CACHE_DIR="${VLLM_CACHE_ROOT}/inductor"
    if is_true "$VLLM_ENFORCE_EAGER"; then
        export TORCHDYNAMO_DISABLE=1
    fi

    echo "[evalplus] dataset=${dataset} devices=${devices} tp=${tp} enforce_eager=${VLLM_ENFORCE_EAGER} batch_size=${BATCH_SIZE} prompt_batch_size=${CODEGEN_PROMPT_BATCH_SIZE}"

    local -a codegen_args=(
        --model "$MODEL_PATH"
        --dataset "$dataset"
        --backend vllm
        --trust_remote_code
        --tp "$tp"
        --root "$MODEL_OUTPUT_ROOT"
        --identifier "$SAMPLES_ID"
        --max-new-tokens "$MAX_TOKENS"
        --max-model-len "$VLLM_MAX_MODEL_LEN"
        --bs "$N_SAMPLES"
        --problem-batch-size "$CODEGEN_PROMPT_BATCH_SIZE"
    )
    if is_true "$VLLM_ENFORCE_EAGER"; then
        codegen_args+=(--enforce-eager)
    fi
    if is_true "$GREEDY"; then
        codegen_args+=(--greedy)
    else
        codegen_args+=(--temperature "$TEMPERATURE" --top-p "$TOP_P" --n-samples "$N_SAMPLES")
    fi
    CUDA_VISIBLE_DEVICES="$devices" python3 -u coding/evalplus/evalplus/codegen.py "${codegen_args[@]}"

    if [[ ! -f "$samples_file" ]]; then
        # Legacy layout: evalplus_results/{dataset}/{full_model_path}_vllm_temp_*.jsonl
        samples_file=$(find "${EVAL_OUTPUT_ROOT}/evalplus_results/${dataset}" -name "*${MODEL_ID}_vllm_temp_*.jsonl" ! -name "*.raw.jsonl" -type f 2>/dev/null | head -n 1)
    fi
    if [[ -z "${samples_file:-}" || ! -f "$samples_file" ]]; then
        echo "[evalplus] samples file not found for dataset=${dataset}" >&2
        return 1
    fi

    if ! python3 -u -m evalplus.evaluate \
        --dataset "$dataset" \
        --samples "$samples_file" \
        --output_file "$result_file" \
        --min-time-limit 10.0 \
        --gt-time-limit-factor 8.0; then
        echo "[evalplus] evaluate failed for dataset=${dataset}" >&2
        return 1
    fi

    if [[ ! -f "$result_file" ]]; then
        echo "[evalplus] missing result file after evaluate: ${result_file}" >&2
        return 1
    fi

    echo "[evalplus] dataset=${dataset} samples=${samples_file}"
    echo "[evalplus] dataset=${dataset} results=${result_file}"
}

restore_xtrace_if_needed() {
    # run_lcb disables xtrace; restore it for the caller (main scheduler).
    case "$-" in
        *x*) ;;
        *) [[ "${CODE_EVAL_XTRACE_WAS_ON:-0}" -eq 1 ]] && set -x ;;
    esac
}

cleanup_vllm_cache() {
    local cache_dir="${MODEL_OUTPUT_ROOT}/.vllm_cache"
    [[ -d "$cache_dir" ]] || return 0
    rm -rf "$cache_dir"
    echo "Removed vLLM cache: ${cache_dir}"
}

link_lcb_benchmark_data() {
    local lcb_lite_dir="${LCB_ROOT}/code_generation_lite"
    if [[ ! -d "$lcb_lite_dir" ]]; then
        echo "LiveCodeBench code_generation_lite dir not found: ${lcb_lite_dir}" >&2
        return 1
    fi
    for filename in test.jsonl test2.jsonl test3.jsonl test4.jsonl test5.jsonl test6.jsonl; do
        if [[ -f "${LCB_DATA_ROOT}/${filename}" ]]; then
            ln -sfn "${LCB_DATA_ROOT}/${filename}" "${lcb_lite_dir}/${filename}"
        fi
    done
}

run_lcb() {
    local lcb_devices="$1"
    local lcb_tp="$2"

    if [[ ! -f "${LCB_ROOT}/lcb_runner/runner/main.py" ]]; then
        echo "LiveCodeBench runner not found at ${LCB_ROOT}" >&2
        echo "Clone G-OPD code_eval/LiveCodeBench or set LCB_ROOT to a valid checkout." >&2
        return 1
    fi
    if ! link_lcb_benchmark_data; then
        return 1
    fi

    LCB_SOURCE_DIR="${LCB_OUTPUT_ROOT}"
    local assigned_gpus
    assigned_gpus="$(count_gpus_in_csv "$lcb_devices")"
    if [[ -n "${LCB_TP:-}" ]]; then
        lcb_tp="$LCB_TP"
    fi
    if [[ "$lcb_tp" -gt "$assigned_gpus" ]]; then
        lcb_tp="$assigned_gpus"
    fi
    if [[ "$lcb_tp" -lt 1 ]]; then
        echo "[lcb] invalid LCB_TP=${lcb_tp}" >&2
        return 1
    fi
    echo "[lcb] model=${LCB_MODEL} local_model_path=${MODEL_PATH} output_root=${LCB_OUTPUT_ROOT}"
    echo "[lcb] devices=${lcb_devices} tensor_parallel_size=${lcb_tp} enforce_eager=${VLLM_ENFORCE_EAGER} max_model_len=${VLLM_MAX_MODEL_LEN}"
    if is_true "$FORCE_RERUN"; then
        rm -rf "${LCB_OUTPUT_ROOT}"
        mkdir -p "${LCB_OUTPUT_ROOT}"
        echo "[lcb] FORCE_RERUN=true: cleared ${LCB_OUTPUT_ROOT}"
    fi
    local lcb_status=0
    local old_pwd="$PWD"
    local old_cuda="${CUDA_VISIBLE_DEVICES:-}"
    local old_pythonpath="${PYTHONPATH:-}"
    local old_vllm_cache="${VLLM_CACHE_ROOT:-}"
    local lcb_pid=""
    local -a lcb_runner_args

    cd "$LCB_ROOT"
    export VLLM_CACHE_ROOT="${MODEL_OUTPUT_ROOT}/.vllm_cache/lcb"
    mkdir -p "$VLLM_CACHE_ROOT"
    export TORCHINDUCTOR_CACHE_DIR="${VLLM_CACHE_ROOT}/inductor"
    if is_true "$VLLM_ENFORCE_EAGER"; then
        export TORCHDYNAMO_DISABLE=1
    else
        unset TORCHDYNAMO_DISABLE
    fi
    export LCB_LOCAL_MODEL_PATH="$MODEL_PATH"
    export LCB_OUTPUT_ROOT="${LCB_OUTPUT_ROOT}"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export PYTHONPATH="${LCB_ROOT}:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="$lcb_devices"
    lcb_runner_args=(
        --model "$LCB_MODEL"
        --local_model_path "$MODEL_PATH"
        --trust_remote_code
        --scenario codegeneration
        --release_version "$LCB_RELEASE_VERSION"
        --tensor_parallel_size "$lcb_tp"
        --max_model_len "$VLLM_MAX_MODEL_LEN"
        --use_cache
        --cache_batch_size "$CODEGEN_PROMPT_BATCH_SIZE"
        --n "$N_SAMPLES"
        --temperature "$TEMPERATURE"
        --max_tokens "$MAX_TOKENS"
        --lcb_output_root "$LCB_OUTPUT_ROOT"
        --custom_output_save_name "$MODEL_NAME"
        --top_p "$TOP_P"
        --timeout 60
        --num_process_evaluate "${LCB_NUM_PROCESS_EVALUATE}"
        --evaluate --continue_existing --continue_existing_with_eval
    )
    if is_true "$VLLM_ENFORCE_EAGER"; then
        lcb_runner_args+=(--enforce_eager)
    else
        lcb_runner_args+=(--no-enforce_eager)
    fi
    local lcb_runner_module="lcb_runner.runner.main"
    local lcb_gen_file="${LCB_OUTPUT_ROOT}/Scenario.codegeneration_${N_SAMPLES}_${TEMPERATURE}.json"
    if [[ -f "$lcb_gen_file" ]] && ! is_true "$FORCE_RERUN"; then
        local pending_gen
        pending_gen="$(python3 - "$lcb_gen_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(sum(1 for row in data if not row.get("output_list")))
PY
)"
        if [[ "${pending_gen:-1}" -eq 0 ]]; then
            lcb_runner_module="lcb_runner.evaluation.run_eval_only"
            export LCB_EVAL_POOL=fork
            echo "[lcb] generations complete -> eval-only module (no vLLM import, pool=fork)"
        fi
    fi
    local heartbeat_file="${MODEL_OUTPUT_ROOT}/lcb/.lcb_eval_heartbeat"
    mkdir -p "$(dirname "$heartbeat_file")"
    : >"$heartbeat_file"
    export LCB_EVAL_HEARTBEAT_FILE="$heartbeat_file"
    echo "[lcb] launching ${lcb_runner_module} (num_process_evaluate=${LCB_NUM_PROCESS_EVALUATE}) heartbeat=${heartbeat_file}"
    CODE_EVAL_XTRACE_WAS_ON=0
    case "$-" in *x*) CODE_EVAL_XTRACE_WAS_ON=1 ;; esac
    set +e
    set +x
    python3 -u -m "$lcb_runner_module" "${lcb_runner_args[@]}" &
    lcb_pid=$!
    echo "[lcb] runner pid=${lcb_pid}"
    # Quiet watchdog: no set -x noise; one beautified progress line per minute.
    (
        set +x
        while kill -0 "$lcb_pid" 2>/dev/null; do
            hb="$(tr -d '\n' <"$heartbeat_file" 2>/dev/null || true)"
            rss="$(ps -o rss= -p "$lcb_pid" 2>/dev/null | tr -d ' ' || true)"
            completed="${hb%%/*}"
            total="${hb##*/}"
            if [[ "$hb" == */* && "$completed" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ && "$total" -gt 0 ]]; then
                prog="$(
                    python3 -c "c,t=int('$completed'),int('$total'); p=100.0*c/t; f=int(16*c/t); b='#'*f+'.'*(16-f); w=len(str(t)); print(f'[lcb-eval] {c:>{w}}/{t} ({p:5.1f}%) [{b}]')"
                )"
            else
                prog="[lcb-eval] progress=${hb:-unknown}"
            fi
            line="[$(opd_log_ts)] ${prog} | rss_kb=${rss:-?}"
            printf '%s\n' "$line"
            sleep 60
        done
    ) &
    local watchdog_pid=$!
    wait "$lcb_pid"
    lcb_status=$?
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
    set -e
    restore_xtrace_if_needed
    local hb_final="unknown"
    if [[ -f "$heartbeat_file" ]]; then
        hb_final="$(tr -d '\n' <"$heartbeat_file" 2>/dev/null || true)"
    fi
    _code_eval_log_direct "[lcb] runner exited: code=${lcb_status} ($(describe_exit_code "$lcb_status")) progress=${hb_final} at $(opd_log_ts)"
    cd "$old_pwd"
    if [[ -n "$old_cuda" ]]; then
        export CUDA_VISIBLE_DEVICES="$old_cuda"
    else
        unset CUDA_VISIBLE_DEVICES
    fi
    export PYTHONPATH="$old_pythonpath"
    if [[ -n "$old_vllm_cache" ]]; then
        export VLLM_CACHE_ROOT="$old_vllm_cache"
    else
        unset VLLM_CACHE_ROOT
    fi
    unset LCB_EVAL_HEARTBEAT_FILE

    if [[ "$lcb_status" -ne 0 ]]; then
        _code_eval_log_direct "[lcb] failed: code=${lcb_status} ($(describe_exit_code "$lcb_status")) at $(opd_log_ts)"
        return "$lcb_status"
    fi
}

run_bench_evalplus() {
    local devices="$1"
    local dataset="$2"
    local tp="$3"
    if should_skip_evalplus "$dataset"; then
        update_summary
        return 0
    fi
    if ! run_evalplus_one "$dataset" "$devices" "$tp"; then
        return 1
    fi
    update_summary
    return 0
}

run_bench_lcb() {
    local devices="$1"
    local tp="$2"
    LCB_SOURCE_DIR="${LCB_OUTPUT_ROOT}"
    if should_skip_lcb; then
        update_summary
        return 0
    fi
    if ! run_lcb "$devices" "$tp"; then
        return 1
    fi
    update_summary
    return 0
}

run_bench() {
    local devices="$1"
    local task="$2"
    local tp="$3"
    case "$task" in
        lcb) run_bench_lcb "$devices" "$tp" ;;
        humaneval|mbpp) run_bench_evalplus "$devices" "$task" "$tp" ;;
        *)
            echo "Unknown bench task: ${task}" >&2
            return 1
            ;;
    esac
}

code_eval_vllm_port_for_bench() {
    # vLLM uses VLLM_PORT as the base for torch.distributed; tp>1 may consume
    # several consecutive ports. Keep 1000 apart so parallel EvalPlus workers
    # (humaneval + mbpp) do not collide (EADDRINUSE).
    local task="$1"
    case "$task" in
        humaneval) echo 30000 ;;
        mbpp) echo 31000 ;;
        lcb) echo 32000 ;;
        *)
            echo "Unknown bench task for VLLM_PORT: ${task}" >&2
            return 1
            ;;
    esac
}

launch_code_eval_bench_worker() {
    local gpu_str="$1"
    local task="$2"
    local tp="$3"
    local vllm_port
    vllm_port="$(code_eval_vllm_port_for_bench "$task")" || return 1
    (
        export CODE_EVAL_SINGLE_BENCH="$task"
        export CODE_EVAL_SINGLE_GPU="$gpu_str"
        export CODE_EVAL_SINGLE_TP="$tp"
        export CODE_EVAL_LOGGING_ACTIVE=1
        export VLLM_PORT="$vllm_port"
        echo "[code] worker bench=${task} devices=${gpu_str} tp=${tp} VLLM_PORT=${VLLM_PORT}"
        cd "$CODE_EVAL_ROOT"
        exec "$BASH" "${BASH_SOURCE[0]}"
    ) &
}

even_gpu_split() {
    local task="$1"
    local task_idx="$2"
    local num_tasks="$3"
    local base=$(( NUM_GPUS / num_tasks ))
    local remainder=$(( NUM_GPUS % num_tasks ))
    if [[ "$task_idx" -lt "$remainder" ]]; then
        echo $(( base + 1 ))
    else
        echo "$base"
    fi
}

schedule_parallel_tasks() {
    local failed=0
    local -a tasks=("$@")
    local num_tasks=${#tasks[@]}
    local -a pids=()
    local gpu_offset=0
    local task_idx task task_gpu_n task_gpus gpu_str tp

    if [[ "$num_tasks" -eq 0 ]]; then
        return 0
    fi

    for task_idx in "${!tasks[@]}"; do
        task="${tasks[$task_idx]}"
        task_gpu_n="$(even_gpu_split "$task" "$task_idx" "$num_tasks")"
        task_gpus=("${ALL_GPUS[@]:gpu_offset:task_gpu_n}")
        gpu_offset=$(( gpu_offset + task_gpu_n ))
        gpu_str=$(IFS=,; echo "${task_gpus[*]}")
        tp=${#task_gpus[@]}
        echo "[code] scheduling ${task} on GPUs: ${gpu_str} (tp=${tp})"
        launch_code_eval_bench_worker "$gpu_str" "$task" "$tp"
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    return "$failed"
}

schedule_serial_tasks() {
    local failed=0
    local task gpu_str="$ALLOCATED_GPU_CSV"
    local tp=$NUM_GPUS
    for task in "$@"; do
        echo "[code] scheduling ${task} serially on GPUs: ${gpu_str} (tp=${tp})"
        if ! run_bench "$gpu_str" "$task" "$tp"; then
            failed=1
        fi
    done
    return "$failed"
}

evalplus_phase_can_parallel() {
    local num_evalplus="$1"
    [[ "$num_evalplus" -le 1 ]] && return 0
    local base=$(( NUM_GPUS / num_evalplus ))
    if is_true "$FORCE_PARALLEL"; then
        [[ "$base" -ge 1 ]]
        return
    fi
    [[ "$base" -ge "$MIN_GPUS_PER_BENCH" ]]
}

schedule_pending_code_eval_benches() {
    local failed=0
    local -a evalplus_pending=()
    local lcb_pending=false
    local task

    for task in "${PENDING_TASKS[@]}"; do
        case "$task" in
            lcb) lcb_pending=true ;;
            humaneval|mbpp) evalplus_pending+=("$task") ;;
            *)
                echo "Unknown bench task: ${task}" >&2
                return 1
                ;;
        esac
    done

    if is_true "$FORCE_SERIAL"; then
        local -a serial_tasks=()
        local ordered_task
        for ordered_task in humaneval mbpp lcb; do
            for task in "${PENDING_TASKS[@]}"; do
                if [[ "$task" == "$ordered_task" ]]; then
                    serial_tasks+=("$task")
                fi
            done
        done
        echo "[code] FORCE_SERIAL=true: running pending benches one after another on all ${NUM_GPUS} GPU(s)"
        if ! schedule_serial_tasks "${serial_tasks[@]}"; then
            failed=1
        fi
        return "$failed"
    fi

    local num_evalplus=${#evalplus_pending[@]}
    if [[ "$num_evalplus" -gt 0 ]]; then
        echo "[code] phase 1/2: EvalPlus benches (${evalplus_pending[*]}) on ${NUM_GPUS} GPU(s)"
        if evalplus_phase_can_parallel "$num_evalplus"; then
            if ! schedule_parallel_tasks "${evalplus_pending[@]}"; then
                failed=1
            fi
        else
            echo "[code] phase 1: insufficient GPUs for parallel EvalPlus; running serially"
            if ! schedule_serial_tasks "${evalplus_pending[@]}"; then
                failed=1
            fi
        fi
    fi

    if [[ "$lcb_pending" == true ]]; then
        if [[ "$failed" -ne 0 ]]; then
            echo "[code] skipping phase 2 (LCB) because phase 1 failed" >&2
            return 1
        fi
        echo "[code] phase 2/2: LCB on all ${NUM_GPUS} GPU(s)"
        if ! run_bench "$ALLOCATED_GPU_CSV" "lcb" "$NUM_GPUS"; then
            failed=1
        fi
    fi

    return "$failed"
}

TASKS=()
is_true "$RUN_LCB" && TASKS+=(lcb)
is_true "$RUN_HUMANEVAL" && TASKS+=(humaneval)
is_true "$RUN_MBPP" && TASKS+=(mbpp)
NUM_TASKS=${#TASKS[@]}
ENABLED_BENCHES_CSV=$(IFS=,; echo "${TASKS[*]}")
LCB_SOURCE_DIR="${LCB_OUTPUT_ROOT}"

update_summary() {
    opd_refresh_log_datetime
    local -a summary_args=(
        --output_root "$MODEL_OUTPUT_ROOT"
        --model_name "$MODEL_NAME"
        --benches "$ENABLED_BENCHES_CSV"
        --n_samples "$N_SAMPLES"
        --temperature "$TEMPERATURE"
        --top_p "$TOP_P"
        --max_tokens "$MAX_TOKENS"
        --batch_size "$BATCH_SIZE"
    )
    if [[ -n "${LCB_SOURCE_DIR:-}" ]]; then
        summary_args+=(--lcb_source_dir "$LCB_SOURCE_DIR")
    fi
    python3 code_eval_summary.py "${summary_args[@]}"
}

bench_needs_gpu() {
    local task="$1"
    case "$task" in
        lcb)
            if should_skip_lcb; then
                return 1
            fi
            ;;
        humaneval|mbpp)
            if should_skip_evalplus "$task"; then
                return 1
            fi
            ;;
        *)
            echo "Unknown bench task: ${task}" >&2
            return 0
            ;;
    esac
    return 0
}

verify_enabled_benches_complete() {
    local task
    local missing=0
    for task in "${TASKS[@]}"; do
        case "$task" in
            humaneval|mbpp)
                if ! should_skip_evalplus "$task"; then
                    echo "[code_eval] incomplete bench: ${task} (missing samples or eval_results.json)" >&2
                    missing=1
                fi
                ;;
            lcb)
                if ! should_skip_lcb; then
                    echo "[code_eval] incomplete bench: lcb (missing eval output)" >&2
                    missing=1
                fi
                ;;
            *)
                echo "[code_eval] unknown enabled bench: ${task}" >&2
                missing=1
                ;;
        esac
    done
    return "$missing"
}

# Single-bench worker mode: isolated bash process avoids background function fall-through
# that can re-execute scheduler lines and trigger spurious syntax errors.
if [[ -n "${CODE_EVAL_SINGLE_BENCH:-}" ]]; then
    echo "[code] worker bench=${CODE_EVAL_SINGLE_BENCH} devices=${CODE_EVAL_SINGLE_GPU} tp=${CODE_EVAL_SINGLE_TP}"
    if ! run_bench "${CODE_EVAL_SINGLE_GPU}" "${CODE_EVAL_SINGLE_BENCH}" "${CODE_EVAL_SINGLE_TP}"; then
        exit 1
    fi
    exit 0
fi

PENDING_TASKS=()
SKIPPED_TASKS=()
for task in "${TASKS[@]}"; do
    if bench_needs_gpu "$task"; then
        PENDING_TASKS+=("$task")
    else
        SKIPPED_TASKS+=("$task")
    fi
done

NUM_PENDING=${#PENDING_TASKS[@]}
NUM_SKIPPED=${#SKIPPED_TASKS[@]}
echo "PENDING_BENCHES (${NUM_PENDING}): ${PENDING_TASKS[*]:-(none)}"
echo "SKIPPED_BENCHES (${NUM_SKIPPED}): ${SKIPPED_TASKS[*]:-(none)}"

if [[ "$NUM_PENDING" -eq 0 ]]; then
    echo "[code] all ${NUM_TASKS} enabled bench(es) already complete; skipping GPU eval."
    update_summary
    if ! verify_enabled_benches_complete; then
        echo "[code_eval] enabled bench outputs are incomplete" >&2
        exit 1
    fi
    exit 0
fi

echo "ENABLED_BENCHES=${TASKS[*]}"
echo "NUM_GPUS=${NUM_GPUS} NUM_PENDING=${NUM_PENDING} SCHEDULE=two-phase(EvalPlus->LCB) FORCE_SERIAL=${FORCE_SERIAL}"

failed=0
if ! schedule_pending_code_eval_benches; then
    failed=1
fi

cleanup_vllm_cache
update_summary

if [[ "$failed" -ne 0 ]]; then
    echo "[code_eval] one or more code eval subprocesses failed"
    exit 1
fi
if ! verify_enabled_benches_complete; then
    echo "[code_eval] enabled bench outputs are incomplete" >&2
    exit 1
fi
echo "[code_eval] model ${MODEL_NAME} code eval done at $(opd_log_ts)"
exit 0
