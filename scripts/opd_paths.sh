#!/bin/bash
# Shared filesystem layout for OPD scripts.
# Override any OPD_* variable when mounts differ across environments.

if [[ -z "${REPO_ROOT:-}" ]]; then
    _OPD_PATHS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${_OPD_PATHS_DIR}/.." && pwd)"
    unset _OPD_PATHS_DIR
fi

export OPD_DATA_ROOT=${OPD_DATA_ROOT:-$REPO_ROOT/data/opd}
export OPD_MODEL_ROOT=${OPD_MODEL_ROOT:-$REPO_ROOT/models}
export OPD_PUBLIC_MODEL_ROOT=${OPD_PUBLIC_MODEL_ROOT:-$OPD_MODEL_ROOT/public}
export OPD_TEACHER_MODEL_ROOT=${OPD_TEACHER_MODEL_ROOT:-$OPD_PUBLIC_MODEL_ROOT}
export OPD_A2OPD_MODEL_ROOT=${OPD_A2OPD_MODEL_ROOT:-$OPD_MODEL_ROOT/a2opd}
export OPD_MULTI_OPD_MODEL_ROOT=${OPD_MULTI_OPD_MODEL_ROOT:-$OPD_MODEL_ROOT/multi_opd}
export OPD_A2OPD_RESULT_ROOT=${OPD_A2OPD_RESULT_ROOT:-$REPO_ROOT/data/a2opd/results}
export OPD_EXP_RESULT_ROOT=${OPD_EXP_RESULT_ROOT:-$OPD_A2OPD_RESULT_ROOT/benchmark_eval}
export OPD_DISTRIBUTION_KL_RESULT_ROOT=${OPD_DISTRIBUTION_KL_RESULT_ROOT:-$OPD_A2OPD_RESULT_ROOT/a2opd_distribution_kl}

# G-OPD training data (override OPD_PUBLIC_DATA_ROOT to your local copy).
export OPD_PUBLIC_DATA_ROOT=${OPD_PUBLIC_DATA_ROOT:-$OPD_DATA_ROOT/training}
export OPD_MATH_TRAIN_FILE=${OPD_MATH_TRAIN_FILE:-$OPD_PUBLIC_DATA_ROOT/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet}
export OPD_MATH_TEST_FILE=${OPD_MATH_TEST_FILE:-$OPD_PUBLIC_DATA_ROOT/G-OPD-Training-Data/MathTestTotal/test.parquet}
export OPD_CODE_TRAIN_FILE=${OPD_CODE_TRAIN_FILE:-$OPD_PUBLIC_DATA_ROOT/G-OPD-Training-Data/Eurus/code_train.parquet}
export OPD_CODE_VAL_FILE=${OPD_CODE_VAL_FILE:-$OPD_PUBLIC_DATA_ROOT/G-OPD-Training-Data/Eurus/code_validation.parquet}

# Vendored verl (see README); not always pip-installed in conda env.
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"

# Log timestamps default to China Standard Time (UTC+8), without a timezone suffix.
export OPD_LOG_TZ=${OPD_LOG_TZ:-Asia/Shanghai}
export OPD_LOG_TS_FMT=${OPD_LOG_TS_FMT:-%Y-%m-%d %H:%M:%S}

_opd_resolve_log_tz() {
    local tz="$OPD_LOG_TZ"
    # Minimal images may lack zoneinfo; Asia/Shanghai then falls back to UTC-8.
    if [[ "$tz" == "Asia/Shanghai" ]] && [[ "$(TZ="$tz" date +%z 2>/dev/null)" != "+0800" ]]; then
        tz='UTC-8'
    fi
    echo "$tz"
}

if ! declare -f opd_log_date >/dev/null 2>&1; then
    opd_log_date() {
        TZ="$(_opd_resolve_log_tz)" date "$@"
    }
fi

if ! declare -f opd_log_ts >/dev/null 2>&1; then
    opd_log_ts() {
        opd_log_date "+${OPD_LOG_TS_FMT}"
    }
fi

# Child processes (vLLM, Python logging) use the same local timezone.
export TZ="$(_opd_resolve_log_tz)"

opd_refresh_log_datetime() {
    export OPD_LOG_DATETIME="$(opd_log_ts)"
}
