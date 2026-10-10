#!/bin/bash
# Shared filesystem layout for A2-OPD scripts.
# Override any A2OPD_* variable when mounts differ across environments.

if [[ -z "${REPO_ROOT:-}" ]]; then
    _OPD_PATHS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${_OPD_PATHS_DIR}/.." && pwd)"
    unset _OPD_PATHS_DIR
fi

export A2OPD_DATA_ROOT=${A2OPD_DATA_ROOT:-$REPO_ROOT/data}
export A2OPD_MODEL_ROOT=${A2OPD_MODEL_ROOT:-$REPO_ROOT/models}
export A2OPD_CKPT_ROOT=${A2OPD_CKPT_ROOT:-$REPO_ROOT/checkpoints}
export A2OPD_RESULT_ROOT=${A2OPD_RESULT_ROOT:-$REPO_ROOT/results}
export A2OPD_EXP_RESULT_ROOT=${A2OPD_EXP_RESULT_ROOT:-$A2OPD_RESULT_ROOT/benchmark_eval}
export A2OPD_DISTRIBUTION_KL_RESULT_ROOT=${A2OPD_DISTRIBUTION_KL_RESULT_ROOT:-$A2OPD_RESULT_ROOT/distribution_kl}

export A2OPD_MATH_TRAIN_FILE=${A2OPD_MATH_TRAIN_FILE:-$A2OPD_DATA_ROOT/math/train.parquet}
export A2OPD_MATH_TEST_FILE=${A2OPD_MATH_TEST_FILE:-$A2OPD_DATA_ROOT/math/test.parquet}
export A2OPD_CODE_TRAIN_FILE=${A2OPD_CODE_TRAIN_FILE:-$A2OPD_DATA_ROOT/code/train.parquet}
export A2OPD_CODE_VAL_FILE=${A2OPD_CODE_VAL_FILE:-$A2OPD_DATA_ROOT/code/validation.parquet}

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
