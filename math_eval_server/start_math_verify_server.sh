#!/usr/bin/env bash
# Source this file from training scripts to auto-start the math verify HTTP server.
#
# Usage:
#   source "${REPO_ROOT}/math_eval_server/start_math_verify_server.sh"
#   start_math_verify_server || exit 1
#   trap stop_math_verify_server EXIT

_math_verify_server_pid=""
_math_verify_log_file=""

_math_verify_resolve_log_file() {
    local repo_root="$1"
    local log_dir="${MATH_VERIFY_LOG_DIR:-${repo_root}/logs/w2s_math}"
    mkdir -p "${log_dir}"
    if [[ -n "${MATH_VERIFY_LOG_FILE:-}" ]]; then
        echo "${MATH_VERIFY_LOG_FILE}"
        return 0
    fi
    echo "${log_dir}/math_verify_$$.log"
}

_math_verify_health_ok() {
    local url="$1"
    local print_err="${2:-0}"
    MATH_VERIFY_HEALTH_URL="$url" PRINT_ERR="$print_err" python3 -c '
import os
import sys
import urllib.request

try:
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    with opener.open(os.environ["MATH_VERIFY_HEALTH_URL"], timeout=3) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception as e:
    if os.environ.get("PRINT_ERR") == "1":
        print(f"Health check error: {e}", file=sys.stderr)
    sys.exit(1)
'
}

_math_verify_fetch_health() {
    local url="$1"
    MATH_VERIFY_HEALTH_URL="$url" python3 -c '
import os
import urllib.request

try:
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    with opener.open(os.environ["MATH_VERIFY_HEALTH_URL"], timeout=3) as resp:
        print(resp.read().decode())
except Exception:
    pass
' 2>/dev/null || true
}

start_math_verify_server() {
    local repo_root="${1:-${REPO_ROOT:-}}"
    if [[ -z "${repo_root}" ]]; then
        echo "ERROR: REPO_ROOT is not set" >&2
        return 1
    fi

    if [[ -n "${MATH_VERIFY_SERVER_URL:-}" ]]; then
        local health_url="${MATH_VERIFY_SERVER_URL%/verify}/health"
        if _math_verify_health_ok "${health_url}"; then
            echo "[math-verify] Reusing existing server: ${MATH_VERIFY_SERVER_URL}"
            return 0
        fi
        echo "[math-verify] MATH_VERIFY_SERVER_URL unreachable (${MATH_VERIFY_SERVER_URL}), starting local server ..."
        unset MATH_VERIFY_SERVER_URL
    fi

    export MATH_VERIFY_HOST="${MATH_VERIFY_HOST:-0.0.0.0}"
    export MATH_VERIFY_PORT="${MATH_VERIFY_PORT:-7683}"

    local node_ip
    node_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [[ -z "$node_ip" ]]; then
        node_ip="127.0.0.1"
    fi

    export MATH_VERIFY_SERVER_URL="http://${node_ip}:${MATH_VERIFY_PORT}/verify"
    local health_url="http://${node_ip}:${MATH_VERIFY_PORT}/health"

    _math_verify_log_file="$(_math_verify_resolve_log_file "${repo_root}")"
    export MATH_VERIFY_LOG_FILE="${_math_verify_log_file}"

    echo "[math-verify] Starting server on port ${MATH_VERIFY_PORT} (log: ${_math_verify_log_file}) ..."
    (
        cd "${repo_root}/math_eval_server" || exit 1
        exec python3 -u reward_verify_server.py >> "${_math_verify_log_file}" 2>&1
    ) &
    _math_verify_server_pid=$!

    local attempt
    for attempt in $(seq 1 120); do
        if _math_verify_health_ok "${health_url}" "0"; then
            echo "[math-verify] Server ready: ${MATH_VERIFY_SERVER_URL}"
            _math_verify_fetch_health "${health_url}"
            echo
            return 0
        fi
        if ! kill -0 "${_math_verify_server_pid}" 2>/dev/null; then
            echo "ERROR: math verify server exited before becoming healthy (see ${_math_verify_log_file})" >&2
            return 1
        fi
        sleep 1
    done

    echo "ERROR: math verify server failed health check: ${health_url} (see ${_math_verify_log_file})" >&2
    _math_verify_health_ok "${health_url}" "1"
    stop_math_verify_server
    return 1
}

stop_math_verify_server() {
    if [[ -n "${_math_verify_server_pid}" ]] && kill -0 "${_math_verify_server_pid}" 2>/dev/null; then
        echo "[math-verify] Stopping server (pid=${_math_verify_server_pid}) ..."
        kill "${_math_verify_server_pid}" 2>/dev/null || true
        wait "${_math_verify_server_pid}" 2>/dev/null || true
    fi
    _math_verify_server_pid=""
}
