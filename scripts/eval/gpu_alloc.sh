# Shared GPU discovery for K8s, containers, and bare-metal eval jobs.
# The scheduler is expected to expose the assigned devices through
# CUDA_VISIBLE_DEVICES. If it does not, all GPUs visible to nvidia-smi are used.

resolve_allocated_gpu_csv() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "${CUDA_VISIBLE_DEVICES// /}"
        return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local count ids=() i
        count=$(nvidia-smi -L 2>/dev/null | wc -l)
        if [[ "$count" -gt 0 ]]; then
            for ((i = 0; i < count; i++)); do
                ids+=("$i")
            done
            local IFS=,
            echo "${ids[*]}"
            return 0
        fi
    fi

    echo "No GPUs are visible. Set CUDA_VISIBLE_DEVICES explicitly." >&2
    return 1
}

count_gpus_in_csv() {
    local csv="$1"
    local -a gpus=()
    IFS=',' read -r -a gpus <<< "$csv"
    echo "${#gpus[@]}"
}
