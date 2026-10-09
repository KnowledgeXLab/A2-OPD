#!/bin/bash
# Resolve A²OPD experiment checkpoints for offline math/code eval.
# Expects A2OPD_EXP_PATH and optional A2OPD_CKPT_SLOT (default: last).

_a2opd_has_model_weights() {
    local path="$1"
    compgen -G "${path}/*.safetensors" >/dev/null 2>&1 \
        || compgen -G "${path}/*.bin" >/dev/null 2>&1 \
        || [[ -f "${path}/model.safetensors.index.json" ]]
}

_a2opd_parse_student_model_name() {
    local exp_name="$1"
    if [[ "$exp_name" =~ ^S_(.+)-T_ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

_a2opd_restore_student_tokenizer() {
    local merged_hf="$1"
    local student_name="$2"
    local student_root="${OPD_PUBLIC_MODEL_ROOT}/${student_name}"
    local tok_file

    if [[ ! -d "$student_root" ]]; then
        echo "Warning: student base not found for tokenizer restore: ${student_root}" >&2
        return 0
    fi

    for tok_file in \
        tokenizer.json tokenizer_config.json vocab.json merges.txt \
        special_tokens_map.json added_tokens.json chat_template.jinja; do
        if [[ -f "${student_root}/${tok_file}" ]]; then
            cp -f "${student_root}/${tok_file}" "${merged_hf}/${tok_file}"
        fi
    done
}

resolve_a2opd_eval_model() {
    local exp_path="${A2OPD_EXP_PATH:-}"
    local ckpt_slot="${A2OPD_CKPT_SLOT:-last}"
    local merged_hf actor student_name

    if [[ -z "$exp_path" ]]; then
        echo "A2OPD_EXP_PATH is empty." >&2
        return 1
    fi

    if [[ ! -d "$exp_path" ]]; then
        echo "A2OPD experiment path not found: ${exp_path}" >&2
        return 1
    fi

    exp_path="$(readlink -f "$exp_path")"
    merged_hf="${exp_path}/${ckpt_slot}/merged_hf"
    actor="${exp_path}/${ckpt_slot}/actor"

    if [[ -d "$merged_hf" ]] && _a2opd_has_model_weights "$merged_hf"; then
        MODEL_PATH="$merged_hf"
        echo "[a2opd] using existing merged_hf: ${MODEL_PATH}"
    elif [[ -d "$actor" && -f "${actor}/fsdp_config.json" ]]; then
        echo "[a2opd] merging FSDP actor -> ${merged_hf}"
        mkdir -p "$merged_hf"
        (
            cd "${REPO_ROOT}/verl"
            export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"
            python3 -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "${actor}" \
                --target_dir "${merged_hf}" \
                --use_cpu_initialization
        )
        if ! _a2opd_has_model_weights "$merged_hf"; then
            echo "Merge finished but no HF weights found at ${merged_hf}" >&2
            return 1
        fi
        if student_name="$(_a2opd_parse_student_model_name "$(basename "$exp_path")")"; then
            _a2opd_restore_student_tokenizer "$merged_hf" "$student_name"
        fi
        MODEL_PATH="$merged_hf"
    else
        echo "No merged_hf or FSDP actor checkpoint under ${exp_path}/${ckpt_slot}" >&2
        return 1
    fi

    MODEL_NAME="${MODEL_NAME:-$(basename "$exp_path")}"
    export MODEL_PATH MODEL_NAME
    export A2OPD_EXP_PATH="$exp_path"
    export A2OPD_CKPT_SLOT="$ckpt_slot"
    return 0
}
