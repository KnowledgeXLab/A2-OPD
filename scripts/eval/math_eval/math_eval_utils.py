"""Shared helpers for math eval scripts."""

from __future__ import annotations

import json
from pathlib import Path

from parquet_io import read_parquet_flexible


def _normalize_prompt_messages(prompt):
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, list) and prompt:
        if isinstance(prompt[0], dict):
            return prompt
        return [{"role": "user", "content": str(prompt[0])}]
    return [{"role": "user", "content": str(prompt)}]


def _normalize_reward_model(reward_model):
    if hasattr(reward_model, "tolist"):
        reward_model = reward_model.tolist()
    if isinstance(reward_model, dict):
        return str(reward_model.get("ground_truth", ""))
    return str(reward_model)


def load_input_data(input_file, val_metric_source=None, begin_idx=-1, end_idx=-1):
    if input_file.endswith(".parquet"):
        df = read_parquet_flexible(input_file, "auto")
        if val_metric_source:
            df = df[df["data_source"] == val_metric_source]
        input_data = []
        for _, row in df.iterrows():
            messages = _normalize_prompt_messages(row["prompt"])
            problem = messages[0]["content"]
            answer = _normalize_reward_model(row["reward_model"])
            item = {
                "problem": problem,
                "answer": answer,
                "data_source": row.get("data_source"),
                "id": row.get("id"),
            }
            input_data.append(item)
    else:
        with open(input_file, "r", encoding="utf-8") as file:
            input_data = [json.loads(line) for line in file if line.strip()]

    if begin_idx >= 0 and end_idx >= 0:
        input_data = input_data[begin_idx:end_idx]
    return input_data


def sample_key(item: dict, index: int | None = None) -> str:
    sample_id = item.get("id")
    if sample_id is not None:
        return str(sample_id)
    if index is not None:
        return f"__idx_{index}__"
    raise ValueError(f"Cannot derive sample key for item: {item!r}")


def is_valid_result_row(row: dict) -> bool:
    accs = row.get("acc_list") or []
    responses = row.get("responses") or []
    return len(accs) > 0 and len(responses) > 0


def load_saved_results_by_key(result_file: str | Path) -> tuple[dict[str, dict], int]:
    path = Path(result_file)
    if not path.is_file():
        return {}, 0

    results: dict[str, dict] = {}
    duplicate_lines = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") is not None:
                key = str(row["id"])
            else:
                key = f"__legacy_line_{line_no}__"
            if key in results:
                duplicate_lines += 1
            results[key] = row
    return results, duplicate_lines


def rewrite_results_deduped(
    result_file: str | Path,
    input_data: list[dict],
    saved_by_key: dict[str, dict],
) -> None:
    path = Path(result_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered: list[dict] = []
    seen: set[str] = set()
    for index, item in enumerate(input_data):
        key = sample_key(item, index)
        if key not in saved_by_key:
            continue
        ordered.append(saved_by_key[key])
        seen.add(key)

    for key, row in saved_by_key.items():
        if key not in seen:
            ordered.append(row)

    with path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_completed_by_id(
    result_file: str | Path,
    input_file: str,
    val_metric_source: str | None = None,
) -> int:
    input_data = load_input_data(input_file, val_metric_source=val_metric_source)
    saved_by_key, _ = load_saved_results_by_key(result_file)
    completed = 0
    for index, item in enumerate(input_data):
        key = sample_key(item, index)
        if key in saved_by_key and is_valid_result_row(saved_by_key[key]):
            completed += 1
    return completed


def prepare_resume(
    input_data: list[dict],
    result_file: str | Path,
) -> tuple[list[dict], int, int, bool]:
    """Return pending items, completed count, total count, whether file was rewritten."""
    saved_by_key, duplicate_lines = load_saved_results_by_key(result_file)
    rewritten = False
    if duplicate_lines > 0:
        rewrite_results_deduped(result_file, input_data, saved_by_key)
        rewritten = True
        print(
            f"[eval_math] deduplicated {duplicate_lines} duplicate row(s) in {result_file}"
        )

    pending: list[dict] = []
    completed = 0
    for index, item in enumerate(input_data):
        key = sample_key(item, index)
        if key in saved_by_key and is_valid_result_row(saved_by_key[key]):
            completed += 1
        else:
            pending.append(item)

    return pending, completed, len(input_data), rewritten


def bench_output_dir(name: str) -> str:
    return name.lower()


def count_expected_samples(input_file: str, val_metric_source: str | None = None) -> int:
    return len(load_input_data(input_file, val_metric_source=val_metric_source))


def count_result_lines(result_file: str | Path) -> int:
    path = Path(result_file)
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def is_result_complete(
    result_file: str | Path,
    input_file: str,
    val_metric_source: str | None = None,
) -> bool:
    expected = count_expected_samples(input_file, val_metric_source)
    if expected <= 0:
        return False
    completed = count_completed_by_id(result_file, input_file, val_metric_source)
    return completed >= expected


def result_file_for(output_root: str | Path, dataset: str, model_name: str) -> Path:
    return Path(output_root) / bench_output_dir(dataset) / f"{model_name}.jsonl"
