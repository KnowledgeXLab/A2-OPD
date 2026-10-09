"""Parallel code execution reward for training and validation (prime_code).

Use with ``reward_manager=batch`` and ``reward_func_batched`` via
``custom_reward_function`` / ``val_custom_reward_function``.
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Sequence

_CODE_SOURCES = frozenset(
    {
        "codecontests",
        "apps",
        "codeforces",
        "taco",
        "code",
    }
)

_DEFAULT_CODE_EVAL_WORKERS = 16


def _is_code(data_source: str, ability: str | None) -> bool:
    ds = (data_source or "").strip().lower()
    if ds in _CODE_SOURCES:
        return True
    return (ability or "").strip().lower() == "code"


def _ground_truth_str(ground_truth: Any) -> str:
    if ground_truth is None:
        return ""
    if isinstance(ground_truth, str):
        return ground_truth
    return str(ground_truth)


def _extract_code_solution(solution_str: str) -> str:
    if "```python" in solution_str:
        return solution_str.split("```python")[-1].split("```")[0]
    if "```" in solution_str:
        parts = solution_str.split("```")
        if len(parts) >= 3:
            return parts[-2]
    return solution_str


def _normalize_val_result(
    result: dict[str, Any],
    *,
    ground_truth: Any,
    pred: str = "",
) -> dict[str, Any]:
    score = float(result.get("score", 0.0))
    out = dict(result)
    out["score"] = score
    out["format_score"] = float(result.get("format_score", 1.0))
    out["acc"] = bool(result.get("acc", score >= 1.0 - 1e-6))
    out["pass_rate"] = float(result.get("pass_rate", score))
    out["extracted_gt"] = result.get("extracted_gt", _ground_truth_str(ground_truth))
    out["pred"] = result.get("pred", pred)
    return out


def _ability_from_extra(extra_info: Any) -> str | None:
    if extra_info is None:
        return None
    if isinstance(extra_info, dict):
        val = extra_info.get("ability")
        return str(val) if val is not None else None
    return None


def _failed_val_result(ground_truth: Any, *, pred: str = "") -> dict[str, Any]:
    return _normalize_val_result(
        {"score": 0.0, "format_score": 0.0, "acc": False, "pass_rate": 0.0},
        ground_truth=ground_truth,
        pred=pred,
    )


def _score_code(solution_str: str, ground_truth: Any, *, code_timeout: int) -> dict[str, Any]:
    from verl.utils.reward_score import prime_code

    del code_timeout
    pred = _extract_code_solution(solution_str)
    success, _metadata = prime_code.compute_score(
        solution_str,
        ground_truth,
        continuous=False,
    )
    if isinstance(success, bool):
        score = 1.0 if success else 0.0
    else:
        score = float(success)
    acc = score >= 1.0 - 1e-6
    return _normalize_val_result(
        {
            "score": score,
            "format_score": 1.0,
            "acc": acc,
            "pass_rate": score,
        },
        ground_truth=ground_truth,
        pred=pred,
    )


def _parallel_score_code_task(
    payload: tuple[int, str, Any, int],
) -> tuple[int, dict[str, Any]]:
    idx, solution_str, ground_truth, code_timeout = payload
    try:
        return idx, _score_code(solution_str, ground_truth, code_timeout=code_timeout)
    except Exception:
        print(f"[code_eval_reward] parallel code eval error idx={idx}", flush=True)
        traceback.print_exc()
        return idx, _failed_val_result(ground_truth)


def _as_batch_list(items: Any, n: int, *, name: str) -> list[Any]:
    if items is None:
        raise ValueError(f"code_eval_reward: {name} is None")
    try:
        length = len(items)
    except TypeError as e:
        raise TypeError(f"code_eval_reward: {name} has no len()") from e
    if length != n:
        raise ValueError(
            f"code_eval_reward batch size mismatch for {name}: len={length}, expected n={n}"
        )
    return list(items)


def _resolve_code_eval_workers(code_eval_workers: int | None) -> int:
    if code_eval_workers is not None and int(code_eval_workers) > 0:
        return int(code_eval_workers)
    cpu = os.cpu_count() or 8
    return max(1, min(_DEFAULT_CODE_EVAL_WORKERS, cpu))


def reward_func(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    sandbox_fusion_url: str | None = None,
    concurrent_semaphore: Any = None,
    memory_limit_mb: int | None = None,
    code_timeout: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    del sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, kwargs
    ability = _ability_from_extra(extra_info)

    try:
        if _is_code(data_source, ability):
            return _score_code(solution_str, ground_truth, code_timeout=int(code_timeout))
        raise NotImplementedError(
            f"code_eval_reward: unsupported data_source={data_source!r} ability={ability!r}; "
            f"expected code sources in {sorted(_CODE_SOURCES)}"
        )
    except Exception:
        print(
            f"[code_eval_reward] error data_source={data_source!r} ability={ability!r}",
            flush=True,
        )
        traceback.print_exc()
        return _failed_val_result(ground_truth)


def reward_func_batched(
    data_sources: Sequence[Any],
    solution_strs: Sequence[str],
    ground_truths: Sequence[Any],
    extra_infos: Sequence[Any] | None = None,
    code_timeout: int = 10,
    code_eval_workers: int | None = None,
    sandbox_fusion_url: str | None = None,
    concurrent_semaphore: Any = None,
    memory_limit_mb: int | None = None,
    val_timing_sink: dict[str, float] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    del sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, kwargs
    t_reward_total = time.perf_counter()
    solution_list = list(solution_strs)
    n = len(solution_list)
    sources = _as_batch_list(data_sources, n, name="data_sources")
    truths = _as_batch_list(ground_truths, n, name="ground_truths")
    if extra_infos is None:
        extras: list[Any] = [None] * n
    else:
        extras = _as_batch_list(extra_infos, n, name="extra_infos")

    results: list[dict[str, Any] | None] = [None] * n
    code_jobs: list[tuple[int, str, Any, int]] = []
    code_count = 0

    for i in range(n):
        data_source = str(sources[i])
        ability = _ability_from_extra(extras[i])
        solution_str = solution_list[i]
        ground_truth = truths[i]

        try:
            if _is_code(data_source, ability):
                code_jobs.append((i, solution_str, ground_truth, int(code_timeout)))
            else:
                raise NotImplementedError(
                    f"code_eval_reward: unsupported data_source={data_source!r} ability={ability!r}"
                )
        except Exception:
            print(
                f"[code_eval_reward] error data_source={data_source!r} ability={ability!r}",
                flush=True,
            )
            traceback.print_exc()
            results[i] = _failed_val_result(ground_truth)

    code_pool_s = 0.0
    if code_jobs:
        code_count = len(code_jobs)
        workers = _resolve_code_eval_workers(code_eval_workers)
        print(
            f"[code_eval_reward] parallel code eval: {code_count} samples, {workers} workers",
            flush=True,
        )
        t_code = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_parallel_score_code_task, job) for job in code_jobs]
            for fut in as_completed(futures):
                idx, scored = fut.result()
                results[idx] = scored
        code_pool_s = time.perf_counter() - t_code

    reward_total_s = time.perf_counter() - t_reward_total
    if isinstance(val_timing_sink, dict):
        val_timing_sink["reward/code_pool_s"] = val_timing_sink.get("reward/code_pool_s", 0.0) + code_pool_s
        val_timing_sink["reward/total_s"] = val_timing_sink.get("reward/total_s", 0.0) + reward_total_s
        val_timing_sink["reward/num_code"] = val_timing_sink.get("reward/num_code", 0.0) + float(code_count)
        val_timing_sink["reward/num_samples"] = val_timing_sink.get("reward/num_samples", 0.0) + float(n)

    print(
        "[code_eval_reward timing] "
        f"n={n} code={code_count} code_pool={code_pool_s:.2f}s total={reward_total_s:.2f}s",
        flush=True,
    )

    out: list[dict[str, Any]] = []
    for i, item in enumerate(results):
        if item is None:
            out.append(_failed_val_result(truths[i]))
        else:
            out.append(item)
    return out
