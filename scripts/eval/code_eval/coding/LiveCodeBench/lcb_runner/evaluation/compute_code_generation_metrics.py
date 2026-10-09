# borrowed and extended from
# https://github.com/Naman-ntc/codescratch/blob/main/evaluation/bigcode-evaluation-harness/lm_eval/tasks/custom_metrics/apps_custom_metrics/utils.py

import os
import sys
import traceback

sys.set_int_max_str_digits(50000)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
from tqdm import tqdm

from lcb_runner.evaluation.testing_util import Capturing, run_test
from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results
from lcb_runner.utils.log_progress import is_interactive, log_progress


def _install_unraisable_noise_filter() -> None:
    """Suppress known-harmless multiprocess ResourceTracker teardown noise on Py3.12."""
    if getattr(sys, "_lcb_unraisable_hook_installed", False):
        return

    def _hook(unraisable):
        exc = unraisable.exc_value
        obj = unraisable.object
        if (
            unraisable.exc_type is AttributeError
            and exc is not None
            and "_recursion_count" in str(exc)
            and obj is not None
            and type(obj).__name__ == "ResourceTracker"
        ):
            return
        sys.__unraisablehook__(unraisable)

    sys.unraisablehook = _hook
    sys._lcb_unraisable_hook_installed = True


def _silence_multiprocess_resource_tracker() -> None:
    try:
        import multiprocess.resource_tracker as rt
    except Exception:
        return
    if getattr(rt.ResourceTracker, "_lcb_patched", False):
        return

    def _noop(self, *args, **kwargs):
        return None

    rt.ResourceTracker._stop = _noop  # type: ignore[method-assign]
    rt.ResourceTracker.__del__ = _noop  # type: ignore[method-assign]
    rt.ResourceTracker._lcb_patched = True


def _pool_worker_init() -> None:
    _install_unraisable_noise_filter()
    _silence_multiprocess_resource_tracker()


_install_unraisable_noise_filter()


def _eval_pool_context():
    """fork when parent is lightweight; spawn only if vLLM/torch already loaded."""
    forced = os.environ.get("LCB_EVAL_POOL", "").strip().lower()
    if forced in ("fork", "spawn"):
        return multiprocessing.get_context(forced)
    if "torch" not in sys.modules and "vllm" not in sys.modules:
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _temp_run(sample, generation, debug, timeout, queue):
    # Suppress stray print() from generated solutions; stdio grading still
    # captures output via an inner Capturing context in grade_stdio().
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        with Capturing():
            res, metadata = run_test(
                sample, test=generation, debug=debug, timeout=timeout
            )
        queue.put((res, metadata))
    except Exception as e:
        queue.put(
            (
                [-4],
                {
                    "error_code": -4,
                    "error_message": f"sandbox exception: {repr(e)}",
                },
            )
        )


def _normalize_result_list(curr_res):
    fixed = []
    for e in curr_res:
        if isinstance(e, np.ndarray):
            e = e.item(0)
        if isinstance(e, np.bool_):
            e = bool(e)
        fixed.append(e)
    return fixed


def _global_timeout_for_sample(sample, timeout: int) -> int:
    n_inputs = len(json.loads(sample["input_output"])["inputs"])
    return (timeout + 1) * n_inputs + 5


def _eval_single_sample(task):
    """Pool worker: one sandbox Process per sample (EvalPlus-style, no nested pool)."""
    index, sample, generation, debug, timeout = task
    curr_res = [-2]
    curr_metadata = {
        "error_code": -5,
        "error_message": "TestRunnerError",
    }
    try:
        curr_res, curr_metadata = check_correctness(
            sample, generation, timeout=timeout, debug=debug
        )
        curr_res = _normalize_result_list(curr_res)
    except Exception as e:
        curr_metadata = {
            "error": repr(e),
            "error_code": -5,
            "error_message": "TestRunnerError",
        }
    return index, [curr_res], [curr_metadata]


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.

    Uses fork + Queue (not Manager, not spawn):
    - Manager leaked server processes and caused mid-run SIGKILL.
    - spawn re-imports torch via `python -m` every sample (pynvml spam +
      multiprocess ResourceTracker noise) and is much slower.
    - fork + a high-enough RLIMIT_AS (default 16GB) avoids the 4GB MemoryError
      while still capping pathological allocations.
    """
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_temp_run,
        args=(sample, generation, debug, timeout, queue),
    )
    p.start()
    p.join(timeout=_global_timeout_for_sample(sample, timeout))
    if p.is_alive():
        p.kill()
        p.join(5)

    result = None
    metadata = None
    try:
        result, metadata = queue.get_nowait()
    except Exception:
        pass
    try:
        queue.close()
        queue.join_thread()
    except Exception:
        pass

    if result is None:
        in_outs = json.loads(sample["input_output"])
        reason = (
            "timeout"
            if p.exitcode in (0, None)
            else f"worker exitcode={p.exitcode}"
        )
        print(
            f"[lcb-eval] sample eval produced no result ({reason})",
            flush=True,
        )
        return (
            [-1 for _ in range(len(in_outs["inputs"]))],
            {"error_code": -1, "error_message": f"no result ({reason})"},
        )
    return result, metadata or {"error_code": -1, "error_message": "no metadata"}


def _write_eval_heartbeat(completed: int, total: int) -> None:
    """Best-effort progress file so SIGKILL still leaves a breadcrumb."""
    path = os.environ.get("LCB_EVAL_HEARTBEAT_FILE")
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{completed}/{total}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    """Run unit tests on generated code (EvalPlus-style ProcessPool + sandbox Process)."""

    tasks = [
        (index, samples_list[index], generations_list[index][0], debug, timeout)
        for index in range(len(generations_list))
    ]

    total = len(tasks)
    completed = 0
    workers = 1 if debug else max(1, num_process_evaluate)
    mp_ctx = _eval_pool_context()
    pool_mode = mp_ctx.get_start_method()
    print(
        f"[lcb-eval] start: {total} samples, workers={workers}, "
        f"timeout={timeout}s, pool={pool_mode}",
        flush=True,
    )
    _write_eval_heartbeat(0, total)
    use_tqdm = is_interactive()
    pbar = tqdm(total=total, disable=not use_tqdm) if use_tqdm else None
    results = {}
    metadata = {}
    executor_kwargs = {
        "max_workers": workers,
        "mp_context": mp_ctx,
        "initializer": _pool_worker_init,
    }
    # spawn workers re-import the whole stack; recycle to limit leaks.
    if pool_mode == "spawn" and hasattr(ProcessPoolExecutor, "__init__"):
        import inspect

        if "max_tasks_per_child" in inspect.signature(ProcessPoolExecutor).parameters:
            executor_kwargs["max_tasks_per_child"] = 32

    try:
        batch_size = max(workers * 4, 16)
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            pending = {}
            next_i = 0

            def _fill():
                nonlocal next_i
                while next_i < len(tasks) and len(pending) < batch_size:
                    fut = executor.submit(_eval_single_sample, tasks[next_i])
                    pending[fut] = tasks[next_i][0]
                    next_i += 1

            _fill()
            while pending:
                for future in as_completed(list(pending.keys())):
                    index = pending.pop(future)
                    try:
                        ret_index, res_list, meta_list = future.result()
                        assert ret_index == index
                        results[index] = res_list
                        metadata[index] = meta_list
                    except Exception as exc:
                        print(
                            f"[lcb-eval] ERROR evaluating sample index={index}: {exc}",
                            flush=True,
                        )
                        traceback.print_exc()
                        raise
                    completed += 1
                    _write_eval_heartbeat(completed, total)
                    if pbar is not None:
                        pbar.update(1)
                    else:
                        log_progress("lcb-eval", completed, total, min_interval=1)
                    _fill()
                    break
    except Exception:
        print(
            f"[lcb-eval] aborted after {completed}/{total} samples",
            flush=True,
        )
        raise
    finally:
        if pbar is not None:
            pbar.close()
    print(f"[lcb-eval] finished {completed}/{total} samples", flush=True)
    _write_eval_heartbeat(completed, total)

    assert len(results) == len(
        tasks
    ), f"results = {len(results)} tasks = {len(tasks)}"

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):

    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample, generation_list) in enumerate(
        zip(samples_list, generations_list)
    ):
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    num_problems = len(samples_list)
    num_cases = len(samples_linear)
    print(
        f"[lcb] evaluating {num_problems} problems ({num_cases} samples, "
        f"{num_cases // num_problems if num_problems else 0} per problem)..."
    )

    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])

    for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)

    final_metadata = []
    for key in sorted(list(metadatas.keys())):
        final_metadata.append(metadatas[key])
    for i in range(len(final_metadata)):
        if type(final_metadata[i]) is not list:
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

        assert len(final_metadata[i]) == len(
            generations_list[0]
        ), f"{len(final_metadata[i])=}"

    return [metrics, results, final_metadata]


if __name__ == "__main__":
    # print(
    #     check_correctness(
    #         {
    #             "input_output": json.dumps(
    #                 {
    #                     "inputs": [
    #                         json.dumps([1] * 100000)
    #                         + "\n"
    #                         + json.dumps([100000, -100000] * (100000 // 2))
    #                     ],
    #                     "outputs": [json.dumps([100000, 0] * (100000 // 2))],
    #                     "fn_name": "mostFrequentIDs",
    #                 }
    #             )
    #         },
    #         "class Solution:\n    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:\n        from collections import defaultdict\n        \n        # Count of each ID\n        count = defaultdict(int)\n        # How many IDs exist for a given frequency\n        freq_of_count = defaultdict(int)\n        \n        max_freq = 0\n        ans = []\n        \n        for i in range(len(nums)):\n            x = nums[i]\n            change = freq[i]\n            \n            old_freq = count[x]\n            new_freq = old_freq + change\n            \n            # If there was an old frequency, decrease its usage\n            if old_freq > 0:\n                freq_of_count[old_freq] -= 1\n                if freq_of_count[old_freq] == 0:\n                    del freq_of_count[old_freq]\n            \n            # Update with the new frequency\n            count[x] = new_freq\n            freq_of_count[new_freq] += 1\n            \n            # Update max_freq if needed\n            if new_freq > max_freq:\n                max_freq = new_freq\n            \n            # If the collection at max_freq is empty, reduce max_freq until we find a non-empty bin\n            while max_freq > 0 and max_freq not in freq_of_count:\n                max_freq -= 1\n            \n            # If the collection is empty, max_freq will be 0\n            ans.append(max_freq)\n        \n        return ans",
    #         6,
    #         debug=True,
    #     )
    # )

    print(
        check_correctness(
            {
                "input_output": json.dumps(
                    {
                        "inputs": ")))))",
                        "outputs": "0",
                    },
                )
            },
            "\nMOD = 998244353\n\nS = input().strip()\nn = len(S)\n\nif n % 2 != 0:\n    print(0)\n    exit()\n\n# Initialize DP table\ndp = [[0] * (n + 2) for _ in range(n + 1)]\ndp[0][0] = 1\n\nfor i in range(1, n + 1):\n    c = S[i-1]\n    for b in range(n + 1):\n        if dp[i-1][b] == 0:\n            continue\n        if c == '(':\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        elif c == ')':\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n        else:  # '?'\n            # Replace with '('\n            new_b = b + 1\n            if new_b <= n:\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n            # Replace with ')'\n            if b > 0:\n                new_b = b - 1\n                dp[i][new_b] = (dp[i][new_b] + dp[i-1][b]) % MOD\n\nprint(dp[n][0] % MOD)\n",
            6,
            debug=True,
        )
    )
