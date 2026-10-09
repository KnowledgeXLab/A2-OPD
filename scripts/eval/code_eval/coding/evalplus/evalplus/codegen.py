import json
import os
from itertools import groupby
from typing import Dict, List, Optional, Tuple

from evalplus.data import get_evalperf_data, get_human_eval_plus, get_mbpp_plus
from evalplus.provider import DecoderBase, make_model
from evalplus.sanitize import sanitize
from evalplus.utils import format_progress, progress


def _load_task_sample_counts(
    target_path: str, resume: bool
) -> Dict[str, int]:
    task2nexist: Dict[str, int] = {}
    if not resume:
        return task2nexist
    if target_path.endswith(".jsonl") and os.path.isfile(target_path):
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                task_id = json.loads(line)["task_id"]
                task2nexist[task_id] = task2nexist.get(task_id, 0) + 1
    return task2nexist


def _resolve_output_paths(target_path: str) -> Tuple[str, str]:
    if target_path.endswith(".jsonl"):
        raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
        all_solutions_path = target_path.replace(".jsonl", ".all_solutions.jsonl")
    else:
        raw_target_path = target_path + ".raw"
        all_solutions_path = target_path + ".all_solutions.jsonl"
        os.makedirs(target_path, exist_ok=True)
    return raw_target_path, all_solutions_path


def _write_task_samples(
    *,
    target_path: str,
    raw_target_path: str,
    task_id: str,
    task: Dict,
    model: DecoderBase,
    outputs: List[str],
    sidx_start: int,
    p_name: Optional[str] = None,
) -> List[str]:
    prompt = task["prompt"].strip() + "\n"
    all_raw_solutions: List[str] = []
    sidx = sidx_start
    for impl in outputs:
        solution = prompt + impl if model.is_direct_completion() else impl
        all_raw_solutions.append(solution)
        sanitized_solution = sanitize(solution, entrypoint=task["entry_point"])
        if target_path.endswith(".jsonl"):
            with open(target_path, "a") as f:
                f.write(
                    json.dumps({"task_id": task_id, "solution": sanitized_solution})
                    + "\n"
                )
            with open(raw_target_path, "a") as f:
                f.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
        else:
            assert p_name is not None
            with open(
                os.path.join(target_path, p_name, f"{sidx}.py"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(sanitized_solution)
            with open(
                os.path.join(raw_target_path, p_name, f"{sidx}.py"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(solution)
        sidx += 1
    return all_raw_solutions


def _codegen_single_task_loop(
    *,
    target_path: str,
    raw_target_path: str,
    all_solutions_path: str,
    model: DecoderBase,
    dataset: Dict,
    greedy: bool,
    n_samples: int,
    id_range,
    resume: bool,
    task2nexist: Dict[str, int],
    progress_note: Optional[str],
) -> None:
    backend_type: str = type(model).__name__
    with progress(progress_note or backend_type) as p:
        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            p_name = None
            if not target_path.endswith(".jsonl"):
                p_name = task_id.replace("/", "_")
                os.makedirs(os.path.join(target_path, p_name), exist_ok=True)
                task2nexist[task_id] = len(
                    [
                        f
                        for f in os.listdir(os.path.join(target_path, p_name))
                        if f.endswith(".py")
                    ]
                )

            n_more_samples = n_samples
            log = f"Codegen: {task_id} @ {model}"
            if resume and task2nexist.get(task_id, 0) > 0:
                log += f" (resuming from {task2nexist[task_id]})"
                n_more_samples -= task2nexist[task_id]
            if n_more_samples <= 0:
                continue

            p.console.print(log)

            sidx = n_samples - n_more_samples
            all_raw_solutions: List[str] = []
            while sidx < n_samples:
                prompt = task["prompt"].strip() + "\n"
                outputs = model.codegen(
                    prompt,
                    do_sample=not greedy,
                    num_samples=n_samples - sidx,
                )
                assert outputs, "No outputs from model!"
                batch_solutions = _write_task_samples(
                    target_path=target_path,
                    raw_target_path=raw_target_path,
                    task_id=task_id,
                    task=task,
                    model=model,
                    outputs=outputs,
                    sidx_start=sidx,
                    p_name=p_name,
                )
                all_raw_solutions.extend(batch_solutions)
                sidx += len(outputs)

            with open(all_solutions_path, "a") as f:
                f.write(
                    json.dumps({"task_id": task_id, "solutions": all_raw_solutions})
                    + "\n"
                )


def _codegen_batched_loop(
    *,
    target_path: str,
    raw_target_path: str,
    all_solutions_path: str,
    model: DecoderBase,
    dataset: Dict,
    greedy: bool,
    n_samples: int,
    id_range,
    resume: bool,
    task2nexist: Dict[str, int],
    progress_note: Optional[str],
    problem_batch_size: int,
) -> None:
    pending = []
    for task_id, task in dataset.items():
        if id_range is not None:
            id_num = int(task_id.split("/")[1])
            low, high = id_range
            if id_num < low or id_num >= high:
                continue

        p_name = None
        existing = task2nexist.get(task_id, 0)
        if not target_path.endswith(".jsonl"):
            p_name = task_id.replace("/", "_")
            if os.path.isdir(os.path.join(target_path, p_name)):
                existing = len(
                    [
                        f
                        for f in os.listdir(os.path.join(target_path, p_name))
                        if f.endswith(".py")
                    ]
                )
            task2nexist[task_id] = existing

        n_more_samples = n_samples - existing if resume else n_samples
        if n_more_samples <= 0:
            continue

        pending.append(
            {
                "task_id": task_id,
                "task": task,
                "p_name": p_name,
                "n_more": n_more_samples,
                "sidx_start": n_samples - n_more_samples,
            }
        )

    if not pending:
        return

    pending.sort(key=lambda item: (-item["n_more"], item["task_id"]))
    bench = progress_note or type(model).__name__
    batch_plan: list[tuple[int, list[dict]]] = []
    for n_more, group_iter in groupby(pending, key=lambda item: item["n_more"]):
        group = list(group_iter)
        for batch_start in range(0, len(group), problem_batch_size):
            batch_plan.append((n_more, group[batch_start : batch_start + problem_batch_size]))

    completed = 0
    total = len(pending)
    num_batches = len(batch_plan)

    with progress(bench) as p:
        for batch_idx, (n_more, batch) in enumerate(batch_plan, start=1):
            task_ids = [item["task_id"] for item in batch]
            if len(task_ids) == 1:
                task_range = task_ids[0]
            else:
                task_range = f"{task_ids[0]}..{task_ids[-1]}"
            start_detail = (
                f"| batch {batch_idx}/{num_batches} START "
                f"| {len(batch)} tasks x{n_more} "
                f"| {task_range}"
            )
            print(format_progress(bench, completed, total, detail=start_detail), flush=True)
            prompts = [item["task"]["prompt"].strip() + "\n" for item in batch]
            batch_outputs = model.codegen_batch(
                prompts,
                do_sample=not greedy,
                num_samples=n_more,
            )
            assert len(batch_outputs) == len(batch), "Batch output size mismatch!"
            for item, outputs in zip(batch, batch_outputs):
                assert outputs, f"No outputs from model for {item['task_id']}!"
                if item["p_name"] is not None:
                    os.makedirs(
                        os.path.join(target_path, item["p_name"]), exist_ok=True
                    )
                all_raw_solutions = _write_task_samples(
                    target_path=target_path,
                    raw_target_path=raw_target_path,
                    task_id=item["task_id"],
                    task=item["task"],
                    model=model,
                    outputs=outputs,
                    sidx_start=item["sidx_start"],
                    p_name=item["p_name"],
                )
                with open(all_solutions_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "task_id": item["task_id"],
                                "solutions": all_raw_solutions,
                            }
                        )
                        + "\n"
                    )
                completed += 1
            detail = (
                f"| batch {batch_idx}/{num_batches} DONE "
                f"| {len(batch)} tasks x{n_more} "
                f"| {task_range}"
            )
            print(
                format_progress(bench, completed, total, detail=detail),
                flush=True,
            )


def codegen(
    target_path: str,
    model: DecoderBase,
    dataset: Dict,
    greedy=False,
    n_samples=1,
    id_range=None,
    resume=True,
    progress_note: Optional[str] = None,
    problem_batch_size: int = 1,
):
    task2nexist = _load_task_sample_counts(target_path, resume)
    raw_target_path, all_solutions_path = _resolve_output_paths(target_path)

    print(f"Sanitized code outputs will be saved to {target_path}")
    print(f"Raw outputs will be saved to {raw_target_path}")
    print(f"All solutions will be saved to {all_solutions_path}")
    if problem_batch_size > 1:
        print(f"EvalPlus multi-problem batch size: {problem_batch_size}")

    if greedy or problem_batch_size <= 1:
        _codegen_single_task_loop(
            target_path=target_path,
            raw_target_path=raw_target_path,
            all_solutions_path=all_solutions_path,
            model=model,
            dataset=dataset,
            greedy=greedy,
            n_samples=n_samples,
            id_range=id_range,
            resume=resume,
            task2nexist=task2nexist,
            progress_note=progress_note,
        )
        return

    _codegen_batched_loop(
        target_path=target_path,
        raw_target_path=raw_target_path,
        all_solutions_path=all_solutions_path,
        model=model,
        dataset=dataset,
        greedy=greedy,
        n_samples=n_samples,
        id_range=id_range,
        resume=resume,
        task2nexist=task2nexist,
        progress_note=progress_note,
        problem_batch_size=problem_batch_size,
    )


def run_codegen(
    model: str,
    dataset: str,
    root: str = "evalplus_results",
    bs: Optional[int] = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 16384,
    resume: bool = True,
    greedy: bool = False,
    id_range: List = None,
    version: str = "default",
    backend: str = "vllm",
    force_base_prompt: bool = False,
    base_url: str = None,
    tp: int = 1,
    evalperf_type: str = None,  # For EvalPerf
    jsonl_fmt: bool = True,
    attn_implementation: str = "eager",
    device_map: Optional[str] = None,
    trust_remote_code: bool = False,
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = False,
    enforce_eager: bool = False,
    max_model_len: int = 32768,
    dtype: str = "bfloat16",
    gptqmodel_backend: str = "auto",  # For GPTQModel
    gguf_file: Optional[str] = None,
    identifier: Optional[str] = None,
    problem_batch_size: int = 1,
):
    assert dataset in ["humaneval", "mbpp", "evalperf"], f"Invalid dataset {dataset}"
    assert evalperf_type is None or evalperf_type in [
        "instruct",
        "perf-instruct",
        "perf-CoT",
        "azr",
    ]

    # Make dir for codes generated by each model
    if identifier is None:
        identifier = model.strip("./").replace("/", "--") + f"_{backend}_temp_{temperature}"
    if evalperf_type:
        identifier += f"-{evalperf_type}"

    target_path = os.path.join(root, dataset, identifier)
    if jsonl_fmt:
        target_path += ".jsonl"
    else:
        os.makedirs(target_path, exist_ok=True)

    if dataset == "humaneval":
        dataset_dict = get_human_eval_plus(version=version)
    elif dataset == "mbpp":
        dataset_dict = get_mbpp_plus(version=version)
    elif dataset == "evalperf":
        original_dataset = {**get_human_eval_plus(), **get_mbpp_plus()}
        dataset_dict = {k: original_dataset[k] for k in get_evalperf_data()}
        assert id_range is None, "id_range not supported for evalperf"
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    all_tasks_complete = False
    if jsonl_fmt and os.path.isfile(target_path):
        task_counts = {}
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_id = data["task_id"]
                task_counts[task_id] = task_counts.get(task_id, 0) + 1

            all_tasks_complete = all(
                task_counts.get(task_id, 0) >= n_samples
                for task_id in dataset_dict.keys()
            )

    if all_tasks_complete:
        print("All samples are already cached. Skipping codegen.")
        return target_path

    if greedy and (temperature != 0 or bs != 1 or n_samples != 1):
        temperature = 0.0
        bs = 1
        n_samples = 1
        problem_batch_size = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if id_range is not None:
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    if bs is None:
        bs = min(n_samples, 32)
        print(f"Setting batch size to {bs}")

    if problem_batch_size > 1 and backend != "vllm":
        print(
            f"problem_batch_size={problem_batch_size} requires backend=vllm; "
            "falling back to single-problem codegen."
        )
        problem_batch_size = 1

    # Make project dir
    os.makedirs(root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(root, dataset), exist_ok=True)

    # Model instructions
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

    if evalperf_type == "perf-instruct":
        instruction_prefix = "Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type == "perf-CoT":
        instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    elif evalperf_type == "azr":
        instruction_prefix = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Assistant: <think>"
    elif evalperf_type is not None and evalperf_type != "instruct":
        raise ValueError(f"Invalid evalperf_type: {evalperf_type}")

    # Model creation
    model_runner = make_model(
        model=model,
        backend=backend,
        batch_size=bs,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        force_base_prompt=force_base_prompt,
        dataset=dataset,
        base_url=base_url,
        tp=tp,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        dtype=dtype,
        gptqmodel_backend=gptqmodel_backend,
        gguf_file=gguf_file,
    )

    codegen(
        target_path=target_path,
        dataset=dataset_dict,
        greedy=greedy,
        model=model_runner,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
        progress_note=dataset,
        problem_batch_size=problem_batch_size,
    )

    # force shutdown the model runner
    del model_runner
    import gc

    gc.collect()

    return target_path


def main():
    from fire import Fire

    Fire(run_codegen)


if __name__ == "__main__":
    main()
