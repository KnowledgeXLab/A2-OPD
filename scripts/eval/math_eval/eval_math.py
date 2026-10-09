import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path

import torch
from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

EVAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

from eval_log_progress import format_batch_detail, log_progress, sample_id_range  # noqa: E402
from math_eval_utils import load_input_data, prepare_resume

BOXED_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def _env_is_true(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def apply_chat_template(toker, messages, chat_template=None, enable_thinking=False):
    if chat_template is None:
        input_prompt = toker.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
    else:
        input_prompt = chat_template.format(prompt=messages[0]["content"])
    return input_prompt


def build_prompt_token_ids(toker, input_data, enable_thinking=False):
    chat_template = None
    prompt_token_ids = []
    for item in input_data:
        item["answer"] = str(item["answer"])
        problem = item["problem"]
        if BOXED_SUFFIX.strip() not in problem:
            problem = problem + BOXED_SUFFIX
        prompt_token_ids.append(
            apply_chat_template(
                toker,
                [{"role": "user", "content": problem}],
                chat_template=chat_template,
                enable_thinking=enable_thinking,
            )
        )
    return prompt_token_ids


def process_generations(input_batch, generations, model_name):
    res_data = []
    for i in range(len(input_batch)):
        d = copy.deepcopy(input_batch[i])
        responses = []
        boxed_answers = []
        acc_list = []
        # There are args.n generations per input
        for j in range(len(generations[i].outputs)):
            response = generations[i].outputs[j].text.strip()
            responses.append(response)
            boxed_answer = remove_boxed(last_boxed_only_string(response))
            boxed_answers.append(boxed_answer)
            # Compare boxed_answer with d["answer"]
            if boxed_answer is None:
                acc = False
            else:
                acc = verify(
                    parse("\\boxed{" + d["answer"] + "}"),
                    parse("\\boxed{" + boxed_answer + "}"),
                )
            acc_list.append(acc)
        d["pred_answers"] = boxed_answers
        d["responses"] = responses
        d["acc_list"] = acc_list
        d["model"] = model_name
        res_data.append(d)
    return res_data


def append_results(output_file, res_data):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        for d in res_data:
            file.write(json.dumps(d, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def print_metrics(toker, res_data, input_file, val_metric_source=None):
    total_preds = 0
    correct_preds = 0
    pass_at_k = 0
    avg_length = 0
    for d in res_data:
        accs = d.get("acc_list", [])
        total_preds += len(accs)
        correct_preds += sum(1 for acc in accs if acc)
        if any(accs):
            pass_at_k += 1
        responses = d.get("responses", [])
        for response in responses:
            length = len(toker.encode(response, add_special_tokens=False))
            avg_length += length / len(responses)

    accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
    pass_at_k = pass_at_k / len(res_data) if len(res_data) > 0 else 0.0
    avg_length = avg_length / len(res_data) if len(res_data) > 0 else 0.0
    dataset_label = input_file
    if val_metric_source:
        dataset_label = f"{input_file} [{val_metric_source}]"
    print(f"dataset: {dataset_label}")
    print(f"num_samples: {len(res_data)}")
    print(f"Total predictions: {total_preds}")
    print(f"Accurate predictions: {correct_preds}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"passs@k: {pass_at_k:.4f}")
    print(f"avg_length: {avg_length:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument(
        "--val_metric_source",
        type=str,
        default=None,
        help="For parquet input, filter rows by data_source (e.g. AIME2024).",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--begin_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", action="store_true")
    args = parser.parse_args()

    input_data = load_input_data(
        args.input_file,
        val_metric_source=args.val_metric_source,
        begin_idx=args.begin_idx,
        end_idx=args.end_idx,
    )
    if not input_data:
        raise ValueError(
            f"No samples loaded from {args.input_file}"
            + (
                f" with val_metric_source={args.val_metric_source!r}"
                if args.val_metric_source
                else ""
            )
        )

    total_samples = len(input_data)
    pending, completed, _, rewritten = prepare_resume(input_data, args.output_file)

    if not pending:
        print(
            f"[eval_math] skip generation: {args.output_file} already has "
            f"{completed}/{total_samples} unique samples by id"
        )
        return

    if completed > 0 or rewritten:
        print(
            f"[eval_math] resume by id: {completed}/{total_samples} done, "
            f"{len(pending)} remaining -> {args.output_file}"
        )
    else:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        if not Path(args.output_file).is_file():
            Path(args.output_file).write_text("", encoding="utf-8")

    input_data = pending

    toker = AutoTokenizer.from_pretrained(args.model_path)
    model_name = os.path.basename(args.model_path)

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=args.max_num_seqs,
        enforce_eager=_env_is_true("VLLM_ENFORCE_EAGER", False),
    )

    try:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=args.n,
            seed=args.seed,
        )

        batch_size = max(1, args.batch_size)
        num_batches = (len(input_data) + batch_size - 1) // batch_size
        progress_note = args.val_metric_source or "math"
        all_res_data = []
        for batch_start in range(0, len(input_data), batch_size):
            batch_idx = batch_start // batch_size + 1
            batch_items = input_data[batch_start : batch_start + batch_size]
            prompt_token_ids = build_prompt_token_ids(
                toker, batch_items, enable_thinking=args.enable_thinking
            )
            generations = llm.generate(prompt_token_ids, sampling_params=sampling_params)
            batch_results = process_generations(batch_items, generations, model_name)
            append_results(args.output_file, batch_results)
            all_res_data.extend(batch_results)

            done = completed + batch_start + len(batch_results)
            detail = format_batch_detail(
                batch_idx,
                num_batches,
                len(batch_items),
                n=args.n,
                sample_range=sample_id_range(batch_items),
            )
            log_progress(progress_note, done, total_samples, detail=detail)

        print_metrics(toker, all_res_data, args.input_file, args.val_metric_source)
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
