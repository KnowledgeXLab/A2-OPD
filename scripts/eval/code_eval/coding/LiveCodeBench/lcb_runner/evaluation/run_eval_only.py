"""Lightweight LCB eval entry: no vLLM/torch import on the main process."""

from __future__ import annotations

import json
import os
import sys
import traceback

# Patch datasets/multiprocess teardown noise before any heavy imports.
from lcb_runner.evaluation.compute_code_generation_metrics import (
    _install_unraisable_noise_filter,
    _silence_multiprocess_resource_tracker,
)

_install_unraisable_noise_filter()
_silence_multiprocess_resource_tracker()

from lcb_runner.evaluation import extract_instance_results
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.runner.parser import get_args
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    get_metrics,
    sort_and_extract_save_results,
)
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.utils.scenarios import Scenario


def main() -> None:
    args = get_args()
    if not args.evaluate:
        print("[lcb-eval-only] --evaluate is required", file=sys.stderr)
        sys.exit(2)

    model = LanguageModelStore[args.model]
    benchmark, _format_prompt = build_prompt_benchmark(args)
    output_path = get_output_path(model.model_repr, args)
    eval_file = output_path.replace(".json", "_eval.json")
    eval_all_file = output_path.replace(".json", "_eval_all.json")

    if not os.path.exists(output_path):
        print(f"[lcb-eval-only] missing generations: {output_path}", file=sys.stderr)
        sys.exit(1)

    with open(output_path, "r", encoding="utf-8") as f:
        save_results = json.load(f)

    save_results = [row for row in save_results if row.get("output_list")]
    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )
    benchmark = sorted(benchmark, key=lambda x: x.question_id)
    if len(benchmark) != len(combined_results):
        print(
            f"[lcb-eval-only] benchmark/combined size mismatch: "
            f"{len(benchmark)} vs {len(combined_results)}",
            file=sys.stderr,
        )
        sys.exit(1)

    old_eval_all_results: list = []
    old_eval_results = None
    if args.continue_existing_with_eval and os.path.exists(eval_all_file):
        with open(eval_all_file, encoding="utf-8") as fp:
            old_eval_all_results = json.load(fp)
        if os.path.exists(eval_file):
            with open(eval_file, encoding="utf-8") as fp:
                old_eval_results = json.load(fp)
        done_ids = {row["question_id"] for row in old_eval_all_results}
        keep = [
            i for i, inst in enumerate(benchmark) if inst.question_id not in done_ids
        ]
        benchmark = [benchmark[i] for i in keep]
        combined_results = [combined_results[i] for i in keep]
        if not benchmark:
            print("[lcb-eval-only] all problems already evaluated")
            return
        print(
            f"[lcb-eval-only] resume: {len(old_eval_all_results)} done, "
            f"{len(benchmark)} remaining"
        )

    metrics = get_metrics(args.scenario, args, benchmark, combined_results)
    graded = extract_instance_results(metrics[1])

    if old_eval_results and isinstance(old_eval_results, list) and old_eval_results:
        old_metrics = old_eval_results[0]
        old_eval_size = len(old_eval_all_results)
        new_eval_size = len(benchmark)
        if new_eval_size > 0:
            for key in metrics[0]:
                if key in old_metrics and key != "detail":
                    metrics[0][key] = (
                        old_eval_size * old_metrics[key]
                        + new_eval_size * metrics[0][key]
                    ) / (old_eval_size + new_eval_size)
            if "detail" in metrics[0] and "detail" in old_metrics:
                for key in metrics[0]["detail"]:
                    if key in old_metrics["detail"]:
                        metrics[0]["detail"][key] = {
                            **metrics[0]["detail"][key],
                            **old_metrics["detail"][key],
                        }
            metrics[1] = {**metrics[1], **old_eval_results[1]}

    if args.scenario == Scenario.codegeneration:
        metadatas = metrics[2] if metrics else [[] for _ in benchmark]
        save_eval_results = [
            instance.insert_output_evaluation(
                outputs_list, extracted_list, graded_list, metadata=meta
            )
            for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                benchmark, combined_results, graded, metadatas
            )
        ]
    else:
        save_eval_results = [
            instance.insert_output_evaluation(
                outputs_list, extracted_list, graded_list
            )
            for instance, (outputs_list, extracted_list), graded_list in zip(
                benchmark, combined_results, graded
            )
        ]

    save_eval_results = old_eval_all_results + save_eval_results
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    with open(eval_all_file, "w", encoding="utf-8") as f:
        json.dump(save_eval_results, f, indent=4)
    print(f"[lcb-eval-only] wrote {eval_file} and {eval_all_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[lcb-eval-only] interrupted", flush=True)
        sys.exit(130)
    except Exception:
        print("[lcb-eval-only] FATAL error:", flush=True)
        traceback.print_exc()
        sys.exit(1)
