# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone vLLM evaluation script for SFT validation benchmarks."""

from __future__ import annotations

import os

# Defensively purge any distributed env vars leaked from the torchrun parent process.
# vLLM's EngineCore (spawned below) must initialize its own distributed world and
# must never try to connect to the training process group's TCPStore.
_DIST_ENV_PREFIXES = (
    "PET_",
    "TORCHELASTIC_",
    "ELASTIC_",
    "TORCH_DIST_",
)
_DIST_ENV_EXACT = {
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "LOCAL_WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
}
for _key in list(os.environ.keys()):
    if _key in _DIST_ENV_EXACT or _key.startswith(_DIST_ENV_PREFIXES):
        os.environ.pop(_key, None)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import argparse
import json
import sys

import pandas as pd
from transformers import AutoTokenizer

from verl.trainer.sft_validation import (
    _format_prompt_for_generation,
    _resolve_ground_truth,
    _resolve_prompt,
    build_math_acc_metric_name,
)
from verl.utils.reward_score.math_verify import compute_score


def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM evaluation for SFT math benchmarks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--val_files", type=str, required=True, help="Path to validation parquet file(s)")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Key for prompt in dataframe")
    parser.add_argument("--max_new_tokens", type=int, default=16384, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Sampling top_p")
    parser.add_argument("--n_rollouts", type=int, default=4, help="Number of rollouts per prompt")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU memory utilization")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max validation samples to evaluate")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON path to save metrics")
    parser.add_argument(
        "--apply_chat_template_kwargs_json",
        type=str,
        default="{}",
        help="JSON string of kwargs for apply_chat_template",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    return parser.parse_args()


def main():
    args = parse_args()
    apply_chat_template_kwargs = json.loads(args.apply_chat_template_kwargs_json)

    # 1. Load validation data
    val_files = args.val_files
    if os.path.isdir(val_files):
        # find all parquet files
        files = [
            os.path.join(val_files, f)
            for f in sorted(os.listdir(val_files))
            if f.endswith(".parquet")
        ]
        dfs = [pd.read_parquet(f) for f in files]
        val_df = pd.concat(dfs, ignore_index=True)
    elif "," in val_files:
        files = [f.strip() for f in val_files.split(",") if f.strip()]
        dfs = [pd.read_parquet(f) for f in files]
        val_df = pd.concat(dfs, ignore_index=True)
    else:
        val_df = pd.read_parquet(val_files)

    if args.max_samples > 0 and args.max_samples < len(val_df):
        val_df = val_df.iloc[:args.max_samples]

    rows: list[tuple[str | list[dict[str, str]], str]] = []
    for _, row in val_df.iterrows():
        prompt = _resolve_prompt(row, args.prompt_key)
        ground_truth = _resolve_ground_truth(row)
        if prompt is None or ground_truth is None:
            continue
        rows.append((prompt, ground_truth))

    if not rows:
        print("[vLLM-Eval] Warning: no valid prompt/ground_truth rows found.")
        result = {"pass_at_n": 0.0, "mean_at_n": 0.0, "num_samples": 0}
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2)
        return

    print(f"[vLLM-Eval] Loaded {len(rows)} validation prompts.")

    # 2. Tokenizer & prompt formatting
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    prompt_texts = []
    ground_truths = []
    for prompt, ground_truth in rows:
        formatted = _format_prompt_for_generation(prompt, tokenizer, apply_chat_template_kwargs)
        prompt_texts.append(formatted)
        ground_truths.append(ground_truth)

    # 3. vLLM Engine
    from vllm import LLM, SamplingParams

    # Calculate model max length
    max_model_len = min(32768, 1024 + args.max_new_tokens)

    sampling_params = SamplingParams(
        n=args.n_rollouts,
        temperature=args.temperature if args.n_rollouts > 1 else 0.0,
        top_p=args.top_p if args.n_rollouts > 1 else 1.0,
        max_tokens=args.max_new_tokens,
    )

    print(
        f"[vLLM-Eval] Initializing vLLM: model={args.model_path}, "
        f"tp={args.tensor_parallel_size}, max_model_len={max_model_len}, "
        f"n_rollouts={args.n_rollouts}, max_new_tokens={args.max_new_tokens}"
    )

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=args.trust_remote_code,
        seed=42,
    )

    # 4. Batch generation
    print(f"[vLLM-Eval] Starting generation for {len(prompt_texts)} prompts (n={args.n_rollouts}) ...")
    outputs = llm.generate(prompt_texts, sampling_params=sampling_params, use_tqdm=True)

    # 5. Evaluate accuracy
    print("[vLLM-Eval] Scoring outputs with math_verify ...")
    pass_total = 0.0
    mean_total = 0.0

    for i, req_output in enumerate(outputs):
        sample_scores = []
        for sample in req_output.outputs:
            response_text = sample.text
            score = float(compute_score(response_text, ground_truths[i]))
            sample_scores.append(score)
        pass_total += int(max(sample_scores) >= 1.0)
        mean_total += sum(sample_scores) / len(sample_scores)

    num_rows = len(outputs)
    pass_at_n = pass_total / num_rows if num_rows > 0 else 0.0
    mean_at_n = mean_total / num_rows if num_rows > 0 else 0.0

    pass_name = build_math_acc_metric_name(args.n_rollouts, metric_kind="pass")
    mean_name = build_math_acc_metric_name(args.n_rollouts, metric_kind="mean")
    print(f"[vLLM-Eval] Finished: {pass_name}={pass_at_n:.4f}, {mean_name}={mean_at_n:.4f} on {num_rows} samples")

    result = {
        "pass_at_n": pass_at_n,
        "mean_at_n": mean_at_n,
        "num_samples": num_rows,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[vLLM-Eval] Metrics saved to {args.output_file}")


if __name__ == "__main__":
    main()
