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

from __future__ import annotations

import gc
from typing import Any

import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from verl.utils.device import get_device_id, get_device_name
from verl.utils.fs import copy_to_local
from verl.utils.reward_score.math_verify import compute_score
from verl.utils.torch_dtypes import PrecisionType


def val_df_has_ground_truth(val_df: pd.DataFrame) -> bool:
    if val_df is None or len(val_df) == 0:
        return False
    if any(key in val_df.columns for key in ("ground_truth", "answer", "reward_model")):
        return True
    return False


def _normalize_ground_truth(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    text = str(value).strip()
    return text or None


def _resolve_ground_truth(row: pd.Series) -> str | None:
    for key in ("ground_truth", "answer"):
        if key in row.index:
            gt = _normalize_ground_truth(row[key])
            if gt is not None:
                return gt

    if "reward_model" in row.index:
        reward_model = row["reward_model"]
        if isinstance(reward_model, dict):
            gt = _normalize_ground_truth(reward_model.get("ground_truth"))
            if gt is not None:
                return gt
    return None


def _resolve_prompt(row: pd.Series, prompt_key: str) -> str | list[dict[str, str]] | None:
    if prompt_key not in row.index:
        return None
    value = row[prompt_key]
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        messages = list(value)
        return messages or None
    text = str(value).strip()
    return text or None


def _format_prompt_for_generation(
    prompt: str | list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    apply_chat_template_kwargs: dict[str, Any],
) -> str:
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )
    prompt_chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        prompt_chat,
        add_generation_prompt=True,
        tokenize=False,
        **apply_chat_template_kwargs,
    )


class SFTMathBenchmarkValDataset(Dataset):
    """Prompt-only math benchmark parquet for SFT validation performance."""

    supports_sft_loss = False

    def __init__(self, parquet_files: str | ListConfig, max_samples: int = -1, use_shm: bool = False):
        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        dataframes = []
        for parquet_file in parquet_files:
            local_path = copy_to_local(parquet_file, verbose=True, use_shm=use_shm)
            dataframes.append(pd.read_parquet(local_path))
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        total = len(self.dataframe)
        print(f"validation benchmark dataset len: {total}")
        if max_samples > 0 and max_samples < total:
            self.dataframe = self.dataframe.iloc[:max_samples]
            print(f"selected {max_samples} validation benchmark samples out of {total}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        raise NotImplementedError("SFTMathBenchmarkValDataset is used for performance eval only.")


@torch.no_grad()
def evaluate_math_accuracy(
    *,
    state_dict: dict[str, torch.Tensor],
    model_config,
    tokenizer: PreTrainedTokenizer,
    val_df: pd.DataFrame,
    prompt_key: str = "prompt",
    max_samples: int = -1,
    max_new_tokens: int = 16384,
    batch_size: int = 4,
    n_rollouts: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 42,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
    trust_remote_code: bool = False,
    model_dtype: str = "bf16",
    shard_rank: int = 0,
    shard_world_size: int = 1,
) -> dict[str, float]:
    """Generate on validation prompts and score with math_verify.

    When shard_world_size > 1, each rank evaluates a disjoint prompt subset.
    Returns local pass/mean totals plus num_rows for distributed aggregation.
    """
    if val_df is None or len(val_df) == 0:
        return {"pass_at_n": 0.0, "mean_at_n": 0.0, "pass_total": 0.0, "mean_total": 0.0, "num_rows": 0.0}

    n_rollouts = max(1, int(n_rollouts))
    shard_world_size = max(1, int(shard_world_size))
    shard_rank = int(shard_rank)
    apply_chat_template_kwargs = apply_chat_template_kwargs or {}
    eval_df = val_df
    if max_samples > 0:
        eval_df = val_df.iloc[:max_samples]

    rows: list[tuple[str | list[dict[str, str]], str]] = []
    for _, row in eval_df.iterrows():
        prompt = _resolve_prompt(row, prompt_key)
        ground_truth = _resolve_ground_truth(row)
        if prompt is None or ground_truth is None:
            continue
        rows.append((prompt, ground_truth))

    if shard_world_size > 1:
        rows = rows[shard_rank::shard_world_size]

    if not rows:
        return {"pass_at_n": 0.0, "mean_at_n": 0.0, "pass_total": 0.0, "mean_total": 0.0, "num_rows": 0.0}

    eval_dtype = PrecisionType.to_dtype(model_dtype)
    if eval_dtype == torch.float32:
        eval_dtype = torch.bfloat16
    device = torch.device(get_device_id())
    device_type = get_device_name()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        eval_model = AutoModelForCausalLM.from_config(
            model_config,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
        )
    except (ValueError, ImportError, OSError):
        eval_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=trust_remote_code)
    eval_model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()
    eval_model.to(device=device, dtype=eval_dtype)
    eval_model.eval()

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    pass_total = 0.0
    mean_total = 0.0
    base_gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            prompt_texts = []
            ground_truths = []
            for prompt, ground_truth in batch_rows:
                prompt_text = _format_prompt_for_generation(
                    prompt, tokenizer, apply_chat_template_kwargs
                )
                prompt_texts.append(prompt_text)
                ground_truths.append(ground_truth)

            encoded = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            input_len = input_ids.shape[1]

            batch_sample_scores: list[list[float]] = [[] for _ in batch_rows]
            for rollout_idx in range(n_rollouts):
                if n_rollouts == 1:
                    rollout_gen_kwargs = {**base_gen_kwargs, "do_sample": False}
                else:
                    rollout_seed = seed + rollout_idx
                    torch.manual_seed(rollout_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(rollout_seed)
                    rollout_gen_kwargs = {
                        **base_gen_kwargs,
                        "do_sample": True,
                        "temperature": temperature,
                        "top_p": top_p,
                    }

                with torch.autocast(device_type=device_type, dtype=eval_dtype):
                    outputs = eval_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **rollout_gen_kwargs,
                    )

                for idx in range(len(batch_rows)):
                    generated_ids = outputs[idx, input_len:]
                    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    batch_sample_scores[idx].append(float(compute_score(response, ground_truths[idx])))

                del outputs

            for sample_scores in batch_sample_scores:
                pass_total += int(max(sample_scores) >= 1.0)
                mean_total += sum(sample_scores) / len(sample_scores)

            del input_ids, attention_mask, encoded
    finally:
        tokenizer.padding_side = original_padding_side

    num_rows = len(rows)
    result = {
        "pass_at_n": pass_total / num_rows,
        "mean_at_n": mean_total / num_rows,
        "pass_total": pass_total,
        "mean_total": mean_total,
        "num_rows": float(num_rows),
    }

    del eval_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def build_math_acc_metric_name(n_rollouts: int, metric_kind: str = "pass") -> str:
    n_rollouts = max(1, int(n_rollouts))
    if metric_kind == "mean":
        return f"val/math_acc/mean@{n_rollouts}"
    return "val/math_acc" if n_rollouts == 1 else f"val/math_acc/pass@{n_rollouts}"


def is_better_metric(current: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best + min_delta
    return current < best - min_delta
