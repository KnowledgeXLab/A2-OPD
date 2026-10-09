"""Offline aggregation helpers for A²OPD distribution KL metrics."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch

from verl.trainer.ppo.a2opd_utils import (
    build_policy_mask,
    compute_a2opd_topk_overlap_metrics,
    estimate_anchor_kl,
    materialize_dataproto_cpu,
    estimate_proxy_base_target_base_kl,
    estimate_proxy_kl,
    estimate_target_transfer_kl,
    estimate_transfer_gap,
    is_topk_log_probs,
)

METRIC_KEYS = (
    "a2opd/proxy_kl",
    "a2opd/proxy_base_target_base_kl",
    "a2opd/target_transfer_kl",
    "a2opd/anchor_kl",
    "a2opd/transfer_gap",
)

OVERLAP_METRIC_KEYS = (
    "a2opd/topk_overlap_target_vs_teacher_ratio",
    "a2opd/topk_overlap_target_vs_teacher_base_ratio",
    "a2opd/topk_overlap_target_vs_base_ratio",
    "a2opd/topk_overlap_teacher_vs_teacher_base_ratio",
)


def format_distribution_kl_metric_summary(metrics: dict[str, float]) -> str:
    """Compact metric string for progress logs."""
    parts: list[str] = []
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        short_key = key.split("/", 1)[-1]
        parts.append(f"{short_key}={value:.4f}")
    overlap_value = metrics.get("a2opd/topk_overlap_teacher_vs_teacher_base_ratio")
    if overlap_value is not None:
        parts.append(f"topk_overlap_teacher_vs_teacher_base={overlap_value:.4f}")
    return ", ".join(parts) if parts else "metrics=pending"


def log_distribution_kl_progress(message: str) -> None:
    print(f"[distribution_kl] {message}", flush=True)


def warn_skip_distribution_kl_batch(
    *,
    batch_idx: int,
    num_batches: int,
    batch_num_samples: int,
    valid_tokens: float,
) -> None:
    log_distribution_kl_progress(
        f"WARNING batch {batch_idx}/{num_batches} skipped: "
        f"no valid OPD tokens (samples={batch_num_samples}, valid_tokens={valid_tokens:.0f}); "
        "likely cross-tokenizer mask filtered all response tokens"
    )


def compute_distribution_kl_metrics(batch) -> dict[str, float]:
    """Compute Proxy / Induced / Actual Target KL and Transfer Gap on one OPD batch."""
    cross_mask = batch.batch.get("cross_token_opd_mask")
    policy_mask = build_policy_mask(batch.batch["response_mask"], cross_mask)

    teacher_log_probs = batch.batch["teacher_log_probs"].detach()
    teacher_base_log_probs = batch.batch["teacher_base_log_probs"].detach()
    old_log_probs = batch.batch["old_log_probs"].detach()
    base_log_probs = batch.batch["base_log_probs"].detach()
    candidate_mask = batch.batch.get("cross_token_topk_mask")
    if candidate_mask is not None:
        candidate_mask = candidate_mask.bool()

    metrics: dict[str, float] = {}
    metrics["a2opd/proxy_kl"] = estimate_proxy_kl(
        teacher_log_probs,
        teacher_base_log_probs,
        policy_mask,
        candidate_mask=candidate_mask,
    )
    metrics["a2opd/proxy_base_target_base_kl"] = estimate_proxy_base_target_base_kl(
        teacher_base_log_probs,
        base_log_probs,
        policy_mask,
        candidate_mask=candidate_mask,
    )
    metrics["a2opd/anchor_kl"] = estimate_anchor_kl(
        old_log_probs,
        base_log_probs,
        policy_mask,
        candidate_mask=candidate_mask,
    )

    if is_topk_log_probs(old_log_probs):
        metrics["a2opd/target_transfer_kl"] = estimate_target_transfer_kl(
            teacher_log_probs=teacher_log_probs,
            teacher_base_log_probs=teacher_base_log_probs,
            target_base_log_probs=base_log_probs,
            policy_mask=policy_mask,
            candidate_mask=candidate_mask,
        )
        metrics["a2opd/transfer_gap"] = estimate_transfer_gap(
            old_log_probs=old_log_probs,
            teacher_log_probs=teacher_log_probs,
            teacher_base_log_probs=teacher_base_log_probs,
            target_base_log_probs=base_log_probs,
            policy_mask=policy_mask,
            candidate_mask=candidate_mask,
        )
        overlap_metrics, overlap_valid_tokens = compute_a2opd_topk_overlap_metrics(batch.batch)
        metrics.update(overlap_metrics)
        metrics["_overlap_valid_tokens"] = overlap_valid_tokens

    valid_tokens = policy_mask.sum().item()
    metrics["_valid_tokens"] = valid_tokens
    metrics["_num_samples"] = float(batch.batch["response_mask"].shape[0])
    return metrics


def aggregate_distribution_kl_metrics(
    batch_metrics: list[dict[str, float]],
    *,
    skipped_batches: int = 0,
    skipped_samples: float = 0.0,
) -> dict[str, float]:
    """Token-weighted average over calibration batches."""
    if not batch_metrics:
        if skipped_batches > 0:
            raise ValueError(
                "No valid response tokens found in calibration batches "
                f"({skipped_batches} batch(es), {skipped_samples:.0f} sample(s) skipped)."
            )
        raise ValueError("No batch metrics to aggregate.")

    total_tokens = sum(m.get("_valid_tokens", 0.0) for m in batch_metrics)
    total_overlap_tokens = sum(m.get("_overlap_valid_tokens", 0.0) for m in batch_metrics)
    total_samples = sum(m.get("_num_samples", 0.0) for m in batch_metrics)
    if total_tokens <= 0:
        raise ValueError(
            "No valid response tokens found in calibration batches "
            f"({skipped_batches} batch(es), {skipped_samples:.0f} sample(s) skipped)."
        )

    aggregated: dict[str, float] = {
        "distribution_kl/num_samples": total_samples,
        "distribution_kl/valid_tokens": total_tokens,
    }
    if total_overlap_tokens > 0:
        aggregated["distribution_kl/overlap_valid_tokens"] = total_overlap_tokens
    if skipped_batches > 0:
        aggregated["distribution_kl/skipped_batches"] = float(skipped_batches)
        aggregated["distribution_kl/skipped_samples"] = skipped_samples
    for key in METRIC_KEYS:
        weighted = 0.0
        seen = False
        for metrics in batch_metrics:
            if key not in metrics:
                continue
            weight = metrics.get("_valid_tokens", 0.0)
            weighted += metrics[key] * weight
            seen = True
        if seen:
            aggregated[key] = weighted / total_tokens
    if total_overlap_tokens > 0:
        for key in OVERLAP_METRIC_KEYS:
            weighted = 0.0
            seen = False
            for metrics in batch_metrics:
                if key not in metrics:
                    continue
                weight = metrics.get("_overlap_valid_tokens", 0.0)
                weighted += metrics[key] * weight
                seen = True
            if seen:
                aggregated[key] = weighted / total_overlap_tokens
    return aggregated


def save_rollout_cache(batches, output_path: str | Path) -> Path:
    """Persist post-rollout batches (pre-OPD) for fixed-rollout reference sweeps."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_batches = [materialize_dataproto_cpu(batch) for batch in batches]
    with output_path.open("wb") as handle:
        pickle.dump(cpu_batches, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def load_rollout_cache(input_path: str | Path):
    """Load cached post-rollout batches written by save_rollout_cache()."""
    input_path = Path(input_path)
    with input_path.open("rb") as handle:
        batches = pickle.load(handle)
    return [materialize_dataproto_cpu(batch) for batch in batches]


def collect_rollout_records(batch, tokenizer) -> list[dict]:
    """Extract decoded prompt/response pairs from a post-rollout batch."""
    batch_size = batch.batch["response_mask"].shape[0]
    records: list[dict] = []
    for i in range(batch_size):
        prompt_ids = batch.batch["prompts"][i]
        response_ids = batch.batch["responses"][i]
        attention_mask = batch.batch["attention_mask"][i]
        prompt_length = prompt_ids.shape[0]
        valid_response_len = int(attention_mask[prompt_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_len]
        record = {
            "prompt_text": tokenizer.decode(prompt_ids, skip_special_tokens=True),
            "response_text": tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        }
        for key in ("uid", "data_source", "index"):
            if key in batch.non_tensor_batch:
                value = batch.non_tensor_batch[key][i]
                if hasattr(value, "item"):
                    value = value.item()
                record[key] = value
        records.append(record)
    return records


def write_distribution_kl_rollouts(
    records: list[dict],
    output_path: str | Path,
) -> Path:
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_path, index=False)
    return output_path


def write_distribution_kl_results(
    metrics: dict[str, Any],
    output_path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics}
    if extra:
        payload["meta"] = extra
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output_path
