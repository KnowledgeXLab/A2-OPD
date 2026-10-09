"""Utilities for A²OPD: trust-region constrained proxy signal absorption."""

from __future__ import annotations

from typing import Literal

import torch

BaselineMode = Literal["none", "state", "batch", "sequence", "group"]


def materialize_dataproto_on_device(data, device):
    """Rebuild an unlocked DataProto on ``device`` (safe after pickle / TensorDict.to)."""
    from tensordict import TensorDict

    from verl import DataProto

    if data.batch is None:
        return data
    tensors = {
        key: tensor.detach().to(device, non_blocking=True)
        for key, tensor in data.batch.items()
    }
    return DataProto(
        batch=TensorDict(tensors, batch_size=data.batch.batch_size),
        non_tensor_batch=dict(data.non_tensor_batch) if data.non_tensor_batch else {},
        meta_info=dict(data.meta_info) if data.meta_info else {},
    )


def materialize_dataproto_cpu(data):
    """Rebuild an unlocked CPU DataProto for rollout-cache persistence."""
    from tensordict import TensorDict

    from verl import DataProto

    if data.batch is None:
        return data
    tensors = {key: tensor.detach().cpu().clone() for key, tensor in data.batch.items()}
    return DataProto(
        batch=TensorDict(tensors, batch_size=data.batch.batch_size),
        non_tensor_batch=dict(data.non_tensor_batch) if data.non_tensor_batch else {},
        meta_info=dict(data.meta_info) if data.meta_info else {},
    )


def prepare_dataproto_for_opd_prep(data, device, *, from_rollout_cache: bool = False):
    """Materialize cached rollout batches on the worker GPU before OPD prep.

    Only distribution-KL fixed-rollout eval loads CPU pickle caches; regular
    training batches must pass through unchanged.
    """
    if not from_rollout_cache:
        return data
    if data.batch is None or len(data.batch.keys()) == 0:
        return data
    return materialize_dataproto_on_device(data, device)


def build_policy_mask(
    response_mask: torch.Tensor,
    cross_token_opd_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    policy_mask = response_mask.float()
    if cross_token_opd_mask is not None:
        policy_mask = policy_mask * cross_token_opd_mask.to(dtype=policy_mask.dtype)
    return policy_mask


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    if dim is None:
        denom = mask.sum().clamp(min=1.0)
        return (values * mask).sum() / denom
    denom = mask.sum(dim=dim, keepdim=True).clamp(min=1.0)
    return (values * mask).sum(dim=dim, keepdim=keepdim) / denom


def is_topk_log_probs(log_probs: torch.Tensor) -> bool:
    return log_probs.dim() == 3


@torch.no_grad()
def mean_topk_overlap_ratio(
    target_topk_ids: torch.Tensor,
    other_topk_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    chunk_size: int = 262144,
) -> float:
    """Mean |topk(target) ∩ topk(other)| / K over valid token positions.

    Sorting the concatenated candidate ids counts the union in O(K log K)
    without materializing a (K, K) pairwise-comparison tensor.
    """
    if target_topk_ids.shape != other_topk_ids.shape:
        raise ValueError(
            "target_topk_ids and other_topk_ids must have the same shape, got "
            f"{tuple(target_topk_ids.shape)} and {tuple(other_topk_ids.shape)}."
        )
    if target_topk_ids.dim() != 3:
        raise ValueError(
            "top-k ids must have shape (batch, response_length, top_k), got "
            f"{tuple(target_topk_ids.shape)}."
        )
    if tuple(valid_mask.shape) != tuple(target_topk_ids.shape[:-1]):
        raise ValueError(
            "valid_mask must have shape (batch, response_length), got "
            f"{tuple(valid_mask.shape)}."
        )

    k = target_topk_ids.shape[-1]
    if k <= 0:
        return 0.0

    target_flat = target_topk_ids.reshape(-1, k)
    other_flat = other_topk_ids.reshape(-1, k)
    valid_flat = valid_mask.reshape(-1).bool()
    total_intersection = torch.zeros((), dtype=torch.long, device=target_topk_ids.device)
    total_positions = valid_flat.sum(dtype=torch.long)

    for start in range(0, target_flat.shape[0], chunk_size):
        end = min(start + chunk_size, target_flat.shape[0])
        chunk_valid = valid_flat[start:end]
        if not chunk_valid.any():
            continue
        merged = torch.cat((target_flat[start:end], other_flat[start:end]), dim=-1)
        sorted_ids = torch.sort(merged, dim=-1).values
        union_count = 1 + (sorted_ids[:, 1:] != sorted_ids[:, :-1]).sum(dim=-1)
        intersection_count = 2 * k - union_count
        total_intersection += intersection_count[chunk_valid].sum(dtype=torch.long)

    return (total_intersection.float() / (total_positions * k).clamp(min=1)).item()


@torch.no_grad()
def compute_a2opd_topk_overlap_metrics(
    batch_tensors,
) -> tuple[dict[str, float], float]:
    """Compute top-k overlap ratios on an OPD-prepared batch.

    Returns metric dict and the number of token positions used for weighting
    batch aggregation (response tokens for same-tokenizer, ref-axis tokens otherwise).
    """
    metrics: dict[str, float] = {}
    overlap_valid_tokens = 0.0

    target_topk_ids = batch_tensors.get("student_topk_ids")
    response_mask = batch_tensors.get("response_mask")
    if target_topk_ids is None or response_mask is None:
        return metrics, overlap_valid_tokens

    response_mask_bool = response_mask.bool()
    candidate_mask = batch_tensors.get("cross_token_topk_mask")
    use_cross_token = candidate_mask is not None
    if use_cross_token:
        candidate_mask = candidate_mask.bool()

    if not use_cross_token:
        base_topk_ids = batch_tensors.get("base_topk_ids")
        if base_topk_ids is not None:
            metrics["a2opd/topk_overlap_target_vs_base_ratio"] = mean_topk_overlap_ratio(
                target_topk_ids,
                base_topk_ids,
                response_mask_bool,
            )

        teacher_topk_ids = batch_tensors.get("teacher_topk_ids")
        teacher_base_topk_ids = batch_tensors.get("teacher_base_topk_ids")
        if teacher_topk_ids is not None:
            metrics["a2opd/topk_overlap_target_vs_teacher_ratio"] = mean_topk_overlap_ratio(
                target_topk_ids,
                teacher_topk_ids,
                response_mask_bool,
            )
        if teacher_base_topk_ids is not None:
            metrics["a2opd/topk_overlap_target_vs_teacher_base_ratio"] = mean_topk_overlap_ratio(
                target_topk_ids,
                teacher_base_topk_ids,
                response_mask_bool,
            )
        if teacher_topk_ids is not None and teacher_base_topk_ids is not None:
            metrics["a2opd/topk_overlap_teacher_vs_teacher_base_ratio"] = mean_topk_overlap_ratio(
                teacher_topk_ids,
                teacher_base_topk_ids,
                response_mask_bool,
            )
        overlap_valid_tokens = float(response_mask_bool.sum().item())
        return metrics, overlap_valid_tokens

    def _log_ref_axis_overlap(
        metric_name: str,
        translated_target_topk_ids: torch.Tensor,
        other_topk_ids: torch.Tensor,
        ref_response_mask: torch.Tensor,
        span_starts: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> None:
        nonlocal overlap_valid_tokens
        primary_valid = (
            response_mask_bool
            & span_mask.bool()
            & candidate_mask.any(dim=-1)
        )
        ref_valid = build_ref_axis_topk_valid_mask(
            primary_valid,
            span_starts,
            ref_response_mask,
        )
        metrics[metric_name] = mean_topk_overlap_ratio(
            translated_target_topk_ids,
            other_topk_ids,
            ref_valid,
        )
        overlap_valid_tokens = max(overlap_valid_tokens, float(ref_valid.sum().item()))

    teacher_topk_ids = batch_tensors.get("teacher_topk_ids")
    ref_topk_ids = batch_tensors.get("ref_topk_ids")
    ref_response_mask = batch_tensors.get("ref_response_mask")
    ref_span_starts = batch_tensors.get("cross_token_ref_span_starts")
    ref_span_mask = batch_tensors.get("cross_token_ref_span_mask")
    if (
        teacher_topk_ids is not None
        and ref_topk_ids is not None
        and ref_response_mask is not None
        and ref_span_starts is not None
        and ref_span_mask is not None
    ):
        _log_ref_axis_overlap(
            "a2opd/topk_overlap_target_vs_teacher_ratio",
            ref_topk_ids,
            teacher_topk_ids,
            ref_response_mask,
            ref_span_starts,
            ref_span_mask,
        )

    teacher_base_topk_ids = batch_tensors.get("teacher_base_topk_ids")
    base_ref_topk_ids = batch_tensors.get("base_ref_topk_ids")
    base_ref_response_mask = batch_tensors.get("base_ref_response_mask")
    base_ref_span_starts = batch_tensors.get("cross_token_base_ref_span_starts")
    base_ref_span_mask = batch_tensors.get("cross_token_base_ref_span_mask")
    if base_ref_response_mask is None and ref_response_mask is not None:
        base_ref_response_mask = ref_response_mask
        base_ref_span_starts = ref_span_starts
        base_ref_span_mask = ref_span_mask
        if base_ref_topk_ids is None:
            base_ref_topk_ids = ref_topk_ids
    if (
        teacher_base_topk_ids is not None
        and base_ref_topk_ids is not None
        and base_ref_response_mask is not None
        and base_ref_span_starts is not None
        and base_ref_span_mask is not None
    ):
        _log_ref_axis_overlap(
            "a2opd/topk_overlap_target_vs_teacher_base_ratio",
            base_ref_topk_ids,
            teacher_base_topk_ids,
            base_ref_response_mask,
            base_ref_span_starts,
            base_ref_span_mask,
        )

    if (
        teacher_topk_ids is not None
        and teacher_base_topk_ids is not None
        and teacher_topk_ids.shape == teacher_base_topk_ids.shape
    ):
        teacher_valid = None
        if (
            ref_response_mask is not None
            and teacher_topk_ids.shape[1] == ref_response_mask.shape[1]
            and ref_span_starts is not None
            and ref_span_mask is not None
        ):
            primary_valid = (
                response_mask_bool
                & ref_span_mask.bool()
                & candidate_mask.any(dim=-1)
            )
            teacher_valid = build_ref_axis_topk_valid_mask(
                primary_valid,
                ref_span_starts,
                ref_response_mask,
            )
        elif tuple(teacher_topk_ids.shape[:-1]) == tuple(response_mask_bool.shape):
            teacher_valid = response_mask_bool
        if teacher_valid is not None:
            metrics["a2opd/topk_overlap_teacher_vs_teacher_base_ratio"] = mean_topk_overlap_ratio(
                teacher_topk_ids,
                teacher_base_topk_ids,
                teacher_valid,
            )

    if overlap_valid_tokens <= 0:
        overlap_valid_tokens = float(response_mask_bool.sum().item())
    return metrics, overlap_valid_tokens


@torch.no_grad()
def build_ref_axis_topk_valid_mask(
    primary_valid: torch.Tensor,
    span_starts: torch.Tensor,
    ref_response_mask: torch.Tensor,
) -> torch.Tensor:
    """Mark ref-axis positions that receive at least one mapped primary token."""
    if primary_valid.shape != span_starts.shape:
        raise ValueError(
            "primary_valid and span_starts must share shape (batch, response_length), got "
            f"{tuple(primary_valid.shape)} and {tuple(span_starts.shape)}."
        )

    ref_length = ref_response_mask.shape[1]
    ref_valid = torch.zeros(
        (*primary_valid.shape[:-1], ref_length),
        dtype=torch.bool,
        device=primary_valid.device,
    )
    mapped = primary_valid & span_starts.ge(0)
    if not mapped.any():
        return ref_valid & ref_response_mask.bool()

    batch_indices = torch.arange(primary_valid.shape[0], device=primary_valid.device).unsqueeze(1)
    batch_indices = batch_indices.expand_as(span_starts)
    flat_batch = batch_indices[mapped]
    flat_ref = span_starts[mapped].long()
    ref_valid[flat_batch, flat_ref] = True
    return ref_valid & ref_response_mask.bool()


def align_ref_topk_ids_to_primary(
    ref_topk_ids: torch.Tensor,
    span_starts: torch.Tensor,
    span_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather ref-axis top-k ids back onto primary token positions."""
    safe_positions = span_starts.clamp(min=0, max=ref_topk_ids.shape[1] - 1)
    gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, ref_topk_ids.shape[-1])
    aligned = torch.gather(ref_topk_ids, dim=1, index=gather_index)
    return aligned * span_mask.unsqueeze(-1).to(dtype=aligned.dtype)


def translate_primary_topk_ids(
    topk_ids: torch.Tensor,
    translation_map: torch.Tensor,
) -> torch.Tensor:
    """Map primary-vocab top-k ids into the ref tokenizer space."""
    translation_map = translation_map.to(device=topk_ids.device)
    in_range = topk_ids.ge(0) & topk_ids.lt(translation_map.numel())
    safe_ids = topk_ids.clamp(min=0, max=max(translation_map.numel() - 1, 0))
    translated = translation_map[safe_ids]
    return translated.masked_fill(~in_range, -1)


def align_topk_opd_advantages(
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Make advantages broadcastable with top-k log-probs (B,T,K).

    A²OPD top-k path stores per-candidate advantages as (B,T,K). Legacy
    position-level (B,T) advantages are expanded to (B,T,1) for PPO loss.
    """
    if not is_topk_log_probs(old_log_probs):
        return advantages
    if advantages.dim() == 2:
        return advantages.unsqueeze(-1)
    if advantages.dim() == 3:
        if advantages.shape[-1] not in (1, old_log_probs.shape[-1]):
            raise ValueError(
                f"topk advantages last dim must be 1 or K={old_log_probs.shape[-1]}, "
                f"got shape {tuple(advantages.shape)}"
            )
        return advantages
    raise ValueError(
        f"Expected advantages of shape (B,T) or (B,T,K), got {tuple(advantages.shape)}"
    )


def compute_proxy_signal(
    teacher_log_probs: torch.Tensor,
    teacher_base_log_probs: torch.Tensor,
) -> torch.Tensor:
    """d_t = log pi_phi^+ - log pi_phi."""
    return teacher_log_probs - teacher_base_log_probs


def compute_frozen_utility(
    proxy_signal: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    lambda_k: float,
) -> torch.Tensor:
    """u = d - lambda * log(pi_old / pi_ref). Supports (B,T) or (B,T,K)."""
    anchor_delta = old_log_probs - ref_log_probs
    return proxy_signal - float(lambda_k) * anchor_delta


def compute_baseline(
    utility: torch.Tensor,
    policy_mask: torch.Tensor,
    mode: BaselineMode,
    uids: list | None = None,
) -> torch.Tensor:
    """Baseline for the single-token OPD path (B,T).

    For top-k OPD use ``compute_topk_baseline`` instead (supports state/none/...).
    """
    if is_topk_log_probs(utility):
        raise ValueError(
            "compute_baseline expects position-level utility (B,T). "
            "For top-k OPD, pass utility.mean(dim=-1) first."
        )

    if mode == "none":
        return torch.zeros_like(utility)

    if mode == "state":
        raise ValueError(
            "baseline_mode='state' applies to top-k OPD (B,T,K) only. "
            "Use sequence, batch, group, or none for single-token OPD."
        )

    if mode == "batch":
        baseline = masked_mean(utility, policy_mask)
        return baseline.expand_as(utility)

    if mode == "sequence":
        baseline = masked_mean(utility, policy_mask, dim=1, keepdim=True)
        return baseline.expand_as(utility)

    if mode == "group":
        if uids is None:
            raise ValueError("baseline_mode='group' requires uid in batch.non_tensor_batch.")
        batch_size = utility.shape[0]
        baseline = torch.zeros_like(utility)
        uid_to_rows: dict[str, list[int]] = {}
        for row_idx in range(batch_size):
            uid = str(uids[row_idx])
            uid_to_rows.setdefault(uid, []).append(row_idx)
        for rows in uid_to_rows.values():
            row_tensor = torch.tensor(rows, device=utility.device, dtype=torch.long)
            group_mask = policy_mask.index_select(0, row_tensor)
            group_util = utility.index_select(0, row_tensor)
            group_mean = masked_mean(group_util, group_mask)
            baseline[row_tensor] = group_mean
        return baseline

    raise ValueError(f"Unknown baseline_mode: {mode}")


def compute_topk_candidate_weights(
    old_log_probs: torch.Tensor,
    dtype: torch.dtype | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Student probabilities p_k renormalized over the top-k candidate set.

    Top-k PG contributions must carry this policy weight (c_k = p_k * q_k);
    see ``compute_opd_advantages_topk``.
    """
    logits = old_log_probs.float()
    if candidate_mask is not None:
        logits = logits.masked_fill(~candidate_mask.bool(), -torch.inf)
    weights = torch.softmax(logits, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if dtype is not None:
        weights = weights.to(dtype=dtype)
    return weights


def compute_topk_baseline(
    utility: torch.Tensor,
    old_log_probs: torch.Tensor,
    policy_mask: torch.Tensor,
    mode: BaselineMode,
    uids: list | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Baseline b for top-k OPD utility (B,T,K), broadcastable for subtraction.

    The baseline is always computed on the per-state expected signal
    s_t = E_{a ~ softmax(old_log_probs)}[u_t(a)] = sum_k p_k * u_k, i.e. the
    K-sum of the PG contributions c_k = p_k * u_k. Callers subtract it as
    ``c_k - p_k * b`` so the policy weight is not applied twice (this keeps b
    a valid score-function baseline on the renormalized top-k distribution).

      - ``none``: zero baseline (advantage = raw contribution c_k)
      - ``state``: b_t = s_t per position (B,T,1)
      - ``sequence``: masked mean of s_t over T within each sequence
      - ``batch``: masked mean of s_t over the whole batch
      - ``group``: masked mean of s_t pooled within each prompt uid group
    """
    if not is_topk_log_probs(utility):
        raise ValueError("compute_topk_baseline requires (B,T,K) utility.")

    if mode == "none":
        return torch.zeros_like(utility)

    weights = compute_topk_candidate_weights(
        old_log_probs,
        dtype=utility.dtype,
        candidate_mask=candidate_mask,
    )
    state_expected = (weights * utility).sum(dim=-1, keepdim=True)  # (B,T,1)
    mask = policy_mask.unsqueeze(-1).to(dtype=utility.dtype)

    if mode == "state":
        return state_expected

    if mode == "sequence":
        seq_sum = (state_expected * mask).sum(dim=1, keepdim=True)
        seq_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return seq_sum / seq_count

    if mode == "batch":
        batch_sum = (state_expected * mask).sum()
        batch_count = mask.sum().clamp_min(1.0)
        return batch_sum / batch_count

    if mode == "group":
        if uids is None:
            raise ValueError("baseline_mode='group' requires uid in batch.non_tensor_batch.")
        baseline = torch.zeros_like(state_expected)
        uid_to_rows: dict[str, list[int]] = {}
        for row_idx in range(utility.shape[0]):
            uid_to_rows.setdefault(str(uids[row_idx]), []).append(row_idx)
        for rows in uid_to_rows.values():
            row_tensor = torch.tensor(rows, device=utility.device, dtype=torch.long)
            group_mask = mask.index_select(0, row_tensor)
            group_signal = state_expected.index_select(0, row_tensor)
            group_mean = (group_signal * group_mask).sum() / group_mask.sum().clamp_min(1.0)
            baseline[row_tensor] = group_mean
        return baseline

    raise ValueError(f"Unknown baseline_mode: {mode}")


def compute_opd_advantages_topk(
    teacher_log_probs: torch.Tensor,
    teacher_base_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    policy_mask: torch.Tensor,
    lambda_k: float,
    baseline_mode: BaselineMode = "state",
    uids: list | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-k OPD: utility (B,T,K), per-candidate PG advantage (B,T,K), importance I_t (B,T).

    Per the P-OPD objective J = sum_a pi_theta(a|s) * r*(s,a), each candidate's
    policy-gradient contribution must carry the student probability weight
    p_k (softmax of ``old_log_probs`` over the K candidates). Advantages are
    therefore returned in contribution form

        A_k = p_k * (u_k - b) = c_k - p_k * b,   c_k = p_k * u_k,

    matching the reference OPD implementation (``token_level_rewards`` are
    contributions ``c_k = p_k * q_k``; baselines are applied as ``c_k - p_k*b``).
    Without the p_k factor the loss would weight every candidate uniformly,
    amplifying low-probability tail candidates by 1/p_k and breaking the
    score-function property of the baseline.

    ``baseline_mode`` controls advantage centering (see ``compute_topk_baseline``):
      - ``none``: A_k = c_k  (no baseline)
      - ``state`` (default): b_t = E_{p}[u_t], so A_k = p_k * (u_k - E_p[u_t])

    Token emphasis I_t always uses the p_k-weighted state-centered candidate
    variance, independent of ``baseline_mode``, so micro-layer sampling stays
    stable when baseline is off.
    """
    if not is_topk_log_probs(old_log_probs):
        raise ValueError("compute_opd_advantages_topk requires (B,T,K) log-prob tensors.")

    utility = compute_frozen_utility(
        compute_proxy_signal(teacher_log_probs, teacher_base_log_probs),
        old_log_probs,
        ref_log_probs,
        lambda_k,
    ) # (B,T,K)
    weights = compute_topk_candidate_weights(
        old_log_probs,
        dtype=utility.dtype,
        candidate_mask=candidate_mask,
    )
    contributions = weights * utility  # c_k = p_k * u_k

    state_baseline = compute_topk_baseline(
        utility,
        old_log_probs,
        policy_mask,
        "state",
        candidate_mask=candidate_mask,
    )
    centered_for_importance = utility - state_baseline
    token_importance = (weights * centered_for_importance**2).sum(dim=-1)

    baseline = compute_topk_baseline(
        utility,
        old_log_probs,
        policy_mask,
        baseline_mode,
        uids=uids,
        candidate_mask=candidate_mask,
    )
    advantages = (contributions - weights * baseline) * policy_mask.unsqueeze(-1)  # (B,T,K)
    if candidate_mask is not None:
        advantages = advantages * candidate_mask.to(dtype=advantages.dtype)

    return utility, advantages, token_importance


def compute_opd_advantages(
    teacher_log_probs: torch.Tensor,
    teacher_base_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    policy_mask: torch.Tensor,
    lambda_k: float,
    baseline_mode: BaselineMode = "sequence",
    uids: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frozen utility u_hat and centered advantage A_hat.

    Top-k path (B,T,K): ``baseline_mode`` via ``compute_topk_baseline`` (default state).
    Single-token path (B,T): ``baseline_mode`` via ``compute_baseline`` (no state).
    """
    if is_topk_log_probs(old_log_probs):
        utility, advantages, _ = compute_opd_advantages_topk(
            teacher_log_probs,
            teacher_base_log_probs,
            old_log_probs,
            ref_log_probs,
            policy_mask,
            lambda_k,
            baseline_mode=baseline_mode,
            uids=uids,
        )
        return utility, advantages

    proxy_signal = compute_proxy_signal(teacher_log_probs, teacher_base_log_probs)
    utility = compute_frozen_utility(proxy_signal, old_log_probs, ref_log_probs, lambda_k)
    baseline = compute_baseline(utility, policy_mask, baseline_mode, uids=uids)
    advantages = (utility - baseline) * policy_mask
    return utility, advantages


def estimate_anchor_kl(
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    policy_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> float:
    """Estimate KL(pi_old || pi_ref) on rollout tokens.

    Top-k path: exact KL with both distributions renormalized over the same
    K-candidate set — identical estimator to ``estimate_target_transfer_kl``,
    so the dual-update ratio K_k/B_k compares like with like. The previous
    k3 estimator is unnormalized and grows super-linearly once the policy
    drifts, which made the ratio explode and destabilized the lambda loop.

    Single-token path falls back to the non-negative k3 (low_var_kl)
    per-token estimator.
    """
    if is_topk_log_probs(old_log_probs):
        old_logits = old_log_probs.float()
        ref_logits = ref_log_probs.float()
        if candidate_mask is not None:
            mask = candidate_mask.bool()
            old_logits = old_logits.masked_fill(~mask, -torch.inf)
            ref_logits = ref_logits.masked_fill(~mask, -torch.inf)
        old_logp = torch.log_softmax(old_logits, dim=-1)
        ref_logp = torch.log_softmax(ref_logits, dim=-1)
        per_candidate_kl = old_logp.exp() * (old_logp - ref_logp)
        if candidate_mask is not None:
            per_candidate_kl = torch.where(
                candidate_mask.bool(),
                per_candidate_kl,
                torch.zeros_like(per_candidate_kl),
            )
        per_token_kl = per_candidate_kl.sum(dim=-1)
        per_token_kl = torch.nan_to_num(per_token_kl, nan=0.0, posinf=0.0, neginf=0.0)
        return masked_mean(per_token_kl, policy_mask).item()

    from verl.trainer.ppo.core_algos import kl_penalty_forward

    kld = kl_penalty_forward(old_log_probs, ref_log_probs, "low_var_kl")
    return masked_mean(kld, policy_mask).item()


def estimate_proxy_kl(
    teacher_log_probs: torch.Tensor,
    teacher_base_log_probs: torch.Tensor,
    policy_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> float:
    """Non-negative estimate of KL(teacher+ || teacher_base) on rollout tokens.

    Top-k path: exact KL with both distributions renormalized over the same
    K-candidate set — identical estimator to ``estimate_target_transfer_kl``
    and the top-k branch of ``estimate_anchor_kl``, so the budget B_k and the
    dual-update ratio K_k/B_k are all measured on the same scale. The signed
    mean log-ratio ``E[log pi+ - log pi_base]`` can be negative on student
    samples and must not be used as a trust-region budget.

    Single-token path falls back to the non-negative k3 (low_var_kl)
    per-token estimator.
    """
    if is_topk_log_probs(teacher_log_probs):
        teacher_logits = teacher_log_probs.float()
        teacher_base_logits = teacher_base_log_probs.float()
        if candidate_mask is not None:
            mask = candidate_mask.bool()
            teacher_logits = teacher_logits.masked_fill(~mask, -torch.inf)
            teacher_base_logits = teacher_base_logits.masked_fill(~mask, -torch.inf)
        teacher_logp = torch.log_softmax(teacher_logits, dim=-1)
        teacher_base_logp = torch.log_softmax(teacher_base_logits, dim=-1)
        per_candidate_kl = teacher_logp.exp() * (teacher_logp - teacher_base_logp)
        if candidate_mask is not None:
            per_candidate_kl = torch.where(
                candidate_mask.bool(),
                per_candidate_kl,
                torch.zeros_like(per_candidate_kl),
            )
        per_token_kl = per_candidate_kl.sum(dim=-1)
        per_token_kl = torch.nan_to_num(per_token_kl, nan=0.0, posinf=0.0, neginf=0.0)
        return masked_mean(per_token_kl, policy_mask).item()

    from verl.trainer.ppo.core_algos import kl_penalty_forward

    kld = kl_penalty_forward(teacher_log_probs, teacher_base_log_probs, "low_var_kl")
    return masked_mean(kld, policy_mask).item()


def estimate_proxy_base_target_base_kl(
    teacher_base_log_probs: torch.Tensor,
    target_base_log_probs: torch.Tensor,
    policy_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> float:
    """Estimate KL(pi_proxy_base || pi_target_base) on rollout tokens.

    Uses the same candidate-set KL estimator as ``estimate_anchor_kl``.
    """
    return estimate_anchor_kl(
        teacher_base_log_probs,
        target_base_log_probs,
        policy_mask,
        candidate_mask=candidate_mask,
    )


def estimate_target_transfer_kl(
    teacher_log_probs: torch.Tensor,
    teacher_base_log_probs: torch.Tensor,
    target_base_log_probs: torch.Tensor,
    policy_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> float:
    """Estimate B_tr on the OPD candidate set.

    On each token position and its K retained student candidates, construct
    pi_tr(a) proportional to pi_ref(a) exp(d(a)), where
    d(a) = log pi_teacher+(a) - log pi_teacher(a).  Both pi_tr and pi_ref are
    normalized over the same candidate set before computing KL(pi_tr || pi_ref).

    This is a top-K approximation of the full-vocabulary transfer KL.  Exact
    full-vocabulary evaluation would require retaining logits for every token,
    which is infeasible for the long-sequence OPD batches.
    """
    if not is_topk_log_probs(teacher_log_probs):
        raise ValueError(
            "target_transfer_kl budget requires top-k OPD log-probs (B, T, K). "
            "Set algorithm.hetero_distill.opd_top_k > 1."
        )
    if teacher_log_probs.shape[-1] <= 1:
        raise ValueError("target_transfer_kl budget requires at least two OPD candidates.")

    proxy_signal = compute_proxy_signal(teacher_log_probs, teacher_base_log_probs)
    ref_logits = target_base_log_probs.float()
    transfer_logits = target_base_log_probs.float() + proxy_signal.float()
    if candidate_mask is not None:
        mask = candidate_mask.bool()
        ref_logits = ref_logits.masked_fill(~mask, -torch.inf)
        transfer_logits = transfer_logits.masked_fill(~mask, -torch.inf)
    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
    transfer_log_probs = torch.log_softmax(transfer_logits, dim=-1)
    per_candidate_kl = transfer_log_probs.exp() * (transfer_log_probs - ref_log_probs)
    if candidate_mask is not None:
        per_candidate_kl = torch.where(
            candidate_mask.bool(),
            per_candidate_kl,
            torch.zeros_like(per_candidate_kl),
        )
    per_token_kl = per_candidate_kl.sum(dim=-1)
    per_token_kl = torch.nan_to_num(per_token_kl, nan=0.0, posinf=0.0, neginf=0.0)
    return masked_mean(per_token_kl, policy_mask).item()


def estimate_transfer_gap(
    old_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    teacher_base_log_probs: torch.Tensor,
    target_base_log_probs: torch.Tensor,
    policy_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> float:
    """Estimate KL(pi_old || pi_transfer) on rollout tokens.

    pi_transfer is the theoretical transferred target distribution:
    pi_transfer(a) proportional to pi_base(a) exp(d(a)), where
    d(a) = log pi_teacher+(a) - log pi_teacher_base(a).

    Together with ``estimate_anchor_kl`` (KL(pi_old || pi_base)) and
    ``estimate_target_transfer_kl`` (KL(pi_transfer || pi_base)), this
    measures how far the current target policy is from the induced transfer
    distribution on the same OPD candidate set.
    """
    if is_topk_log_probs(old_log_probs):
        proxy_signal = compute_proxy_signal(teacher_log_probs, teacher_base_log_probs)
        old_logits = old_log_probs.float()
        transfer_logits = target_base_log_probs.float() + proxy_signal.float()
        if candidate_mask is not None:
            mask = candidate_mask.bool()
            old_logits = old_logits.masked_fill(~mask, -torch.inf)
            transfer_logits = transfer_logits.masked_fill(~mask, -torch.inf)
        old_logp = torch.log_softmax(old_logits, dim=-1)
        transfer_logp = torch.log_softmax(transfer_logits, dim=-1)
        per_candidate_kl = old_logp.exp() * (old_logp - transfer_logp)
        if candidate_mask is not None:
            per_candidate_kl = torch.where(
                candidate_mask.bool(),
                per_candidate_kl,
                torch.zeros_like(per_candidate_kl),
            )
        per_token_kl = per_candidate_kl.sum(dim=-1)
        per_token_kl = torch.nan_to_num(per_token_kl, nan=0.0, posinf=0.0, neginf=0.0)
        return masked_mean(per_token_kl, policy_mask).item()

    from verl.trainer.ppo.core_algos import kl_penalty_forward

    proxy_signal = teacher_log_probs - teacher_base_log_probs
    transfer_log_probs = target_base_log_probs + proxy_signal
    kld = kl_penalty_forward(old_log_probs, transfer_log_probs, "low_var_kl")
    return masked_mean(kld, policy_mask).item()


def compute_token_sampling_prob(
    teacher_log_probs: torch.Tensor,
    teacher_base_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    policy_mask: torch.Tensor,
    lambda_k: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Build q_t ∝ sqrt(I_t + eps) using top-k candidate variance.

    Requires ``opd_top_k > 0`` so log-prob tensors are (B,T,K). At each position:
      I_t = Var_{a in topK}[u_t(a)] ≈ E_a[(u_t(a) - E_a[u_t(a)])^2]

    Returns q_t with sum 1 over valid response positions (token-position emphasis).
    """
    if not is_topk_log_probs(old_log_probs):
        raise ValueError(
            "compute_token_sampling_prob requires opd_top_k > 0 with (B,T,K) log-prob tensors. "
            "Enable algorithm.hetero_distill.opd_top_k before token_sampling_enable."
        )

    _, _, token_importance = compute_opd_advantages_topk(
        teacher_log_probs,
        teacher_base_log_probs,
        old_log_probs,
        ref_log_probs,
        policy_mask,
        lambda_k,
    )
    return token_sampling_prob_from_importance(token_importance, policy_mask, epsilon)


def token_sampling_prob_from_importance(
    token_importance: torch.Tensor,
    policy_mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Build q_t ∝ sqrt(I_t + eps) from precomputed per-position importance."""
    weights = torch.sqrt(token_importance + epsilon) * policy_mask
    return weights / weights.sum().clamp(min=epsilon)