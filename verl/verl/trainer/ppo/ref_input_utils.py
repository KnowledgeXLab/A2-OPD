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
"""
Utilities for preparing reference / proxy model inputs when the ref model uses a
different tokenizer, chat template, or both.

Phase 0/1 cross-tokenizer hetero OPD:
  - Text-bridge responses: primary decode -> ref encode
  - Exact text-span alignment from each primary token to one or more ref tokens
  - Teacher log-prob sums aligned back to the primary response length

Cross-tokenizer top-k OPD:
  - Map a primary token id to a ref token id when both single tokens decode to
    exactly the same text (ids do not need to match)
  - Align primary and ref response positions by decoded-text character offsets
  - Score the translated candidates with one normal ref teacher-forcing pass
"""

from __future__ import annotations


def _cross_token_metric_role(namespace: str) -> str:
    """Map internal namespace to wandb role names: target=student, proxy=teacher."""
    if namespace == "ref":
        return "proxy"
    if namespace == "base_ref":
        return "proxy_base"
    return namespace

import logging
import os
from typing import Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def tokenizers_need_cross_token_bridge(primary_tokenizer, ref_tokenizer) -> bool:
    """Return True when primary and ref tokenizers are not byte-identical vocab maps."""
    if ref_tokenizer is None:
        return False
    if primary_tokenizer is ref_tokenizer:
        return False
    return primary_tokenizer.get_vocab() != ref_tokenizer.get_vocab()


def _decode_single_token_pieces(tokenizer, token_ids: list[int], chunk_size: int = 4096) -> list[str]:
    """Decode token ids independently without cleaning up whitespace."""
    pieces: list[str] = []
    for start in range(0, len(token_ids), chunk_size):
        chunk = token_ids[start : start + chunk_size]
        pieces.extend(
            tokenizer.batch_decode(
                [[token_id] for token_id in chunk],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    return pieces


def build_single_token_translation_map(primary_tokenizer, ref_tokenizer) -> tuple[torch.Tensor, dict[str, float]]:
    """Build primary-id -> ref-id mapping by decoded single-token text.

    Only unambiguous, non-special, non-empty pieces that are represented by one
    token on both sides are retained. Invalid or multi-token translations are
    encoded as -1. This deliberately compares decoded text rather than token ids
    or tokenizer-internal token strings.
    """
    primary_vocab = primary_tokenizer.get_vocab()
    ref_vocab = ref_tokenizer.get_vocab()
    primary_ids = sorted(set(primary_vocab.values()))
    ref_ids = sorted(set(ref_vocab.values()))
    primary_special = set(primary_tokenizer.all_special_ids) | set(
        primary_tokenizer.get_added_vocab().values()
    )
    ref_special = set(ref_tokenizer.all_special_ids) | set(
        ref_tokenizer.get_added_vocab().values()
    )

    ref_text_to_id: dict[str, int] = {}
    ambiguous_ref_texts: set[str] = set()
    for token_id, text in zip(ref_ids, _decode_single_token_pieces(ref_tokenizer, ref_ids), strict=True):
        if token_id in ref_special or not text or "\ufffd" in text:
            continue
        if text in ref_text_to_id:
            ambiguous_ref_texts.add(text)
        else:
            ref_text_to_id[text] = token_id
    for text in ambiguous_ref_texts:
        ref_text_to_id.pop(text, None)

    map_size = max(primary_ids, default=-1) + 1
    translation = torch.full((map_size,), -1, dtype=torch.long)
    mapped = 0
    for token_id, text in zip(
        primary_ids,
        _decode_single_token_pieces(primary_tokenizer, primary_ids),
        strict=True,
    ):
        if token_id in primary_special or not text or "\ufffd" in text:
            continue
        ref_id = ref_text_to_id.get(text)
        if ref_id is not None:
            translation[token_id] = ref_id
            mapped += 1

    stats = {
        "single_token_mapped": float(mapped),
        "single_token_mapped_ratio": float(mapped / max(len(primary_ids), 1)),
    }
    return translation, stats


def _response_token_offsets(tokenizer, text: str, expected_ids: torch.Tensor) -> list[tuple[int, int]] | None:
    """Return canonical character offsets, or None when decode->encode changes ids."""
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError):
        return None

    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    expected = expected_ids.detach().cpu().tolist()
    if token_ids != expected:
        return None

    offsets = encoded["offset_mapping"]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    return [(int(start), int(end)) for start, end in offsets]


def _align_primary_to_ref_positions(
    primary_tokenizer,
    ref_tokenizer,
    response_text: str,
    primary_ids: torch.Tensor,
    ref_ids: torch.Tensor,
    primary_response_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Map primary token text spans to exact ref-token spans."""
    span_starts = torch.full((primary_response_length,), -1, dtype=torch.long)
    span_ends = torch.full((primary_response_length,), -1, dtype=torch.long)
    span_mask = torch.zeros(primary_response_length, dtype=torch.long)
    primary_offsets = _response_token_offsets(primary_tokenizer, response_text, primary_ids)
    ref_offsets = _response_token_offsets(ref_tokenizer, response_text, ref_ids)
    if primary_offsets is None or ref_offsets is None:
        return span_starts, span_ends, span_mask, False

    ref_start_to_index = {
        start: index
        for index, (start, end) in enumerate(ref_offsets)
        if end > start
    }
    ref_end_to_index = {
        end: index + 1
        for index, (start, end) in enumerate(ref_offsets)
        if end > start
    }
    for primary_index, (start, end) in enumerate(primary_offsets):
        ref_start = ref_start_to_index.get(start)
        ref_end = ref_end_to_index.get(end)
        if ref_start is not None:
            span_starts[primary_index] = ref_start
        if end > start and ref_start is not None and ref_end is not None and ref_end > ref_start:
            span_ends[primary_index] = ref_end
            span_mask[primary_index] = 1
    return span_starts, span_ends, span_mask, True


def _get_primary_response_mask(batch: DataProto, batch_idx: int, response_length: int) -> torch.Tensor:
    if "response_mask" in batch.batch:
        return batch.batch["response_mask"][batch_idx].to(dtype=torch.long)
    return torch.ones(response_length, dtype=torch.long)


def _decode_primary_response_text(primary_tokenizer, responses_row: torch.Tensor, response_mask_row: torch.Tensor) -> str:
    valid_len = int(response_mask_row.sum().item())
    if valid_len <= 0:
        return ""
    valid_ids = responses_row[:valid_len]
    return primary_tokenizer.decode(valid_ids, skip_special_tokens=False)


def _right_pad_1d(tokens: torch.Tensor, target_len: int, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    cur_len = len(tokens)
    if cur_len >= target_len:
        trimmed = tokens[:target_len]
        mask = torch.ones(target_len, dtype=torch.long)
        return trimmed, mask
    pad_len = target_len - cur_len
    padding = torch.full((pad_len,), pad_token_id, dtype=tokens.dtype)
    padded = torch.cat([tokens, padding], dim=0)
    mask = torch.cat([torch.ones(cur_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)], dim=0)
    return padded, mask


def prepare_ref_model_inputs(
    batch: DataProto,
    ref_tokenizer,
    primary_tokenizer=None,
    cross_token_bridge: bool = False,
    apply_chat_template_kwargs: Optional[dict] = None,
    namespace: str = "ref",
) -> tuple[DataProto, dict]:
    """Prepare ref-side tensors for proxy / teacher log-prob computation.

    When ``cross_token_bridge`` is False (same vocab, possibly different chat template),
    responses are reused from the primary rollout as before.

    When ``cross_token_bridge`` is True, responses are bridged through text and
    exact primary-token -> ref-token text spans are recorded. ``namespace``
    keeps teacher+ (``ref``) and teacher-base (``base_ref``) tensors separate.
    """
    if apply_chat_template_kwargs is None:
        apply_chat_template_kwargs = {}
    
    # Check if raw_prompt is available
    if "raw_prompt" not in batch.non_tensor_batch:
        raise ValueError(
            "raw_prompt not found in batch.non_tensor_batch. "
            "Please set data.return_raw_chat=True in config to enable re-tokenization for ref model."
        )
    if cross_token_bridge and primary_tokenizer is None:
        raise ValueError("primary_tokenizer is required when cross_token_bridge=True.")

    batch_size = len(batch)
    raw_prompts = batch.non_tensor_batch["raw_prompt"]
    responses = batch.batch["responses"]
    primary_response_length = responses.shape[1]

    if "response_mask" in batch.batch:
        primary_response_attention_mask = batch.batch["response_mask"].to(dtype=torch.long)
    else:
        primary_response_attention_mask = torch.ones_like(responses, dtype=torch.long)

    ref_prompt_ids_list = []
    ref_response_ids_list = []
    cross_token_ref_span_starts = []
    cross_token_ref_span_ends = []
    cross_token_ref_span_masks = []
    offset_alignment_successes = 0

    for i in range(batch_size):
        # Get the raw messages for this sample
        messages = raw_prompts[i]
        if not isinstance(messages, (list, np.ndarray)):
            raise TypeError(f"raw_prompt must be a list or numpy array, got {type(messages)}")
        messages = list(messages)
        
        # Apply chat template to get the prompt string using ref tokenizer
        ref_prompt_str = ref_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Tokenize prompt
        ref_prompt_output = ref_tokenizer(
            ref_prompt_str, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        ref_prompt_ids = ref_prompt_output["input_ids"][0]
        ref_prompt_ids_list.append(ref_prompt_ids)

        if cross_token_bridge:
            response_mask_row = _get_primary_response_mask(batch, i, primary_response_length)
            response_text = _decode_primary_response_text(
                primary_tokenizer, responses[i], response_mask_row
            )
            ref_resp_ids = torch.tensor(
                ref_tokenizer.encode(response_text, add_special_tokens=False),
                dtype=responses.dtype,
            )
            valid_len = int(response_mask_row.sum().item())
            primary_resp_valid = responses[i, :valid_len]
            ref_span_starts, ref_span_ends, ref_span_mask, offsets_ok = _align_primary_to_ref_positions(
                primary_tokenizer=primary_tokenizer,
                ref_tokenizer=ref_tokenizer,
                response_text=response_text,
                primary_ids=primary_resp_valid,
                ref_ids=ref_resp_ids,
                primary_response_length=primary_response_length,
            )
            cross_token_ref_span_starts.append(ref_span_starts)
            cross_token_ref_span_ends.append(ref_span_ends)
            cross_token_ref_span_masks.append(ref_span_mask * response_mask_row)
            offset_alignment_successes += int(offsets_ok)
            ref_response_ids_list.append(ref_resp_ids)
        else:
            ref_response_ids_list.append(responses[i])

    max_prompt_len = max(len(ids) for ids in ref_prompt_ids_list)
    ref_prompt_ids_padded = []
    ref_prompt_attention_mask_padded = []

    for ref_prompt_ids in ref_prompt_ids_list:
        prompt_len = len(ref_prompt_ids)
        pad_len = max_prompt_len - prompt_len
        if pad_len > 0:
            # Left pad prompt ids
            padding = torch.full((pad_len,), ref_tokenizer.pad_token_id, dtype=ref_prompt_ids.dtype)
            ref_prompt_ids_padded_item = torch.cat([padding, ref_prompt_ids], dim=0)
            # Create attention mask: 0 for padding, 1 for real tokens
            attention_mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(prompt_len, dtype=torch.long)
            ], dim=0)
        else:
            ref_prompt_ids_padded_item = ref_prompt_ids
            attention_mask = torch.ones(prompt_len, dtype=torch.long)
        ref_prompt_ids_padded.append(ref_prompt_ids_padded_item)
        ref_prompt_attention_mask_padded.append(attention_mask)

    ref_prompt_ids_tensor = torch.stack(ref_prompt_ids_padded, dim=0)
    ref_prompt_attention_mask_tensor = torch.stack(ref_prompt_attention_mask_padded, dim=0)

    if cross_token_bridge:
        max_ref_response_len = max(len(ids) for ids in ref_response_ids_list)
        ref_responses_padded = []
        ref_response_mask_padded = []
        primary_ref_len_diffs = []

        for i, ref_resp_ids in enumerate(ref_response_ids_list):
            padded_resp, ref_resp_mask = _right_pad_1d(
                ref_resp_ids,
                max_ref_response_len,
                ref_tokenizer.pad_token_id,
            )
            ref_responses_padded.append(padded_resp)
            ref_response_mask_padded.append(ref_resp_mask)
            primary_valid_len = int(primary_response_attention_mask[i].sum().item())
            primary_ref_len_diffs.append(abs(primary_valid_len - len(ref_resp_ids)))

        ref_responses_tensor = torch.stack(ref_responses_padded, dim=0)
        ref_response_mask_tensor = torch.stack(ref_response_mask_padded, dim=0)
        cross_token_ref_span_starts_tensor = torch.stack(cross_token_ref_span_starts, dim=0)
        cross_token_ref_span_ends_tensor = torch.stack(cross_token_ref_span_ends, dim=0)
        cross_token_ref_span_mask_tensor = torch.stack(cross_token_ref_span_masks, dim=0)

        ref_input_ids_tensor = torch.cat([ref_prompt_ids_tensor, ref_responses_tensor], dim=1)
        ref_attention_mask_tensor = torch.cat(
            [ref_prompt_attention_mask_tensor, ref_response_mask_tensor], dim=1
        )

        batch.batch[f"{namespace}_responses"] = ref_responses_tensor
        batch.batch[f"{namespace}_response_mask"] = ref_response_mask_tensor
        batch.batch[f"cross_token_{namespace}_span_starts"] = cross_token_ref_span_starts_tensor
        batch.batch[f"cross_token_{namespace}_span_ends"] = cross_token_ref_span_ends_tensor
        batch.batch[f"cross_token_{namespace}_span_mask"] = cross_token_ref_span_mask_tensor

        valid_primary_tokens = primary_response_attention_mask.sum().clamp(min=1).float()
        aligned_tokens = cross_token_ref_span_mask_tensor.sum().float()
        role = _cross_token_metric_role(namespace)
        stats = {
            f"hetero/cross_token_target_{role}_aligned_span_ratio": (
                aligned_tokens / valid_primary_tokens
            ).item(),
            f"hetero/cross_token_target_{role}_len_diff_mean": float(
                np.mean(primary_ref_len_diffs)
            ),
            f"hetero/cross_token_{role}_response_len_mean": float(
                ref_response_mask_tensor.sum(dim=-1).float().mean().item()
            ),
            f"hetero/cross_token_target_{role}_offset_success_ratio": float(
                offset_alignment_successes / max(batch_size, 1)
            ),
        }
    else:
        ref_input_ids_tensor = torch.cat([ref_prompt_ids_tensor, responses], dim=1)
        ref_attention_mask_tensor = torch.cat(
            [ref_prompt_attention_mask_tensor, primary_response_attention_mask], dim=1
        )
        stats = {}

    ref_position_ids_tensor = compute_position_id_with_mask(ref_attention_mask_tensor)
    
    # Add to batch
    batch.batch[f"{namespace}_input_ids"] = ref_input_ids_tensor
    batch.batch[f"{namespace}_attention_mask"] = ref_attention_mask_tensor
    batch.batch[f"{namespace}_position_ids"] = ref_position_ids_tensor

    logger.info(
        "Prepared %s model inputs: primary_responses=%s ref_input_ids=%s cross_token_bridge=%s",
        namespace,
        tuple(responses.shape),
        tuple(ref_input_ids_tensor.shape),
        cross_token_bridge,
    )

    return batch, stats


def align_log_probs_by_text_spans(
    batch: DataProto,
    key: str,
    namespace: str,
) -> tuple[DataProto, dict]:
    """Sum ref-token log probs over each exact primary-token text span."""
    if key not in batch.batch:
        return batch, {}
    log_probs = batch.batch[key]
    # Cross-token top-k worker outputs are already on the primary (B,T,K) axis.
    if log_probs.dim() == 3:
        return batch, {}

    starts_key = f"cross_token_{namespace}_span_starts"
    ends_key = f"cross_token_{namespace}_span_ends"
    mask_key = f"cross_token_{namespace}_span_mask"
    if starts_key not in batch.batch:
        return batch, {}

    starts = batch.batch[starts_key].to(device=log_probs.device)
    ends = batch.batch[ends_key].to(device=log_probs.device)
    span_mask = batch.batch[mask_key].to(device=log_probs.device, dtype=torch.bool)
    prefix_sums = torch.cat(
        [
            torch.zeros(
                (log_probs.shape[0], 1),
                device=log_probs.device,
                dtype=log_probs.dtype,
            ),
            log_probs.cumsum(dim=1),
        ],
        dim=1,
    )
    safe_starts = starts.clamp(min=0, max=log_probs.shape[1])
    safe_ends = ends.clamp(min=0, max=log_probs.shape[1])
    span_log_probs = torch.gather(prefix_sums, 1, safe_ends) - torch.gather(
        prefix_sums, 1, safe_starts
    )
    batch.batch[key] = torch.where(
        span_mask,
        span_log_probs,
        torch.zeros_like(span_log_probs),
    )
    return batch, {}


def align_teacher_log_probs_to_primary(batch: DataProto) -> tuple[DataProto, dict]:
    """Align teacher+ and teacher-base scores using their own token spans."""
    stats: dict[str, float] = {}
    batch, teacher_stats = align_log_probs_by_text_spans(batch, "teacher_log_probs", "ref")
    base_namespace = (
        "base_ref"
        if "cross_token_base_ref_span_starts" in batch.batch
        else "ref"
    )
    batch, base_stats = align_log_probs_by_text_spans(
        batch,
        "teacher_base_log_probs",
        base_namespace,
    )
    stats.update(teacher_stats)
    stats.update(base_stats)
    return batch, stats
