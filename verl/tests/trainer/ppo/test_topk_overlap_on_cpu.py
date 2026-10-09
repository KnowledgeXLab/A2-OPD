import pytest
import torch

from verl.trainer.ppo.a2opd_utils import (
    build_ref_axis_topk_valid_mask,
    compute_a2opd_topk_overlap_metrics,
    mean_topk_overlap_ratio,
)


def test_mean_topk_overlap_ratio_uses_valid_response_positions_only():
    target = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]])
    other = torch.tensor([[[4, 3, 20, 21], [8, 7, 6, 5], [99, 100, 101, 102]]])
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    # Valid positions overlap by 2/4 and 4/4, respectively.
    assert mean_topk_overlap_ratio(target, other, response_mask) == pytest.approx(0.75)


def test_mean_topk_overlap_ratio_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        mean_topk_overlap_ratio(
            torch.tensor([[[1, 2]]]),
            torch.tensor([[[1, 2, 3]]]),
            torch.tensor([[1]], dtype=torch.bool),
        )


def test_compute_a2opd_topk_overlap_metrics_same_tokenizer():
    batch_tensors = {
        "student_topk_ids": torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]]),
        "teacher_topk_ids": torch.tensor([[[4, 3, 20, 21], [8, 7, 6, 5]]]),
        "teacher_base_topk_ids": torch.tensor([[[4, 3, 2, 1], [8, 7, 6, 5]]]),
        "response_mask": torch.tensor([[1, 1]], dtype=torch.bool),
    }
    metrics, overlap_valid_tokens = compute_a2opd_topk_overlap_metrics(batch_tensors)
    assert metrics["a2opd/topk_overlap_target_vs_teacher_ratio"] == pytest.approx(0.75)
    assert metrics["a2opd/topk_overlap_teacher_vs_teacher_base_ratio"] == pytest.approx(0.75)
    assert overlap_valid_tokens == 2.0


def test_build_ref_axis_topk_valid_mask_marks_mapped_ref_positions():
    primary_valid = torch.tensor([[True, False, True]], dtype=torch.bool)
    span_starts = torch.tensor([[1, -1, 0]])
    ref_response_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)

    ref_valid = build_ref_axis_topk_valid_mask(primary_valid, span_starts, ref_response_mask)

    assert ref_valid.tolist() == [[True, True, False]]
