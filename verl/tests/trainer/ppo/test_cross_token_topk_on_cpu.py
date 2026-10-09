import torch
import pytest

from verl.trainer.ppo.a2opd_utils import (
    compute_opd_advantages_topk,
    compute_topk_candidate_weights,
    estimate_target_transfer_kl,
    estimate_transfer_gap,
)
from verl.trainer.ppo.a2opd_controller import A2OPDController
from verl.trainer.ppo.ref_input_utils import (
    _align_primary_to_ref_positions,
    align_log_probs_by_text_spans,
    build_single_token_translation_map,
)
from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class _FakeTokenizer:
    def __init__(self, pieces: dict[int, str], encodings: dict[str, tuple[list[int], list[tuple[int, int]]]]):
        self._pieces = pieces
        self._encodings = encodings
        self.all_special_ids = []

    def get_vocab(self):
        return {f"token_{token_id}": token_id for token_id in self._pieces}

    def get_added_vocab(self):
        return {}

    def batch_decode(self, rows, **_kwargs):
        return [self._pieces[row[0]] for row in rows]

    def __call__(self, text, **_kwargs):
        token_ids, offsets = self._encodings[text]
        return {"input_ids": token_ids, "offset_mapping": offsets}


def test_single_token_translation_uses_text_not_token_id():
    primary = _FakeTokenizer(
        {0: "a", 1: " b", 2: "multi"},
        {},
    )
    ref = _FakeTokenizer(
        {5: "a", 7: " b", 9: "other"},
        {},
    )

    translation, stats = build_single_token_translation_map(primary, ref)

    assert translation.tolist() == [5, 7, -1]
    assert stats["single_token_mapped"] == 2


def test_primary_positions_align_by_text_offsets_not_token_index():
    text = "abcd"
    primary = _FakeTokenizer(
        {0: "a", 1: "bc", 2: "d"},
        {text: ([0, 1, 2], [(0, 1), (1, 3), (3, 4)])},
    )
    ref = _FakeTokenizer(
        {5: "ab", 6: "c", 7: "d"},
        {text: ([5, 6, 7], [(0, 2), (2, 3), (3, 4)])},
    )

    starts, ends, span_mask, success = _align_primary_to_ref_positions(
        primary,
        ref,
        text,
        primary_ids=torch.tensor([0, 1, 2]),
        ref_ids=torch.tensor([5, 6, 7]),
        primary_response_length=3,
    )

    assert success
    assert starts.tolist() == [0, -1, 2]
    assert ends.tolist() == [-1, -1, 3]
    assert span_mask.tolist() == [0, 0, 1]


def test_single_token_path_sums_independent_ref_token_spans():
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.zeros((1, 2), dtype=torch.long),
            "teacher_base_log_probs": torch.tensor([[1.0, 2.0, 3.0]]),
            "cross_token_base_ref_span_starts": torch.tensor([[0, 2]]),
            "cross_token_base_ref_span_ends": torch.tensor([[2, 3]]),
            "cross_token_base_ref_span_mask": torch.tensor([[1, 1]]),
        }
    )

    batch, _ = align_log_probs_by_text_spans(
        batch,
        "teacher_base_log_probs",
        "base_ref",
    )

    assert batch.batch["teacher_base_log_probs"].tolist() == [[3.0, 3.0]]


def test_masked_topk_advantage_renormalizes_over_translated_candidates():
    old = torch.tensor([[[3.0, 2.0, 1.0, 0.0]]])
    ref = torch.tensor([[[2.5, 1.5, 0.5, -0.5]]])
    teacher = torch.tensor([[[1.0, 3.0, 9.0, 9.0]]])
    teacher_base = torch.zeros_like(teacher)
    candidate_mask = torch.tensor([[[True, True, False, False]]])
    policy_mask = torch.ones((1, 1))

    utility, advantages, _ = compute_opd_advantages_topk(
        teacher,
        teacher_base,
        old,
        ref,
        policy_mask,
        lambda_k=0.5,
        baseline_mode="state",
        candidate_mask=candidate_mask,
    )
    weights = compute_topk_candidate_weights(old, candidate_mask=candidate_mask)

    assert torch.allclose(weights.sum(dim=-1), torch.ones((1, 1)))
    assert torch.equal(weights[..., 2:], torch.zeros_like(weights[..., 2:]))
    assert torch.equal(advantages[..., 2:], torch.zeros_like(advantages[..., 2:]))
    assert torch.allclose(advantages.sum(dim=-1), torch.zeros((1, 1)), atol=1e-6)

    transfer_kl = estimate_target_transfer_kl(
        teacher,
        teacher_base,
        ref,
        policy_mask,
        candidate_mask=candidate_mask,
    )
    assert torch.isfinite(torch.tensor(transfer_kl))
    assert utility.shape == old.shape

    transfer_gap = estimate_transfer_gap(
        old,
        teacher,
        teacher_base,
        ref,
        policy_mask,
        candidate_mask=candidate_mask,
    )
    assert torch.isfinite(torch.tensor(transfer_gap))
    assert transfer_gap >= 0.0


def test_proxy_kl_ema_tracks_exponential_average():
    controller = A2OPDController(
        budget_adaptive_enable=False,
        budget_transfer_ema_tau=0.5,
    )
    assert controller.update_proxy_kl_ema(0.2) == pytest.approx(0.2)
    assert controller.update_proxy_kl_ema(0.4) == pytest.approx(0.3)

def test_align_topk_output_unpacks_tuple_before_primary_alignment():
    ref_log_probs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    topk_ids = torch.tensor([[[10, 20], [30, 40]]])
    data = DataProto.from_dict(
        tensors={
            "cross_token_topk_mask": torch.tensor([[[True, False]]]),
            "cross_token_ref_span_starts": torch.tensor([[1]]),
        }
    )

    aligned_log_probs, aligned_topk_ids = ActorRolloutRefWorker._align_topk_output_to_primary(
        (ref_log_probs, topk_ids),
        data,
        "ref",
        return_topk_ids=True,
    )

    assert aligned_topk_ids is topk_ids
    assert aligned_log_probs.tolist() == [[[3.0, 0.0]]]