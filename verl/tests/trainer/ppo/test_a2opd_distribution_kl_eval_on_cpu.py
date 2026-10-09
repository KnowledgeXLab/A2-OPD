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

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.a2opd_distribution_kl_eval import (
    aggregate_distribution_kl_metrics,
    compute_distribution_kl_metrics,
    load_rollout_cache,
    save_rollout_cache,
)
from verl.trainer.ppo.a2opd_utils import materialize_dataproto_cpu, prepare_dataproto_for_opd_prep


def test_aggregate_distribution_kl_metrics_token_weighted():
    batch_a = {
        "a2opd/proxy_kl": 1.0,
        "a2opd/proxy_base_target_base_kl": 1.5,
        "a2opd/target_transfer_kl": 2.0,
        "a2opd/anchor_kl": 3.0,
        "a2opd/transfer_gap": 4.0,
        "_valid_tokens": 10.0,
        "_num_samples": 1.0,
    }
    batch_b = {
        "a2opd/proxy_kl": 3.0,
        "a2opd/proxy_base_target_base_kl": 3.5,
        "a2opd/target_transfer_kl": 4.0,
        "a2opd/anchor_kl": 5.0,
        "a2opd/transfer_gap": 6.0,
        "_valid_tokens": 30.0,
        "_num_samples": 1.0,
    }
    aggregated = aggregate_distribution_kl_metrics([batch_a, batch_b])
    assert aggregated["a2opd/proxy_kl"] == pytest.approx((1 * 10 + 3 * 30) / 40)
    assert aggregated["distribution_kl/valid_tokens"] == 40.0


def test_aggregate_distribution_kl_metrics_reports_skipped_batches():
    batch_a = {
        "a2opd/proxy_kl": 1.0,
        "a2opd/proxy_base_target_base_kl": 1.5,
        "a2opd/target_transfer_kl": 2.0,
        "a2opd/anchor_kl": 3.0,
        "a2opd/transfer_gap": 4.0,
        "_valid_tokens": 10.0,
        "_num_samples": 4.0,
    }
    aggregated = aggregate_distribution_kl_metrics(
        [batch_a],
        skipped_batches=1,
        skipped_samples=128.0,
    )
    assert aggregated["distribution_kl/skipped_batches"] == 1.0
    assert aggregated["distribution_kl/skipped_samples"] == 128.0


def test_aggregate_distribution_kl_metrics_fails_when_all_batches_skipped():
    with pytest.raises(ValueError, match="No valid response tokens"):
        aggregate_distribution_kl_metrics([], skipped_batches=4, skipped_samples=500.0)


def test_collect_rollout_records_decodes_prompt_and_response():
    from verl import DataProto
    from tensordict import TensorDict

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "text_" + "_".join(str(int(x)) for x in ids.tolist())

    batch = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([[1, 2, 0], [3, 4, 0]]),
                "responses": torch.tensor([[5, 6, 0], [7, 8, 0]]),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 0, 1, 1, 0],
                        [1, 1, 0, 1, 1, 0],
                    ]
                ),
                "response_mask": torch.ones(2, 3),
            },
            batch_size=2,
        ),
        non_tensor_batch={"uid": np.array(["a", "b"], dtype=object)},
    )
    from verl.trainer.ppo.a2opd_distribution_kl_eval import collect_rollout_records

    records = collect_rollout_records(batch, _Tok())
    assert len(records) == 2
    assert records[0]["uid"] == "a"
    assert "prompt_text" in records[0]
    assert "response_text" in records[0]


def test_compute_distribution_kl_metrics_on_synthetic_topk_batch():
    batch_size, response_len, top_k = 2, 3, 2
    old_log_probs = torch.randn(batch_size, response_len, top_k)
    teacher_log_probs = torch.randn(batch_size, response_len, top_k)
    teacher_base_log_probs = torch.randn(batch_size, response_len, top_k)
    base_log_probs = torch.randn(batch_size, response_len, top_k)
    response_mask = torch.ones(batch_size, response_len)
    student_topk_ids = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ]
    )
    teacher_topk_ids = torch.tensor(
        [
            [[1, 2], [3, 99], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ]
    )
    teacher_base_topk_ids = torch.tensor(
        [
            [[1, 20], [3, 4], [5, 6]],
            [[7, 80], [9, 10], [11, 12]],
        ]
    )
    base_topk_ids = torch.tensor(
        [
            [[1, 2], [30, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ]
    )

    batch = DataProto(
        batch=TensorDict(
            {
                "old_log_probs": old_log_probs,
                "teacher_log_probs": teacher_log_probs,
                "teacher_base_log_probs": teacher_base_log_probs,
                "base_log_probs": base_log_probs,
                "response_mask": response_mask,
                "student_topk_ids": student_topk_ids,
                "teacher_topk_ids": teacher_topk_ids,
                "teacher_base_topk_ids": teacher_base_topk_ids,
                "base_topk_ids": base_topk_ids,
            },
            batch_size=batch_size,
        )
    )
    metrics = compute_distribution_kl_metrics(batch)
    assert "a2opd/proxy_kl" in metrics
    assert "a2opd/proxy_base_target_base_kl" in metrics
    assert "a2opd/target_transfer_kl" in metrics
    assert "a2opd/anchor_kl" in metrics
    assert "a2opd/transfer_gap" in metrics
    assert metrics["a2opd/topk_overlap_target_vs_teacher_ratio"] == pytest.approx(11 / 12)
    assert metrics["a2opd/topk_overlap_target_vs_teacher_base_ratio"] == pytest.approx(10 / 12)
    assert metrics["a2opd/topk_overlap_target_vs_base_ratio"] == pytest.approx(11 / 12)
    assert metrics["a2opd/topk_overlap_teacher_vs_teacher_base_ratio"] == pytest.approx(9 / 12)
    assert metrics["_overlap_valid_tokens"] == 6.0


def test_prepare_dataproto_for_opd_prep_unlocks_cpu_batch():
    batch = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "responses": torch.tensor([[4, 5]]),
                "response_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            },
            batch_size=1,
        )
    )
    cpu_batch = materialize_dataproto_cpu(batch)
    prepared = prepare_dataproto_for_opd_prep(cpu_batch, "cpu", from_rollout_cache=True)
    prepared.batch["student_topk_ids"] = torch.tensor([[[1, 2]]])
    assert prepared.batch["student_topk_ids"].shape == (1, 1, 2)


def test_prepare_dataproto_for_opd_prep_skips_regular_training_batch():
    batch = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "responses": torch.tensor([[4, 5]]),
                "response_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            },
            batch_size=1,
        )
    )
    prepared = prepare_dataproto_for_opd_prep(batch, "cpu", from_rollout_cache=False)
    assert prepared is batch
    prepared.batch["student_topk_ids"] = torch.tensor([[[1, 2]]])
    assert prepared.batch["student_topk_ids"].shape == (1, 1, 2)


def test_rollout_cache_roundtrip(tmp_path):
    batch = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "responses": torch.tensor([[4, 5]]),
                "response_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            },
            batch_size=1,
        )
    )
    cache_path = tmp_path / "rollout_cache.pkl"
    save_rollout_cache([batch], cache_path)
    loaded = load_rollout_cache(cache_path)
    assert len(loaded) == 1
    assert torch.equal(loaded[0].batch["responses"], batch.batch["responses"])


def test_aggregate_distribution_kl_metrics_overlap_token_weighted():
    batch_a = {
        "a2opd/topk_overlap_teacher_vs_teacher_base_ratio": 0.5,
        "_overlap_valid_tokens": 4.0,
        "_valid_tokens": 4.0,
        "_num_samples": 1.0,
        "a2opd/proxy_kl": 1.0,
        "a2opd/proxy_base_target_base_kl": 1.0,
        "a2opd/target_transfer_kl": 1.0,
        "a2opd/anchor_kl": 1.0,
        "a2opd/transfer_gap": 1.0,
    }
    batch_b = {
        "a2opd/topk_overlap_teacher_vs_teacher_base_ratio": 1.0,
        "_overlap_valid_tokens": 2.0,
        "_valid_tokens": 2.0,
        "_num_samples": 1.0,
        "a2opd/proxy_kl": 3.0,
        "a2opd/proxy_base_target_base_kl": 3.0,
        "a2opd/target_transfer_kl": 3.0,
        "a2opd/anchor_kl": 3.0,
        "a2opd/transfer_gap": 3.0,
    }
    aggregated = aggregate_distribution_kl_metrics([batch_a, batch_b])
    assert aggregated["a2opd/topk_overlap_teacher_vs_teacher_base_ratio"] == pytest.approx(
        (0.5 * 4 + 1.0 * 2) / 6
    )
