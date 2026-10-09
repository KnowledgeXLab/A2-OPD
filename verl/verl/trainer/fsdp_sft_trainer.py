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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
import time
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from omegaconf import DictConfig, ListConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.logger import log_with_rank
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.trainer.sft_validation import (
    SFTMathBenchmarkValDataset,
    build_math_acc_metric_name,
    evaluate_math_accuracy,
    is_better_metric,
    val_df_has_ground_truth,
)
from verl.utils.fsdp_utils import get_fsdp_full_state_dict
from verl.workers.config.optimizer import build_optimizer
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_loss_dataset: Dataset | None,
        val_perf_dataset: Dataset | None = None,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self.val_perf_dataset = val_perf_dataset
        self._build_dataloader(train_dataset, val_loss_dataset)

        self.lora = self.config.model.get("lora_adapter_path") is not None or self.config.model.lora_rank > 0

        # Initialize resume-related variables
        self.resume_global_step = 0

        # build model
        self._build_model_optimizer()

        # Initialize checkpoint manager
        self._init_checkpoint_manager()

        self.load_checkpoint()

        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = self.config.trainer.device

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_loss_dataset):
        # build dataset
        config = self.config
        self.train_dataset = train_dataset
        self.val_loss_dataset = val_loss_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

        self.val_sampler = None
        self.val_dataloader = None
        if val_loss_dataset is not None and getattr(val_loss_dataset, "supports_sft_loss", True):
            self.val_sampler = DistributedSampler(
                val_loss_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=val_loss_dataset,
                batch_size=config.data.micro_batch_size_per_gpu,
                sampler=self.val_sampler,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
                pin_memory_device=device_name,
            )
        elif self.device_mesh.get_rank() == 0:
            print("No SFT validation loss dataset configured; skip val/loss.")

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.lora:
                self.model.enable_input_require_grads()

                lora_adapter_path = self.config.model.get("lora_adapter_path")
                if lora_adapter_path is not None:
                    from peft import PeftModel

                    print(f"Loading pre-trained LoRA adapter for sft from: {lora_adapter_path}")

                    local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.use_shm)

                    self.model = PeftModel.from_pretrained(self.model, local_adapter_path, is_trainable=True)
                    peft_config = self.model.peft_config["default"]
                    # Ensure task_type is TaskType enum, not string
                    if isinstance(peft_config.task_type, str):
                        peft_config.task_type = TaskType.CAUSAL_LM
                else:
                    # Convert config to regular Python types before creating PEFT model
                    lora_config = {
                        "task_type": TaskType.CAUSAL_LM,
                        "r": self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        "target_modules": convert_to_regular_types(self.config.model.target_modules),
                        "bias": "none",
                    }
                    self.model = get_peft_model(self.model, LoraConfig(**lora_config))
                self.model = self.model.to(torch_dtype)

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.lora,
        )

        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = build_optimizer(self.fsdp_model.parameters(), self.config.optim)

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.lr_warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_backward(self, batch, do_backward=True, n_micro_batches=1):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, 1:].reshape(-1).to(self.device_name)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            loss = loss / n_micro_batches  # normalize loss

            if do_backward:
                loss.backward()
            return loss

    def training_step(self, batch: TensorDict):
        start_time = time.time()

        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch, n_micro_batches=n_micro_batches)
            step_loss += loss.item()

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).to(self.device_name)

        # compute time spent per step
        end_time = time.time()
        spend_time_per_step = end_time - start_time

        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)
        return {
            "train/loss": step_loss.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/time(s)": spend_time_per_step,
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
        return loss

    def save_checkpoint(self, step, ckpt_name=None, update_tracker=True, max_ckpt_to_keep=None):
        """Save checkpoint using FSDPCheckpointManager with improved tracking."""
        import shutil

        from verl.utils.fs import local_mkdir_safe

        folder_name = ckpt_name if ckpt_name is not None else f"global_step_{step}"
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, folder_name)

        if self.device_mesh.get_rank() == 0:
            if ckpt_name == "best_valid" and os.path.exists(local_global_step_folder):
                shutil.rmtree(local_global_step_folder)
            print(f"Saving checkpoint to: {local_global_step_folder}")

        torch.distributed.barrier()

        if max_ckpt_to_keep is None:
            max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        if ckpt_name == "best_valid":
            max_ckpt_to_keep = None

        self.checkpoint_manager.save_checkpoint(
            local_path=local_global_step_folder,
            global_step=step,
            max_ckpt_to_keep=max_ckpt_to_keep,
            track_rotation=ckpt_name is None,
        )

        # Save dataloader state
        if self.device_mesh.get_rank() == 0:
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")

            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"Saved dataloader state to: {dataloader_local_path}")

            if update_tracker and ckpt_name is None:
                tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
                temp_tracker_file = tracker_file + ".tmp"
                with open(temp_tracker_file, "w") as f:
                    f.write(str(step))
                os.rename(temp_tracker_file, tracker_file)
                print(f"Updated checkpoint tracker: {tracker_file}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and getattr(self.config.trainer, "default_hdfs_dir", None):
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=local_global_step_folder, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def _get_best_metric_name(self):
        metric_name = self.config.trainer.get("best_metric_name", None)
        if metric_name is not None:
            return metric_name
        if self.config.trainer.get("val_perf_enable", False) and self.val_perf_dataset is not None:
            n_rollouts = self.config.trainer.get("val_perf_n", 1)
            return build_math_acc_metric_name(n_rollouts, metric_kind="mean")
        return "val/loss"

    def _get_best_metric_mode(self):
        mode = self.config.trainer.get("best_metric_mode", None)
        if mode is not None:
            assert mode in ["max", "min"], "trainer.best_metric_mode must be 'max' or 'min'"
            return mode
        if self._get_best_metric_name() == "val/loss":
            return "min"
        return "max"

    def _evaluate_validation_loss(self):
        if self.val_dataloader is None:
            return None

        val_losses = []
        for val_data in self.val_dataloader:
            val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(
                self.device_name
            )
            val_loss = self.validation_step(val_data)
            val_losses.append(val_loss)
        val_loss = torch.mean(torch.stack(val_losses))
        if is_cuda_available:
            torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(val_loss)
            val_loss /= self.device_mesh.size(0)
        return val_loss.detach().item()

    def _evaluate_validation_math_acc(self):
        backend = str(self.config.trainer.get("val_perf_backend", "vllm")).lower()
        if backend == "vllm":
            return self._evaluate_validation_math_acc_vllm()
        elif backend == "hf":
            return self._evaluate_validation_math_acc_hf()
        else:
            raise ValueError(f"Unknown val_perf_backend: {backend}, expected 'vllm' or 'hf'")

    def _evaluate_validation_math_acc_vllm(self):
        import gc
        import json
        import shutil
        import subprocess
        import sys
        import uuid
        from safetensors.torch import save_file
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

        if self.val_perf_dataset is None:
            return None

        val_df = getattr(self.val_perf_dataset, "dataframe", None)
        if val_df is None or not val_df_has_ground_truth(val_df):
            if self.device_mesh.get_rank() == 0:
                print(
                    "Warning: val_perf_enable=true but val_perf_files has no usable ground truth; "
                    "skip math accuracy eval."
                )
            return None

        val_perf_files = self.config.data.get("val_perf_files", None)
        if val_perf_files is None:
            return None

        n_rollouts = max(1, int(self.config.trainer.get("val_perf_n", 1)))
        tp_size = int(self.config.trainer.get("val_perf_vllm_tp", 1))
        gpu_util = float(self.config.trainer.get("val_perf_gpu_memory_utilization", 0.85))

        # 1. Gather full state dict on CPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state_dict = get_fsdp_full_state_dict(
            self.fsdp_model,
            offload_to_cpu=True,
            rank0_only=True,
        )

        # 2. Offload FSDP training model to CPU to completely free GPU memory for vLLM
        offload_fsdp_model_to_cpu(self.fsdp_model, empty_cache=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        acc_metrics = None
        eval_id = uuid.uuid4().hex[:8]
        tmp_eval_dir = os.path.join(self.config.trainer.default_local_dir, f".val_perf_vllm_tmp_{eval_id}")
        result_json_path = os.path.join(tmp_eval_dir, "eval_result.json")

        if self.device_mesh.get_rank() == 0:
            try:
                os.makedirs(tmp_eval_dir, exist_ok=True)
                base_model_dir = copy_to_local(src=self.config.model.partial_pretrain)
                for item in os.listdir(base_model_dir):
                    src = os.path.join(base_model_dir, item)
                    dst = os.path.join(tmp_eval_dir, item)
                    if item.endswith((".safetensors", ".bin", ".pt", ".pth")) or "index.json" in item:
                        continue
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)

                self.model_config.save_pretrained(tmp_eval_dir)

                clean_state_dict = {
                    k: v.detach().cpu().contiguous().to(torch.bfloat16).clone()
                    for k, v in state_dict.items()
                    if isinstance(v, torch.Tensor)
                }
                del state_dict
                gc.collect()

                weight_file = os.path.join(tmp_eval_dir, "model.safetensors")
                print(f"[vLLM-Val] Saving temporary evaluation weights to {weight_file} ...")
                save_file(clean_state_dict, weight_file)
                del clean_state_dict
                gc.collect()

                sub_env = os.environ.copy()
                sub_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
                sub_env["VLLM_USE_V1"] = "1"
                # Thoroughly purge any distributed/env vars leaked from torchrun so that
                # the vLLM subprocess never tries to join the training process group.
                _dist_prefixes = (
                    "PET_",
                    "TORCHELASTIC_",
                    "ELASTIC_",
                    "TORCH_DIST_",
                    "TORCH_SHOW_CPP_STACKTRACES",
                )
                _dist_exact = {
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
                    "OMP_NUM_THREADS",
                }
                for key in list(sub_env.keys()):
                    if key in _dist_exact or key.startswith(_dist_prefixes):
                        sub_env.pop(key, None)
                sub_env["OMP_NUM_THREADS"] = "1"

                if tp_size <= 1:
                    sub_env["CUDA_VISIBLE_DEVICES"] = "0"
                else:
                    dev_count = torch.cuda.device_count() if torch.cuda.is_available() else 8
                    sub_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(min(tp_size, dev_count)))

                if isinstance(val_perf_files, (list, ListConfig)):
                    val_files_arg = ",".join(str(f) for f in val_perf_files)
                else:
                    val_files_arg = str(val_perf_files)

                chat_template_kwargs = convert_to_regular_types(
                    self.config.data.get("apply_chat_template_kwargs", {})
                )
                chat_kwargs_json = json.dumps(chat_template_kwargs)

                cmd = [
                    sys.executable,
                    "-m",
                    "verl.trainer.sft_vllm_eval",
                    "--model_path",
                    tmp_eval_dir,
                    "--val_files",
                    val_files_arg,
                    "--prompt_key",
                    str(self.config.data.prompt_key),
                    "--max_new_tokens",
                    str(self.config.trainer.get("val_perf_max_new_tokens", 16384)),
                    "--temperature",
                    str(self.config.trainer.get("val_perf_temperature", 1.0)),
                    "--top_p",
                    str(self.config.trainer.get("val_perf_top_p", 1.0)),
                    "--n_rollouts",
                    str(n_rollouts),
                    "--tensor_parallel_size",
                    str(tp_size),
                    "--gpu_memory_utilization",
                    str(gpu_util),
                    "--max_samples",
                    str(self.config.trainer.get("val_perf_max_samples", -1)),
                    "--output_file",
                    result_json_path,
                    "--apply_chat_template_kwargs_json",
                    chat_kwargs_json,
                ]
                if self.config.model.trust_remote_code:
                    cmd.append("--trust_remote_code")

                print(f"[vLLM-Val] Running evaluation subprocess: {' '.join(cmd)}")
                subprocess.run(cmd, env=sub_env, check=True)

                if os.path.exists(result_json_path):
                    with open(result_json_path, "r") as f:
                        acc_metrics = json.load(f)
                else:
                    print(f"[vLLM-Val] Error: result file {result_json_path} not found.")
            except Exception as e:
                print(f"[vLLM-Val] Exception during vLLM evaluation: {e}")
            finally:
                if os.path.exists(tmp_eval_dir):
                    shutil.rmtree(tmp_eval_dir, ignore_errors=True)
        else:
            del state_dict

        # 3. Reload FSDP training model back to GPU
        load_fsdp_model_to_gpu(self.fsdp_model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Broadcast evaluation metrics
        payload = acc_metrics if acc_metrics is not None else {"pass_at_n": -1.0, "mean_at_n": -1.0}
        metric_tensor = torch.tensor(
            [payload["pass_at_n"], payload["mean_at_n"]],
            device=self.device_name,
            dtype=torch.float32,
        )
        torch.distributed.broadcast(metric_tensor, src=0)
        if metric_tensor[0].item() < 0:
            return None

        pass_name = build_math_acc_metric_name(n_rollouts, metric_kind="pass")
        mean_name = build_math_acc_metric_name(n_rollouts, metric_kind="mean")
        return {
            pass_name: metric_tensor[0].item(),
            mean_name: metric_tensor[1].item(),
        }

    def _evaluate_validation_math_acc_hf(self):
        if self.val_perf_dataset is None:
            return None

        val_df = getattr(self.val_perf_dataset, "dataframe", None)
        if val_df is None or not val_df_has_ground_truth(val_df):
            if self.device_mesh.get_rank() == 0:
                print(
                    "Warning: val_perf_enable=true but val_perf_files has no usable ground truth; "
                    "skip math accuracy eval."
                )
            return None

        n_rollouts = max(1, int(self.config.trainer.get("val_perf_n", 1)))
        val_perf_world_size = int(self.config.trainer.get("val_perf_world_size", 0))
        if val_perf_world_size <= 0:
            val_perf_world_size = torch.distributed.get_world_size()
        val_perf_world_size = min(val_perf_world_size, torch.distributed.get_world_size())
        val_perf_rank = torch.distributed.get_rank() % val_perf_world_size
        use_distributed_val_perf = val_perf_world_size > 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state_dict = get_fsdp_full_state_dict(
            self.fsdp_model,
            offload_to_cpu=True,
            rank0_only=not use_distributed_val_perf,
        )

        model_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        run_local_eval = use_distributed_val_perf or self.device_mesh.get_rank() == 0
        if run_local_eval:
            local_metrics = evaluate_math_accuracy(
                state_dict=state_dict,
                model_config=self.model_config,
                tokenizer=self.tokenizer,
                val_df=val_df,
                prompt_key=self.config.data.prompt_key,
                max_samples=self.config.trainer.get("val_perf_max_samples", -1),
                max_new_tokens=self.config.trainer.get("val_perf_max_new_tokens", 16384),
                batch_size=self.config.trainer.get("val_perf_batch_size", 4),
                n_rollouts=n_rollouts,
                temperature=self.config.trainer.get("val_perf_temperature", 1.0),
                top_p=self.config.trainer.get("val_perf_top_p", 1.0),
                seed=self.config.trainer.get("seed", 42),
                apply_chat_template_kwargs=convert_to_regular_types(
                    self.config.data.get("apply_chat_template_kwargs", {})
                ),
                trust_remote_code=self.config.model.trust_remote_code,
                model_dtype=model_dtype,
                shard_rank=val_perf_rank if use_distributed_val_perf else 0,
                shard_world_size=val_perf_world_size if use_distributed_val_perf else 1,
            )
        else:
            local_metrics = {
                "pass_total": 0.0,
                "mean_total": 0.0,
                "num_rows": 0.0,
            }
        del state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stats = torch.tensor(
            [
                local_metrics["pass_total"],
                local_metrics["mean_total"],
                local_metrics["num_rows"],
            ],
            device=self.device_name,
            dtype=torch.float64,
        )
        if use_distributed_val_perf:
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        elif torch.distributed.get_world_size() > 1:
            torch.distributed.broadcast(stats, src=0)

        num_rows = stats[2].item()
        if num_rows <= 0:
            return None

        acc_metrics = {
            "pass_at_n": (stats[0] / num_rows).item(),
            "mean_at_n": (stats[1] / num_rows).item(),
        }
        if self.device_mesh.get_rank() == 0:
            pass_name = build_math_acc_metric_name(n_rollouts, metric_kind="pass")
            mean_name = build_math_acc_metric_name(n_rollouts, metric_kind="mean")
            print(
                f"Validation math accuracy ({val_perf_world_size} GPU(s)): "
                f"{pass_name}={acc_metrics['pass_at_n']:.4f}, "
                f"{mean_name}={acc_metrics['mean_at_n']:.4f}"
            )

        payload = acc_metrics
        metric_tensor = torch.tensor(
            [payload["pass_at_n"], payload["mean_at_n"]],
            device=self.device_name,
            dtype=torch.float32,
        )
        torch.distributed.broadcast(metric_tensor, src=0)
        if metric_tensor[0].item() < 0:
            return None

        pass_name = build_math_acc_metric_name(n_rollouts, metric_kind="pass")
        mean_name = build_math_acc_metric_name(n_rollouts, metric_kind="mean")
        return {
            pass_name: metric_tensor[0].item(),
            mean_name: metric_tensor[1].item(),
        }

    def _collect_validation_metrics(self):
        metrics = {}
        val_loss = self._evaluate_validation_loss()
        if val_loss is not None:
            metrics["val/loss"] = val_loss
        if self.config.trainer.get("val_perf_enable", False):
            math_metrics = self._evaluate_validation_math_acc()
            if math_metrics is not None:
                metrics.update(math_metrics)
        return metrics

    def _write_best_checkpoint_meta(self, step, metric_name, metric_value):
        if self.device_mesh.get_rank() != 0:
            return

        base_dir = self.config.trainer.default_local_dir
        best_dir = os.path.join(base_dir, "best_valid")
        meta = {
            "global_step": int(step),
            "metric_name": metric_name,
            "metric_value": float(metric_value),
        }
        torch.save(meta, os.path.join(best_dir, "meta.pt"))
        with open(os.path.join(base_dir, "best_checkpoint.txt"), "w") as f:
            f.write(str(step))
        print(f"Updated BEST checkpoint: step={step}, {metric_name}={metric_value:.6f}")

    def _init_checkpoint_manager(self):
        """Initialize checkpoint manager with proper configuration"""
        # Get checkpoint configuration from config, with defaults
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})

        # Set default values if not specified
        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])
        load_contents = checkpoint_config.get("load_contents", save_contents)

        # Create checkpoint config dict
        checkpoint_config_dict = {
            "load_contents": load_contents,
            "save_contents": save_contents,
        }

        # Convert to DictConfig for compatibility
        checkpoint_config_dict = DictConfig(checkpoint_config_dict)

        # Initialize checkpoint manager
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer,
            checkpoint_config=checkpoint_config_dict,
        )

    def load_checkpoint(self):
        # Determine resume path based on configuration
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # extract resume step from checkpoint path
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        # Use checkpoint manager to load model state
        self.checkpoint_manager.load_checkpoint(checkpoint_path)
        log_with_rank(
            f"Successfully loaded model checkpoint from {checkpoint_path} (step {resume_step})",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # Always load dataloader state for StatefulDataLoader
        self._load_dataloader_state(checkpoint_path)

        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):
        """Load dataloader state from checkpoint"""
        dataloader_path = os.path.join(checkpoint_path, "data.pt")

        if os.path.exists(dataloader_path):
            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

            log_with_rank(
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        else:
            log_with_rank(
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _determine_resume_path(self):
        """Determine the path to resume from based on resume_mode configuration"""
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
                return resume_from_path
            # Try to find the latest checkpoint in the default directory
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
            return resume_from_path
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the default local directory"""
        checkpoint_dir = self.config.trainer.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.device_mesh.get_rank() == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None
        best_metric_name = self._get_best_metric_name()
        best_metric_mode = self._get_best_metric_mode()
        best_metric_value = None
        best_step = None
        early_stop_counter = 0
        stop_training = False
        early_stop_enable = self.config.trainer.get("early_stop_enable", False)
        early_stop_patience = self.config.trainer.get("early_stop_patience", 3)
        early_stop_min_delta = self.config.trainer.get("early_stop_min_delta", 0.0)
        save_best_ckpt = self.config.trainer.get("save_best_ckpt", True)
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        train_time = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0,
                )
            ):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                metric = self.training_step(data)
                train_time += metric["train/time(s)"]
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                # early exit or validation step
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    val_metrics = self._collect_validation_metrics()
                    if rank == 0:
                        tracking.log(data=val_metrics, step=global_step)
                        last_valid_metric = val_metrics

                    if best_metric_name not in val_metrics:
                        if rank == 0:
                            available = ", ".join(sorted(val_metrics))
                            print(
                                f"Warning: best metric '{best_metric_name}' not found in validation metrics "
                                f"({available}). Skip best/early-stop update at step {global_step}."
                            )
                    else:
                        current_metric = float(val_metrics[best_metric_name])
                        improved = is_better_metric(
                            current_metric,
                            best_metric_value,
                            best_metric_mode,
                            early_stop_min_delta,
                        )
                        if improved:
                            best_metric_value = current_metric
                            best_step = global_step
                            early_stop_counter = 0
                            if save_best_ckpt:
                                self.save_checkpoint(
                                    step=global_step, ckpt_name="best_valid", update_tracker=False
                                )
                                self._write_best_checkpoint_meta(
                                    global_step, best_metric_name, current_metric
                                )
                        else:
                            early_stop_counter += 1
                            if rank == 0:
                                best_display = (
                                    "None"
                                    if best_metric_value is None
                                    else f"{best_metric_value:.6f}"
                                )
                                print(
                                    f"No {best_metric_name} improvement at step {global_step}: "
                                    f"current={current_metric:.6f}, best={best_display} "
                                    f"(step {best_step}), patience={early_stop_counter}/{early_stop_patience}"
                                )
                            if early_stop_enable and early_stop_counter >= early_stop_patience:
                                stop_training = True
                                if rank == 0:
                                    print(
                                        f"Early stopping triggered at step {global_step}. "
                                        f"Best {best_metric_name}={best_metric_value:.6f} at step {best_step}."
                                    )
                    torch.distributed.barrier()

                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                if is_last_step or stop_training:
                    if rank == 0:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                        if best_step is not None and best_metric_value is not None:
                            print(
                                f"Best validation {best_metric_name}={best_metric_value:.6f} at step {best_step}"
                            )
                            if save_best_ckpt:
                                print(
                                    f"Best checkpoint saved to: "
                                    f"{os.path.join(self.config.trainer.default_local_dir, 'best_valid')}"
                                )
                    return


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(
        config.data.train_files, config.data, tokenizer, max_samples=config.data.get("train_max_samples", -1)
    )
    val_loss_dataset = create_sft_loss_dataset(
        config.data.val_files, config.data, tokenizer, max_samples=config.data.get("val_max_samples", -1)
    )
    val_perf_dataset = create_sft_perf_dataset(
        config.data.get("val_perf_files", None),
        config.data,
        max_samples=-1,
    )

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_loss_dataset=val_loss_dataset,
        val_perf_dataset=val_perf_dataset,
    )

    trainer.fit()

    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, max_samples=-1):
    """Create a standard SFT dataset for training or val loss."""
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    else:
        dataset_cls = SFTDataset

    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config, max_samples=max_samples)
    return dataset


def create_sft_loss_dataset(data_paths, data_config, tokenizer, max_samples=-1):
    if _is_empty_dataset_path(data_paths):
        return None
    return create_sft_dataset(data_paths, data_config, tokenizer, max_samples=max_samples)


def create_sft_perf_dataset(data_paths, data_config, max_samples=-1):
    if _is_empty_dataset_path(data_paths):
        return None
    return SFTMathBenchmarkValDataset(
        parquet_files=data_paths,
        max_samples=max_samples,
        use_shm=data_config.get("use_shm", False),
    )


def _is_empty_dataset_path(data_paths) -> bool:
    if data_paths is None:
        return True
    if isinstance(data_paths, str):
        return len(data_paths.strip()) == 0
    if isinstance(data_paths, (list, tuple, ListConfig)):
        return len(data_paths) == 0
    return False


if __name__ == "__main__":
    main()
