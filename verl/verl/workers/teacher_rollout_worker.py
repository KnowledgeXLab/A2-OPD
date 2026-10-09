from __future__ import annotations

import datetime
import os
from typing import Optional

import asyncio
import threading
import numpy as np

import torch
try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import warnings
from peft import LoraConfig, TaskType, get_peft_model
from verl import DataProto
from verl.single_controller.base import Worker
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.workers.rollout import get_rollout_class
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.model import compute_position_id_with_mask, convert_weight_keys, get_generation_config
from verl.utils.py_functional import convert_to_regular_types

from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.config.optimizer import build_optimizer

from omegaconf import DictConfig, OmegaConf
from verl.models.transformers.monkey_patch import apply_monkey_patch

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register

from verl.utils.config import omega_conf_to_dataclass

from torch.distributed.device_mesh import init_device_mesh

logger = __import__("logging").getLogger(__name__)

device_name = get_device_name()


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_vl_model_vision_tower(vl_model_instance):
    """
    Util to extract Vision Tower from a VL model instance
    """
    if hasattr(vl_model_instance, "model") and hasattr(vl_model_instance.model, "visual"):
        # transformers >= 4.52.0
        return vl_model_instance.model.visual
    elif hasattr(vl_model_instance, "visual"):
        # transformers < 4.52.0
        return vl_model_instance.visual
    return None


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class TeacherRolloutWorker(Worker, DistProfilerExtension):
    """
    Teacher worker that builds actor model + rollout exactly like ActorRolloutRefWorker,
    but only exposes generate_sequences and never updates optimizer.
    """

    def __init__(self, config: DictConfig, role: str = "teacher_rollout", **kwargs):
        Worker.__init__(self)

        self.config = config
        self.role = role

        import torch
        import torch.distributed
        from torch.distributed.device_mesh import init_device_mesh

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # ===== build device mesh for FSDP, same style as ActorRolloutRefWorker =====
        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(
            world_size=world_size,
            fsdp_size=self.config.teacher_rollout.actor.fsdp_config.fsdp_size,
        )

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.teacher_rollout.actor.get(
            "ulysses_sequence_parallel_size", 1
        )
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                get_device_name(),
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor",
                dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(),
                is_collect=is_collect,
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)

        omega_profiler_config = self.config.teacher_rollout.rollout.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(
            omega_profiler_config, dataclass_type=ProfilerConfig
        )
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(
                    omega_profiler_config.get("tool")
                )
            )
        else:
            tool_config = None

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self._is_actor = True
        self._is_rollout = True
        self._is_ref = False

        self.use_orig_params = self.config.teacher_rollout.actor.fsdp_config.get(
            "use_orig_params", False
        )
        self._is_offload_param = self.config.teacher_rollout.actor.fsdp_config.get(
            "param_offload", True
        )
        self._is_offload_optimizer = self.config.teacher_rollout.actor.fsdp_config.get(
            "optimizer_offload", False
        )

        self._lora_rank = self.config.teacher_rollout.model.get("lora_rank", 0)
        self._is_lora = (
            self.config.teacher_rollout.model.get("lora_adapter_path") is not None
            or self._lora_rank > 0
        )

        self.rollout = None
        self.tokenizer = None
        self.processor = None
        self.generation_config = None
        self.model_config = None
        self.actor_module_fsdp = None
        self.actor_module = None
        self.actor_optimizer = None
        self.actor_lr_scheduler = None
        self.actor_model_config = None
        self.base_sync_done = True
        self.layered_summon = False

        self.keep_rollout_mode = True
        self._rollout_mode_ready = False

    def _ensure_background_loop(self):
        if getattr(self, "_bg_loop", None) is not None:
            return

        self._bg_loop = asyncio.new_event_loop()

        def _runner():
            asyncio.set_event_loop(self._bg_loop)
            self._bg_loop.run_forever()

        self._bg_thread = threading.Thread(target=_runner, daemon=True)
        self._bg_thread.start()

    def _run_coro_blocking(self, coro):
        self._ensure_background_loop()
        fut = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
        return fut.result()

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForVision2Seq,
        )

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        # print("[DEBUG] S tokenizer class:", type(self.tokenizer))
        # print("[DEBUG] S processor class:", type(self.processor) if self.processor is not None else None)
        # print("[DEBUG] S bos:", self.tokenizer.bos_token_id)
        # print("[DEBUG] S eos:", self.tokenizer.eos_token_id)
        # print("[DEBUG] S pad:", self.tokenizer.pad_token_id)
        # print("[DEBUG] S unk:", self.tokenizer.unk_token_id)
        # print("[DEBUG] S chat_template:", getattr(self.tokenizer, "chat_template", None))

        if self.config.teacher_rollout.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.teacher_rollout.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.teacher_rollout.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        # print("[DEBUG] S gen eos:", getattr(self.generation_config, "eos_token_id", None))
        # print("[DEBUG] S gen pad:", getattr(self.generation_config, "pad_token_id", None))
        # print("[DEBUG] S gen bos:", getattr(self.generation_config, "bos_token_id", None))

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"[DEBUG] S Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.teacher_rollout.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to actor module")
            actor_module.enable_input_require_grads()

            lora_adapter_path = self.config.teacher_rollout.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to {role} from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.teacher_rollout.model.get("use_shm", False))

                actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, is_trainable=True)
                peft_config = actor_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM

            else:
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.teacher_rollout.model.lora_rank,
                    "lora_alpha": self.config.teacher_rollout.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.teacher_rollout.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.teacher_rollout.model.exclude_modules),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.teacher_rollout.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[actor model] No vision tower found.")

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora,
        )

        if self._is_rollout and self.config.teacher_rollout.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.teacher_rollout.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config


    def _ensure_rollout_mode_ready(self):
        """
        For teacher model: enter rollout mode only once.

        Teacher model is fixed, so we only need to sync FSDP weights
        to rollout engine once.
        """
        if self._rollout_mode_ready:
            return

        if self.rollout is None:
            raise RuntimeError(
                f"Teacher rollout engine is not initialized on rank={self.rank}. "
                f"Please make sure _build_rollout() has been called before entering rollout mode."
            )

        self._run_coro_blocking(self.rollout_mode())

        if self.actor_module_fsdp is not None:
            self.actor_module_fsdp.eval()

        self._rollout_mode_ready = True

        log_gpu_memory_usage(
            "Teacher rollout mode is ready and will be kept resident",
            logger=logger,
        )


    def _build_rollout(self, trust_remote_code: bool = False):
        from torch.distributed.device_mesh import init_device_mesh
        from verl.workers.config import HFModelConfig, RolloutConfig
        import torch
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            ShardedStateDictConfig,
            StateDictType,
            FullyShardedDataParallel as FSDP,
        )

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.teacher_rollout.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(
            self.config.teacher_rollout.model, dataclass_type=HFModelConfig
        )
        self.model_config = model_config

        infer_tp = (
            self.config.teacher_rollout.rollout.tensor_model_parallel_size
            * self.config.teacher_rollout.rollout.data_parallel_size
        )
        infer_pp = self.config.teacher_rollout.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size

        assert self.world_size % infer_world_size == 0, (
            f"teacher rollout world_size: {self.world_size} is not divisible by "
            f"infer_world_size: {infer_world_size}"
        )

        rollout_device_mesh = init_device_mesh(
            get_device_name(),
            mesh_shape=(dp, infer_tp, infer_pp),
            mesh_dim_names=["dp", "infer_tp", "infer_pp"],
        )

        rollout_name = self.config.teacher_rollout.rollout.name
        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout",
                dp_rank=rollout_device_mesh["dp"].get_local_rank(),
                is_collect=is_collect,
            )

        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        log_gpu_memory_usage(
            f"Before building teacher {rollout_name} rollout", logger=logger
        )
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config,
            model_config=model_config,
            device_mesh=rollout_device_mesh,
        )
        log_gpu_memory_usage(
            f"After building teacher {rollout_name} rollout", logger=logger
        )

        # 和 ActorRolloutRefWorker 保持一致
        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.base_sync_done = "dummy" not in self.config.teacher_rollout.rollout.load_format
        self.layered_summon = self.config.teacher_rollout.rollout.get("layered_summon", False)

        if rollout_config.mode == "sync":
            self._run_coro_blocking(self.trainer_mode())

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):  # LoRA
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.teacher_rollout.rollout.get("layered_summon", False),
                base_sync_done=self.base_sync_done,
            )
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = self.actor_module_fsdp.state_dict()

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )

        # Special handling for LoRA with sleep_level=2:
        # When sleep_level=2, base model weights are destroyed during each sleep cycle.
        # separately collect and update LoRA weights and base model weights through their respective interfaces.
        # Here: params contains LoRA weights, base_model_params contains base model weights.
        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=False,
            )
            base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(
                base_model_params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        if peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params  # Fixed: handle dict case
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )

        if self.config.teacher_rollout.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params

        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param
        aggressive_empty_cache(force_sync=True)
        if self.config.teacher_rollout.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        if self.config.teacher_rollout.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)

        self.actor_module_fsdp.train()

        # add empty cache after each compute
        aggressive_empty_cache(force_sync=True)

        set_expandable_segments(True)

        # restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.teacher_rollout.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(
            OmegaConf.create(self.config.teacher_rollout.model.get("override_config", {}))
        )
        use_remove_padding = self.config.teacher_rollout.model.get("use_remove_padding", False)
        use_shm = self.config.teacher_rollout.model.get("use_shm", False)
        use_fused_kernels = self.config.teacher_rollout.model.get("use_fused_kernels", False)

        local_path = copy_to_local(self.config.teacher_rollout.model.path, use_shm=use_shm)

        self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = (
            self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.teacher_rollout.actor.fsdp_config),
                optim_config=None,  # teacher不训练
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.teacher_rollout.model.get(
                    "enable_gradient_checkpointing", False
                ),
                trust_remote_code=self.config.teacher_rollout.model.get("trust_remote_code", False),
                use_liger=self.config.teacher_rollout.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.teacher_rollout.model.get(
                    "enable_activation_offload", False
                ),
            )
        )

        if fsdp_version(self.actor_module_fsdp) == 1:
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload teacher actor model during init", logger=logger)

        self._build_rollout(
            trust_remote_code=self.config.teacher_rollout.model.get("trust_remote_code", False)
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="magenta", role="teacher_rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        assert self.rollout is not None, "Teacher rollout engine is not initialized."

        from verl.utils.memory_utils import aggressive_empty_cache
        from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max

        prompts = prompts.to(get_device_id())

        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        bs, _ = input_ids.shape

        eos_token_id = (
            self.generation_config.eos_token_id
            if self.generation_config is not None and self.generation_config.eos_token_id is not None
            else self.tokenizer.eos_token_id
        )
        pad_token_id = (
            self.generation_config.pad_token_id
            if self.generation_config is not None and self.generation_config.pad_token_id is not None
            else self.tokenizer.pad_token_id
        )

        prompts.meta_info.update({
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        })

        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        position_ids = position_ids * attention_mask.long()
        prompts.batch["position_ids"] = position_ids

        raw_prompt_ids = []
        for i in range(bs):
            valid_ids = input_ids[i][attention_mask[i].bool()].tolist()
            raw_prompt_ids.append(valid_ids)

        non_tensor_batch = dict(prompts.non_tensor_batch)
        non_tensor_batch["raw_prompt_ids"] = np.array(raw_prompt_ids, dtype=object)
        prompts.non_tensor_batch = non_tensor_batch

        temperature = prompts.meta_info.get(
            "temperature",
            self.config.teacher_rollout.rollout.get("temperature", 1.0),
        )
        top_p = prompts.meta_info.get(
            "top_p",
            self.config.teacher_rollout.rollout.get("top_p", 1.0),
        )
        top_k = prompts.meta_info.get(
            "top_k",
            self.config.teacher_rollout.rollout.get("top_k", -1),
        )

        timing_generate = {}

        if self.keep_rollout_mode:
            self._ensure_rollout_mode_ready()
        else:
            self._run_coro_blocking(self.rollout_mode())
            log_gpu_memory_usage("After switch teacher to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(
                prompts=prompts,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        if not self.keep_rollout_mode:
            self._run_coro_blocking(self.trainer_mode())
            log_gpu_memory_usage("After switch teacher to trainer mode", logger=logger)

        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        aggressive_empty_cache(force_sync=True)
        return output