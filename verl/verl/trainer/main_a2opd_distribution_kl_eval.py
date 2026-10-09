"""
Offline A²OPD distribution KL evaluation entry point.

For each 4-model job:
  1) rollout responses from prompts using the target model (student_ckpt)
  2) teacher-force those responses through OPD prep
  3) report Proxy / Induced / Actual Target KL and Transfer Gap
"""

from __future__ import annotations

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local


def has_hf_weights(path: str) -> bool:
    from pathlib import Path

    root = Path(path)
    if not root.is_dir():
        return False
    return (
        any(root.glob("*.safetensors"))
        or any(root.glob("*.bin"))
        or (root / "model.safetensors.index.json").is_file()
    )


def resolve_distribution_kl_student_ckpt(config) -> tuple[str | None, str | None]:
    """Resolve student checkpoint root and load kind (fsdp or merged_hf)."""
    student_ckpt_root = config.trainer.get("distribution_kl_student_ckpt", None)
    if not student_ckpt_root:
        return None, None

    student_ckpt_kind = str(config.trainer.get("distribution_kl_student_ckpt_kind", "fsdp")).lower()
    actor_ckpt = os.path.join(student_ckpt_root, "actor")
    merged_hf_ckpt = os.path.join(student_ckpt_root, "merged_hf")

    if student_ckpt_kind == "merged_hf":
        if not has_hf_weights(merged_hf_ckpt):
            raise FileNotFoundError(
                f"distribution_kl_student_ckpt_kind=merged_hf but no HF weights found at {merged_hf_ckpt}"
            )
        print(f"Using merged HuggingFace checkpoint: {merged_hf_ckpt}")
        with open_dict(config.actor_rollout_ref.model):
            config.actor_rollout_ref.model.path = merged_hf_ckpt
        return student_ckpt_root, student_ckpt_kind

    if student_ckpt_kind != "fsdp":
        raise ValueError(
            "distribution_kl_student_ckpt_kind must be one of: fsdp, merged_hf "
            f"(got: {student_ckpt_kind})"
        )

    if os.path.isdir(actor_ckpt):
        return student_ckpt_root, "fsdp"

    if has_hf_weights(merged_hf_ckpt):
        print(f"FSDP actor checkpoint missing; falling back to merged_hf: {merged_hf_ckpt}")
        with open_dict(config.actor_rollout_ref.model):
            config.actor_rollout_ref.model.path = merged_hf_ckpt
        return student_ckpt_root, "merged_hf"

    raise FileNotFoundError(
        "distribution_kl_student_ckpt must contain actor/ FSDP weights or merged_hf/ HF weights, "
        f"not found under: {student_ckpt_root}"
    )


def apply_distribution_kl_eval_config(config) -> None:
    """Apply safe defaults so offline KL eval passes validate_config."""
    OmegaConf.set_struct(config, False)
    config.trainer.distribution_kl_eval_only = True
    config.trainer.val_before_train = False
    config.trainer.val_only = False

    # Default ppo_trainer.yaml uses GAE, which enables critic validation we do not need.
    if str(config.algorithm.get("adv_estimator", "gae")).lower() == "gae":
        config.algorithm.adv_estimator = "grpo"
    config.algorithm.use_kl_in_reward = False
    config.critic.enable = False

    # Eval only needs student weights; skip optimizer/extra to reduce CPU memory.
    with open_dict(config.actor_rollout_ref.actor.checkpoint):
        config.actor_rollout_ref.actor.checkpoint.load_contents = ["model"]

    do_sample = config.trainer.get("distribution_kl_do_sample", True)
    with open_dict(config.actor_rollout_ref.rollout.val_kwargs):
        config.actor_rollout_ref.rollout.val_kwargs.do_sample = bool(do_sample)
        if do_sample:
            temperature = config.actor_rollout_ref.rollout.val_kwargs.get("temperature", 0)
            if temperature is None or float(temperature) <= 0:
                config.actor_rollout_ref.rollout.val_kwargs.temperature = float(
                    config.actor_rollout_ref.rollout.get("temperature", 1.0)
                )
            top_p = config.actor_rollout_ref.rollout.val_kwargs.get("top_p", None)
            if top_p is None:
                config.actor_rollout_ref.rollout.val_kwargs.top_p = float(
                    config.actor_rollout_ref.rollout.get("top_p", 1.0)
                )


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_distribution_kl_eval(config)


def run_distribution_kl_eval(config, task_runner_class=None) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(DistributionKLEvalTaskRunner)

    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))


class DistributionKLEvalTaskRunner(TaskRunner):
    """Ray worker that loads four models once and runs calibration KL eval."""

    def run(self, config):
        from pprint import pprint

        print(f"DistributionKLEvalTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        apply_distribution_kl_eval_config(config)
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        if need_critic(config):
            self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        use_base_target = bool(config.trainer.get("distribution_kl_use_base_target", False))
        if use_base_target:
            student_ckpt_root, student_ckpt_kind = None, None
            print(
                "Using base target weights from actor_rollout_ref.model.path="
                f"{config.actor_rollout_ref.model.path} (no trained checkpoint)."
            )
        else:
            student_ckpt_root, student_ckpt_kind = resolve_distribution_kl_student_ckpt(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        ref_tokenizer = None
        ref_base_tokenizer = None
        ref_model_config = config.actor_rollout_ref.ref.get("model", {})
        ref_model_path = ref_model_config.get("path", None) if ref_model_config else None
        if ref_model_path is not None:
            ref_local_path = copy_to_local(
                ref_model_path,
                use_shm=config.actor_rollout_ref.model.get("use_shm", False),
            )
            ref_tokenizer = hf_tokenizer(ref_local_path, trust_remote_code=trust_remote_code)
            ref_base_model_path = ref_model_config.get("base_model_path", None)
            if ref_base_model_path is not None:
                ref_base_local_path = copy_to_local(
                    ref_base_model_path,
                    use_shm=config.actor_rollout_ref.model.get("use_shm", False),
                )
                ref_base_tokenizer = hf_tokenizer(ref_base_local_path, trust_remote_code=trust_remote_code)

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        prompt_files = config.data.get("prompt_files", None) or config.data.val_files
        placeholder_dataset = create_rl_dataset(
            prompt_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=min(config.data.get("distribution_kl_max_samples", 1), 1),
        )

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=None,
            train_dataset=placeholder_dataset,
            val_dataset=placeholder_dataset,
            collate_fn=collate_fn,
            train_sampler=create_rl_sampler(config.data, placeholder_dataset),
            ref_tokenizer=ref_tokenizer,
            ref_base_tokenizer=ref_base_tokenizer,
        )

        trainer.init_workers()

        if student_ckpt_root and student_ckpt_kind == "fsdp":
            actor_ckpt = os.path.join(student_ckpt_root, "actor")
            print(f"Loading target checkpoint from {actor_ckpt}")
            trainer.actor_rollout_wg.load_checkpoint(actor_ckpt, None)
        elif student_ckpt_root and student_ckpt_kind == "merged_hf":
            print(
                "Target weights loaded from merged_hf via actor_rollout_ref.model.path="
                f"{config.actor_rollout_ref.model.path}"
            )

        metrics = trainer.eval_distribution_kl()
        print("Final distribution KL metrics:")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
