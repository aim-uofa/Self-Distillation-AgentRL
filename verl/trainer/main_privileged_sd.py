"""Shared isolated launcher for ICSD and PCSD training."""

from __future__ import annotations

import math

import ray
from omegaconf import OmegaConf


def run_privileged_sd(config, method: str) -> None:
    if method not in {"icsd", "pcsd"}:
        raise ValueError(f"Unsupported privileged-distillation method: {method}")

    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create(
            {**ray_init_kwargs, "runtime_env": runtime_env}
        )
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = PrivilegedSDTaskRunner.remote()
    ray.get(runner.run.remote(config, method))


def _finite_float(config, key: str, default: float, *, minimum=None) -> float:
    value = float(config.get(key, default))
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{key} must be finite and >= {minimum}, got {value}")
    return value


@ray.remote(num_cpus=1)
class PrivilegedSDTaskRunner:
    def run(self, config, method: str):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        method_cfg = config.algorithm.get(method, {})
        actor_values = {}
        if method == "icsd":
            actor_values = {
                "icsd_loss_coef": _finite_float(method_cfg, "coef", 0.01, minimum=0.0),
                "icsd_trust_split": _finite_float(method_cfg, "trust_split", 0.4),
                "icsd_influence_split": _finite_float(method_cfg, "influence_split", 0.5),
                "icsd_scale_floor": _finite_float(method_cfg, "scale_floor", 1e-4, minimum=0.0),
                "icsd_min_group_tokens": int(method_cfg.get("min_group_tokens", 8)),
            }
            if not 0.0 < actor_values["icsd_trust_split"] < 1.0:
                raise ValueError("algorithm.icsd.trust_split must be in (0, 1)")
            if not 0.0 < actor_values["icsd_influence_split"] < 1.0:
                raise ValueError("algorithm.icsd.influence_split must be in (0, 1)")
            if actor_values["icsd_scale_floor"] <= 0.0:
                raise ValueError("algorithm.icsd.scale_floor must be positive")
            if actor_values["icsd_min_group_tokens"] < 1:
                raise ValueError("algorithm.icsd.min_group_tokens must be positive")
        else:
            actor_values = {
                "pcsd_loss_coef": _finite_float(method_cfg, "coef", 0.01, minimum=0.0),
                "pcsd_n_min": int(method_cfg.get("n_min", 1)),
                "pcsd_n_max": int(method_cfg.get("n_max", 8)),
                "pcsd_decay": _finite_float(method_cfg, "decay", 0.8),
                "pcsd_variance_low": _finite_float(method_cfg, "variance_low", 0.05),
                "pcsd_variance_high": _finite_float(method_cfg, "variance_high", 0.5),
                "pcsd_trend_gamma": _finite_float(method_cfg, "trend_gamma", 0.3, minimum=0.0),
                "pcsd_gate_beta": _finite_float(method_cfg, "gate_beta", 5.0, minimum=0.0),
                "pcsd_eps": 1e-8,
            }
            if actor_values["pcsd_n_min"] < 1 or actor_values["pcsd_n_max"] < actor_values["pcsd_n_min"]:
                raise ValueError("PCSD windows must satisfy 1 <= n_min <= n_max")
            if not 0.0 < actor_values["pcsd_decay"] < 1.0:
                raise ValueError("algorithm.pcsd.decay must be in (0, 1)")
            if actor_values["pcsd_variance_high"] <= actor_values["pcsd_variance_low"]:
                raise ValueError("PCSD variance_high must be greater than variance_low")

        # These entry points are isolated: exactly one auxiliary method is on.
        with open_dict(config):
            actor_cfg = config.actor_rollout_ref.actor
            for flag in (
                "use_sdl_loss",
                "use_sdar_loss",
                "use_icsd_loss",
                "use_pcsd_loss",
                "use_ovcsd_loss",
            ):
                actor_cfg[flag] = False
            actor_cfg[f"use_{method}_loss"] = True
            for key, value in actor_values.items():
                actor_cfg[key] = value
            config.algorithm.skillsd = {
                "sdl_lambda": 0.0,
                "warmdown_steps": -1,
            }

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs
        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True
        )

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0 and not is_version_ge(pkg="vllm", minver="0.7.3"):
                raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import (
                ActorRolloutRefWorker,
                AsyncActorRolloutRefWorker,
                CriticWorker,
            )
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        elif method == "pcsd":
            raise ValueError("PCSD requires the frozen reference policy; keep actor.use_kl_loss enabled")

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager
        else:
            raise NotImplementedError
        reward_fn = EpisodeRewardManager(
            tokenizer=tokenizer, num_examine=0, normalize_by_length=False
        )
        val_reward_fn = EpisodeRewardManager(
            tokenizer=tokenizer, num_examine=1, normalize_by_length=False
        )

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )
        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl+env, GRPO group sampling is configured by env.rollout.n"
        )

        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(
            config=config, tokenizer=tokenizer, processor=processor
        )

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from verl.trainer.ppo.rlsd_utils import SkillProvider
        skills_dir = method_cfg.get("skills_dir", "skills/alfworld")
        skill_all = method_cfg.get("skill_all", False)
        skill_provider = SkillProvider(skills_dir=skills_dir, skill_all=skill_all)
        method_name = method.upper()
        print(f"[{method_name}] Loaded skills from {skills_dir}")
        print(f"[{method_name}] Available skills: {list(skill_provider.skill_contents.keys())}")
        print(f"[{method_name}] Task-to-skill mapping: {skill_provider.task_to_skill}")
        print(f"[{method_name}] actor settings: {actor_values}")

        if method == "icsd":
            from verl.trainer.ppo.icsd_ray_trainer import ICSDRayTrainer as TrainerClass
        else:
            from verl.trainer.ppo.pcsd_ray_trainer import PCSDRayTrainer as TrainerClass

        trainer = TrainerClass(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            skill_provider=skill_provider,
        )
        trainer.init_workers()
        trainer.fit()
