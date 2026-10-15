"""Isolated training entry point for Observation-Calibrated SD (OCSD)."""

import math

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ocsd(config)


def run_ocsd(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        runtime = get_ppo_ray_runtime_env()
        ray_kwargs = config.get("ray_init", {})
        runtime = OmegaConf.merge(runtime, ray_kwargs.get("runtime_env", {}))
        ray_kwargs = OmegaConf.create({**ray_kwargs, "runtime_env": runtime})
        print(f"ray init kwargs: {ray_kwargs}")
        ray.init(**OmegaConf.to_container(ray_kwargs))

    ray.get(OCSDTaskRunner.remote().run.remote(config))


@ray.remote(num_cpus=1)
class OCSDTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        ocsd = config.algorithm.get("oc_opsd", {})
        beta = float(ocsd.get("beta", 0.5))
        advantage_epsilon = float(ocsd.get("eps_A", 1e-6))
        future_horizon = int(ocsd.get("K_future", 2))
        top_step_ratio = float(ocsd.get("top_step_ratio", 0.2))
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError(f"algorithm.oc_opsd.beta must be in [0, 1], got {beta}")
        if not math.isfinite(advantage_epsilon) or advantage_epsilon < 0.0:
            raise ValueError(
                "algorithm.oc_opsd.eps_A must be finite and non-negative, "
                f"got {advantage_epsilon}"
            )
        if future_horizon not in (1, 2):
            raise ValueError(
                f"algorithm.oc_opsd.K_future must be 1 or 2, got {future_horizon}"
            )
        if not math.isfinite(top_step_ratio) or not 0.0 < top_step_ratio <= 1.0:
            raise ValueError(
                "algorithm.oc_opsd.top_step_ratio must be in (0, 1], "
                f"got {top_step_ratio}"
            )

        # OCSD changes only the post-GRPO advantages. Actor-side losses from
        # every other method are explicitly off in this entry point.
        with open_dict(config):
            for key in (
                "use_sdar_loss",
                "use_sdl_loss",
                "use_ovcsd_loss",
            ):
                config.actor_rollout_ref.actor[key] = False
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

        if config.actor_rollout_ref.rollout.name == "vllm":
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0 and not is_version_ge(
                pkg="vllm", minver="0.7.3"
            ):
                raise NotImplementedError("PPO LoRA requires vllm>=0.7.3")

        if config.actor_rollout_ref.actor.strategy in ("fsdp", "fsdp2"):
            assert config.critic.strategy in ("fsdp", "fsdp2")
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

        workers = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        pool_id = "global_pool"
        resource_pool_spec = {
            pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        }
        mapping = {Role.ActorRollout: pool_id, Role.Critic: pool_id}

        if config.reward_model.enable:
            if config.reward_model.strategy in ("fsdp", "fsdp2"):
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            workers[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            workers[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = pool_id

        if config.reward_model.get("reward_manager", "episode") != "episode":
            raise NotImplementedError
        from agent_system.reward_manager import EpisodeRewardManager

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
            "Keep actor_rollout_ref.rollout.n=1; environment grouping uses env.rollout.n"
        )
        from agent_system.multi_turn_rollout.ocsd_rollout import OCSDTrajectoryCollector

        collector = OCSDTrajectoryCollector(
            config=config, tokenizer=tokenizer, processor=processor
        )

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        print(
            "[OCSD] "
            f"beta={beta}, advantage_epsilon={advantage_epsilon}, "
            f"future_horizon={future_horizon}, top_step_ratio={top_step_ratio}, "
            f"env={ocsd.get('env_name')}"
        )
        from verl.trainer.ppo.ocsd_ray_trainer import OCSDRayTrainer

        trainer = OCSDRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=workers,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=collector,
            envs=envs,
            val_envs=val_envs,
            skill_provider=None,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
