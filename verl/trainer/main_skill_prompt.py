"""Evaluation-only Skill-Prompt entry point, with and without retrieved skills."""

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_skill_prompt(config)


def run_skill_prompt(config) -> None:
    if not config.trainer.get("val_only", False) or not config.trainer.get(
        "val_before_train", False
    ):
        raise ValueError("Skill-Prompt only supports val_only=True and val_before_train=True")
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = SkillPromptTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class SkillPromptTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

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

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
            "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        # Load skill provider
        from verl.trainer.ppo.rlsd_utils import SkillProvider

        # Preserve existing Skill-Prompt CLI keys without retaining training.
        skill_prompt_cfg = config.algorithm.get("skill_grpo", {})
        skills_dir = skill_prompt_cfg.get("skills_dir", "skills/alfworld")
        skill_all = skill_prompt_cfg.get("skill_all", False)
        skill_provider = SkillProvider(skills_dir=skills_dir, skill_all=skill_all)
        print(f"[Skill-Prompt] Loaded skills from {skills_dir}")
        print(f"[Skill-Prompt] Available skills: {list(skill_provider.skill_contents.keys())}")
        print(f"[Skill-Prompt] Task-to-skill mapping: {skill_provider.task_to_skill}")

        # Create skill-aware trajectory collector
        traj_collector = SkillTrajectoryCollector(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            skill_provider=skill_provider,
        )

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = SkillPromptEvaluator(
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
        )
        trainer.init_workers()
        trainer.fit()


class SkillTrajectoryCollector:
    """
    TrajectoryCollector wrapper that injects skill information into observation text.
    Delegates all actual work to the underlying TrajectoryCollector instance.
    """

    def __init__(self, config, tokenizer, processor=None, skill_provider=None):
        from agent_system.multi_turn_rollout import TrajectoryCollector

        self._collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
        self.skill_provider = skill_provider
        self.inject_skill = True
        # Expose config for external access
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def _inject_skills_into_obs(self, obs):
        """Prepend skill text to each observation's text field."""
        if not self.inject_skill or self.skill_provider is None:
            return obs
        obs_texts = obs.get("text", None)
        if obs_texts is None:
            return obs
        for i in range(len(obs_texts)):
            if obs_texts[i] is not None:
                skill_text = self.skill_provider.get_privileged_info_from_prompt(obs_texts[i])
                if skill_text:
                    obs_texts[i] = f"[Privileged Skill Information]\n{skill_text}\n\n{obs_texts[i]}"
        return obs

    def preprocess_single_sample(self, item, gen_batch, obs):
        return self._collector.preprocess_single_sample(item, gen_batch, obs)

    def preprocess_batch(self, gen_batch, obs):
        obs = self._inject_skills_into_obs(obs)
        return self._collector.preprocess_batch(gen_batch, obs)

    def vanilla_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
        return self._collector.vanilla_multi_turn_loop(gen_batch, actor_rollout_wg, envs)

    def dynamic_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
        return self._collector.dynamic_multi_turn_loop(gen_batch, actor_rollout_wg, envs)

    def gather_rollout_data(self, *args, **kwargs):
        return self._collector.gather_rollout_data(*args, **kwargs)

    def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
        """
        Override multi_turn_loop to inject skills into observations.
        We monkey-patch the inner collector's preprocess_batch to inject skills.
        """
        if is_train:
            raise ValueError("Skill-Prompt does not support training rollouts")
        # For validation, inject_skill is controlled externally by the evaluator

        # Temporarily replace the inner collector's preprocess_batch
        original_preprocess_batch = self._collector.preprocess_batch

        def skill_preprocess_batch(gen_batch, obs):
            obs = self._inject_skills_into_obs(obs)
            return original_preprocess_batch(gen_batch, obs)

        self._collector.preprocess_batch = skill_preprocess_batch
        try:
            result = self._collector.multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train)
        finally:
            self._collector.preprocess_batch = original_preprocess_batch

        return result


class SkillPromptEvaluator:
    """
    Wrapper around RayPPOTrainer that performs dual validation
    (with skill and without skill) during evaluation.
    """

    def __init__(self, **kwargs):
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        if not kwargs["config"].trainer.get("val_only", False) or not kwargs[
            "config"
        ].trainer.get("val_before_train", False):
            raise ValueError("Skill-Prompt requires val_only=True and val_before_train=True")
        self._trainer = RayPPOTrainer(**kwargs)
        self._traj_collector = kwargs["traj_collector"]

    def init_workers(self):
        self._trainer.init_workers()

    def fit(self):
        """
        Override fit to intercept _validate calls for dual evaluation.
        We monkey-patch _validate on the trainer instance.
        """
        original_validate = self._trainer._validate
        traj_collector = self._traj_collector

        def dual_validate():
            # First validation: without skill
            traj_collector.inject_skill = False
            metrics_no_skill = original_validate()
            prefixed_no = {f"val_no_skill/{k.replace('val/', '')}": v for k, v in metrics_no_skill.items()}

            # Second validation: with skill
            traj_collector.inject_skill = True
            metrics_with_skill = original_validate()
            prefixed_with = {f"val_with_skill/{k.replace('val/', '')}": v for k, v in metrics_with_skill.items()}

            # Combine both
            combined = {}
            combined.update(prefixed_with)
            combined.update(prefixed_no)
            # Also keep original val/ keys from with_skill run for backward compatibility
            combined.update(metrics_with_skill)
            return combined

        self._trainer._validate = dual_validate
        self._trainer.fit()


if __name__ == "__main__":
    main()
