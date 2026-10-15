"""PCSD trainer with a frozen privileged-context teacher."""

import torch

from verl import DataProto
from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer


class PCSDRayTrainer(SkillSDRayTrainer):
    """Evaluate skill-augmented tokens with the frozen reference checkpoint."""

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        if not self.use_reference_policy:
            raise RuntimeError("PCSD requires a frozen reference policy for its teacher")

        teacher_batch = build_teacher_batch(
            batch=batch,
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
        )
        if not self.ref_in_actor:
            teacher_output = self.ref_policy_wg.compute_ref_log_prob(teacher_batch)
        else:
            teacher_output = self.actor_rollout_wg.compute_ref_log_prob(teacher_batch)
        return teacher_output.batch["ref_log_prob"].detach()
