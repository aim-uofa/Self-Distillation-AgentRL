"""ADRS TVA reward and GRPO* advantage, isolated from existing estimators."""

import torch

from verl import DataProto
from verl.trainer.ppo.adrs_grpo import compute_grpo_outcome_advantage
from verl.trainer.ppo.adrs_utils import compute_adrs_reward
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer


class ADRSRayTrainer(SkillSDRayTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adrs_cfg = self.config.algorithm.get("adrs", {})
        if self.config.algorithm.adv_estimator != "grpo":
            raise ValueError("ADRS local entry supports the released GRPO* recipe only")
        if self.adrs_cfg.get("use_rtg_advantage", False):
            raise ValueError("ADRS use_rtg_advantage=True is not part of the supported paper recipe")
        if self.adrs_cfg.get("eta_mode", "fixed") != "fixed":
            raise ValueError("ADRS local entry supports eta_mode=fixed")
        if self.adrs_cfg.get("tva_level", "L2") == "L3":
            raise ValueError("ADRS L3 requires GiGPO step groups; this entry uses GRPO/L2")

    def _prepare_method_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        cfg = self.adrs_cfg
        warmup = int(cfg.get("wn_warmup_steps", 20))
        active_steps = int(cfg.get("wn_active_steps", 50))
        teacher_active = self.global_steps > warmup and (
            active_steps < 0 or self.global_steps <= warmup + active_steps
        )
        eta = float(cfg.get("eta", 0.02)) if teacher_active else 0.0
        response_mask = batch.batch["response_mask"]
        turn_steps = batch.non_tensor_batch.get("turn_step")
        step_boundary = None
        if turn_steps is not None:
            step_boundary = torch.zeros_like(response_mask, dtype=torch.long)
            for i in range(response_mask.shape[0]):
                step_boundary[i] = int(turn_steps[i])

        teacher_reward, modulation, metrics = compute_adrs_reward(
            teacher_log_probs=batch.batch["teacher_log_probs"],
            response_mask=response_mask,
            step_boundary_mask=step_boundary,
            token_level_rewards=batch.batch["token_level_rewards"],
            prompt_uids=batch.non_tensor_batch.get("uid"),
            ref_log_probs=batch.batch.get("ref_log_prob"),
            eta=eta,
            tva_temperature=float(cfg.get("tva_temperature", 1.0)),
            tva_tau=float(cfg.get("tva_tau", 2.0)),
            baseline_mode=str(cfg.get("baseline_mode", "step")),
            normalize_mode=str(cfg.get("normalize_mode", "per_seq")),
            tva_gate_norm=bool(cfg.get("tva_gate_norm", False)),
            tva_level=str(cfg.get("tva_level", "L2")),
        )
        batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"] + teacher_reward
        batch.batch["teacher_reward_raw"] = teacher_reward.detach()
        batch.batch["tva_modulation"] = modulation
        calculation_mask = response_mask
        if self.config.actor_rollout_ref.rollout.multi_turn.enable:
            calculation_mask = batch.batch["loss_mask"][:, -response_mask.shape[1]:]
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=batch.batch["token_level_rewards"],
            response_mask=calculation_mask,
            index=batch.non_tensor_batch["uid"],
            traj_index=batch.non_tensor_batch["traj_uid"],
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            eta=eta,
            teacher_reward=teacher_reward,
            wn_clip=float(cfg.get("wn_clip", 0.2)),
            wn_sign=float(cfg.get("wn_sign", 1.0)),
            wn_adaptive_clip=bool(cfg.get("wn_adaptive_clip", False)),
            wn_mode=str(cfg.get("wn_mode", "additive")),
            student_log_probs=batch.batch["old_log_probs"],
            teacher_log_probs=batch.batch["teacher_log_probs"],
            wn_neg_reset=bool(cfg.get("wn_neg_reset", False)),
        )
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = returns
        if teacher_active and float(cfg.get("ent_coeff_active", 0.0)) > 0:
            batch.meta_info["entropy_coeff_override"] = float(cfg["ent_coeff_active"])
        if teacher_active and float(cfg.get("kl_boost_mult", 1.0)) != 1.0:
            batch.meta_info["kl_loss_coef_override"] = (
                float(self.config.actor_rollout_ref.actor.kl_loss_coef)
                * float(cfg["kl_boost_mult"])
            )
        metrics["adrs/eta_effective"] = eta
        metrics["adrs/teacher_active"] = float(teacher_active)
        return batch, metrics
