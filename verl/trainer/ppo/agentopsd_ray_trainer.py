"""AgentOPSD's released Bayesian turn/token credit on the local trainer loop."""

import torch

from verl import DataProto
from verl.trainer.ppo.agentopsd_utils import (
    compute_group_mean_v0,
    compute_opsd_token_advantage,
    compute_opsd_turn_advantage,
)
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer
from verl.utils.torch_functional import masked_mean


class AgentOPSDRayTrainer(SkillSDRayTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.algorithm.get("agentopsd", {})
        self.opsd_cfg = cfg
        self.opsd_granularity = str(cfg.get("granularity", "turn"))
        if self.opsd_granularity not in ("turn", "token"):
            raise ValueError("AgentOPSD granularity must be turn or token")

    def _get_mult_lambda(self):
        initial = float(self.opsd_cfg.get("mult_lambda", 0.5))
        warmdown = int(self.opsd_cfg.get("warmdown_steps", -1))
        if warmdown <= 0:
            return initial
        return initial * max(0.0, 1.0 - self.global_steps / warmdown)

    def _prepare_method_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        cfg = self.opsd_cfg
        rewards = batch.batch["token_level_rewards"]
        mask = batch.batch["response_mask"]
        student_lp = batch.batch["old_log_probs"]
        teacher_lp = batch.batch["teacher_log_probs"]
        seq_adv = batch.batch["advantages"]
        first_valid = mask.long().argmax(dim=-1)
        rows = torch.arange(seq_adv.shape[0], device=seq_adv.device)
        seq_per_row = seq_adv[rows, first_valid]
        uid = batch.non_tensor_batch.get("uid")
        traj_uid = batch.non_tensor_batch.get("traj_uid")
        prior = (
            compute_group_mean_v0(rewards, mask, uid, traj_uid)
            if uid is not None and traj_uid is not None else None
        )
        options = dict(
            token_level_rewards=rewards,
            student_log_probs=student_lp,
            teacher_log_probs=teacher_lp,
            response_mask=mask,
            v0_prior=float(cfg.get("v0_prior", 0.5)),
            v0_per_traj=prior,
            seq_advantage_per_row=seq_per_row,
            uid=uid,
            use_anchoring=bool(cfg.get("use_anchoring", False)),
            use_group_norm=bool(cfg.get("use_group_norm", False)),
            center_delta=bool(cfg.get("center_delta", False)),
            belief_decay_gamma=float(cfg.get("belief_decay_gamma", 1.0)),
            tether_seq=bool(cfg.get("tether_seq", False)),
            tether_lambda=float(cfg.get("tether_lambda", 0.5)),
            tether_band=float(cfg.get("tether_band", 0.2)),
            belief_mult=bool(cfg.get("belief_mult", True)),
            mult_lambda=self._get_mult_lambda(),
            mult_band=float(cfg.get("mult_band", 0.2)),
            belief_gate=bool(cfg.get("belief_gate", True)),
            mult_signed=bool(cfg.get("mult_signed", True)),
            signed=bool(cfg.get("signed", True)),
        )
        if self.opsd_granularity == "turn":
            if traj_uid is None or "turn_step" not in batch.non_tensor_batch:
                raise KeyError("AgentOPSD turn credit requires traj_uid and turn_step")
            advantages = compute_opsd_turn_advantage(
                **options,
                traj_uid=traj_uid,
                turn_step=batch.non_tensor_batch["turn_step"],
                episode_rewards=batch.non_tensor_batch.get("episode_rewards"),
            )
        else:
            advantages = compute_opsd_token_advantage(**options)
        batch.batch["advantages"] = advantages
        delta = (teacher_lp - student_lp) * mask
        return batch, {
            "agentopsd/granularity_turn": float(self.opsd_granularity == "turn"),
            "agentopsd/mult_lambda": self._get_mult_lambda(),
            "agentopsd/teacher_student_gap_mean": masked_mean(delta, mask).item(),
        }
