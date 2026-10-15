"""
SkillSD (Skill-based Self-Distillation) utilities.

Provides the SDL (Self Distillation Loss) computation for the SkillSD algorithm.
Skill retrieval and teacher batch construction are reused from rlsd_utils.py
and rlsd_ray_trainer.py without modification.
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_sdl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    """
    Compute the Self Distillation Loss (SDL) for SkillSD.

    Args:
        student_log_probs: (bs, response_length) — log π_θ(y_t | x, y_<t).
            Current policy forward pass; retains gradients.
        teacher_log_probs: (bs, response_length) — log π_θ(y_t | x ⊕ S(x), y_<t).
            Frozen (no grad). Teacher sees skill-augmented input.
        old_log_probs: (bs, response_length) — log π_old(y_t | x, y_<t).
            Frozen (no grad). Rollout-time policy.
        response_mask: (bs, response_length) — mask for valid response tokens.
        loss_agg_mode: aggregation mode passed to agg_loss.

    Returns:
        sdl_loss: scalar — the aggregated SDL loss.
    """
    teacher_log_probs = teacher_log_probs.detach()
    old_log_probs = old_log_probs.detach()

    ell = student_log_probs - teacher_log_probs

    neg_ell_clamped = (-ell).clamp(max=20.0)
    k3 = torch.exp(neg_ell_clamped) - 1.0 + ell

    log_rho_on = (student_log_probs - old_log_probs).clamp(max=10.0)
    rho_on = torch.exp(log_rho_on)

    sdl_per_token = rho_on * k3

    sdl_loss = agg_loss(loss_mat=sdl_per_token, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return sdl_loss
