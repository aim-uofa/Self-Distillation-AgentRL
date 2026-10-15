"""Influence Calibration for Self-Distillation (ICSD) utilities.

The coefficient construction follows arXiv:2608.14945.  All allocation
statistics are detached; gradients flow only through the student log
probabilities in :func:`compute_icsd_loss`.
"""

from __future__ import annotations

import torch

from verl.trainer.ppo.core_algos import agg_loss


def _fixed_asymmetric_laplace_cdf(
    values: torch.Tensor,
    split: float,
) -> torch.Tensor:
    """Paper-compatible fixed monotone fallback (location 0, unit scales)."""
    left = split * torch.exp(torch.clamp(values, max=0.0))
    right = 1.0 - (1.0 - split) * torch.exp(torch.clamp(-values, max=0.0))
    return torch.where(values < 0.0, left, right)


def _fit_asymmetric_laplace_cdf(
    values: torch.Tensor,
    fit_values: torch.Tensor,
    global_values: torch.Tensor,
    *,
    split: float,
    scale_floor: float,
    min_group_tokens: int,
) -> tuple[torch.Tensor, bool]:
    """Fit the robust asymmetric CDF, falling back group -> batch -> fixed."""
    fallback_used = False
    source = fit_values
    if source.numel() < min_group_tokens or not torch.isfinite(source).all():
        source = global_values
        fallback_used = True

    if source.numel() < min_group_tokens or not torch.isfinite(source).all():
        return _fixed_asymmetric_laplace_cdf(values, split), True

    location = source.median()
    lower = source[source < location]
    upper = source[source >= location]
    lower_scale = (
        (lower - location).abs().mean()
        if lower.numel() > 0
        else location.new_tensor(scale_floor)
    )
    upper_scale = (
        (upper - location).abs().mean()
        if upper.numel() > 0
        else location.new_tensor(scale_floor)
    )
    lower_scale = lower_scale.clamp_min(scale_floor)
    upper_scale = upper_scale.clamp_min(scale_floor)

    lower_cdf = split * torch.exp(
        torch.clamp((values - location) / lower_scale, max=0.0)
    )
    upper_cdf = 1.0 - (1.0 - split) * torch.exp(
        torch.clamp(-(values - location) / upper_scale, max=0.0)
    )
    return torch.where(values < location, lower_cdf, upper_cdf), fallback_used


@torch.no_grad()
def compute_icsd_coefficients(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    turn_indices: torch.Tensor,
    trust_split: float = 0.4,
    influence_split: float = 0.5,
    scale_floor: float = 1e-4,
    min_group_tokens: int = 8,
) -> tuple[torch.Tensor, dict]:
    """Compute detached ICSD coefficients for one actor mini-batch.

    Calibration pools valid tokens with the same interaction-turn index across
    trajectories.  Mass matching is performed separately for every response
    row, which corresponds to one concrete action turn in this repository.
    """
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("student and teacher log-probability shapes must match")
    if old_log_probs.shape != student_log_probs.shape:
        raise ValueError("old and student log-probability shapes must match")
    if advantages.shape != student_log_probs.shape:
        raise ValueError("advantages and student log-probability shapes must match")
    if response_mask.shape != student_log_probs.shape:
        raise ValueError("response_mask and student log-probability shapes must match")
    if turn_indices.ndim != 1 or turn_indices.shape[0] != student_log_probs.shape[0]:
        raise ValueError("turn_indices must have shape (batch_size,)")
    if not 0.0 < trust_split < 1.0:
        raise ValueError("trust_split must be in (0, 1)")
    if not 0.0 < influence_split < 1.0:
        raise ValueError("influence_split must be in (0, 1)")
    if scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive")
    if min_group_tokens < 1:
        raise ValueError("min_group_tokens must be positive")

    mask = response_mask.bool()
    student = student_log_probs.detach().float()
    teacher = teacher_log_probs.detach().float()
    old = old_log_probs.detach().float()
    advantage = advantages.detach().float()

    gap = teacher - student
    ratio = torch.exp(student - old)
    influence = advantage * ratio * gap
    global_gap = gap[mask]
    global_influence = influence[mask]

    trust = torch.zeros_like(gap)
    calibrated_influence = torch.zeros_like(gap)
    fallback_groups = 0

    for turn_index in torch.unique(turn_indices.detach()):
        row_mask = turn_indices == turn_index
        group_mask = mask & row_mask.unsqueeze(-1)
        if not group_mask.any():
            continue
        trust_values, trust_fallback = _fit_asymmetric_laplace_cdf(
            gap[group_mask],
            gap[group_mask],
            global_gap,
            split=trust_split,
            scale_floor=scale_floor,
            min_group_tokens=min_group_tokens,
        )
        influence_values, influence_fallback = _fit_asymmetric_laplace_cdf(
            influence[group_mask],
            influence[group_mask],
            global_influence,
            split=influence_split,
            scale_floor=scale_floor,
            min_group_tokens=min_group_tokens,
        )
        trust[group_mask] = trust_values
        calibrated_influence[group_mask] = influence_values
        fallback_groups += int(trust_fallback or influence_fallback)

    fallback_mask = mask & (advantage < 0.0) & (gap < 0.0)
    calibrated_mask = mask & ~fallback_mask
    coefficients = torch.zeros_like(gap)
    coefficients[fallback_mask] = trust[fallback_mask]

    mass_errors = []
    degenerate_turns = 0
    for row in range(gap.shape[0]):
        valid_row = mask[row]
        calibrated_row = calibrated_mask[row]
        numerator = trust[row][calibrated_row].sum()
        denominator = (
            trust[row][calibrated_row]
            * calibrated_influence[row][calibrated_row]
        ).sum()
        if calibrated_row.any() and denominator > scale_floor:
            alpha = numerator / denominator
            coefficients[row][calibrated_row] = (
                alpha
                * trust[row][calibrated_row]
                * calibrated_influence[row][calibrated_row]
            )
        else:
            coefficients[row][valid_row] = trust[row][valid_row]
            degenerate_turns += 1
        mass_errors.append(
            (coefficients[row][valid_row].sum() - trust[row][valid_row].sum()).abs()
        )

    valid_count = mask.sum().clamp_min(1)
    mass_error = (
        torch.stack(mass_errors).max()
        if mass_errors
        else gap.new_tensor(0.0)
    )
    metrics = {
        "icsd/trust_mean": ((trust * mask).sum() / valid_count).item(),
        "icsd/coefficient_mean": ((coefficients * mask).sum() / valid_count).item(),
        "icsd/influence_mean": ((influence * mask).sum() / valid_count).item(),
        "icsd/influence_abs_mean": ((influence.abs() * mask).sum() / valid_count).item(),
        "icsd/fallback_token_ratio": (fallback_mask.sum() / valid_count).item(),
        "icsd/calibration_fallback_groups": float(fallback_groups),
        "icsd/degenerate_turns": float(degenerate_turns),
        "icsd/max_turn_mass_error": mass_error.item(),
    }
    return coefficients.to(student_log_probs.dtype).detach(), metrics


def compute_icsd_loss(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    coefficients: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """Apply precomputed detached ICSD allocation to the KD loss."""
    teacher = teacher_log_probs.detach()
    weights = coefficients.detach()
    loss_mat = weights * (teacher - student_log_probs)
    loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
    )
    return loss, {"icsd/loss": loss.detach().item()}
