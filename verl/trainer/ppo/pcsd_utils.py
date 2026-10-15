"""Persistent Consistency Self-Distillation (PCSD) utilities."""

from __future__ import annotations

import torch

from verl.trainer.ppo.core_algos import agg_loss


def _shift_left(values: torch.Tensor, offset: int) -> torch.Tensor:
    shifted = torch.zeros_like(values)
    if offset == 0:
        shifted.copy_(values)
    elif offset < values.shape[1]:
        shifted[:, :-offset] = values[:, offset:]
    return shifted


def _window_statistics(
    values: torch.Tensor,
    mask: torch.Tensor,
    window_size: int,
    decay: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weighted_sum = torch.zeros_like(values)
    weight_sum = torch.zeros_like(values)
    square_sum = torch.zeros_like(values)
    for offset in range(window_size):
        shifted_mask = _shift_left(mask, offset)
        shifted_values = _shift_left(values, offset)
        weight = 1.0 if decay is None else decay**offset
        weighted_sum += weight * shifted_mask * shifted_values
        weight_sum += weight * shifted_mask
        square_sum += shifted_mask * shifted_values.square()
    mean = weighted_sum / weight_sum.clamp_min(1.0)
    return mean, weight_sum, square_sum


def compute_pcsd_weights(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    n_min: int = 1,
    n_max: int = 8,
    decay: float = 0.8,
    variance_low: float = 0.05,
    variance_high: float = 0.5,
    trend_gamma: float = 0.3,
    gate_beta: float = 5.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """Compute detached PCSD token weights independently within each row."""
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("student and teacher log-probability shapes must match")
    if response_mask.shape != student_log_probs.shape:
        raise ValueError("response_mask and log-probability shapes must match")
    if n_min < 1 or n_max < n_min:
        raise ValueError("window sizes must satisfy 1 <= n_min <= n_max")
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be in (0, 1)")
    if variance_high <= variance_low:
        raise ValueError("variance_high must be greater than variance_low")
    if trend_gamma < 0.0 or gate_beta < 0.0 or eps <= 0.0:
        raise ValueError("trend_gamma/gate_beta must be non-negative and eps positive")

    mask = response_mask.detach().float()
    gap = (teacher_log_probs.detach().float() - student_log_probs.detach().float()) * mask

    short_mean, _, _ = _window_statistics(gap, mask, n_min, decay)
    long_mean, _, _ = _window_statistics(gap, mask, n_max, decay)

    local_mean, local_count, local_square_sum = _window_statistics(
        gap, mask, n_max, None
    )
    local_variance = (
        local_square_sum / local_count.clamp_min(1.0) - local_mean.square()
    ).clamp_min(0.0)
    interpolation = ((local_variance - variance_low) / (variance_high - variance_low)).clamp(0.0, 1.0)
    adaptive_gap = (1.0 - interpolation) * short_mean + interpolation * long_mean

    position_sum = torch.zeros_like(gap)
    slope_numerator = torch.zeros_like(gap)
    slope_denominator = torch.zeros_like(gap)
    for offset in range(n_max):
        shifted_mask = _shift_left(mask, offset)
        position_sum += shifted_mask * float(offset)
    mean_position = position_sum / local_count.clamp_min(1.0)
    for offset in range(n_max):
        shifted_mask = _shift_left(mask, offset)
        shifted_gap = _shift_left(gap, offset)
        centered = float(offset) - mean_position
        slope_numerator += shifted_mask * centered * shifted_gap
        slope_denominator += shifted_mask * centered.square()
    slope = slope_numerator / (slope_denominator + eps)
    slope = torch.where(local_count >= 2.0, slope, torch.zeros_like(slope))

    response_scale = (
        gap.abs().sum(dim=-1, keepdim=True)
        / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        + eps
    )
    trend_factor = (
        1.0 - trend_gamma * torch.relu(-slope / response_scale)
    ).clamp(0.0, 1.0)
    weights = torch.sigmoid(gate_beta * adaptive_gap) * trend_factor * mask
    weights = weights.detach()

    valid_count = mask.sum().clamp_min(1.0)
    metrics = {
        "pcsd/weight_mean": (weights.sum() / valid_count).item(),
        "pcsd/adaptive_gap_mean": ((adaptive_gap * mask).sum() / valid_count).item(),
        "pcsd/local_variance_mean": ((local_variance * mask).sum() / valid_count).item(),
        "pcsd/trend_factor_mean": ((trend_factor * mask).sum() / valid_count).item(),
        "pcsd/negative_trend_ratio": (((slope < 0.0).float() * mask).sum() / valid_count).item(),
    }
    return weights.to(student_log_probs.dtype), metrics


def compute_pcsd_loss(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    n_min: int = 1,
    n_max: int = 8,
    decay: float = 0.8,
    variance_low: float = 0.05,
    variance_high: float = 0.5,
    trend_gamma: float = 0.3,
    gate_beta: float = 5.0,
    eps: float = 1e-8,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    weights, metrics = compute_pcsd_weights(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        n_min=n_min,
        n_max=n_max,
        decay=decay,
        variance_low=variance_low,
        variance_high=variance_high,
        trend_gamma=trend_gamma,
        gate_beta=gate_beta,
        eps=eps,
    )
    loss_mat = weights * (teacher_log_probs.detach() - student_log_probs)
    loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
    )
    metrics["pcsd/loss"] = loss.detach().item()
    return loss, metrics
