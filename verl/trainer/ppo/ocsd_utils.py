"""Pure helpers for Observation-Calibrated Self-Distillation (OCSD).

The implementation follows the released OCSD/OC-OPSD code while keeping the
method isolated from the shared environment and actor implementations.
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch


ASSISTANT_MARKERS = (
    "<|im_start|>assistant",
    "<|start_header_id|>assistant<|end_header_id|>",
    "<|assistant|>",
    "[ASSISTANT]",
    "### Assistant:",
    "Assistant:",
)


def extract_action_text(response: str) -> str:
    """Extract the environment action from a decoded model response."""
    for pattern in (
        r"<action>(.*?)</action>",
        r"(<search>.*?</search>)",
        r"(<answer>.*?</answer>)",
    ):
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    match = re.search(r"(?i)^action:\s*(.+)", response, re.MULTILINE)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return response.strip() or "a future action"


def build_action_schema(action: str, env_name: str) -> str:
    """Remove entity values while retaining the next action's intent."""
    action = action.strip()
    env_name = env_name.lower()
    lowered = action.lower()

    if "alfworld" in env_name:
        patterns = (
            (r"^take .+ from .+", "take an item from a receptacle"),
            (r"^move .+ to .+", "move an item to a location"),
            (r"^heat .+ with .+", "heat an item with an appliance"),
            (r"^cool .+ with .+", "cool an item with an appliance"),
            (r"^clean .+ with .+", "clean an item with a fixture"),
            (r"^slice .+ with .+", "slice an item with a tool"),
            (r"^go to .+", "go to a location"),
            (r"^open .+", "open a receptacle"),
            (r"^close .+", "close a receptacle"),
            (r"^use .+", "use a lamp"),
            (r"^examine .+", "examine a target"),
        )
        for pattern, schema in patterns:
            if re.match(pattern, lowered):
                return schema
        return "a future action"

    if env_name == "scienceworld":
        for pattern, schema in (
            (r"^focus on .+", "focus on a task object"),
            (r"^use .+ on .+", "use a tool on a target"),
            (r"^connect .+ to .+", "connect two electrical terminals"),
            (r"^move .+ to .+", "move an object to a container"),
            (r"^pour .+ into .+", "pour a liquid into a container"),
            (r"^pick up .+", "pick up an object"),
            (r"^go to .+", "go to a room"),
            (r"^(open|close|activate|deactivate|mix|read|look at|look in) .+", None),
        ):
            match = re.match(pattern, lowered)
            if match:
                return schema or f"{match.group(1)} a target"
        return lowered if lowered in {"wait", "wait1", "look around", "inventory", "task"} else "a future action"

    if "search" in env_name:
        if re.match(r"<search>", action, re.IGNORECASE):
            return "<search>a query</search>"
        if re.match(r"<answer>", action, re.IGNORECASE):
            return "<answer>an answer</answer>"
        return "a future action"

    if "webshop" in env_name:
        if re.match(r"^search\[", lowered):
            return "search[a query]"
        match = re.match(r"^([a-z_][a-z0-9_]*)\[(.*)\]$", lowered, re.DOTALL)
        if not match:
            return "a future action"
        tool, argument = match.group(1), match.group(2).strip()
        if tool != "click":
            return f"{tool}[an argument]"
        if argument == "buy now":
            return "click[purchase button]"
        if argument in {"back to search", "next page", "prev page", "previous page"}:
            return "click[navigation or pagination]"
        if len(argument) <= 20 and "'" not in argument:
            return "click[a product attribute]"
        return "click[a specific product]"

    return "a future action"


def insert_future_evidence_before_response(prompt: str, future_block: str) -> str:
    """Insert privileged evidence immediately before the last assistant turn."""
    insert_at = max(prompt.rfind(marker) for marker in ASSISTANT_MARKERS)
    block = "\n\n" + future_block.strip() + "\n\n"
    if insert_at < 0:
        return prompt.rstrip() + block
    return prompt[:insert_at].rstrip() + block + prompt[insert_at:].lstrip()


def build_future_block(
    available_future_len: int,
    obs_1: str,
    action_1: str,
    obs_2: str,
    env_name: str,
    full: bool,
) -> str:
    """Build structurally matched Full or Ablated future evidence."""
    shown_obs_1 = obs_1.strip() if full and obs_1 else "not provided"
    if available_future_len <= 1:
        return (
            "Future evidence:\n"
            "After current action:\n"
            f"Observation: {shown_obs_1}"
        )

    shown_obs_2 = obs_2.strip() if full and obs_2 else "not provided"
    schema = build_action_schema(action_1, env_name)
    return (
        "Future evidence:\n"
        "After current action:\n"
        f"Observation: {shown_obs_1}\n\n"
        "Next action:\n"
        f"{schema}\n\n"
        "After next action:\n"
        f"Observation: {shown_obs_2}"
    )


def normalize_advantages_shape(
    advantages: torch.Tensor, response_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token-level advantages and their per-step scalar value."""
    batch_size, response_length = response_mask.shape
    if advantages.ndim == 1:
        sequence_advantages = advantages
        token_advantages = advantages[:, None].expand(batch_size, response_length).clone()
    elif advantages.ndim == 2 and advantages.shape[1] == 1:
        sequence_advantages = advantages[:, 0]
        token_advantages = advantages.expand(batch_size, response_length).clone()
    elif advantages.shape == response_mask.shape:
        token_advantages = advantages
        denominator = response_mask.sum(dim=-1).clamp(min=1)
        sequence_advantages = (advantages * response_mask).sum(dim=-1) / denominator
    else:
        raise ValueError(
            f"Unsupported advantages shape {tuple(advantages.shape)} for response "
            f"mask {tuple(response_mask.shape)}"
        )
    return token_advantages, sequence_advantages


def select_uncertain_steps(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    available_future_len: np.ndarray,
    sequence_advantages: np.ndarray,
    trajectory_ids: np.ndarray,
    top_step_ratio: float,
    advantage_epsilon: float,
) -> np.ndarray:
    """Select the highest-NLL eligible steps independently per trajectory."""
    eligible = (available_future_len > 0) & (
        np.abs(sequence_advantages) > advantage_epsilon
    )
    if top_step_ratio >= 1.0:
        return eligible

    mask = response_mask.float()
    nll = (-(old_log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)).detach()
    nll = nll.cpu().numpy()
    selected = np.zeros(len(eligible), dtype=bool)
    for trajectory_id in np.unique(trajectory_ids):
        indices = np.where((trajectory_ids == trajectory_id) & eligible)[0]
        if len(indices) == 0:
            continue
        count = max(1, math.ceil(len(indices) * top_step_ratio))
        selected[indices[np.argsort(nll[indices])[::-1][:count]]] = True
    return eligible & selected


def compute_ocsd_advantages(
    advantages: torch.Tensor,
    full_log_probs: torch.Tensor,
    ablated_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    available_future_len: torch.Tensor,
    beta: float = 0.5,
    advantage_epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the released OCSD observation-residual advantage modulation."""
    token_advantages, sequence_advantages = normalize_advantages_shape(
        advantages, response_mask
    )
    future_mask = (available_future_len > 0).float()[:, None]
    advantage_mask = (sequence_advantages.abs() > advantage_epsilon).float()[:, None]
    oc_mask = response_mask.float() * future_mask * advantage_mask

    residual = (full_log_probs - ablated_log_probs) * response_mask
    quality = torch.tanh(residual / 2.0)
    signed_quality = torch.sign(sequence_advantages)[:, None] * quality
    modulation = 1.0 + beta * oc_mask * signed_quality
    calibrated = token_advantages * modulation

    with torch.no_grad():
        fallback = (~oc_mask.bool()) & response_mask.bool()
        if fallback.any():
            maximum_difference = (
                calibrated[fallback] - token_advantages[fallback]
            ).abs().max()
            assert maximum_difference.item() < 1e-6, (
                "OCSD fallback tokens must exactly retain their GRPO advantages"
            )

    with torch.no_grad():
        valid = response_mask.bool()
        selected = oc_mask.bool()
        valid_count = valid.sum().clamp(min=1)
        oc_count = oc_mask.sum().clamp(min=1)
        selected_quality = quality[selected] if selected.any() else quality.new_zeros(1)
        selected_residual = residual[selected] if selected.any() else residual.new_zeros(1)
        selected_modulation = modulation[selected] if selected.any() else modulation.new_ones(1)
        sign = torch.sign(sequence_advantages)[:, None].expand_as(quality)
        align_ratio = (((sign * quality > 0).float() * oc_mask).sum() / oc_count).item()
        valid_advantages = token_advantages[valid]
        valid_calibrated = calibrated[valid]
        std_before = valid_advantages.std().item() if valid_advantages.numel() > 1 else 0.0
        std_after = valid_calibrated.std().item() if valid_calibrated.numel() > 1 else 0.0
        metrics = {
            "oc_opsd/q_mean": selected_quality.mean().item(),
            "oc_opsd/q_abs_mean": selected_quality.abs().mean().item(),
            "oc_opsd/q_pos_ratio": (selected_quality > 0).float().mean().item(),
            "oc_opsd/q_neg_ratio": (selected_quality < 0).float().mean().item(),
            "oc_opsd/residual_mean": selected_residual.mean().item(),
            "oc_opsd/residual_abs_mean": selected_residual.abs().mean().item(),
            "oc_opsd/m_future_ratio": future_mask.mean().item(),
            "oc_opsd/adv_nonzero_ratio": advantage_mask.mean().item(),
            "oc_opsd/m_oc_ratio": (oc_mask.sum() / valid_count).item(),
            "oc_opsd/available_future_len_mean": available_future_len.float()
            .mean()
            .item(),
            "oc_opsd/A_oc_mean": (calibrated * response_mask).sum()
            .div(valid_count)
            .item(),
            "oc_opsd/A_oc_abs_mean": (calibrated.abs() * response_mask)
            .sum()
            .div(valid_count)
            .item(),
            "oc_opsd/modulation_factor_mean": selected_modulation.mean().item(),
            "oc_opsd/modulation_factor_min": selected_modulation.min().item(),
            "oc_opsd/modulation_factor_max": selected_modulation.max().item(),
            "oc_opsd/adv_delta_abs_mean": (
                (calibrated - token_advantages).abs() * response_mask
            ).sum().div(valid_count).item(),
            "oc_opsd/sign_flip_count": ((calibrated * token_advantages < 0) & valid)
            .sum()
            .item(),
            "oc_opsd/align_ratio": align_ratio,
            "oc_opsd/adv_std_before": std_before,
            "oc_opsd/adv_std_after": std_after,
            "oc_opsd/adv_std_ratio": std_after / (std_before + 1e-8)
            if valid_advantages.numel() > 1
            else 1.0,
        }
    return calibrated, metrics
