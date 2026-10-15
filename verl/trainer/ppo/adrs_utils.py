"""Copied from gitrxh/ADRS-arxiv (Apache-2.0), adrs_utils.py.

https://github.com/gitrxh/ADRS-arxiv

ADRS: Agentic Reinforcement Learning with Self-Distilled Reward Shaping.

ADRS injects teacher knowledge as token-level reward shaping (no distillation loss).
TVA (Teacher Value Advantage) measures teacher usefulness at multiple granularities,
modulating the teacher reward so only competent teacher signals are used.

Three-level Teacher Usefulness Estimation hierarchy:
    Level 3 (step-level):      Uses anchor state grouping (multi-turn, GiGPO)
    Level 2 (completion-level): Uses prompt grouping (any K>1 setting, including GRPO)
    Level 1 (PAS):             Uses teacher-ref logprob gap (any setting, zero overhead)

Core data flow:
    teacher_log_probs → raw teacher reward (centered, normalized)
    TVA (auto-selected level) → per-token modulation coefficient
    modulated_reward = TVA_modulation × raw_teacher_reward
    token_level_rewards += modulated_reward  (before advantage computation)
"""

import numpy as np
import torch
from collections import defaultdict


# ------------------------------------------------------------------ #
# -------------------- Teacher Reward Shaping ---------------------- #
# ------------------------------------------------------------------ #

def compute_step_mean(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    step_boundary_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-step mean of values, broadcast back to token level.

    Args:
        values: (bs, response_length)
        response_mask: (bs, response_length)
        step_boundary_mask: (bs, response_length) — integer step IDs per token.

    Returns:
        (bs, response_length) — mean of each token's step, masked.
    """
    bs, seq_len = values.shape
    masked_vals = values * response_mask

    max_steps = int(step_boundary_mask.max().item()) + 1

    step_sums = torch.zeros(bs, max_steps, device=values.device)
    step_sums.scatter_add_(1, step_boundary_mask.long(), masked_vals)

    step_counts = torch.zeros(bs, max_steps, device=values.device)
    step_counts.scatter_add_(1, step_boundary_mask.long(), response_mask.float())
    step_counts = step_counts.clamp(min=1)

    step_means = step_sums / step_counts
    step_means_expanded = step_means.gather(1, step_boundary_mask.long())
    return step_means_expanded * response_mask


def compute_teacher_reward(
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_boundary_mask: torch.Tensor = None,
    baseline_mode: str = "step",
    normalize: bool = True,
    normalize_mode: str = "global",
) -> torch.Tensor:
    """
    Compute raw (un-scaled) teacher reward shaping signal.

    r_teacher_t = normalize(teacher_lp_t - baseline_t)

    Args:
        teacher_log_probs: (bs, response_length)
        response_mask: (bs, response_length)
        step_boundary_mask: (bs, response_length) — integer step IDs. None → global baseline.
        baseline_mode: "step" or "global".
        normalize: whether to normalize to unit variance.

    Returns:
        teacher_reward: (bs, response_length) — zero-mean, optionally unit-variance.
    """
    teacher_lp = teacher_log_probs.detach()

    if baseline_mode == "step" and step_boundary_mask is not None:
        baseline = compute_step_mean(teacher_lp, response_mask, step_boundary_mask)
    else:
        valid_lp = teacher_lp[response_mask.bool()]
        baseline = valid_lp.mean() if valid_lp.numel() > 0 else torch.tensor(0.0, device=teacher_lp.device)

    teacher_reward = (teacher_lp - baseline) * response_mask

    if normalize:
        if normalize_mode == "per_seq":
            cnt = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            seq_mean = (teacher_reward * response_mask).sum(dim=-1, keepdim=True) / cnt
            seq_var = (((teacher_reward - seq_mean) * response_mask) ** 2).sum(dim=-1, keepdim=True) / cnt
            seq_std = seq_var.clamp(min=0).sqrt() + 1e-8
            teacher_reward = (teacher_reward / seq_std) * response_mask
        else:
            valid_tr = teacher_reward[response_mask.bool()]
            tr_std = valid_tr.std() + 1e-8 if valid_tr.numel() > 1 else torch.tensor(1.0, device=teacher_lp.device)
            teacher_reward = teacher_reward / tr_std

    return teacher_reward


# ------------------------------------------------------------------ #
# -------------------- TVA Computation ----------------------------- #
# ------------------------------------------------------------------ #

def compute_teacher_alignment_per_step(
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_boundary_mask: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-step teacher alignment score (scalar per step in the batch).

    For each (batch_idx, step_id) pair, computes:
        alpha = sigmoid((mean_teacher_logprob_in_step - global_mean) / temperature)

    Returns a flat tensor of shape (num_valid_steps,) in the order encountered
    when scanning (batch 0 step 0, batch 0 step 1, ..., batch 1 step 0, ...).
    Also returns a mapping from flat index to (batch_idx, step_id).
    """
    bs, seq_len = teacher_log_probs.shape
    masked_tlp = teacher_log_probs.detach() * response_mask

    max_steps = int(step_boundary_mask.max().item()) + 1

    step_tlp_sum = torch.zeros(bs, max_steps, device=teacher_log_probs.device)
    step_count = torch.zeros(bs, max_steps, device=teacher_log_probs.device)
    step_tlp_sum.scatter_add_(1, step_boundary_mask.long(), masked_tlp)
    step_count.scatter_add_(1, step_boundary_mask.long(), response_mask.float())

    global_mean = (masked_tlp.sum() / (response_mask.sum() + eps)).detach()

    step_mean_tlp = step_tlp_sum / (step_count + eps)
    centered = step_mean_tlp - global_mean
    step_alignment_2d = torch.sigmoid(centered / (temperature + eps))

    valid_mask = step_count > 0
    if not valid_mask.any():
        return torch.tensor([], device=teacher_log_probs.device), []

    valid_indices = valid_mask.nonzero(as_tuple=False)  # (N, 2) with [b, sid]
    alignment_flat = step_alignment_2d[valid_indices[:, 0], valid_indices[:, 1]]
    flat_to_bs = valid_indices.tolist()

    return alignment_flat, flat_to_bs


def compute_tva_vectorized(
    step_alignment: torch.Tensor,
    step_rewards: torch.Tensor,
    step_group_uids: np.ndarray,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Vectorized TVA: E[R | teacher-aligned] - E[R | teacher-divergent] per group.

    Args:
        step_alignment: (num_steps,) alignment scores in (0, 1).
        step_rewards: (num_steps,) reward per step.
        step_group_uids: (num_steps,) anchor state group IDs.

    Returns:
        tva: (num_steps,) Teacher Value Advantage per step.
    """
    num_steps = step_rewards.shape[0]
    device = step_rewards.device

    unique_uids, inverse_indices = np.unique(step_group_uids, return_inverse=True)
    group_ids = torch.tensor(inverse_indices, device=device, dtype=torch.long)
    num_groups = len(unique_uids)

    a = step_alignment.to(device)
    r = step_rewards.to(device)

    aligned_wr = torch.zeros(num_groups, device=device)
    aligned_w = torch.zeros(num_groups, device=device)
    divergent_wr = torch.zeros(num_groups, device=device)
    divergent_w = torch.zeros(num_groups, device=device)
    group_count = torch.zeros(num_groups, device=device)

    aligned_wr.scatter_add_(0, group_ids, a * r)
    aligned_w.scatter_add_(0, group_ids, a)
    divergent_wr.scatter_add_(0, group_ids, (1.0 - a) * r)
    divergent_w.scatter_add_(0, group_ids, 1.0 - a)
    group_count.scatter_add_(0, group_ids, torch.ones(num_steps, device=device))

    aligned_mean = aligned_wr / (aligned_w + eps)
    divergent_mean = divergent_wr / (divergent_w + eps)
    group_tva = aligned_mean - divergent_mean

    group_tva[group_count < 2] = 0.0

    tva = group_tva[group_ids]
    return tva


# ------------------------------------------------------------------ #
# ---------- Level 2: Completion-Level TVA (for GRPO) -------------- #
# ------------------------------------------------------------------ #

def compute_completion_level_tva(
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    token_level_rewards: torch.Tensor,
    prompt_uids: np.ndarray,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """
    Level 2 TVA: Completion-level teacher usefulness estimation.

    Works with ANY K>1 setting (including GRPO). Uses the prompt as the
    "shared starting point" instead of anchor states.

    For each prompt group: computes whether completions that are more
    aligned with the teacher tend to have higher rewards.

    TVA_prompt = E[R | teacher-aligned completions] - E[R | teacher-divergent completions]

    Args:
        teacher_log_probs: (bs, response_length)
        response_mask: (bs, response_length)
        token_level_rewards: (bs, response_length) — env rewards per token.
        prompt_uids: (bs,) — prompt group IDs (completions sharing same prompt).
        temperature: sigmoid temperature for alignment score.

    Returns:
        tva_per_sample: (bs,) — one TVA value per completion, broadcast from prompt group.
        metrics: dict.
    """
    bs = teacher_log_probs.shape[0]
    device = teacher_log_probs.device

    masked_tlp = teacher_log_probs.detach() * response_mask
    completion_teacher_mean = masked_tlp.sum(dim=1) / (response_mask.sum(dim=1) + eps)

    completion_reward = token_level_rewards.sum(dim=1)

    global_teacher_mean = completion_teacher_mean.mean()
    alpha = torch.sigmoid((completion_teacher_mean - global_teacher_mean) / (temperature + eps))

    unique_uids, inverse = np.unique(prompt_uids, return_inverse=True)
    group_ids = torch.tensor(inverse, device=device, dtype=torch.long)
    num_groups = len(unique_uids)

    aligned_wr = torch.zeros(num_groups, device=device)
    aligned_w = torch.zeros(num_groups, device=device)
    divergent_wr = torch.zeros(num_groups, device=device)
    divergent_w = torch.zeros(num_groups, device=device)
    group_count = torch.zeros(num_groups, device=device)

    aligned_wr.scatter_add_(0, group_ids, alpha * completion_reward)
    aligned_w.scatter_add_(0, group_ids, alpha)
    divergent_wr.scatter_add_(0, group_ids, (1.0 - alpha) * completion_reward)
    divergent_w.scatter_add_(0, group_ids, 1.0 - alpha)
    group_count.scatter_add_(0, group_ids, torch.ones(bs, device=device))

    group_tva = (aligned_wr / (aligned_w + eps)) - (divergent_wr / (divergent_w + eps))
    group_tva[group_count < 2] = 0.0

    tva_per_sample = group_tva[group_ids]

    with torch.no_grad():
        metrics = {
            "tva_l2/mean": tva_per_sample.mean().item(),
            "tva_l2/std": tva_per_sample.std().item() if bs > 1 else 0.0,
            "tva_l2/positive_ratio": (tva_per_sample > 0).float().mean().item(),
            "tva_l2/num_groups": float(num_groups),
        }

    return tva_per_sample, metrics


# ------------------------------------------------------------------ #
# ---------- Level 1: PAS (Privilege Advantage Signal) ------------- #
# ------------------------------------------------------------------ #

def compute_pas(
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """
    Level 1 PAS: Token-level privilege advantage signal.

    PAS_t = log π_teacher(y_t) - log π_ref(y_t)

    Measures how much the teacher (with privilege/skills) prefers this token
    compared to the reference policy (without privilege). Works in ANY setting
    with zero additional computation (ref logprobs already computed for KL penalty).

    Args:
        teacher_log_probs: (bs, response_length)
        ref_log_probs: (bs, response_length)
        response_mask: (bs, response_length)

    Returns:
        pas: (bs, response_length) — normalized PAS signal.
        metrics: dict.
    """
    pas = (teacher_log_probs.detach() - ref_log_probs.detach()) * response_mask

    valid_pas = pas[response_mask.bool()]
    pas_std = valid_pas.std() + eps if valid_pas.numel() > 1 else torch.tensor(1.0, device=pas.device)
    pas_normalized = pas / pas_std

    with torch.no_grad():
        metrics = {
            "pas/mean": valid_pas.mean().item() if valid_pas.numel() > 0 else 0.0,
            "pas/std": (pas_std - eps).item(),
            "pas/positive_ratio": (valid_pas > 0).float().mean().item() if valid_pas.numel() > 0 else 0.0,
        }

    return pas_normalized, metrics


# ------------------------------------------------------------------ #
# -------------------- Combined STAR + TVA ------------------------- #
# ------------------------------------------------------------------ #

def compute_adrs_reward(
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_rewards: torch.Tensor = None,
    step_group_uids: np.ndarray = None,
    step_boundary_mask: torch.Tensor = None,
    token_level_rewards: torch.Tensor = None,
    prompt_uids: np.ndarray = None,
    ref_log_probs: torch.Tensor = None,
    eta: float = 0.1,
    tva_temperature: float = 1.0,
    tva_tau: float = 2.0,
    baseline_mode: str = "step",
    normalize: bool = True,
    normalize_mode: str = "global",
    tva_gate_norm: bool = False,
    tva_level: str = "auto",
) -> tuple[torch.Tensor, dict]:
    """
    ADRS: Compute TVA-modulated teacher reward shaping.

    Automatically selects the finest available TVA level:
        "L3" / step-level:       requires step_rewards, step_group_uids, step_boundary_mask
        "L2" / completion-level: requires token_level_rewards, prompt_uids
        "L1" / PAS:              requires ref_log_probs
        "none":                  fixed eta, no modulation

    Args:
        teacher_log_probs: (bs, response_length)
        response_mask: (bs, response_length)
        step_rewards: (num_flat_steps,) — for L3. From GiGPO discounted returns.
        step_group_uids: (num_flat_steps,) — for L3. From build_step_group.
        step_boundary_mask: (bs, response_length) — for L3. Integer step IDs.
        token_level_rewards: (bs, response_length) — for L2. Env rewards.
        prompt_uids: (bs,) — for L2. Prompt group IDs.
        ref_log_probs: (bs, response_length) — for L1. Reference policy logprobs.
        eta: base teacher reward coefficient.
        tva_temperature: temperature for teacher alignment sigmoid.
        tva_tau: temperature for TVA modulation sigmoid.
        baseline_mode: "step" or "global" for teacher reward centering.
        normalize: whether to normalize teacher reward to unit variance.
        tva_level: "auto", "L3", "L2", "L1", or "none".

    Returns:
        modulated_reward: (bs, response_length) — add this to token_level_rewards.
        tva_modulation: (bs, response_length) — per-token competence gate σ(τ·TVA),
            reusable as a distillation-loss gate (TVA-gated distillation); 0.5 where no TVA.
        metrics: dict with diagnostic statistics.
    """
    bs, seq_len = teacher_log_probs.shape
    device = teacher_log_probs.device

    # Auto-select TVA level based on available data
    if tva_level == "auto":
        if step_rewards is not None and step_group_uids is not None and step_boundary_mask is not None:
            tva_level = "L3"
        elif token_level_rewards is not None and prompt_uids is not None:
            tva_level = "L2"
        elif ref_log_probs is not None:
            tva_level = "L1"
        else:
            tva_level = "none"

    # Baseline mode: use "global" if no step boundary available
    effective_baseline_mode = baseline_mode
    if baseline_mode == "step" and step_boundary_mask is None:
        effective_baseline_mode = "global"

    # Step 1: Raw teacher reward
    raw_teacher_reward = compute_teacher_reward(
        teacher_log_probs, response_mask, step_boundary_mask,
        baseline_mode=effective_baseline_mode, normalize=normalize,
        normalize_mode=normalize_mode,
    )

    metrics = {"adrs/tva_level": {"none": 0, "L1": 1, "L2": 2, "L3": 3}[tva_level]}

    # ---- Level: none — fixed eta, no modulation ----
    if tva_level == "none":
        modulated_reward = eta * raw_teacher_reward
        tva_modulation = torch.full((bs, seq_len), 0.5, device=device)
        with torch.no_grad():
            valid = response_mask.bool()
            vr = modulated_reward[valid]
            metrics.update({
                "adrs/teacher_reward_mean": vr.mean().item() if vr.numel() > 0 else 0.0,
                "adrs/teacher_reward_std": vr.std().item() if vr.numel() > 1 else 0.0,
                "adrs/eta": eta,
            })
        return modulated_reward, tva_modulation, metrics

    # ---- Level 1: PAS modulation ----
    if tva_level == "L1":
        pas, pas_metrics = compute_pas(teacher_log_probs, ref_log_probs, response_mask)
        # PAS > 0 means teacher prefers this token more than ref → modulate up
        tva_modulation = torch.sigmoid(tva_tau * pas)
        modulated_reward = eta * tva_modulation * raw_teacher_reward

        with torch.no_grad():
            valid = response_mask.bool()
            vr = modulated_reward[valid]
            metrics.update({
                "adrs/teacher_reward_mean": vr.mean().item() if vr.numel() > 0 else 0.0,
                "adrs/teacher_reward_std": vr.std().item() if vr.numel() > 1 else 0.0,
                "adrs/eta": eta,
                "tva/modulation_mean": tva_modulation[valid].mean().item(),
            })
            metrics.update(pas_metrics)
        return modulated_reward, tva_modulation, metrics

    # ---- Level 2: Completion-level TVA modulation ----
    if tva_level == "L2":
        tva_per_sample, l2_metrics = compute_completion_level_tva(
            teacher_log_probs, response_mask, token_level_rewards,
            prompt_uids, temperature=tva_temperature,
        )
        # Gate input: optionally std-normalize so the sigmoid is responsive.
        # Raw TVA magnitude is often ~1e-3 → sigmoid(tau·tva)≈0.5 → gate inert
        # (50% of even a useless teacher leaks through). Dividing by std (NOT
        # mean-centered → sign preserved: useful teacher >0→amplify, harmful
        # <0→suppress) restores the intended behavior. Mirrors L1/PAS, which
        # already normalizes. Off by default to keep existing runs unchanged.
        gate_in = tva_per_sample
        if tva_gate_norm:
            g_std = gate_in.std()
            if g_std > 1e-8:
                gate_in = gate_in / (g_std + 1e-8)
        # Broadcast sample-level TVA to all tokens in each completion
        tva_modulation = torch.sigmoid(tva_tau * gate_in).unsqueeze(1).expand(bs, seq_len)
        modulated_reward = eta * tva_modulation * raw_teacher_reward

        with torch.no_grad():
            valid = response_mask.bool()
            vr = modulated_reward[valid]
            metrics.update({
                "adrs/teacher_reward_mean": vr.mean().item() if vr.numel() > 0 else 0.0,
                "adrs/teacher_reward_std": vr.std().item() if vr.numel() > 1 else 0.0,
                "adrs/eta": eta,
                "tva/modulation_mean": tva_modulation[valid].mean().item(),
            })
            metrics.update(l2_metrics)
        return modulated_reward, tva_modulation, metrics

    # ---- Level 3: Step-level TVA modulation ----
    step_alignment, flat_to_bs = compute_teacher_alignment_per_step(
        teacher_log_probs, response_mask, step_boundary_mask,
        temperature=tva_temperature,
    )
    num_flat_steps = len(flat_to_bs)

    if num_flat_steps == 0 or len(step_group_uids) == 0:
        modulated_reward = eta * raw_teacher_reward
        tva_modulation = torch.full((bs, seq_len), 0.5, device=device)
        metrics.update({"adrs/eta": eta, "tva/no_steps": 1.0})
        return modulated_reward, tva_modulation, metrics

    sr_flat = torch.zeros(num_flat_steps, device=device)
    uid_flat = np.empty(num_flat_steps, dtype=object)

    # In the flat multi-turn batch, each row IS one step, so step_rewards
    # and step_group_uids are indexed by batch row (b), not by 2D (b, sid).
    for flat_idx, (b, sid) in enumerate(flat_to_bs):
        if b < len(step_rewards):
            sr_flat[flat_idx] = step_rewards[b].float() if isinstance(step_rewards[b], torch.Tensor) else float(step_rewards[b])
        if b < len(step_group_uids):
            uid_flat[flat_idx] = step_group_uids[b]
        else:
            uid_flat[flat_idx] = f"fallback_{b}"

    tva = compute_tva_vectorized(step_alignment, sr_flat, uid_flat)

    # Gate input: optionally std-normalize (see L2 note) so the sigmoid responds
    # to small TVA magnitudes instead of staying pinned at 0.5.
    tva_gate = tva
    if tva_gate_norm:
        g_std = tva.std()
        if g_std > 1e-8:
            tva_gate = tva / (g_std + 1e-8)

    tva_modulation = torch.ones(bs, seq_len, device=device) * 0.5
    if len(flat_to_bs) > 0:
        tva_sigmoid = torch.sigmoid(tva_tau * tva_gate)
        flat_to_bs_t = torch.tensor(flat_to_bs, device=device, dtype=torch.long)
        for flat_idx in range(len(flat_to_bs)):
            b, sid = flat_to_bs_t[flat_idx, 0].item(), flat_to_bs_t[flat_idx, 1].item()
            step_mask = (step_boundary_mask[b] == sid) & response_mask[b].bool()
            if step_mask.any():
                tva_modulation[b, step_mask] = tva_sigmoid[flat_idx]

    modulated_reward = eta * tva_modulation * raw_teacher_reward

    with torch.no_grad():
        valid = response_mask.bool()
        vr = modulated_reward[valid]
        metrics.update({
            "adrs/teacher_reward_mean": vr.mean().item() if vr.numel() > 0 else 0.0,
            "adrs/teacher_reward_std": vr.std().item() if vr.numel() > 1 else 0.0,
            "adrs/teacher_reward_abs_max": vr.abs().max().item() if vr.numel() > 0 else 0.0,
            "adrs/eta": eta,
            "tva/mean": tva.mean().item() if tva.numel() > 0 else 0.0,
            "tva/std": tva.std().item() if tva.numel() > 1 else 0.0,
            "tva/positive_ratio": (tva > 0).float().mean().item() if tva.numel() > 0 else 0.0,
            "tva/modulation_mean": tva_modulation[valid].mean().item(),
            "tva/num_steps": float(num_flat_steps),
        })

    return modulated_reward, tva_modulation, metrics
