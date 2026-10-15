"""ADRS-only GRPO* advantage from the Apache-2.0 released ADRS core_algos.py.

https://github.com/gitrxh/ADRS-arxiv
Kept separate so existing methods retain the original GRPO estimator.
"""

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    compute_mean_std_cross_steps: bool = True,
    eta: float = 0.0,
    teacher_reward: torch.Tensor = None,
    wn_clip: float = 0.0,
    wn_sign: float = 1.0,
    wn_adaptive_clip: bool = False,
    wn_mode: str = "additive",
    student_log_probs: torch.Tensor = None,
    teacher_log_probs: torch.Tensor = None,
    wn_neg_reset: bool = False,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group.
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                id2std[idx] = torch.std(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        means = torch.stack([id2mean[index[i]] for i in range(bsz)]).to(scores.device)
        stds = torch.stack([id2std[index[i]] for i in range(bsz)]).to(scores.device)
        if norm_adv_by_std_in_grpo:
            scores = (scores - means) / (stds + epsilon)
        else:
            scores = scores - means
        scores = scores.unsqueeze(-1) * response_mask

        if eta > 0 and teacher_reward is not None:
            if wn_mode == "multiplicative" and student_log_probs is not None and teacher_log_probs is not None:
                # Multiplicative gate (RLSD-style): per-token teacher/student ratio rescaling.
                # When scores (env signal) are 0, multiplicative naturally produces 0
                # (no artificial nonzero signal like additive mode),
                # so wn_adaptive_clip is unnecessary. eta is analogous to RLSD mixing coeff.
                from verl.trainer.ppo.rlsd_utils import compute_rlsd_token_advantage
                scores = compute_rlsd_token_advantage(
                    seq_advantages=scores,
                    student_log_probs=student_log_probs,
                    teacher_log_probs=teacher_log_probs,
                    response_mask=response_mask,
                    rlsd_lambda=eta,
                    rlsd_clip_eps=wn_clip if wn_clip > 0 else 0.2,
                    neg_reset=wn_neg_reset,
                )
            else:
                id2indices = defaultdict(list)
                for i in range(bsz):
                    id2indices[index[i]].append(i)
                median_std = None
                if wn_adaptive_clip and wn_clip > 0:
                    valid_stds = [id2std[idx] for idx in id2indices if idx in id2std and id2std[idx].item() >= epsilon]
                    median_std = torch.stack(valid_stds).median() if valid_stds else torch.tensor(1.0)
                for idx, indices in id2indices.items():
                    if len(indices) <= 1:
                        continue
                    if id2std.get(idx, torch.tensor(1.0)).item() < epsilon:
                        continue
                    group_teacher = teacher_reward[indices]
                    group_mask = response_mask[indices]
                    group_scores = scores[indices]
                    if wn_neg_reset:
                        # Apply teacher signal only to positive-advantage samples;
                        # Use first valid token advantage to determine sign.
                        first_valid = group_mask.int().argmax(dim=-1)
                        sample_adv = group_scores[torch.arange(len(indices), device=group_scores.device), first_valid]
                        pos_mask = (sample_adv > 0).unsqueeze(-1)  # (group_size, 1)
                    raw_masked = group_teacher * group_mask
                    w_mean = verl_F.masked_mean(raw_masked, group_mask)
                    w_var = verl_F.masked_var(raw_masked, group_mask)
                    within_norm = (group_teacher - w_mean) * torch.rsqrt(w_var + 1e-8)
                    wn_contribution = wn_sign * eta * within_norm * group_mask
                    if wn_clip > 0:
                        effective_clip = wn_clip
                        if wn_adaptive_clip:
                            # When env reward discrimination is below batch median,
                            # reduce teacher clip to prevent teacher from dominating sparse env signals.
                            relative_strength = torch.clamp(id2std[idx] / (median_std + epsilon), max=1.0)
                            effective_clip = wn_clip * relative_strength
                        wn_contribution = wn_contribution.clamp(-effective_clip, effective_clip)
                    if wn_neg_reset:
                        wn_contribution = wn_contribution * pos_mask
                    scores[indices] = scores[indices] + wn_contribution

    return scores, scores
