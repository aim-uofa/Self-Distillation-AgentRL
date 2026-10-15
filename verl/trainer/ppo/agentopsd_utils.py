"""Copied from ZethWang/AgentOPSD (Apache-2.0), opsd_utils.py.

https://github.com/ZethWang/AgentOPSD

OPSD utilities.

OPSD ("Bayesian Value Recursion for Token-Level Credit Assignment") reuses the
same self-oracle teacher/student log-ratio that RLSD already computes, but
interprets it as a Bayesian sufficient statistic for the per-token success
belief instead of a policy-mixing weight.

Key insight: the per-token log-ratio

    delta_t = log pi_teacher(y_t | s, skill) - log pi_student(y_t | s)

is BOTH the one-step expert log-ratio AND the Bayesian sufficient statistic for
the success belief V_t = P(R = 1 | prefix up to token t). The belief recursion
lives in logit space:

    logit(V_t) = logit(V_{t-1}) + delta_t          # cumulative sum along tokens
    V_t        = sigmoid(logit(V_t))

The token-level advantage is the belief increment

    A_t = V_t - V_{t-1}   ( ~= V_t * (1 - V_t) * delta_t )

which telescopes: sum_t A_t = V_T - V_0. To make it a proper credit
decomposition of the sparse episode return R, we renormalize the per-trajectory
sum to equal the budget (R - V_0).

The SkillProvider (privileged-info loading) is shared with RLSD and imported
from ``rlsd_utils`` so it is not duplicated here.
"""

import numpy as np
import torch

# Re-export SkillProvider so callers can import it from either module.
from verl.trainer.ppo.rlsd_utils import SkillProvider  # noqa: F401


def _group_normalize(adv: torch.Tensor, response_mask: torch.Tensor, uid, token_level: bool = True) -> torch.Tensor:
    """Standardize (adv - mean) / (std + 1e-6) over valid token positions per uid."""
    from collections import defaultdict

    rmask = response_mask.to(adv.dtype)
    bsz = adv.shape[0]
    id2rows = defaultdict(list)
    for i in range(bsz):
        id2rows[uid[i]].append(i)

    out = adv.clone()
    for _, rows in id2rows.items():
        idx = torch.tensor(rows, dtype=torch.long, device=adv.device)
        vals = adv[idx]
        m = rmask[idx].bool()
        if m.sum() <= 1:
            continue
        sel = vals[m]
        mean = sel.mean()
        std = sel.std(unbiased=False)
        out[idx] = torch.where(m, (vals - mean) / (std + 1e-6), vals)
    return out * rmask


def _group_normalize_turns(adv_row: torch.Tensor, uid) -> torch.Tensor:
    """Standardize a (bs,) per-turn scalar advantage within each uid group."""
    from collections import defaultdict

    bsz = adv_row.shape[0]
    id2rows = defaultdict(list)
    for i in range(bsz):
        id2rows[uid[i]].append(i)

    out = adv_row.clone()
    for _, rows in id2rows.items():
        idx = torch.tensor(rows, dtype=torch.long, device=adv_row.device)
        if idx.numel() <= 1:
            continue
        sel = adv_row[idx]
        mean = sel.mean()
        std = sel.std(unbiased=False)
        out[idx] = (sel - mean) / (std + 1e-6)
    return out


def _discounted_cumsum(x: torch.Tensor, gamma: float, dim: int) -> torch.Tensor:
    """Leaky cumulative sum along ``dim``: out_t = gamma * out_{t-1} + x_t.

    Equivalent to out_t = sum_{j<=t} gamma^{t-j} x_j (geometric decay of the
    evidence by turn/token distance). gamma >= 1 reduces to a plain cumulative
    sum. Uses the O(T) recurrence rather than the gamma^{-j} closed form, which
    overflows for long horizons.
    """
    if gamma >= 1.0:
        return torch.cumsum(x, dim=dim)
    xm = x.movedim(dim, 0)  # (T, ...)
    out = torch.empty_like(xm)
    acc = torch.zeros_like(xm[0])
    for t in range(xm.shape[0]):
        acc = gamma * acc + xm[t]
        out[t] = acc
    return out.movedim(0, dim)


def compute_opsd_token_advantage(
    token_level_rewards: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    v0_prior: float = 0.5,
    v0_per_traj: torch.Tensor = None,
    eps: float = 1e-6,
    seq_advantage_per_row: torch.Tensor = None,
    uid=None,
    use_anchoring: bool = False,
    use_group_norm: bool = False,
    center_delta: bool = False,
    belief_decay_gamma: float = 1.0,
    tether_seq: bool = False,
    tether_lambda: float = 0.5,
    tether_band: float = 0.2,
    belief_mult: bool = False,
    mult_lambda: float = 1.0,
    mult_band: float = 0.2,
    belief_gate: bool = True,
    mult_signed: bool = True,
    signed: bool = False,
) -> torch.Tensor:
    """
    Compute token-level OPSD advantages via Bayesian belief recursion.

    Beyond the three paper stabilizers, three DIAGNOSTIC FIXES address the
    E[delta] = -KL(student||teacher) <= 0 systematic negative drift that makes
    the raw recursion saturate (V -> 0) over long sequences:

        (fix 1) center_delta: subtract each trajectory's own mean delta before
            the cumsum, removing the -KL drift so the log-odds walk is a
            martingale (zero-mean evidence). Cheapest, most on-target fix.
        (fix 2) belief_decay_gamma (gamma): decay accumulated evidence
            geometrically via logit(V_t) = logit(V_0) + cumsum_gamma(delta),
            i.e. c_t = gamma*c_{t-1} + delta_t. gamma < 1 counters over-counting
            from correlated (autoregressive) evidence; gamma = 1 is plain cumsum.
        (fix 3) tether_seq: RLSD-style magnitude anchoring. Bypass the recursion
            for MAGNITUDE and pin it to the trusted GRPO seq advantage A_seq:
                w_t   = clamp(exp(sign(A_seq) * delta_t), 1-band, 1+band)
                A_tok = A_seq * ((1 - lambda) + lambda * w_t)
            so |A_tok| stays within +-(lambda*band) of A_seq; delta only nudges.

    The paper stabilizers are INDEPENDENT toggles so the leave-one-out
    ablations are all expressible:

        full     : use_anchoring=True,  use_group_norm=True
        noanchor : anchoring off (sign comes from the belief increment itself),
                   group norm still on
        nonorm   : anchoring on, group norm off -> keep the (R - V0) telescoping
                   renormalization of the anchored magnitudes

    Baseline (all off) reproduces the raw telescoping recursion.

    Args:
        token_level_rewards: (bs, response_length) — sparse episode reward tensor.
            The per-trajectory return R is its sum over the response dimension.
        student_log_probs: (bs, response_length) — log pi_theta(y_t | x, y_<t).
        teacher_log_probs: (bs, response_length) — log pi_theta(y_t | x, r, y_<t).
        response_mask: (bs, response_length) — mask for valid response tokens.
        v0_prior: scalar prior success belief V_0 used when ``v0_per_traj`` is None.
        v0_per_traj: optional (bs,) tensor of per-trajectory priors V_0 (e.g. the
            GRPO group mean of R). Overrides ``v0_prior`` when provided.
        eps: numerical clamp bound; V is clamped to (eps, 1 - eps) before logit.
        seq_advantage_per_row: (bs,) GRPO sequence advantage per row. Required for
            direction anchoring (supplies the sign).
        uid: (bs,) GRPO group ids. Required for group normalization.
        use_anchoring: apply A_hat = sign(A_seq) * |A_raw| (GRPO sets the sign).
            When False, the belief increment keeps its own sign.
        use_group_norm: standardize (x - mean)/(std+eps) within each uid group,
            REPLACING the (R - V0) renormalization. When False, the anchored (or
            raw) advantage is renormalized by the telescoping budget (R - V0).

    Returns:
        token_advantages: (bs, response_length) — token-level advantage A_t,
            zero outside ``response_mask``, detached (no grad).
    """
    with torch.no_grad():
        response_mask = response_mask.to(student_log_probs.dtype)

        # Per-trajectory return R = sum of the sparse reward over the response dim.
        R = (token_level_rewards * response_mask).sum(dim=-1)  # (bs,)

        # Prior success belief V_0 per trajectory.
        if v0_per_traj is not None:
            V0 = v0_per_traj.to(student_log_probs.dtype).to(student_log_probs.device)
        else:
            V0 = torch.full_like(R, float(v0_prior))
        V0 = torch.clamp(V0, 1e-4, 1.0 - 1e-4)  # (bs,)

        # One-step expert log-ratio == Bayesian sufficient statistic.
        delta_t = (teacher_log_probs - student_log_probs) * response_mask  # (bs, L)

        # (fix 3) A_seq-tether (RLSD route): pin MAGNITUDE to the trusted GRPO seq
        # advantage; delta only supplies a bounded +-(lambda*band) nudge. Fully
        # bypasses the belief recursion, so it short-circuits everything below.
        # Placed BEFORE center_delta so the tether nudge uses the RAW evidence
        # (RLSD semantics), and mutually exclusive with it in practice.
        if tether_seq and seq_advantage_per_row is not None:
            a_seq = seq_advantage_per_row.to(delta_t.dtype).to(delta_t.device).unsqueeze(-1)  # (bs,1)
            sgn = torch.sign(a_seq)
            lo, hi = 1.0 - float(tether_band), 1.0 + float(tether_band)
            # Clamp the EXPONENT (not just its result) so unclipped large delta
            # cannot overflow exp() in fp16/bf16; the band bounds it either way.
            import math

            w_t = torch.clamp(torch.exp(torch.clamp(sgn * delta_t, math.log(lo), math.log(hi))), lo, hi)
            lam = float(tether_lambda)
            token_advantages = a_seq * ((1.0 - lam) + lam * w_t) * response_mask
            return token_advantages * response_mask

        # (fix 1) Delta centering: subtract each trajectory's own masked mean so
        # the accumulated evidence has zero drift (removes the -KL bias). This
        # de-biases the INPUT; scale is still restored by (R - V0) telescoping.
        if center_delta:
            n_valid = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (bs,1)
            mean_delta = (delta_t * response_mask).sum(dim=-1, keepdim=True) / n_valid
            delta_t = (delta_t - mean_delta) * response_mask

        # Belief recursion in logit space: logit(V_t) = logit(V_0) + cumsum_gamma(delta_t).
        # With belief_decay_gamma < 1 the cumulative sum decays geometrically by
        # token distance (recency-weighted belief); gamma = 1 is the plain cumsum.
        logit_V0 = torch.log(V0 / (1.0 - V0)).unsqueeze(-1)  # (bs, 1)
        cum_delta = _discounted_cumsum(delta_t, float(belief_decay_gamma), dim=-1)  # (bs, L)
        logit_Vt = logit_V0 + cum_delta  # (bs, L)
        V_t = torch.sigmoid(logit_Vt)  # (bs, L)
        V_t = torch.clamp(V_t, eps, 1.0 - eps)

        # V_{t-1}: shift right, with V_{-1} = V_0.
        V_prev = torch.cat([V0.unsqueeze(-1), V_t[:, :-1]], dim=-1)  # (bs, L)

        # Belief increment A_t = V_t - V_{t-1}; zero outside the response.
        A_t = (V_t - V_prev) * response_mask  # (bs, L)

        # (belief_mult route) RLSD-style MULTIPLICATIVE credit driven by the
        # accumulated-belief increment MAGNITUDE (direction discarded). Only the
        # size of each token's belief move matters: standardize |A_t| within the
        # trajectory to a bounded multiplier m_t in [1-band, 1+band], then scale
        # the trusted GRPO seq advantage by it. GRPO keeps the sign; delta only
        # re-weights magnitude. Inherits RLSD's safety (sign-preserving, bounded,
        # no renorm) but the multiplier comes from the belief RECURSION rather
        # than a local exp(delta). Bypasses anchoring / group-norm / telescoping.
        if belief_mult and seq_advantage_per_row is not None:
            a_seq = seq_advantage_per_row.to(A_t.dtype).to(A_t.device).unsqueeze(-1)  # (bs,1)
            # (ablation) belief-saturation gate: the default s_t = |V_t - V_{t-1}|
            # carries the implicit V(1-V) gate that discounts already-settled beliefs;
            # belief_gate=False strips it down to the raw evidence |delta_t|.
            n_valid = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (bs,1)
            b = float(mult_band)
            if signed:
                # (paper Eq.6) outcome-consistency: standardize the SIGNED per-token
                # increment and steer by sign(A^{(i)}) so turns whose belief moved with
                # the outcome are amplified, those that disagree are shrunk.
                q_t = A_t if belief_gate else (delta_t * response_mask)  # signed increment
                q_mean = (q_t * response_mask).sum(dim=-1, keepdim=True) / n_valid
                q_var = (((q_t - q_mean) ** 2) * response_mask).sum(dim=-1, keepdim=True) / n_valid
                z_t = (q_t - q_mean) / (torch.sqrt(q_var + eps) + eps)
                m_t = torch.clamp(1.0 + b * torch.sign(a_seq) * z_t, 1.0 - b, 1.0 + b)
            else:
                # (ablation) belief-saturation gate: the default s_t = |V_t - V_{t-1}|
                # carries the implicit V(1-V) gate that discounts already-settled beliefs;
                # belief_gate=False strips it down to the raw evidence |delta_t|.
                if belief_gate:
                    s_t = A_t.abs()  # (bs, L) belief increment magnitude (V(1-V)-gated)
                else:
                    s_t = delta_t.abs() * response_mask  # raw evidence, gate removed
                s_mean = (s_t * response_mask).sum(dim=-1, keepdim=True) / n_valid
                s_var = (((s_t - s_mean) ** 2) * response_mask).sum(dim=-1, keepdim=True) / n_valid
                s_std = torch.sqrt(s_var + eps)
                m_t = torch.clamp(1.0 + b * (s_t - s_mean) / (s_std + eps), 1.0 - b, 1.0 + b)
            lam = float(mult_lambda)
            # (ablation) signed direction: default keeps GRPO's sign; mult_signed=False
            # keeps only the magnitude (drops the verifier's direction).
            dir_seq = a_seq if mult_signed else a_seq.abs()
            token_advantages = dir_seq * ((1.0 - lam) + lam * m_t) * response_mask
            return token_advantages * response_mask

        # (2) Direction anchoring: recursion sets magnitude, GRPO sets the sign.
        if use_anchoring and seq_advantage_per_row is not None:
            sgn = torch.sign(seq_advantage_per_row.to(A_t.dtype).to(A_t.device)).unsqueeze(-1)  # (bs, 1)
            A_hat = sgn * A_t.abs() * response_mask  # (bs, L)
        else:
            A_hat = A_t

        # (3) Group normalization REPLACES the (R - V0) renormalization when on.
        if use_group_norm and uid is not None:
            token_advantages = _group_normalize(A_hat, response_mask, uid, token_level=True)
            return token_advantages * response_mask

        # Telescoping renormalization so that sum_t A_hat == (R - V_0).
        budget = (R - V0).unsqueeze(-1)  # (bs, 1)
        A_sum = A_hat.sum(dim=-1, keepdim=True)  # (bs, 1)
        # Guard the zero-sum case: where |sum| is ~0, fall back to zero advantage.
        safe = A_sum.abs() > eps
        scale = torch.where(safe, budget / (A_sum + (~safe) * 1.0), torch.zeros_like(A_sum))
        token_advantages = A_hat * scale
        token_advantages = token_advantages * response_mask

    return token_advantages


def compute_opsd_turn_advantage(
    token_level_rewards: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    traj_uid,
    turn_step,
    episode_rewards=None,
    uid=None,
    v0_prior: float = 0.5,
    v0_per_traj: torch.Tensor = None,
    eps: float = 1e-6,
    seq_advantage_per_row: torch.Tensor = None,
    use_anchoring: bool = False,
    use_group_norm: bool = False,
    center_delta: bool = False,
    belief_decay_gamma: float = 1.0,
    tether_seq: bool = False,
    tether_lambda: float = 0.5,
    tether_band: float = 0.2,
    belief_mult: bool = False,
    mult_lambda: float = 1.0,
    mult_band: float = 0.2,
    belief_gate: bool = True,
    mult_signed: bool = True,
    signed: bool = False,
) -> torch.Tensor:
    """
    Compute TURN-level (step-level) OPSD advantages via Bayesian belief recursion.

    Unlike the token-level variant (where the belief evolves within a single
    action's tokens), here the belief evolves ACROSS the turns of an episode.
    In this codebase each row of the batch is ONE turn (one env step); rows
    sharing a ``traj_uid`` are the same episode, ordered by ``turn_step``.

    Algorithm (per episode):
        1. Aggregate each turn's token log-ratios into a scalar turn-delta:
               delta_turn = sum_t (delta_t * response_mask)   # over response tokens
           where delta_t = teacher_log_probs - student_log_probs.
        2. Order the episode's turns by ``turn_step`` ascending.
        3. Logit-space recursion over turns:
               logit(V_k) = logit(V_0) + sum_{j<=k} delta_turn_j
               V_k        = sigmoid(...), clamped to (eps, 1 - eps)
           V_0 is the per-episode prior (GRPO group mean of R via ``uid``).
        4. Turn advantage A_turn_k = V_k - V_{k-1}, with V_{-1} = V_0.
           Telescopes to V_last - V_0 per episode.
        5. Renormalize per EPISODE so sum_k A_turn_k == (R_episode - V_0), with
           R_episode the episode's total return. Guard zero-sum episodes.
        6. Broadcast each turn's scalar A_turn_k back to ALL response tokens of
           that row (masked outside ``response_mask``).

    Args:
        token_level_rewards: (bs, response_length) sparse reward tensor. In this
            codebase the EpisodeRewardManager places the FULL episode return on
            the last response token of EVERY turn-row, so a per-row sum already
            equals R_episode; ``episode_rewards`` is used directly when provided.
        student_log_probs: (bs, response_length) log pi_theta(y_t | x, y_<t).
        teacher_log_probs: (bs, response_length) log pi_theta(y_t | x, r, y_<t).
        response_mask: (bs, response_length) mask for valid response tokens.
        traj_uid: (bs,) array of trajectory ids; rows sharing one are an episode.
        turn_step: (bs,) array of int turn indices within each episode.
        episode_rewards: optional (bs,) array/tensor of the full episode return
            broadcast to every turn-row. Preferred source for R_episode.
        uid: optional (bs,) array of GRPO prompt-group ids for the V_0 group mean.
        v0_prior: scalar prior V_0 used when ``v0_per_traj`` is None.
        v0_per_traj: optional (bs,) tensor of per-trajectory priors V_0. Overrides
            ``v0_prior`` when provided.
        eps: numerical clamp bound; V is clamped to (eps, 1 - eps).
        seq_advantage_per_row: optional (bs,) GRPO sequence advantage per row. When
            given, direction anchoring is applied per turn (magnitude from the
            belief recursion, sign from the turn's episode seq advantage) and the
            per-episode (R - V0) renormalization is REPLACED by group norm (via uid).

            carries that turn's scalar A_turn_k, zero outside ``response_mask``,
            detached (no grad). Matches the token-level function's contract.
    """
    from collections import defaultdict

    with torch.no_grad():
        dtype = student_log_probs.dtype
        device = student_log_probs.device
        bsz, resp_len = student_log_probs.shape
        rmask = response_mask.to(dtype)

        # (1) Scalar turn-delta per row: sum of masked token log-ratios.
        delta_t = (teacher_log_probs - student_log_probs) * rmask  # (bs, L)
        delta_turn = delta_t.sum(dim=-1)  # (bs,)

        # Per-episode prior V_0 (one per episode; identical across a traj's turns).
        if v0_per_traj is not None:
            V0 = v0_per_traj.to(dtype).to(device)
        else:
            V0 = torch.full((bsz,), float(v0_prior), dtype=dtype, device=device)
        V0 = torch.clamp(V0, 1e-4, 1.0 - 1e-4)  # (bs,)

        # R_episode per row: prefer the broadcast ``episode_rewards`` (the full
        # return sits on every turn-row); else fall back to a per-row sum, which
        # in this codebase also equals R_episode.
        if episode_rewards is not None:
            R_ep = torch.as_tensor(
                np.asarray(episode_rewards, dtype=np.float32), dtype=dtype, device=device
            )
        else:
            R_ep = (token_level_rewards * rmask).sum(dim=-1).to(dtype)  # (bs,)

        # (2) Group rows by traj_uid, ordered by turn_step ascending.
        traj_to_rows = defaultdict(list)
        for i in range(bsz):
            traj_to_rows[traj_uid[i]].append(i)

        A_row = torch.zeros(bsz, dtype=dtype, device=device)  # scalar adv per turn
        for tuid, rows in traj_to_rows.items():
            rows_sorted = sorted(rows, key=lambda r: int(turn_step[r]))
            idx = torch.tensor(rows_sorted, dtype=torch.long, device=device)

            # (3) Logit recursion over the ordered turn sequence.
            v0_ep = V0[idx[0]]  # one V_0 per episode
            logit_v0 = torch.log(v0_ep / (1.0 - v0_ep))
            d_seq = delta_turn[idx]  # (K,)

            # (fix 3) A_seq-tether (RLSD route) at turn granularity: pin each turn's
            # magnitude to its GRPO seq advantage; the (mean-token) turn delta only
            # supplies a bounded nudge. Bypasses the belief recursion for this episode.
            if tether_seq and seq_advantage_per_row is not None:
                a_seq_turn = seq_advantage_per_row[idx].to(dtype).to(device)  # (K,)
                nvalid_turn = rmask[idx].sum(dim=-1).clamp(min=1.0)  # (K,) tokens per turn
                d_mean = d_seq / nvalid_turn  # keep exp() bounded (mean, not sum)
                sgn = torch.sign(a_seq_turn)
                lo, hi = 1.0 - float(tether_band), 1.0 + float(tether_band)
                w = torch.clamp(torch.exp(sgn * d_mean), lo, hi)  # (K,)
                lam = float(tether_lambda)
                A_row[idx] = (a_seq_turn * ((1.0 - lam) + lam * w)).to(dtype)
                continue

            # (fix 1) Delta centering: remove the episode's average per-turn drift.
            if center_delta and d_seq.numel() > 1:
                d_seq = d_seq - d_seq.mean()

            # (fix 2) belief_decay_gamma < 1 decays evidence geometrically by turn
            # distance (recency-weighted belief): c_k = gamma*c_{k-1} + delta_k,
            # logit(V_k) = logit(V_0) + c_k; gamma = 1 is the plain cumsum.
            cum_d = _discounted_cumsum(d_seq, float(belief_decay_gamma), dim=0)  # (K,)
            logit_Vk = logit_v0 + cum_d  # (K,)
            V_k = torch.clamp(torch.sigmoid(logit_Vk), eps, 1.0 - eps)  # (K,)

            # (4) A_turn_k = V_k - V_{k-1}, with V_{-1} = V_0.
            V_prev = torch.cat([v0_ep.view(1), V_k[:-1]], dim=0)  # (K,)
            A_seq = V_k - V_prev  # (K,)

            # (belief_mult route) RLSD-style MULTIPLICATIVE credit driven by the
            # per-turn belief increment MAGNITUDE (direction discarded). z-score
            # |V_k - V_{k-1}| within the episode to a bounded multiplier m_k, then
            # scale the trusted GRPO turn seq advantage by it. GRPO keeps the sign;
            # the accumulated belief only re-weights magnitude. Bypasses anchoring /
            # group-norm / telescoping (mirrors the token-level belief_mult path).
            if belief_mult and seq_advantage_per_row is not None:
                a_seq_turn = seq_advantage_per_row[idx].to(dtype).to(device)  # (K,) GRPO adv
                b = float(mult_band)
                if signed:
                    # (paper Eq.6) outcome-consistency: standardize the SIGNED per-turn
                    # increment and steer by sign(A^{(i)}) so turns whose belief moved
                    # with the episode outcome are amplified, those that disagree shrunk.
                    q_k = A_seq if belief_gate else d_seq  # (K,) signed increment
                    if q_k.numel() > 1:
                        q_mean = q_k.mean()
                        q_std = q_k.std(unbiased=False)
                    else:
                        q_mean = q_k.mean()
                        q_std = torch.zeros_like(q_mean)
                    z_k = (q_k - q_mean) / (q_std + eps)
                    m_k = torch.clamp(1.0 + b * torch.sign(a_seq_turn) * z_k, 1.0 - b, 1.0 + b)
                else:
                    # (ablation) belief-saturation gate: the default s_k = |V_k - V_{k-1}|
                    # carries the implicit V(1-V) gate; belief_gate=False strips it down to
                    # the raw per-turn evidence |delta_turn|.
                    if belief_gate:
                        s_k = A_seq.abs()  # (K,) belief increment magnitude (V(1-V)-gated)
                    else:
                        s_k = d_seq.abs()  # (K,) raw evidence, gate removed
                    if s_k.numel() > 1:
                        s_mean = s_k.mean()
                        s_std = s_k.std(unbiased=False)
                    else:
                        s_mean = s_k.mean()
                        s_std = torch.zeros_like(s_mean)
                    m_k = torch.clamp(1.0 + b * (s_k - s_mean) / (s_std + eps), 1.0 - b, 1.0 + b)
                lam = float(mult_lambda)
                # (ablation) signed direction: default keeps GRPO's sign; mult_signed=False
                # keeps only the magnitude (drops the verifier's direction).
                dir_seq = a_seq_turn if mult_signed else a_seq_turn.abs()
                A_row[idx] = (dir_seq * ((1.0 - lam) + lam * m_k)).to(dtype)
                continue

            # (2) Direction anchoring: recursion sets magnitude, GRPO sets the sign.
            if use_anchoring and seq_advantage_per_row is not None:
                sgn = torch.sign(seq_advantage_per_row[idx].to(dtype).to(device))  # (K,)
                A_base = sgn * A_seq.abs()
            else:
                A_base = A_seq

            # (3)/(5) group norm (deferred to after the loop) REPLACES telescoping.
            if use_group_norm and uid is not None:
                A_row[idx] = A_base.to(dtype)
            else:
                # Telescoping renorm so sum_k A_turn_k == R_episode - V_0.
                budget = R_ep[idx[0]] - v0_ep
                A_sum = A_base.sum()
                if A_sum.abs() > eps:
                    A_base = A_base * (budget / A_sum)
                else:
                    A_base = torch.zeros_like(A_base)  # guard zero-sum episode
                A_row[idx] = A_base.to(dtype)

        # (3) Group normalization over turns within each uid (post-loop). Skipped
        # under tether_seq / belief_mult: both already pin magnitude to A_seq
        # (+-band multiplicatively), and re-standardizing would destroy that
        # contract (matches the token path, whose tether/mult early-return before
        # any group norm).
        if use_group_norm and uid is not None and not tether_seq and not belief_mult:
            A_row = _group_normalize_turns(A_row, uid)

        # (6) Broadcast each turn's scalar advantage to all its response tokens.
        turn_advantages = A_row.unsqueeze(-1) * rmask  # (bs, L)

    return turn_advantages


def compute_group_mean_v0(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    traj_index,
) -> torch.Tensor:
    """
    Compute the per-trajectory prior V_0 as the GRPO group mean of the return R.

    Mirrors GRPO's grouping in ``core_algos.compute_grpo_outcome_advantage``:
    group by prompt ``index`` (uid), deduplicating repeated steps within a
    trajectory via ``traj_index`` (traj_uid). V_0 is the standard group success
    rate S/G (the same group mean GRPO uses to center the advantage), clamped to
    (1e-4, 1 - 1e-4) so logit(V_0) stays finite for all-correct / all-wrong groups.

    Args:
        token_level_rewards: (bs, response_length) sparse reward tensor.
        response_mask: (bs, response_length) response mask.
        index: (bs,) array of prompt-group ids (uid).
        traj_index: (bs,) array of trajectory ids (traj_uid) for dedup.

    Returns:
        v0: (bs,) tensor of per-trajectory group-mean priors.
    """
    from collections import defaultdict

    with torch.no_grad():
        scores = (token_level_rewards * response_mask.to(token_level_rewards.dtype)).sum(dim=-1)  # (bs,)
        bsz = scores.shape[0]

        id2score = defaultdict(list)
        id2mean = {}
        seen_pairs = set()
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            t = torch.tensor(id2score[idx], dtype=torch.float32)
            id2mean[idx] = torch.mean(t)

        v0 = torch.tensor(
            [float(id2mean[index[i]]) for i in range(bsz)],
            dtype=scores.dtype,
            device=scores.device,
        )
        v0 = torch.clamp(v0, 1e-4, 1.0 - 1e-4)

    return v0
