"""Utilities for Outcome-Verified Comparative Self-Distillation (OVCSD).

The paper leaves transition canonicalisation and top-k tail aggregation in an
unreleased supplement.  This module keeps those choices deterministic and
local to OVCSD so existing methods are unaffected.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


_WHITESPACE_RE = re.compile(r"\s+")
_SCIENCEWORLD_SURFACE_LIST_RE = re.compile(
    r"^(?P<prefix>(?:Observation: )?\t?(?:a|an) (?P<surface>[a-z][a-z -]*)\. "
    r"On the (?P=surface) is: )(?P<items>[^\n]+?)(?P<ending>\.[ \t]*)$",
    re.MULTILINE,
)
_SCIENCEWORLD_LIST_ITEM_RE = re.compile(
    r"(?:a|an) [a-zA-Z0-9][a-zA-Z0-9 '-]*"
    r"(?: \(containing [a-zA-Z0-9][a-zA-Z0-9 ,'-]*\))?"
    r"(?:, which is (?:on|off))?"
    r"(?=, (?:a|an) |$)"
)


def canonicalize_scienceworld_comparison(value: str) -> str:
    """Compare known unordered surface lists; never normalize model inputs.

    Keep duplicate objects, contents, container ownership and on/off clauses.
    Unknown syntax (including numbered choices and nested containers) is left
    untouched. All other characters, including case/whitespace, remain exact.
    """
    def sort_surface_items(match):
        body = match.group("items")
        items = []
        position = 0
        while position < len(body):
            item = _SCIENCEWORLD_LIST_ITEM_RE.match(body, position)
            if item is None:
                return match.group(0)
            items.append(item.group(0))
            position = item.end()
            if position == len(body):
                break
            if not body.startswith(", ", position):
                return match.group(0)
            position += 2
            if position == len(body):
                return match.group(0)
        return match.group("prefix") + ", ".join(sorted(items)) + match.group("ending")

    return _SCIENCEWORLD_SURFACE_LIST_RE.sub(sort_surface_items, value)


def scienceworld_observations_equal(left, right) -> bool:
    """Fail closed except for the explicitly recognized list permutations."""
    return (
        isinstance(left, str) and isinstance(right, str)
        and canonicalize_scienceworld_comparison(left)
        == canonicalize_scienceworld_comparison(right)
    )


def same_prompt_tokens(left: Dict, right: Dict) -> bool:
    """Compare actual skill-free model inputs, excluding left padding."""
    def valid_tokens(row):
        prompt = row["prompts"]
        mask = row["attention_mask"][:prompt.shape[-1]].bool()
        return prompt[mask].detach().cpu()

    return torch.equal(valid_tokens(left), valid_tokens(right))


def canonicalize_text(value) -> str:
    """Return a stable, conservative text representation."""
    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", str(value)).strip().lower()


def state_fingerprint(value) -> str:
    """Hash an observation without placing long environment text in tree keys."""
    canonical = canonicalize_text(value)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PrefixCandidate:
    """A restorable student state and the failed branches it covers."""

    key: Tuple
    depth: int
    members: Tuple[int, ...]
    representative: int
    prefix_actions: Tuple[str, ...]
    state_fingerprint: str
    shared: bool


def build_prefix_candidates(
    trajectories: Sequence[Sequence[Dict]],
    decoded_actions: Sequence[Sequence[str]],
    *,
    strict_prompt: bool = False,
) -> List[PrefixCandidate]:
    """Build deepest-first state-aligned candidates from failed trajectories.

    A candidate at depth ``d`` is the state immediately before action ``d``.
    Only states with a following student action can form a comparative pair.
    """
    grouped: Dict[Tuple, List[int]] = {}
    payloads: Dict[Tuple, Tuple[int, Tuple[str, ...], str]] = {}

    for traj_idx, (trajectory, actions) in enumerate(zip(trajectories, decoded_actions)):
        usable = min(len(trajectory), len(actions))
        for depth in range(usable):
            prefix = tuple(canonicalize_text(action) for action in actions[:depth])
            fingerprint = state_fingerprint(trajectory[depth].get("anchor_obs"))
            key = (prefix, fingerprint)
            if strict_prompt:
                # No case/whitespace folding of histories, and no grouping of
                # different surface-list renderings seen by the students.
                prefix = tuple(actions[:depth])
                plain_text = trajectory[depth]["ovcsd_plain_obs_text"]
                scores = tuple(float(turn["scienceworld_score"]) for turn in trajectory[:depth])
                key = (prefix, fingerprint, plain_text, scores)
            grouped.setdefault(key, []).append(traj_idx)
            payloads[key] = (depth, tuple(actions[:depth]), fingerprint)

    candidates: List[PrefixCandidate] = []
    for key, members in grouped.items():
        depth, prefix_actions, fingerprint = payloads[key]
        unique_members = tuple(sorted(set(members)))
        candidates.append(
            PrefixCandidate(
                key=key,
                depth=depth,
                members=unique_members,
                representative=unique_members[0],
                prefix_actions=prefix_actions,
                state_fingerprint=fingerprint,
                shared=len(unique_members) >= 2,
            )
        )

    return sorted(
        candidates,
        key=lambda item: (item.depth, len(item.members), item.key),
        reverse=True,
    )


def select_deepest_candidate(
    candidates: Sequence[PrefixCandidate],
    uncovered: Iterable[int],
    attempted: Iterable[Tuple],
) -> PrefixCandidate | None:
    """Select the deepest untried shared node, then an individual fallback."""
    uncovered_set = set(uncovered)
    attempted_set = set(attempted)
    if not uncovered_set:
        return None

    shared_candidates = []
    for candidate in candidates:
        if candidate.key in attempted_set or not candidate.shared or candidate.depth == 0:
            continue
        active_members = tuple(member for member in candidate.members if member in uncovered_set)
        if active_members:
            shared_candidates.append((candidate, active_members))
    if shared_candidates:
        candidate, active_members = max(
            shared_candidates,
            key=lambda item: (item[0].depth, len(item[1]), item[0].key),
        )
        return PrefixCandidate(
            key=candidate.key,
            depth=candidate.depth,
            members=active_members,
            representative=active_members[0],
            prefix_actions=candidate.prefix_actions,
            state_fingerprint=candidate.state_fingerprint,
            shared=True,
        )

    # Paper fallback: when no non-trivial shared node exists, use a deep
    # restorable state from one individual failed trajectory.
    individual_candidates = []
    for member in sorted(uncovered_set):
        for candidate in candidates:
            if member not in candidate.members:
                continue
            individual_key = ("individual", member, candidate.key)
            if individual_key in attempted_set:
                continue
            individual_candidates.append((candidate, member, individual_key))
    if individual_candidates:
        candidate, member, individual_key = max(
            individual_candidates,
            key=lambda item: (item[0].depth, item[1]),
        )
        return PrefixCandidate(
            key=individual_key,
            depth=candidate.depth,
            members=(member,),
            representative=member,
            prefix_actions=candidate.prefix_actions,
            state_fingerprint=candidate.state_fingerprint,
            shared=False,
        )
    return None


def select_nearest_ancestor(
    failed: PrefixCandidate,
    candidates: Sequence[PrefixCandidate],
    uncovered: Iterable[int],
    attempted: Iterable[Tuple],
) -> PrefixCandidate | None:
    """Back off to the nearest untried state on the representative branch."""
    uncovered_set = set(uncovered)
    attempted_set = set(attempted)
    representative = failed.representative
    eligible = []
    for candidate in candidates:
        if candidate.key in attempted_set or candidate.depth >= failed.depth:
            continue
        if representative not in candidate.members:
            continue
        active_members = tuple(member for member in candidate.members if member in uncovered_set)
        if not active_members:
            continue
        eligible.append((candidate, active_members))
    if not eligible:
        return None
    candidate, active_members = max(
        eligible,
        key=lambda item: (item[0].depth, len(item[1]), item[0].key),
    )
    return PrefixCandidate(
        key=candidate.key,
        depth=candidate.depth,
        members=active_members,
        representative=representative,
        prefix_actions=candidate.prefix_actions,
        state_fingerprint=candidate.state_fingerprint,
        shared=candidate.shared,
    )


def find_first_aligned_divergence(
    student_trajectory: Sequence[Dict],
    student_actions: Sequence[str],
    teacher_trajectory: Sequence[Dict],
    teacher_actions: Sequence[str],
    student_start_depth: int,
    *,
    strict_prompt: bool = False,
) -> Tuple[int, int] | None:
    """Return ``(student_turn, teacher_turn)`` for the first aligned action difference."""
    student_remaining = len(student_actions) - student_start_depth
    limit = min(student_remaining, len(teacher_actions), len(teacher_trajectory))
    for teacher_turn in range(limit):
        student_turn = student_start_depth + teacher_turn
        if student_turn >= len(student_trajectory):
            return None
        if strict_prompt:
            student_row = student_trajectory[student_turn]
            teacher_row = teacher_trajectory[teacher_turn]
            if (
                not teacher_row.get("ovcsd_prompt_aligned", False)
                or not isinstance(student_row.get("ovcsd_plain_obs_text"), str)
                or student_row["ovcsd_plain_obs_text"] != teacher_row.get("ovcsd_plain_obs_text")
                or not scienceworld_observations_equal(
                    student_row.get("anchor_obs"), teacher_row.get("anchor_obs")
                )
            ):
                return None
            if teacher_turn > 0 and (
                float(student_trajectory[student_turn - 1]["scienceworld_score"])
                != float(teacher_trajectory[teacher_turn - 1]["scienceworld_score"])
            ):
                return None
        student_state = state_fingerprint(student_trajectory[student_turn].get("anchor_obs"))
        teacher_state = state_fingerprint(teacher_trajectory[teacher_turn].get("anchor_obs"))
        if not strict_prompt and student_state != teacher_state:
            return None
        if canonicalize_text(student_actions[student_turn]) != canonicalize_text(
            teacher_actions[teacher_turn]
        ):
            return student_turn, teacher_turn
        if strict_prompt and student_actions[student_turn] != teacher_actions[teacher_turn]:
            # Same parsed action but different literal history: do not pretend
            # subsequent prompts are identical.
            return None
    return None


def local_comparative_advantages(member_count: int) -> Tuple[float, float]:
    """Return the balanced teacher and per-student local advantages."""
    if member_count <= 0:
        raise ValueError("member_count must be positive")
    root = float(member_count) ** 0.5
    return root, -1.0 / root


def compute_topk_tail_forward_kl(
    teacher_topk_log_probs: torch.Tensor,
    teacher_tail_log_prob: torch.Tensor,
    student_topk_log_probs: torch.Tensor,
    student_tail_log_prob: torch.Tensor,
    suffix_mask: torch.Tensor,
    suffix_weights: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute forward KL on teacher top-k support plus one tail bucket."""
    q_log = teacher_topk_log_probs.detach().float()
    q_tail_log = teacher_tail_log_prob.detach().float()
    p_log = student_topk_log_probs.float()
    p_tail_log = student_tail_log_prob.float()

    q = q_log.exp()
    q_tail = q_tail_log.exp()
    token_kl = (q * (q_log - p_log)).sum(dim=-1)
    token_kl = token_kl + q_tail * (q_tail_log - p_tail_log)

    mask = suffix_mask.to(dtype=token_kl.dtype)
    if suffix_weights is None:
        weights = mask
        denominator = weights.sum()
        if denominator.detach().item() <= 0:
            loss = student_topk_log_probs.sum() * 0.0
            return loss, {"ovcsd/suffix_kl": 0.0, "ovcsd/suffix_tokens": 0.0}
        loss = (token_kl * weights).sum() / denominator
    else:
        # These weights are normalized over each complete intervention pair.
        # Keep their absolute scale across PPO micro-batches; the actor's
        # normal gradient accumulation supplies the mini-batch normalization.
        weights = suffix_weights.to(dtype=token_kl.dtype) * mask
        if weights.sum().detach().item() <= 0:
            loss = student_topk_log_probs.sum() * 0.0
            return loss, {"ovcsd/suffix_kl": 0.0, "ovcsd/suffix_tokens": 0.0}
        loss = (token_kl * weights).sum()

    valid_kl = token_kl[mask.bool()]
    metrics = {
        "ovcsd/suffix_kl": float(valid_kl.mean().detach().item()),
        "ovcsd/suffix_kl_objective": float(loss.detach().item()),
        "ovcsd/suffix_tokens": float(mask.sum().detach().item()),
        "ovcsd/suffix_kl_max": float(valid_kl.max().detach().item()),
    }
    return loss.to(student_topk_log_probs.dtype), metrics
