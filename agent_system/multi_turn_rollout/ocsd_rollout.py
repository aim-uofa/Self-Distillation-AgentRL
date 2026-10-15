"""OCSD-only trajectory metadata collection.

Keeping this subclass separate avoids changing the rollout records consumed by
GRPO, SDAR, and the other methods.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from verl import DataProto
from verl.trainer.ppo.ocsd_utils import extract_action_text

from .rollout_loop import TrajectoryCollector


def _is_active(step: dict) -> bool:
    active = step.get("active_masks", False)
    if hasattr(active, "item"):
        return bool(active.item())
    return bool(active)


class OCSDTrajectoryCollector(TrajectoryCollector):
    """Attach the next one/two observations needed by OCSD teacher replay."""

    def _annotate_future_evidence(self, trajectories: List[List[Dict]]) -> None:
        debug = os.environ.get("OCSD_DEBUG_ROLLOUT", "0") == "1"
        debug_count = 0
        for env_index, trajectory in enumerate(trajectories):
            for step_index, step in enumerate(trajectory):
                if not _is_active(step):
                    continue

                has_next = (
                    step_index + 1 < len(trajectory)
                    and _is_active(trajectory[step_index + 1])
                )
                has_second = (
                    step_index + 2 < len(trajectory)
                    and _is_active(trajectory[step_index + 2])
                )
                available = 2 if has_second else 1 if has_next else 0

                step["step_index"] = int(step_index)
                step["available_future_len"] = available
                step["future_obs_text_1"] = (
                    str(trajectory[step_index + 1].get("anchor_obs", "") or "")
                    if has_next
                    else ""
                )
                step["future_obs_text_2"] = (
                    str(trajectory[step_index + 2].get("anchor_obs", "") or "")
                    if has_second
                    else ""
                )

                next_response = (
                    trajectory[step_index + 1].get("responses") if has_next else None
                )
                if next_response is None:
                    step["future_action_text_1"] = ""
                else:
                    decoded = self.tokenizer.decode(
                        next_response, skip_special_tokens=False
                    )
                    step["future_action_text_1"] = extract_action_text(decoded)

                if debug and debug_count < 2:
                    print(
                        "\n=== OCSD rollout evidence ===\n"
                        f"env={env_index}, step={step_index}, available={available}\n"
                        f"next_obs={step['future_obs_text_1'][:120]!r}\n"
                        f"next_action={step['future_action_text_1'][:120]!r}\n"
                        f"second_obs={step['future_obs_text_2'][:120]!r}"
                    )
                    debug_count += 1

    def gather_rollout_data(
        self,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
    ) -> DataProto:
        self._annotate_future_evidence(total_batch_list)
        return super().gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success,
            traj_uid=traj_uid,
            tool_callings=tool_callings,
        )
