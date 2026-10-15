"""Privileged rollout helpers used by OVCSD."""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

from .rollout_loop import TrajectoryCollector


def _select_values(values, indices: Sequence[int]):
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        return values[np.asarray(indices)]
    return [values[index] for index in indices]


def _select_observations(obs: Dict, indices: Sequence[int]) -> Dict:
    return {key: _select_values(value, indices) for key, value in obs.items()}


def _assign_observations(target: Dict, indices: Sequence[int], source: Dict) -> None:
    for key, source_values in source.items():
        if source_values is None:
            continue
        target_values = target[key]
        for local_idx, env_idx in enumerate(indices):
            target_values[env_idx] = source_values[local_idx]


class IndexedEnvironmentView:
    """Expose selected physical environment slots as a normal environment manager."""

    def __init__(
        self,
        manager,
        indices: Sequence[int],
        initial_obs: Dict,
        initial_infos: Sequence[Dict],
        prefix_lengths: Optional[Sequence[int]] = None,
        max_total_steps: Optional[int] = None,
    ):
        self.manager = manager
        self.indices = list(indices)
        self.initial_obs = initial_obs
        self.initial_infos = list(initial_infos)
        self.total_steps = np.asarray(
            prefix_lengths if prefix_lengths is not None else [0] * len(self.indices),
            dtype=np.int64,
        )
        self.max_total_steps = max_total_steps
        self.done = np.zeros(len(self.indices), dtype=bool)

    def reset(self, kwargs=None):
        return self.initial_obs, self.initial_infos

    def step(self, text_actions: List[str]):
        obs, rewards, dones, infos = self.manager.step_at(self.indices, text_actions)
        dones = np.asarray(dones, dtype=bool)
        active = np.logical_not(self.done)
        self.total_steps[active] += 1
        if self.max_total_steps is not None:
            dones = np.logical_or(dones, self.total_steps >= self.max_total_steps)
        dones = np.logical_or(dones, self.done)
        self.done = np.logical_or(self.done, dones)
        return obs, rewards, dones, infos

    def success_evaluator(self, *args, **kwargs):
        return self.manager.success_evaluator(*args, **kwargs)


class SkillTeacherTrajectoryCollector(TrajectoryCollector):
    """Generate trajectories with existing privileged skill prompts."""

    def __init__(self, *args, skill_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_provider = skill_provider

    def preprocess_batch(self, gen_batch: DataProto, obs: Dict) -> DataProto:
        obs_with_skill = dict(obs)
        obs_texts = obs.get("text")
        if obs_texts is not None:
            obs_with_skill["text"] = list(obs_texts)
            gamefiles = obs.get("gamefile")
            data_sources = gen_batch.non_tensor_batch.get("data_source")
            for index, prompt_text in enumerate(obs_with_skill["text"]):
                gamefile = None if gamefiles is None else gamefiles[index]
                if gamefile:
                    skill_text = self.skill_provider.get_privileged_info(str(gamefile))
                else:
                    data_source = "unknown" if data_sources is None else str(data_sources[index])
                    skill_text = self.skill_provider.get_privileged_info_from_data_source(
                        data_source,
                        prompt_text,
                    )
                if skill_text:
                    obs_with_skill["text"][index] = (
                        f"[Privileged Skill Information]\n{skill_text}\n\n{prompt_text}"
                    )
        return super().preprocess_batch(gen_batch, obs_with_skill)


def build_no_skill_prefix_turn(
    teacher_turn: Dict,
    prompt_row: Dict,
    reward,
    info: Dict,
) -> Dict:
    """Pair a skilled teacher response with the ordinary student prompt."""
    rebuilt = dict(teacher_turn)
    rebuilt.pop("rollout_log_probs", None)
    rebuilt.pop("raw_prompt_ids", None)

    prompt_ids = prompt_row["input_ids"]
    prompt_attention_mask = prompt_row["attention_mask"]
    response_ids = teacher_turn["responses"].to(prompt_ids.device)
    response_length = response_ids.shape[-1]
    response_attention_mask = teacher_turn["attention_mask"][
        -response_length:
    ].to(
        device=prompt_attention_mask.device,
        dtype=prompt_attention_mask.dtype,
    )
    full_attention_mask = torch.cat(
        [prompt_attention_mask, response_attention_mask],
        dim=-1,
    )

    rebuilt.update(
        {
            "prompts": prompt_ids,
            "responses": response_ids,
            "input_ids": torch.cat([prompt_ids, response_ids], dim=-1),
            "attention_mask": full_attention_mask,
            "position_ids": compute_position_id_with_mask(
                full_attention_mask.unsqueeze(0)
            )[0],
            "rewards": reward,
            "is_action_valid": info.get("is_action_valid", True),
            "active_masks": True,
        }
    )
    for key in (
        "anchor_obs",
        "index",
        "data_source",
        "gamefile",
        "raw_prompt",
    ):
        if key in prompt_row:
            rebuilt[key] = prompt_row[key]
    return rebuilt
