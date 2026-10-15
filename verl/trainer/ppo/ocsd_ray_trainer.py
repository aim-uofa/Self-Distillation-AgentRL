"""Ray trainer adapter for Observation-Calibrated Self-Distillation (OCSD)."""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ocsd_utils import (
    ASSISTANT_MARKERS,
    build_future_block,
    compute_ocsd_advantages,
    insert_future_evidence_before_response,
    normalize_advantages_shape,
    select_uncertain_steps,
)
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer
from verl.utils.model import compute_position_id_with_mask


def build_ocsd_teacher_batch(
    batch: DataProto,
    *,
    full: bool,
    env_name: str,
    tokenizer,
    max_prompt_length: int,
) -> tuple[DataProto, float]:
    """Create a Full/Ablated replay batch that scores the student response."""
    batch_size = batch.batch["input_ids"].shape[0]
    response_length = batch.batch["responses"].shape[1]
    future_obs_1 = batch.non_tensor_batch.get(
        "future_obs_text_1", np.array([""] * batch_size, dtype=object)
    )
    future_action_1 = batch.non_tensor_batch.get(
        "future_action_text_1", np.array([""] * batch_size, dtype=object)
    )
    future_obs_2 = batch.non_tensor_batch.get(
        "future_obs_text_2", np.array([""] * batch_size, dtype=object)
    )
    available = batch.non_tensor_batch.get(
        "available_future_len", np.zeros(batch_size, dtype=np.int64)
    )

    input_ids = []
    attention_masks = []
    position_ids = []
    truncated = 0
    debug = os.environ.get("OCSD_DEBUG_TEACHER", "0") == "1"

    for index in range(batch_size):
        original_ids = batch.batch["input_ids"][index]
        original_mask = batch.batch["attention_mask"][index]
        prompt_length = original_ids.shape[0] - response_length
        prompt_ids = original_ids[:prompt_length]
        prompt_mask = original_mask[:prompt_length].bool()
        prompt_text = tokenizer.decode(
            prompt_ids[prompt_mask], skip_special_tokens=False
        )

        future_length = int(available[index])
        if future_length > 0:
            evidence = build_future_block(
                available_future_len=future_length,
                obs_1=str(future_obs_1[index] or ""),
                action_1=str(future_action_1[index] or ""),
                obs_2=str(future_obs_2[index] or ""),
                env_name=env_name,
                full=full,
            )
            prompt_text = insert_future_evidence_before_response(
                prompt_text, evidence
            )

        encoded = tokenizer.encode(prompt_text, add_special_tokens=False)
        was_truncated = len(encoded) > max_prompt_length
        if was_truncated:
            truncated += 1
            encoded = encoded[-max_prompt_length:]
            tail = tokenizer.decode(encoded, skip_special_tokens=False)
            if not any(marker in tail for marker in ASSISTANT_MARKERS):
                warnings.warn(
                    "OCSD teacher prompt lost its assistant marker after left "
                    "truncation; consider increasing data.max_prompt_length.",
                    stacklevel=2,
                )

        actual_length = len(encoded)
        pad_length = max_prompt_length - actual_length
        teacher_prompt_ids = torch.tensor(encoded, dtype=torch.long)
        if pad_length:
            teacher_prompt_ids = torch.cat(
                [
                    torch.full(
                        (pad_length,), tokenizer.pad_token_id, dtype=torch.long
                    ),
                    teacher_prompt_ids,
                ]
            )
        teacher_prompt_mask = torch.cat(
            [
                torch.zeros(pad_length, dtype=torch.long),
                torch.ones(actual_length, dtype=torch.long),
            ]
        )

        response_ids = batch.batch["responses"][index].detach().cpu()
        response_mask = batch.batch["response_mask"][index].detach().cpu().long()
        valid_length = int(response_mask.sum().item())
        valid_response = response_ids[:valid_length]
        response_pad_length = response_length - valid_length
        if response_pad_length:
            response_ids = torch.cat(
                [
                    valid_response,
                    torch.full(
                        (response_pad_length,),
                        tokenizer.pad_token_id,
                        dtype=torch.long,
                    ),
                ]
            )
            response_mask = torch.cat(
                [
                    torch.ones(valid_length, dtype=torch.long),
                    torch.zeros(response_pad_length, dtype=torch.long),
                ]
            )
        else:
            response_ids = valid_response

        full_ids = torch.cat([teacher_prompt_ids, response_ids])
        full_mask = torch.cat([teacher_prompt_mask, response_mask])
        input_ids.append(full_ids)
        attention_masks.append(full_mask)
        position_ids.append(compute_position_id_with_mask(full_mask[None])[0])

        if debug and index < 1:
            label = "Full" if full else "Ablated"
            print(
                f"\n=== OCSD {label} teacher ===\n"
                f"available_future_len={future_length}, truncated={was_truncated}\n"
                f"prompt_tail={prompt_text[-500:]!r}"
            )

    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "position_ids": torch.stack(position_ids),
            "responses": batch.batch["responses"].detach().cpu(),
        },
        meta_info=batch.meta_info.copy(),
    )
    return teacher_batch, truncated / max(batch_size, 1)


class OCSDRayTrainer(SkillSDRayTrainer):
    """Reuse the stable GRPO loop and replace only its computed advantages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.config.algorithm.get("oc_opsd", {})
        self.ocsd_beta = float(config.get("beta", 0.5))
        self.ocsd_advantage_epsilon = float(config.get("eps_A", 1e-6))
        self.ocsd_future_horizon = int(config.get("K_future", 2))
        self.ocsd_top_step_ratio = float(config.get("top_step_ratio", 0.2))
        self.ocsd_env_name = str(config.get("env_name", "alfworld"))
        self.ocsd_max_prompt_length = int(self.config.data.max_prompt_length)

    def _requires_teacher_forward(self) -> bool:
        # OCSD has its own paired Full/Ablated replay after GRPO advantages exist.
        return False

    def _compute_full_ablated_log_probs(
        self, batch: DataProto, selected: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        batch_size, response_length = batch.batch["response_mask"].shape
        device = batch.batch["advantages"].device
        full_log_probs = torch.zeros(batch_size, response_length, device=device)
        ablated_log_probs = torch.zeros_like(full_log_probs)
        selected_indices = np.flatnonzero(selected)
        if len(selected_indices) == 0:
            return full_log_probs, ablated_log_probs, 0.0

        selected_batch = batch.select_idxs(selected_indices)
        full_batch, truncated_ratio = build_ocsd_teacher_batch(
            selected_batch,
            full=True,
            env_name=self.ocsd_env_name,
            tokenizer=self.tokenizer,
            max_prompt_length=self.ocsd_max_prompt_length,
        )
        ablated_batch, _ = build_ocsd_teacher_batch(
            selected_batch,
            full=False,
            env_name=self.ocsd_env_name,
            tokenizer=self.tokenizer,
            max_prompt_length=self.ocsd_max_prompt_length,
        )
        combined = DataProto.from_dict(
            tensors={
                key: torch.cat([full_batch.batch[key], ablated_batch.batch[key]])
                for key in ("input_ids", "attention_mask", "position_ids", "responses")
            },
            meta_info=batch.meta_info.copy(),
        )
        padded, pad_size = pad_dataproto_to_divisor(
            combined, self.actor_rollout_wg.world_size
        )
        scored = self.actor_rollout_wg.compute_log_prob(padded)
        scored = unpad_dataproto(scored, pad_size=pad_size)
        paired_log_probs = scored.batch["old_log_probs"].detach().to(device)

        selected_count = len(selected_indices)
        selected_mask = selected_batch.batch["response_mask"].float().to(device)
        index_tensor = torch.as_tensor(selected_indices, dtype=torch.long, device=device)
        full_log_probs[index_tensor] = paired_log_probs[:selected_count] * selected_mask
        ablated_log_probs[index_tensor] = (
            paired_log_probs[selected_count:] * selected_mask
        )
        return full_log_probs, ablated_log_probs, truncated_ratio

    def _prepare_method_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        if self.ocsd_beta == 0.0:
            return batch, {"oc_opsd/enabled": 0.0}

        batch_size = batch.batch["response_mask"].shape[0]
        if "available_future_len" not in batch.non_tensor_batch:
            raise KeyError(
                "OCSD rollout metadata is missing. Use OCSDTrajectoryCollector "
                "from verl.trainer.main_ocsd."
            )
        available = np.asarray(
            batch.non_tensor_batch["available_future_len"], dtype=np.int64
        )
        available = np.minimum(available, self.ocsd_future_horizon)
        batch.non_tensor_batch["available_future_len"] = available

        _, sequence_advantages = normalize_advantages_shape(
            batch.batch["advantages"], batch.batch["response_mask"]
        )
        trajectory_ids = batch.non_tensor_batch.get(
            "traj_uid", np.arange(batch_size, dtype=object)
        )
        sequence_advantages_np = sequence_advantages.detach().cpu().numpy()
        selected = select_uncertain_steps(
            old_log_probs=batch.batch["old_log_probs"],
            response_mask=batch.batch["response_mask"],
            available_future_len=available,
            sequence_advantages=sequence_advantages_np,
            trajectory_ids=np.asarray(trajectory_ids),
            top_step_ratio=self.ocsd_top_step_ratio,
            advantage_epsilon=self.ocsd_advantage_epsilon,
        )
        eligible = (available > 0) & (
            np.abs(sequence_advantages_np) > self.ocsd_advantage_epsilon
        )
        full_log_probs, ablated_log_probs, truncated_ratio = (
            self._compute_full_ablated_log_probs(batch, selected)
        )
        calibrated, metrics = compute_ocsd_advantages(
            advantages=batch.batch["advantages"],
            full_log_probs=full_log_probs,
            ablated_log_probs=ablated_log_probs,
            response_mask=batch.batch["response_mask"],
            available_future_len=torch.as_tensor(
                available,
                device=batch.batch["advantages"].device,
            ),
            beta=self.ocsd_beta,
            advantage_epsilon=self.ocsd_advantage_epsilon,
        )
        batch.batch["advantages"] = calibrated
        metrics.update(
            {
                "oc_opsd/enabled": 1.0,
                "oc_opsd/nll_eligible_step_ratio": float(eligible.mean()),
                "oc_opsd/nll_selected_step_ratio": float(selected.mean()),
                "oc_opsd/nll_selected_step_count": float(selected.sum()),
                "oc_opsd/future_block_truncated_ratio": truncated_ratio,
            }
        )
        return batch, metrics
