"""Trainer integration for Outcome-Verified Comparative Self-Distillation."""

from __future__ import annotations

import uuid

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer


_MODEL_KEYS = ("responses", "input_ids", "attention_mask", "position_ids")


def _action_mask(batch: DataProto) -> torch.Tensor:
    """Use VERL's multi-turn loss mask when present, otherwise response mask."""
    response_length = batch.batch["responses"].shape[-1]
    if "loss_mask" in batch.batch:
        return batch.batch["loss_mask"][:, -response_length:]
    if "response_mask" in batch.batch:
        return batch.batch["response_mask"]
    return batch.batch["attention_mask"][:, -response_length:]


class OVCSDRayTrainer(SkillSDRayTrainer):
    """Reuse the stable SkillSD loop while injecting only verified OVCSD rows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ovcsd_cfg = self.config.algorithm.get("ovcsd", {})
        self.ovcsd_top_k = int(ovcsd_cfg.get("suffix_top_k", 32))
        if self.ovcsd_top_k <= 0:
            raise ValueError("algorithm.ovcsd.suffix_top_k must be positive")
        if float(ovcsd_cfg.get("suffix_kl_coef", 0.1)) < 0:
            raise ValueError("algorithm.ovcsd.suffix_kl_coef must be non-negative")
        actor_cfg = self.config.actor_rollout_ref.actor
        if actor_cfg.get("use_dynamic_bsz", False):
            raise ValueError("OVCSD currently requires actor.use_dynamic_bsz=false")
        if actor_cfg.get("use_fused_kernels", False):
            raise ValueError("OVCSD currently requires actor.use_fused_kernels=false")
        if int(actor_cfg.get("ulysses_sequence_parallel_size", 1)) != 1:
            raise ValueError("OVCSD currently requires ulysses_sequence_parallel_size=1")
        # SkillSD's auxiliary loss is disabled on the OVCSD route; keeping this at
        # zero also makes the inherited diagnostic unambiguous.
        self.sdl_lambda = 0.0

    def _requires_teacher_forward(self) -> bool:
        # OVCSD computes a distribution-level teacher target only for verified
        # suffix rows, rather than chosen-token scores for the entire batch.
        return False

    def _reporting_batch(self, batch: DataProto) -> DataProto:
        if "ovcsd_is_extra" not in batch.batch:
            return batch
        keep = (~batch.batch["ovcsd_is_extra"].bool()).cpu().numpy()
        return batch.select_idxs(keep)

    def _prepare_method_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        payload = self.traj_collector.pop_pending_interventions()
        metrics = payload["metrics"]
        records = payload["records"]
        failed_traj_uids = set(str(value) for value in payload["failed_traj_uids"])

        ordinary_mask = _action_mask(batch).clone()
        pg_mask = ordinary_mask.clone()

        traj_uids = batch.non_tensor_batch["traj_uid"]
        turn_steps = batch.non_tensor_batch["turn_step"]
        failed_rows = np.asarray(
            [str(value) in failed_traj_uids for value in traj_uids],
            dtype=bool,
        )
        if failed_rows.any():
            failed_tensor = torch.as_tensor(failed_rows, device=pg_mask.device)
            pg_mask[failed_tensor] = 0
            batch.batch["advantages"][failed_tensor] = 0
            batch.batch["returns"][failed_tensor] = 0

        row_lookup = {}
        for index, (traj_uid, turn_step) in enumerate(zip(traj_uids, turn_steps)):
            row_lookup.setdefault((str(traj_uid), int(turn_step)), []).append(index)
        student_records = [record for record in records if record["kind"] == "student_divergence"]
        matched_student_rows = 0
        for record in student_records:
            row_indices = row_lookup.get((record["traj_uid"], int(record["turn_step"])), [])
            if not row_indices:
                metrics["ovcsd/missing_student_divergence_rows"] = (
                    metrics.get("ovcsd/missing_student_divergence_rows", 0.0) + 1.0
                )
                continue
            for row_idx in row_indices:
                active = ordinary_mask[row_idx]
                pg_mask[row_idx] = active
                advantage = float(record["advantage"])
                batch.batch["advantages"][row_idx] = advantage * active
                batch.batch["returns"][row_idx] = advantage * active
                matched_student_rows += 1

        batch.batch["ovcsd_pg_mask"] = pg_mask
        batch.batch["ovcsd_regularization_mask"] = ordinary_mask
        batch.batch["ovcsd_suffix_mask"] = torch.zeros_like(pg_mask)
        batch.batch["ovcsd_suffix_weights"] = torch.zeros_like(pg_mask, dtype=torch.float32)
        batch.batch["ovcsd_is_extra"] = torch.zeros(
            len(batch), dtype=torch.bool, device=pg_mask.device
        )
        metrics["ovcsd/matched_student_divergence_rows"] = float(matched_student_rows)

        extra_records = [
            record
            for record in records
            if record["kind"] in ("teacher_divergence", "suffix")
        ]
        if extra_records:
            batch, extra_metrics = self._append_verified_rows(batch, extra_records)
            metrics.update(extra_metrics)
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics, logging_prefix="ovcsd_seqlen")
        else:
            self._attach_empty_teacher_targets(batch)

        batch.meta_info["global_token_num"] = torch.sum(
            batch.batch["attention_mask"], dim=-1
        ).tolist()
        return batch, metrics

    def _append_verified_rows(self, batch: DataProto, records):
        world_size = int(self.actor_rollout_wg.world_size)
        real_count = len(records)
        pad_count = (-real_count) % world_size
        padded_records = list(records)
        for _ in range(pad_count):
            padded_records.append(
                {
                    **records[0],
                    "kind": "padding",
                    "advantage": 0.0,
                    "pair_id": f"padding-{uuid.uuid4()}",
                }
            )

        no_skill_proto = self._model_proto(
            [record["no_skill_row"] for record in padded_records]
        )
        privileged_proto = self._model_proto(
            [record["privileged_row"] for record in padded_records]
        )

        old_output = self.actor_rollout_wg.compute_log_prob(no_skill_proto)
        privileged_proto.meta_info["ovcsd_top_k"] = self.ovcsd_top_k
        teacher_output = self.actor_rollout_wg.compute_ovcsd_topk_log_prob(privileged_proto)

        extra = batch.select_idxs(np.zeros(len(padded_records), dtype=np.int64))
        for key in _MODEL_KEYS:
            extra.batch[key] = no_skill_proto.batch[key]
        if "prompts" in extra.batch:
            extra.batch["prompts"] = torch.stack(
                [record["no_skill_row"]["prompts"] for record in padded_records]
            )
        if "loss_mask" in extra.batch:
            extra.batch["loss_mask"] = torch.stack(
                [
                    record["no_skill_row"].get(
                        "loss_mask", record["no_skill_row"]["attention_mask"]
                    )
                    for record in padded_records
                ]
            )

        response_length = extra.batch["responses"].shape[-1]
        response_mask = extra.batch["attention_mask"][:, -response_length:].clone()
        extra.batch["response_mask"] = response_mask
        action_loss_mask = _action_mask(extra).clone()
        extra.batch["old_log_probs"] = old_output.batch["old_log_probs"]
        extra.batch["teacher_log_probs"] = old_output.batch["old_log_probs"].detach().clone()

        if self.use_reference_policy:
            if not self.ref_in_actor:
                ref_output = self.ref_policy_wg.compute_ref_log_prob(no_skill_proto)
            else:
                ref_output = self.actor_rollout_wg.compute_ref_log_prob(no_skill_proto)
            extra.batch["ref_log_prob"] = ref_output.batch["ref_log_prob"]

        for key in ("token_level_scores", "token_level_rewards", "advantages", "returns"):
            if key in extra.batch:
                extra.batch[key].zero_()

        pg_mask = torch.zeros_like(action_loss_mask)
        regularization_mask = torch.zeros_like(action_loss_mask)
        suffix_mask = torch.zeros_like(action_loss_mask)
        for index, record in enumerate(padded_records):
            if record["kind"] == "teacher_divergence":
                pg_mask[index] = action_loss_mask[index]
                regularization_mask[index] = action_loss_mask[index]
                advantage = float(record["advantage"])
                extra.batch["advantages"][index] = advantage * action_loss_mask[index]
                extra.batch["returns"][index] = advantage * action_loss_mask[index]
            elif record["kind"] == "suffix":
                suffix_mask[index] = action_loss_mask[index]
            if record["kind"] != "teacher_divergence" and "loss_mask" in extra.batch:
                extra.batch["loss_mask"][index].zero_()

        suffix_weights = torch.zeros_like(suffix_mask, dtype=torch.float32)
        pair_token_counts = {}
        for index, record in enumerate(padded_records):
            if record["kind"] == "suffix":
                pair_token_counts.setdefault(record["pair_id"], 0.0)
                pair_token_counts[record["pair_id"]] += float(suffix_mask[index].sum().item())
        for index, record in enumerate(padded_records):
            token_count = pair_token_counts.get(record["pair_id"], 0.0)
            if record["kind"] == "suffix" and token_count > 0:
                suffix_weights[index] = suffix_mask[index].float() / token_count

        extra.batch["ovcsd_pg_mask"] = pg_mask
        extra.batch["ovcsd_regularization_mask"] = regularization_mask
        extra.batch["ovcsd_suffix_mask"] = suffix_mask
        extra.batch["ovcsd_suffix_weights"] = suffix_weights
        extra.batch["ovcsd_is_extra"] = torch.ones(
            len(extra), dtype=torch.bool, device=pg_mask.device
        )
        extra.batch["ovcsd_teacher_topk_token_ids"] = teacher_output.batch[
            "ovcsd_teacher_topk_token_ids"
        ]
        extra.batch["ovcsd_teacher_topk_log_probs"] = teacher_output.batch[
            "ovcsd_teacher_topk_log_probs"
        ]
        extra.batch["ovcsd_teacher_tail_log_prob"] = teacher_output.batch[
            "ovcsd_teacher_tail_log_prob"
        ]

        self._set_extra_metadata(extra, padded_records)
        self._attach_empty_teacher_targets(batch)
        batch = DataProto.concat([batch, extra])
        return batch, {
            "ovcsd/teacher_divergence_rows": float(
                sum(record["kind"] == "teacher_divergence" for record in records)
            ),
            "ovcsd/suffix_rows": float(sum(record["kind"] == "suffix" for record in records)),
            "ovcsd/padding_rows": float(pad_count),
            "ovcsd/suffix_tokens_prepared": float(suffix_mask.sum().item()),
        }

    def _model_proto(self, rows) -> DataProto:
        tensors = {
            key: torch.stack([row[key] for row in rows], dim=0)
            for key in _MODEL_KEYS
        }
        return DataProto.from_dict(tensors=tensors)

    def _attach_empty_teacher_targets(self, batch: DataProto):
        batch_size, response_length = batch.batch["responses"].shape
        device = batch.batch["responses"].device
        batch.batch["ovcsd_teacher_topk_token_ids"] = torch.zeros(
            (batch_size, response_length, self.ovcsd_top_k),
            dtype=torch.long,
            device=device,
        )
        batch.batch["ovcsd_teacher_topk_log_probs"] = torch.zeros(
            (batch_size, response_length, self.ovcsd_top_k),
            dtype=torch.float32,
            device=device,
        )
        batch.batch["ovcsd_teacher_tail_log_prob"] = torch.zeros(
            (batch_size, response_length),
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def _set_extra_metadata(extra: DataProto, records):
        for key, values in extra.non_tensor_batch.items():
            replacement = values.copy()
            for index, record in enumerate(records):
                row = record["no_skill_row"]
                if key in row:
                    replacement[index] = row[key]
            extra.non_tensor_batch[key] = replacement

        size = len(records)
        extra.non_tensor_batch["uid"] = np.asarray(
            [f"ovcsd-{record['pair_id']}-{index}" for index, record in enumerate(records)],
            dtype=object,
        )
        extra.non_tensor_batch["traj_uid"] = np.asarray(
            [f"ovcsd-{uuid.uuid4()}" for _ in range(size)], dtype=object
        )
        extra.non_tensor_batch["turn_step"] = np.arange(size, dtype=object)
        for key, value in (
            ("episode_rewards", 0.0),
            ("episode_lengths", 1.0),
            ("tool_callings", 0.0),
            ("is_action_valid", True),
        ):
            extra.non_tensor_batch[key] = np.asarray([value] * size, dtype=object)
