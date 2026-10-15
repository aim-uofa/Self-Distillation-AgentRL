"""Write multi-turn validation rollouts as one JSON object per episode."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _select(config, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(config, path, default=default)
    return _jsonable(value)


def _value_at(mapping: dict[str, Any], key: str, index: int, default: Any = None) -> Any:
    values = mapping.get(key)
    if values is None:
        return default
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    value = _jsonable(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    value = _jsonable(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalized_gamefile(gamefile: Any) -> str | None:
    if gamefile is None:
        return None
    gamefile = str(_jsonable(gamefile))
    marker = "json_2.1.1/"
    if marker in gamefile:
        return gamefile.split(marker, 1)[1]
    return gamefile


def _alfworld_category(gamefile: str | None) -> str | None:
    if not gamefile:
        return None
    for category in (
        "pick_and_place",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    ):
        if category in gamefile:
            return category
    return None


class ValidationRolloutWriter:
    """Stream complete validation episodes to ``rollouts.jsonl``."""

    schema_version = 2

    def __init__(self, output_dir: str, tokenizer, config, global_step: int):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.config = config
        self.global_step = int(global_step)
        self.task_position = 0
        self.record_count = 0

        self.benchmark = self._infer_benchmark()
        self.rollout_seed = _as_int(_select(config, "actor_rollout_ref.rollout.seed", 0))
        self.eval_seed_mode = str(_select(config, "actor_rollout_ref.rollout.eval_seed_mode", "legacy"))
        self.seed_derivation_version = 1 if self.eval_seed_mode == "per_task_turn" else 0
        self.task_seed = _as_int(_select(config, "env.seed", 0))
        self.attempt_index = _as_int(_select(config, "trainer.validation_attempt_index", 0))
        self.model_kind = str(_select(config, "trainer.validation_model_kind", "unknown"))
        self.model_tag = str(_select(config, "trainer.validation_model_tag", "unknown"))

        self.rollout_path = self.output_dir / "rollouts.jsonl"
        self.manifest_path = self.output_dir / "manifest.json"
        self._handle = self.rollout_path.open("w", encoding="utf-8", buffering=1)
        self._write_manifest(completed=False)

    def _infer_benchmark(self) -> str:
        explicit = _select(self.config, "trainer.validation_benchmark", None)
        if explicit:
            return str(explicit).lower()
        env_name = str(_select(self.config, "env.env_name", "unknown")).lower()
        if "alfworld" in env_name or "alfred" in env_name:
            return "alfworld"
        if "webshop" in env_name:
            return "webshop"
        if "search" in env_name:
            return "search"
        return env_name

    def _write_manifest(self, completed: bool) -> None:
        manifest = {
            "schema_version": self.schema_version,
            "completed": completed,
            "benchmark": self.benchmark,
            "model_kind": self.model_kind,
            "model_tag": self.model_tag,
            "model_path": _select(self.config, "actor_rollout_ref.model.path"),
            "task_seed": self.task_seed,
            "alfworld_strict_eval": os.environ.get("ALFWORLD_STRICT_EVAL", "0").lower() in {"1", "true", "yes"},
            "rollout_seed": self.rollout_seed,
            "eval_seed_mode": self.eval_seed_mode,
            "seed_derivation_version": self.seed_derivation_version,
            "attempt_index": self.attempt_index,
            "temperature": _select(self.config, "actor_rollout_ref.rollout.val_kwargs.temperature"),
            "top_p": _select(self.config, "actor_rollout_ref.rollout.val_kwargs.top_p"),
            "top_k": _select(self.config, "actor_rollout_ref.rollout.val_kwargs.top_k"),
            "do_sample": _select(self.config, "actor_rollout_ref.rollout.val_kwargs.do_sample"),
            "max_env_steps": _select(self.config, "env.max_steps"),
            "val_batch_size": _select(self.config, "data.val_batch_size"),
            "global_step": self.global_step,
            "record_count": self.record_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.benchmark == "scienceworld":
            from agent_system.scienceworld_protocol import EVALUATION_PROTOCOL
            manifest.update({
                "scienceworld_protocol": EVALUATION_PROTOCOL,
                "scienceworld_reward_mode": _select(self.config, "env.scienceworld.reward_mode", "score"),
                "max_prompt_length": _select(self.config, "data.max_prompt_length"),
                "max_response_length": _select(self.config, "data.max_response_length"),
            })
        temporary_path = self.manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self.manifest_path)

    def capture_source_batch(self, batch) -> list[dict[str, Any]]:
        """Capture stable dataset metadata before the trainer pops it."""
        metadata = []
        non_tensor = batch.non_tensor_batch
        for index in range(len(batch)):
            extra_info = _jsonable(_value_at(non_tensor, "extra_info", index, {})) or {}
            env_kwargs = _jsonable(_value_at(non_tensor, "env_kwargs", index, {})) or {}
            data_source = _jsonable(_value_at(non_tensor, "data_source", index, "unknown"))
            source_index = extra_info.get("index") if isinstance(extra_info, dict) else None
            question = None
            ground_truth = None
            if isinstance(env_kwargs, dict):
                question = env_kwargs.get("question")
                ground_truth = env_kwargs.get("ground_truth")
            if question is None and isinstance(extra_info, dict):
                question = extra_info.get("question")
            metadata.append(
                {
                    "source_index": _jsonable(source_index),
                    "data_source": data_source,
                    "question": _jsonable(question),
                    "ground_truth": _jsonable(ground_truth),
                }
            )
        return metadata

    def _decode(self, token_ids: Any) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _task_identity(
        self,
        source_metadata: dict[str, Any],
        gamefile: str | None,
        first_prompt: str,
        data_source: str,
    ) -> tuple[str, str]:
        source_index = source_metadata.get("source_index")
        if self.benchmark == "scienceworld" and source_index is not None:
            task_id = str(source_index)
            fingerprint_source = f"scienceworld:agentgym:v1:{task_id}"
        elif self.benchmark == "search" and source_index is not None:
            task_id = f"{data_source}:{source_index}"
            fingerprint_source = f"{task_id}\n{source_metadata.get('question') or first_prompt}"
        elif self.benchmark == "alfworld" and gamefile:
            task_id = gamefile
            fingerprint_source = gamefile
        else:
            task_id = f"position:{self.task_position}"
            fingerprint_source = first_prompt
        fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
        return task_id, fingerprint

    def write_batch(self, output_batch, source_metadata: list[dict[str, Any]]) -> None:
        non_tensor = output_batch.non_tensor_batch
        if self.eval_seed_mode == "per_task_turn" and (
            "eval_request_seed" not in non_tensor or "eval_task_key" not in non_tensor
        ):
            raise RuntimeError("Missing per-task-turn seed trace in validation rollout")
        traj_uids = [_jsonable(value) for value in non_tensor["traj_uid"]]
        grouped_rows: OrderedDict[str, list[int]] = OrderedDict()
        for row_index, traj_uid in enumerate(traj_uids):
            grouped_rows.setdefault(str(traj_uid), []).append(row_index)

        if len(grouped_rows) != len(source_metadata):
            raise RuntimeError(
                "Validation rollout/source size mismatch: "
                f"episodes={len(grouped_rows)}, source_items={len(source_metadata)}. "
                "Keep actor_rollout_ref.rollout.val_kwargs.n=1 for multi-turn evaluation."
            )

        prompts = output_batch.batch["prompts"]
        responses = output_batch.batch["responses"]

        for episode_index, (traj_uid, row_indices) in enumerate(grouped_rows.items()):
            row_indices.sort(key=lambda idx: _as_int(_value_at(non_tensor, "turn_step", idx, idx)))
            first_row = row_indices[0]
            source_item = source_metadata[episode_index]
            first_prompt = self._decode(prompts[first_row])
            gamefile = _normalized_gamefile(_value_at(non_tensor, "gamefile", first_row))
            data_source = str(
                _jsonable(
                    _value_at(
                        non_tensor,
                        "data_source",
                        first_row,
                        source_item.get("data_source", "unknown"),
                    )
                )
            )
            task_id, task_fingerprint = self._task_identity(source_item, gamefile, first_prompt, data_source)

            turns = []
            for row_index in row_indices:
                turns.append(
                    {
                        "turn": _as_int(_value_at(non_tensor, "turn_step", row_index, len(turns))),
                        "request_seed": _as_int(
                            _value_at(non_tensor, "eval_request_seed", row_index, self.rollout_seed)
                        ),
                        "prompt": self._decode(prompts[row_index]),
                        "response": self._decode(responses[row_index]),
                        "reward": _as_float(_value_at(non_tensor, "rewards", row_index, 0.0)),
                        "is_action_valid": bool(_jsonable(_value_at(non_tensor, "is_action_valid", row_index, True))),
                    }
                )

            episode_reward = _as_float(_value_at(non_tensor, "episode_rewards", first_row, 0.0))
            record = {
                "schema_version": self.schema_version,
                "benchmark": self.benchmark,
                "model_kind": self.model_kind,
                "model_tag": self.model_tag,
                "task_seed": self.task_seed,
                "rollout_seed": self.rollout_seed,
                "eval_seed_mode": self.eval_seed_mode,
                "seed_derivation_version": self.seed_derivation_version,
                "eval_task_key": _jsonable(_value_at(non_tensor, "eval_task_key", first_row)),
                "attempt_index": self.attempt_index,
                "task_position": self.task_position,
                "task_id": task_id,
                "task_fingerprint": task_fingerprint,
                "data_source": data_source,
                "category": _alfworld_category(gamefile),
                "source_index": source_item.get("source_index"),
                "question": source_item.get("question"),
                "ground_truth": source_item.get("ground_truth"),
                "gamefile": gamefile,
                "traj_uid": traj_uid,
                "episode_reward": episode_reward,
                "success": bool(episode_reward > 0.0),
                "episode_length": _as_int(
                    _value_at(non_tensor, "episode_lengths", first_row, len(turns))
                ),
                "tool_call_count": _as_float(_value_at(non_tensor, "tool_callings", first_row, 0.0)),
                "turns": turns,
            }
            if self.benchmark == "scienceworld":
                score = _value_at(non_tensor, "scienceworld_score", row_indices[-1])
                if score is None:
                    raise RuntimeError("ScienceWorld rollout is missing its final native score")
                score = _as_float(score)
                record.update({
                    "category": data_source.removeprefix("scienceworld/"),
                    "task_score": score, "success": score >= 100.0,
                    "positive_score": score > 0.0,
                })
            self._handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
            self.task_position += 1
            self.record_count += 1

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.flush()
        self._handle.close()
        self._write_manifest(completed=True)
        print(f"Dumped {self.record_count} validation episodes to {self.rollout_path}")
