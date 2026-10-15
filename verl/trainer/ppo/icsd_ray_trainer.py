"""ICSD trainer built on the isolated SkillSD training loop."""

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer


class ICSDRayTrainer(SkillSDRayTrainer):
    """Expose interaction-turn indices to the ICSD actor allocator."""

    def _prepare_method_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        turn_steps = batch.non_tensor_batch.get("turn_step")
        if turn_steps is None:
            raise KeyError("ICSD requires turn_step metadata for turn-conditioned calibration")
        turn_array = np.asarray(turn_steps, dtype=np.int64)
        if turn_array.shape != (len(batch),):
            raise ValueError(
                f"ICSD expected turn_step shape {(len(batch),)}, got {turn_array.shape}"
            )
        batch.batch["icsd_turn_indices"] = torch.as_tensor(
            turn_array,
            dtype=torch.long,
            device=batch.batch["responses"].device,
        )
        return batch, {}
