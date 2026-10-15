"""Stable request seeds for opt-in multi-turn evaluation."""

import hashlib
import json


def derive_request_seed(
    *, benchmark: str, task_seed: int, rollout_seed: int, task_key: str, turn_index: int
) -> int:
    payload = [
        "eval_request_seed_v1",
        benchmark,
        int(task_seed),
        int(rollout_seed),
        task_key,
        int(turn_index),
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)
