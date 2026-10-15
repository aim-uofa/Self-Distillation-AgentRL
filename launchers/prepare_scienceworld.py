#!/usr/bin/env python3
"""Prepare a disjoint 2059/200 split from the pinned AgentGym-RL task IDs.

Only the selected evaluation Parquet path is written to stdout. Existing
benchmarks, original upstream IDs, and the full 200-task test set are untouched.
The original 2120-task train.parquet is retained for provenance; launchers use
train_disjoint.parquet, excluding all 200 evaluation IDs even for smoke evals.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_system.scienceworld_protocol import (
    DATA_REPO, DATA_REVISION, EVAL_COUNTS, EVALUATION_PROTOCOL, PROTOCOL_VERSION,
    SCIENCEWORLD_VERSION, SOURCE_FILES, resolve_item,
)
from prepare_search_eval_subset import atomic_target, file_sha256, locked, write_json


def load_source(root, split, download=False):
    source_name, expected_size, expected_hash = SOURCE_FILES[split]
    path = root / "source" / Path(source_name).name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"Missing {path}; run bash launchers/prepare_scienceworld.sh on an online host, "
                "then copy data/scienceworld_agentgym to the offline host."
            )
        url = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REVISION}/{source_name}"
        print(f"Downloading pinned ScienceWorld IDs: {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"Upstream ScienceWorld {split} SHA256 mismatch")
        with atomic_target(path) as temporary:
            temporary.write_bytes(payload)
    if file_sha256(path) != expected_hash:
        raise ValueError(f"ScienceWorld source checksum mismatch: {path}; refusing to overwrite")
    source = json.loads(path.read_text())
    items = [entry["item_id"] for entry in source]
    if len(items) != expected_size or len(set(items)) != expected_size:
        raise ValueError(f"Expected {expected_size} distinct {split} tasks")
    tasks = [resolve_item(item) for item in items]
    if split == "test" and dict(Counter(t["category"] for t in tasks)) != EVAL_COUNTS:
        raise ValueError("ScienceWorld evaluation categories do not match Table 5's 200 tasks")
    return tasks


def task_rows(tasks, split):
    return [
        {
            "data_source": f"scienceworld/{task['category']}",
            "prompt": [{"role": "user", "content": "Complete the ScienceWorld task."}],
            "ability": "scienceworld",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"split": split, "index": task["item_id"],
                           "category": task["category"]},
            "env_kwargs": task,
        }
        for task in tasks
    ]


def prepare(data_root, download=False, eval_size="all"):
    eval_size = str(eval_size)
    if eval_size != "all" and (not eval_size.isdecimal() or not 1 <= int(eval_size) <= 200):
        raise ValueError("ScienceWorld EVAL_SIZE must be 'all' or an integer from 1 to 200")
    root = Path(data_root).resolve() / "scienceworld_agentgym"
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / "prepare.lock"):
        tasks = {split: load_source(root, split, download) for split in SOURCE_FILES}
        eval_ids = {t["item_id"] for t in tasks["test"]}
        overlap = sorted({t["item_id"] for t in tasks["train"]} & eval_ids)
        if len(overlap) != 61:
            raise ValueError("Unexpected change to the published train/eval overlap")
        # Filter against the FULL test pool, never the optional smoke prefix.
        # Keep upstream files/manifests intact, so previous evaluation and
        # pass@k artifacts retain exactly the same identity.
        disjoint_train = [task for task in tasks["train"] if task["item_id"] not in eval_ids]
        outputs = {"train": tasks["train"], "test": tasks["test"],
                   "train_disjoint": disjoint_train}
        selection = "test"
        if eval_size not in ("all", "200"):
            selection = f"test_first_{int(eval_size)}"
            outputs[selection] = tasks["test"][:int(eval_size)]
        for name, selected in outputs.items():
            path = root / f"{name}.parquet"
            manifest_path = root / f"{name}.manifest.json"
            identity = {
                "protocol_version": PROTOCOL_VERSION, "source_repo": DATA_REPO,
                "source_revision": DATA_REVISION, "simulator_version": SCIENCEWORLD_VERSION,
                "source_sha256": {k: v[2] for k, v in SOURCE_FILES.items()},
                "item_ids": [t["item_id"] for t in selected], "row_count": len(selected),
                "categories": dict(Counter(t["category"] for t in selected)),
                "train_eval_overlap_count": len(overlap), "train_eval_overlap_ids": overlap,
            }
            if name == "train_disjoint":
                identity.update({
                    "selection": "exclude_all_eval_ids_v1",
                    "source_train_row_count": len(tasks["train"]),
                    "excluded_eval_count": len(overlap), "excluded_eval_ids": overlap,
                    "train_eval_overlap_count": 0, "train_eval_overlap_ids": [],
                })
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if any(manifest.get(key) != value for key, value in identity.items()):
                    raise ValueError(f"ScienceWorld manifest differs from pinned protocol: {manifest_path}")
                if not path.exists() or file_sha256(path) != manifest["parquet_sha256"]:
                    raise ValueError(f"ScienceWorld Parquet missing/corrupted: {path}")
            else:
                if path.exists():
                    raise FileExistsError(f"Unmanaged file exists; refusing to overwrite: {path}")
                split = "train" if name in ("train", "train_disjoint") else "test"
                with atomic_target(path) as temporary:
                    pq.write_table(pa.Table.from_pylist(task_rows(selected, split)), temporary)
                write_json(manifest_path, {**identity, "parquet_sha256": file_sha256(path)})
        print(f"ScienceWorld: train={len(disjoint_train)} (train_disjoint.parquet), "
              f"fixed eval=200, overlap=0, categories={EVAL_COUNTS}", file=sys.stderr)
        print(f"Excluded {len(overlap)} evaluation IDs from the 2120-task source train pool; "
              "original train.parquet retained for provenance only.", file=sys.stderr)
        return root / f"{selection}.parquet"


def check_attempt(data_root, attempt_dir, eval_size="all"):
    """Reject completed runs from other task pools or scoring protocols."""
    selected = prepare(data_root, eval_size=eval_size)
    identity = json.loads(selected.with_suffix(".manifest.json").read_text())
    attempt_dir = Path(attempt_dir)
    manifest = json.loads((attempt_dir / "manifest.json").read_text())
    if (not manifest.get("completed") or manifest.get("benchmark") != "scienceworld"
            or manifest.get("scienceworld_protocol") != EVALUATION_PROTOCOL):
        raise ValueError(f"Incomplete or incompatible ScienceWorld attempt: {attempt_dir}")
    records = [json.loads(line) for line in (attempt_dir / "rollouts.jsonl").read_text().splitlines() if line.strip()]
    if (len(records) != manifest.get("record_count")
            or [row.get("task_id") for row in records] != identity["item_ids"]):
        raise ValueError(f"ScienceWorld attempt task IDs differ from the fixed evaluation set: {attempt_dir}")
    for position, row in enumerate(records):
        task = resolve_item(row["task_id"])
        expected_fingerprint = hashlib.sha256(f"scienceworld:agentgym:v1:{task['item_id']}".encode()).hexdigest()
        score = float(row["task_score"])
        if (row.get("task_position") != position or row.get("benchmark") != "scienceworld"
                or row.get("task_fingerprint") != expected_fingerprint
                or row.get("category") != task["category"]
                or row.get("data_source") != f"scienceworld/{task['category']}"
                or not math.isfinite(score) or row.get("success") != (score >= 100)
                or row.get("positive_score") != (score > 0)):
            raise ValueError(f"Invalid ScienceWorld attempt record at position {position}: {attempt_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--eval-size", default="all")
    parser.add_argument("--check-attempt", type=Path)
    args = parser.parse_args()
    if args.check_attempt:
        check_attempt(args.data_root, args.check_attempt, args.eval_size)
    else:
        print(prepare(args.data_root, args.download, args.eval_size))


if __name__ == "__main__":
    main()
