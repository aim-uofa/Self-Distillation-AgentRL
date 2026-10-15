#!/usr/bin/env python3
"""Compute pass@k and K-attempt benchmark statistics from rollout JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


WEBSHOP_SCORE_RE = re.compile(
    r"\bval/webshop_task_score \(not success_rate\):"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--max-pass-k", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def pass_at_k_estimate(n: int, c: int, k: int) -> float:
    """Return 1 - C(n-c, k) / C(n, k), evaluated stably."""
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    failure_probability = 1.0
    for offset in range(k):
        failure_probability *= (n - c - offset) / (n - offset)
    return 1.0 - failure_probability


def seed_protocol(manifest: dict[str, Any]) -> tuple[str, int]:
    """Treat pre-change manifests as the original legacy_v0 protocol."""
    mode = manifest.get("eval_seed_mode", "legacy")
    version = int(manifest.get("seed_derivation_version", 0))
    if (mode, version) not in {("legacy", 0), ("per_task_turn", 1)}:
        raise RuntimeError(f"Unknown evaluation seed protocol: {mode}_v{version}")
    return mode, version


def load_attempts(input_dir: Path) -> list[dict[str, Any]]:
    rollout_paths = sorted(input_dir.glob("attempt_*_seed_*/rollouts.jsonl"))
    if not rollout_paths:
        raise RuntimeError(f"No attempt rollout files found under {input_dir}")

    attempts = []
    seen_rollout_seeds = set()
    for rollout_path in rollout_paths:
        attempt_dir = rollout_path.parent
        if not (attempt_dir / "_SUCCESS").is_file():
            raise RuntimeError(f"Incomplete attempt (missing _SUCCESS): {attempt_dir}")
        manifest_path = attempt_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("completed"):
            raise RuntimeError(f"Writer did not complete: {manifest_path}")
        mode, version = seed_protocol(manifest)
        manifest["eval_seed_mode"] = mode
        manifest["seed_derivation_version"] = version
        manifest["alfworld_strict_eval"] = bool(manifest.get("alfworld_strict_eval", False))

        rollout_seed = int(manifest["rollout_seed"])
        if rollout_seed in seen_rollout_seeds:
            raise RuntimeError(f"Duplicate rollout_seed={rollout_seed}")
        seen_rollout_seeds.add(rollout_seed)

        records = {}
        with rollout_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if mode == "per_task_turn":
                    if (
                        record.get("eval_seed_mode") != mode
                        or record.get("seed_derivation_version") != version
                        or record.get("eval_task_key") is None
                        or any(
                            turn.get("request_seed") is None for turn in record.get("turns", [])
                        )
                    ):
                        raise RuntimeError(f"Missing per-task-turn seed trace in {rollout_path}:{line_number}")
                position = int(record["task_position"])
                if position in records:
                    raise RuntimeError(f"Duplicate task_position={position} in {rollout_path}:{line_number}")
                records[position] = record
        if len(records) != int(manifest["record_count"]):
            raise RuntimeError(
                f"Manifest/rollout count mismatch in {attempt_dir}: "
                f"manifest={manifest['record_count']}, jsonl={len(records)}"
            )
        attempts.append({"manifest": manifest, "records": records, "path": rollout_path})

    attempts.sort(
        key=lambda item: (
            int(item["manifest"]["attempt_index"]),
            int(item["manifest"]["rollout_seed"]),
        )
    )
    return attempts


def validate_alignment(attempts: list[dict[str, Any]]) -> list[int]:
    reference = attempts[0]
    reference_positions = sorted(reference["records"])
    if reference_positions != list(range(len(reference_positions))):
        raise RuntimeError("Reference attempt task positions are not contiguous from zero")

    reference_manifest = reference["manifest"]
    if reference_manifest.get("benchmark") == "scienceworld":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from agent_system.scienceworld_protocol import EVALUATION_PROTOCOL
        for attempt in attempts:
            if attempt["manifest"].get("scienceworld_protocol") != EVALUATION_PROTOCOL:
                raise RuntimeError("Missing or incompatible ScienceWorld evaluation protocol")
            for key in ("max_prompt_length", "max_response_length", "scienceworld_reward_mode"):
                if attempt["manifest"].get(key) != reference_manifest.get(key):
                    raise RuntimeError(f"ScienceWorld attempt manifest mismatch for {key}")
            for record in attempt["records"].values():
                score = float(record["task_score"])
                if not math.isfinite(score) or record["success"] != (score >= 100):
                    raise RuntimeError("ScienceWorld success must mean final native score >= 100")
    for attempt in attempts[1:]:
        manifest = attempt["manifest"]
        for key in (
            "benchmark", "model_kind", "model_tag", "model_path", "task_seed",
            "temperature", "top_p", "top_k", "do_sample", "max_env_steps",
            "val_batch_size", "alfworld_strict_eval", "eval_seed_mode",
            "seed_derivation_version",
        ):
            if manifest.get(key) != reference_manifest.get(key):
                raise RuntimeError(
                    f"Attempt manifest mismatch for {key}: "
                    f"{manifest.get(key)!r} != {reference_manifest.get(key)!r}"
                )
        if sorted(attempt["records"]) != reference_positions:
            raise RuntimeError(f"Task positions differ in {attempt['path']}")
        for position in reference_positions:
            expected = reference["records"][position]
            actual = attempt["records"][position]
            if actual.get("task_fingerprint") != expected.get("task_fingerprint"):
                raise RuntimeError(
                    "Task alignment mismatch at position "
                    f"{position}: {actual.get('task_id')} != {expected.get('task_id')}"
                )
    return reference_positions


def group_name(record: dict[str, Any]) -> str:
    benchmark = record["benchmark"]
    if benchmark in {"alfworld", "scienceworld"} and record.get("category"):
        return str(record["category"])
    if benchmark == "search" and record.get("data_source"):
        return str(record["data_source"])
    return "overall"


def mean_and_sample_std(values: list[float]) -> tuple[float, float | None]:
    """Return the mean and sample standard deviation (ddof=1)."""
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    mean = math.fsum(values) / len(values)
    if len(values) == 1:
        return mean, None
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def webshop_task_score(attempt: dict[str, Any]) -> float | None:
    """Read WebShop's native mean task score from an attempt console log."""
    manifest_metrics = attempt["manifest"].get("benchmark_metrics", {})
    if "webshop_task_score" in manifest_metrics:
        return float(manifest_metrics["webshop_task_score"])

    console_path = attempt["path"].parent / "console.log"
    if not console_path.is_file():
        return None
    matches = WEBSHOP_SCORE_RE.findall(
        console_path.read_text(encoding="utf-8", errors="replace")
    )
    return float(matches[-1]) if matches else None


def benchmark_statistics(
    attempts: list[dict[str, Any]],
    grouped_tasks: dict[str, list[dict[str, Any]]],
    max_pass_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Summarize per-attempt benchmark results using exactly K attempts once."""
    selected_attempts = attempts[:max_pass_k]
    mode, version = seed_protocol(selected_attempts[0]["manifest"])
    selected_seeds = [
        int(attempt["manifest"]["rollout_seed"]) for attempt in selected_attempts
    ]
    rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for group, tasks in sorted(
        grouped_tasks.items(), key=lambda item: (item[0] != "overall", item[0])
    ):
        values = []
        for attempt in selected_attempts:
            successes = [
                float(bool(attempt["records"][int(task["task_position"])]["success"]))
                for task in tasks
            ]
            values.append(math.fsum(successes) / len(successes))
        mean, std = mean_and_sample_std(values)
        metric_key = "success_rate" if group == "overall" else f"success_rate/{group}"
        metric = {
            "group": group,
            "task_count": len(tasks),
            "num_rollouts": max_pass_k,
            "mean": mean,
            "std": std,
            "values_by_rollout_seed": {
                str(seed): value for seed, value in zip(selected_seeds, values)
            },
        }
        metrics[metric_key] = metric
        rows.append({"metric": "success_rate", "eval_seed_mode": mode, "seed_derivation_version": version, **metric})

    benchmark = str(selected_attempts[0]["manifest"]["benchmark"])
    primary_metric = "success_rate"
    if benchmark == "scienceworld":
        primary_metric = "scienceworld_score"
        for group, tasks in sorted(grouped_tasks.items(), key=lambda item: (item[0] != "overall", item[0])):
            for metric_name in ("scienceworld_score", "positive_score_rate"):
                values = []
                for attempt in selected_attempts:
                    scores = [float(attempt["records"][int(task["task_position"])]["task_score"]) for task in tasks]
                    if not all(math.isfinite(value) for value in scores):
                        raise ValueError("Non-finite ScienceWorld task score")
                    values.append(math.fsum(scores if metric_name == "scienceworld_score" else
                                            [float(value > 0) for value in scores]) / len(scores))
                mean, std = mean_and_sample_std(values)
                metric = {
                    "group": group, "task_count": len(tasks), "num_rollouts": max_pass_k,
                    "mean": mean, "std": std,
                    "values_by_rollout_seed": dict(zip(map(str, selected_seeds), values)),
                }
                key = metric_name if group == "overall" else f"{metric_name}/{group}"
                metrics[key] = metric
                rows.append({"metric": metric_name, "eval_seed_mode": mode, "seed_derivation_version": version, **metric})
        warnings.append("ScienceWorld pass@k uses full success (score>=100), not AgentGym's positive-score Pass@k.")
    if benchmark == "webshop":
        primary_metric = "webshop_task_score"
        scores = [webshop_task_score(attempt) for attempt in selected_attempts]
        if all(score is not None for score in scores):
            score_values = [float(score) for score in scores if score is not None]
            mean, std = mean_and_sample_std(score_values)
            metric = {
                "group": "overall",
                "task_count": len(grouped_tasks["overall"]),
                "num_rollouts": max_pass_k,
                "mean": mean,
                "std": std,
                "values_by_rollout_seed": {
                    str(seed): value for seed, value in zip(selected_seeds, score_values)
                },
            }
            metrics[primary_metric] = metric
            rows.append({"metric": primary_metric, "eval_seed_mode": mode, "seed_derivation_version": version, **metric})
        else:
            missing_seeds = [
                seed for seed, score in zip(selected_seeds, scores) if score is None
            ]
            warnings.append(
                "WebShop task score is unavailable for rollout seed(s) "
                f"{missing_seeds}; their console.log files do not contain the native metric. "
                "Binary success-rate mean/std is still available."
            )

    return (
        {
            "eval_seed_mode": mode,
            "seed_derivation_version": version,
            "num_rollouts": max_pass_k,
            "rollout_seeds": selected_seeds,
            "selection": "first K attempts ordered by attempt_index and rollout_seed",
            "std_definition": "sample standard deviation (ddof=1); null when K=1",
            "primary_metric": primary_metric,
            "metrics": metrics,
        },
        rows,
        warnings,
    )


def main() -> None:
    args = parse_args()
    if args.max_pass_k < 1:
        raise ValueError("--max-pass-k must be positive")
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or input_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    attempts = load_attempts(input_dir)
    n = len(attempts)
    if n < args.max_pass_k:
        raise RuntimeError(f"Need at least {args.max_pass_k} completed attempts, found {n}")
    positions = validate_alignment(attempts)

    reference_records = attempts[0]["records"]
    per_task = []
    grouped_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position in positions:
        reference = reference_records[position]
        successes = [bool(attempt["records"][position]["success"]) for attempt in attempts]
        correct = sum(successes)
        estimates = {
            f"pass@{k}": pass_at_k_estimate(n=n, c=correct, k=k)
            for k in range(1, args.max_pass_k + 1)
        }
        task_result = {
            "eval_seed_mode": attempts[0]["manifest"]["eval_seed_mode"],
            "seed_derivation_version": attempts[0]["manifest"]["seed_derivation_version"],
            "task_position": position,
            "task_id": reference.get("task_id"),
            "task_fingerprint": reference.get("task_fingerprint"),
            "data_source": reference.get("data_source"),
            "category": reference.get("category"),
            "n": n,
            "c": correct,
            "success_by_rollout_seed": {
                str(attempt["manifest"]["rollout_seed"]): success
                for attempt, success in zip(attempts, successes)
            },
            **estimates,
        }
        per_task.append(task_result)
        grouped_tasks["overall"].append(task_result)
        subgroup = group_name(reference)
        if subgroup != "overall":
            grouped_tasks[subgroup].append(task_result)

    summary_rows = []
    for group, tasks in sorted(
        grouped_tasks.items(), key=lambda item: (item[0] != "overall", item[0])
    ):
        for k in range(1, args.max_pass_k + 1):
            metric = f"pass@{k}"
            summary_rows.append(
                {
                    "eval_seed_mode": attempts[0]["manifest"]["eval_seed_mode"],
                    "seed_derivation_version": attempts[0]["manifest"]["seed_derivation_version"],
                    "group": group,
                    "task_count": len(tasks),
                    "num_rollouts": n,
                    "k": k,
                    "pass_at_k": math.fsum(task[metric] for task in tasks) / len(tasks),
                }
            )

    manifest = attempts[0]["manifest"]
    stats, stats_rows, stats_warnings = benchmark_statistics(
        attempts=attempts,
        grouped_tasks=grouped_tasks,
        max_pass_k=args.max_pass_k,
    )
    summary = {
        "benchmark": manifest["benchmark"],
        "model_kind": manifest["model_kind"],
        "model_tag": manifest["model_tag"],
        "task_seed": manifest["task_seed"],
        "eval_seed_mode": manifest["eval_seed_mode"],
        "seed_derivation_version": manifest["seed_derivation_version"],
        "temperature": manifest["temperature"],
        "do_sample": manifest["do_sample"],
        "num_rollouts": n,
        "max_pass_k": args.max_pass_k,
        "rollout_seeds": [attempt["manifest"]["rollout_seed"] for attempt in attempts],
        "groups": {
            group: {
                "task_count": len(tasks),
                **{
                    f"pass@{k}": math.fsum(task[f"pass@{k}"] for task in tasks) / len(tasks)
                    for k in range(1, args.max_pass_k + 1)
                },
            }
            for group, tasks in grouped_tasks.items()
        },
        "benchmark_statistics_at_k": stats,
    }

    (output_dir / "pass_at_k.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "pass_at_k.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "eval_seed_mode", "seed_derivation_version", "group", "task_count",
                "num_rollouts", "k", "pass_at_k",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "per_task.jsonl").open("w", encoding="utf-8") as handle:
        for task in per_task:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    with (output_dir / "benchmark_mean_std.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "eval_seed_mode",
                "seed_derivation_version",
                "metric",
                "group",
                "task_count",
                "num_rollouts",
                "mean",
                "std",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(stats_rows)

    print(f"Loaded {n} attempts and {len(positions)} aligned tasks.")
    print(
        f"Benchmark={manifest['benchmark']} model={manifest['model_tag']} "
        f"temperature={manifest['temperature']}"
    )
    print("Overall:")
    overall = summary["groups"]["overall"]
    for k in range(1, args.max_pass_k + 1):
        print(f"  pass@{k}: {overall[f'pass@{k}']:.6f}")
    print(f"Benchmark mean/std at K={args.max_pass_k} (sample std, ddof=1):")
    print(f"  rollout_seeds: {stats['rollout_seeds']}")
    for metric_name, metric in stats["metrics"].items():
        std_text = "n/a" if metric["std"] is None else f"{metric['std']:.6f}"
        print(f"  {metric_name}: {metric['mean']:.6f} +/- {std_text}")
    for warning in stats_warnings:
        print(f"Warning: {warning}")
    print(f"Wrote {output_dir / 'pass_at_k.json'}")
    print(f"Wrote {output_dir / 'benchmark_mean_std.csv'}")


if __name__ == "__main__":
    main()
