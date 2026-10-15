"""Episode-weighted ScienceWorld metrics, kept separate from legacy benchmarks."""

import numpy as np

from agent_system.scienceworld_protocol import CATEGORY_LABELS, EVAL_CATEGORIES


def scienceworld_metrics(scores, data_sources):
    scores = np.asarray(scores, dtype=np.float64)
    categories = np.asarray([str(source).removeprefix("scienceworld/") for source in data_sources])
    if scores.ndim != 1 or len(scores) != len(categories) or not len(scores):
        raise ValueError("ScienceWorld metrics require aligned, nonempty episode scores")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite ScienceWorld score")
    if set(categories) - set(EVAL_CATEGORIES):
        raise ValueError(f"Unexpected ScienceWorld evaluation categories: {set(categories)}")
    metrics = {"val/success_rate": float(np.mean(scores >= 100.0))}
    for category in (*EVAL_CATEGORIES, "overall"):
        values = scores if category == "overall" else scores[categories == category]
        prefix = f"val/scienceworld/{category}"
        metrics[f"{prefix}/count"] = int(len(values))
        if not len(values):
            continue  # Smoke subsets need not contain every category.
        metrics[f"{prefix}/score"] = float(np.mean(values))
        metrics[f"{prefix}/success_rate"] = float(np.mean(values >= 100.0))
        metrics[f"{prefix}/positive_score_rate"] = float(np.mean(values > 0.0))
        metrics[f"{prefix}/test_score"] = float(np.mean(values) / 100.0)
    return metrics


def print_scienceworld_metrics(metrics):
    pieces = []
    for category in (*EVAL_CATEGORIES, "overall"):
        prefix = f"val/scienceworld/{category}"
        if f"{prefix}/score" in metrics:
            label = CATEGORY_LABELS.get(category, "Overall")
            pieces.append(f"{label}={metrics[prefix + '/score']:.4f} (n={metrics[prefix + '/count']})")
    print("[ScienceWorld] Final native score, episode-weighted: " + ", ".join(pieces))
    print("[ScienceWorld] success_rate means score>=100; positive_score_rate means score>0 (AgentGym Pass@1).")
