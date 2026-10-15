"""Pinned AgentGym ScienceWorld task IDs and evaluation protocol (no simulator import).

AgentGym enumerates ScienceWorld 1.2.3 tasks in ID2TASK order, omitting seven
task families. Never sort names or substitute ScienceWorld's train/test split:
that silently changes the meaning of the published ``sciworld_N`` IDs.
"""

PROTOCOL_VERSION = 1
SCIENCEWORLD_VERSION = "1.2.3"
DATA_REPO = "AgentGym/AgentGym-RL-Data-ID"
DATA_REVISION = "99d0b7923bf126d9c6cdea5f362d2d41ccb5d493"
EVALUATION_PROTOCOL = {
    "version": PROTOCOL_VERSION, "source_repo": DATA_REPO,
    "source_revision": DATA_REVISION, "simulator_version": SCIENCEWORLD_VERSION,
    "simplifications": "",
    "observation_format": "canonical-flat-object-lists-v1",
}
SOURCE_FILES = {
    "train": ("train/sciworld_train.json", 2120,
              "9b7ce7f214c738de96d0048919e08e8241c5c2bca48843c53dae5098b938e28d"),
    "test": ("eval/sciworld_test.json", 200,
             "78fbe5ed7ba545f1a2ee08397a5e740f622872b6a3ee70315d349c4287675c79"),
}
EVAL_CATEGORIES = ("measure", "test_conductivity", "find", "chemistry_mix", "lifespan")
EVAL_COUNTS = dict(zip(EVAL_CATEGORIES, (53, 83, 44, 5, 15)))
CATEGORY_LABELS = {
    "measure": "Measure", "test_conductivity": "Test-Cond.", "find": "Find",
    "chemistry_mix": "Chem-Mix", "lifespan": "Lifespan",
}
# Verified against the packaged 1.2.3 simulator and AgentGym commit
# 640f8bca6901a6a6d540ff61522b813988da47c4, agentenv_sciworld/environment.py.
TASK_CATALOG = (
    ("1-1", "boil", 30, "state_change"),
    ("1-2", "melt", 30, "state_change"),
    ("1-3", "freeze", 30, "state_change"),
    ("1-4", "change-the-state-of-matter-of", 30, "state_change"),
    ("2-1", "use-thermometer", 540, "measure"),
    ("2-2", "measure-melting-point-known-substance", 436, "measure"),
    ("2-3", "measure-melting-point-unknown-substance", 300, "measure"),
    ("3-1", "power-component", 20, "power"),
    ("3-2", "power-component-renewable-vs-nonrenewable-energy", 20, "power"),
    ("3-3", "test-conductivity", 900, "test_conductivity"),
    ("3-4", "test-conductivity-of-unknown-substances", 600, "test_conductivity"),
    ("4-1", "find-living-thing", 300, "find"),
    ("4-2", "find-non-living-thing", 300, "find"),
    ("4-3", "find-plant", 300, "find"),
    ("4-4", "find-animal", 300, "find"),
    ("6-1", "chemistry-mix", 32, "chemistry_mix"),
    ("6-2", "chemistry-mix-paint-secondary-color", 36, "chemistry_mix"),
    ("6-3", "chemistry-mix-paint-tertiary-color", 36, "chemistry_mix"),
    ("7-1", "lifespan-longest-lived", 125, "lifespan"),
    ("7-2", "lifespan-shortest-lived", 125, "lifespan"),
    ("7-3", "lifespan-longest-lived-then-shortest-lived", 125, "lifespan"),
    ("8-1", "identify-life-stages-1", 14, "life_stages"),
    ("8-2", "identify-life-stages-2", 10, "life_stages"),
)


def resolve_item(item_id):
    """Resolve one released ID into an immutable simulator task/variation."""
    if not isinstance(item_id, str) or not item_id.startswith("sciworld_"):
        raise ValueError(f"Invalid AgentGym ScienceWorld ID: {item_id!r}")
    suffix = item_id.removeprefix("sciworld_")
    if not suffix.isdecimal():
        raise ValueError(f"Invalid AgentGym ScienceWorld ID: {item_id!r}")
    index = int(suffix)
    if suffix != str(index):
        raise ValueError(f"Noncanonical ScienceWorld ID: {item_id!r}")
    offset = 0
    for task_type, name, count, category in TASK_CATALOG:
        if offset <= index < offset + count:
            return {"item_id": item_id, "task_type": task_type, "task_name": name,
                    "variation_id": index - offset, "category": category}
        offset += count
    raise ValueError(f"ScienceWorld ID outside the pinned {offset}-task catalog: {item_id}")


def verify_simulator_catalog(env):
    """Fail loudly on an incompatible simulator instead of remapping task IDs."""
    excluded = {"5-1", "5-2", "9-1", "9-2", "9-3", "10-1", "10-2"}
    actual = [(key, name, env.get_max_variations(name))
              for key, name in env.tasks.items() if key not in excluded]
    expected = [entry[:3] for entry in TASK_CATALOG]
    if actual != expected:
        raise RuntimeError("ScienceWorld simulator catalog differs from pinned AgentGym IDs")
