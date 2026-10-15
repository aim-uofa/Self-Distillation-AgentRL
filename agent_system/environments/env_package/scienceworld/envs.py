"""Ray workers for the pinned, unsimplified AgentGym ScienceWorld protocol."""

from importlib.metadata import version
import re
from subprocess import TimeoutExpired

import ray

from agent_system.scienceworld_protocol import SCIENCEWORLD_VERSION, resolve_item


def canonicalize_observation(text):
    """Stabilize flat object lists emitted in JVM-dependent hash/set order.

    Only sort contiguous, single-tab descriptive rows (``a/an ...``). Never
    reorder numbered disambiguation choices, instructions, or nested lists.
    This changes presentation, not the simulator state or available actions.
    """
    text = str(text)
    if "\n\t\t" in text:
        return text
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        end = index
        while end < len(lines) and re.match(r"^\t(?:a|an) ", lines[end], re.IGNORECASE):
            end += 1
        if end > index:
            lines[index:end] = sorted(lines[index:end])
            index = end
        else:
            index += 1
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def scienceworld_projection(text):
    """Accept one text action, never execute model-generated Python/code."""
    match = re.search(r"<action>(.*?)</action>", str(text), re.DOTALL | re.IGNORECASE)
    if match is None:
        match = re.search(r"^Action:\s*(.+)$", str(text), re.MULTILINE | re.IGNORECASE)
    action = match.group(1).strip() if match else ""
    valid = bool(action) and "\n" not in action and "<action>" not in action.lower()
    return (action if valid else ""), valid


class ScienceWorldWorker:
    """One independent JVM; recycle between tasks to bound simulator leakage."""

    def __init__(self, max_steps=20, reward_mode="score"):
        if version("scienceworld") != SCIENCEWORLD_VERSION:
            raise RuntimeError(f"ScienceWorld {SCIENCEWORLD_VERSION} is required for pinned task IDs")
        if max_steps < 1 or reward_mode not in {"score", "success"}:
            raise ValueError("Invalid ScienceWorld horizon or reward mode")
        self.max_steps = int(max_steps)
        self.reward_mode = reward_mode
        self.env = None
        self.done = True

    def reset(self, task):
        from scienceworld import ScienceWorldEnv

        expected = resolve_item(task["item_id"])
        if any(task.get(key) != value for key, value in expected.items()):
            raise ValueError("ScienceWorld task metadata does not match the published item ID")
        self.close()
        # A reset performs one initial look-around, as AgentGym does. The agent
        # gets exactly max_steps further actions; the simulator's > limit check
        # must not prematurely consume the reset observation as an agent turn.
        self.env = ScienceWorldEnv(envStepLimit=self.max_steps + 1)
        self.env.load(expected["task_name"], expected["variation_id"], simplificationStr="")
        obs, _, _, info = self.env.step("look around")
        self.task = expected
        self.task_description = self.env.get_task_description()
        self.steps = 0
        self.done = False
        self.observation = canonicalize_observation(obs)
        self.score = float(info["score"])
        return self.observation, self._info(True)

    def _info(self, valid):
        return {
            **self.task, "task_description": self.task_description,
            "data_source": f"scienceworld/{self.task['category']}",
            "eval_task_id": self.task["item_id"],
            "task_score": self.score, "won": self.score >= 100.0,
            "is_action_valid": bool(valid),
        }

    def step(self, text_action):
        if self.done:
            return self.observation, 0.0, True, self._info(True)
        action, valid = scienceworld_projection(text_action)
        observation, _, completed, info = self.env.step(action)
        self.steps += 1
        self.score = float(info["score"])
        self.observation = canonicalize_observation(observation)
        self.done = bool(completed or self.steps >= self.max_steps)
        valid = valid and "no known action matches" not in self.observation.lower()
        reward = 0.0
        if self.done:
            reward = self.score / 100.0 if self.reward_mode == "score" else float(self.score >= 100.0)
        return self.observation, reward, self.done, self._info(valid)

    def close(self):
        if self.env is not None:
            env, self.env = self.env, None
            process = env._gateway.java_process
            try:
                env.close()
            finally:
                # Upstream close() requests shutdown but does not reap the JVM
                # or close its stdin. Repeated resets otherwise leak handles.
                try:
                    process.wait(timeout=5)
                except TimeoutExpired:
                    process.kill()  # Only this worker's own simulator child.
                    process.wait(timeout=5)
                if process.stdin is not None:
                    process.stdin.close()
                env._obj_tree_tempdir.cleanup()
                self.done = True


class ScienceWorldMultiProcessEnv:
    """Dataset-directed task reset, including short batches and OVCSD replicas.

No independent task RNG is used: the dataloader selects train tasks and repeats
each task for the group. Every validation pass reads the same fixed test file.
    """

    def __init__(self, env_num, group_n, config, resources_per_worker):
        self.capacity = int(env_num) * int(group_n)
        self.workers = []
        self.active_size = 0
        self.max_steps = int(config.max_steps)
        science = config.get("scienceworld", {})
        self.reward_mode = science.get("reward_mode", "score")
        self.worker_cls = ray.remote(**resources_per_worker)(ScienceWorldWorker).options(
            # Scoped to these workers, never exported to other environments.
            runtime_env={"env_vars": {"JAVA_TOOL_OPTIONS": science.get("java_options", "-Xmx512m")}},
        )

    def reset(self, kwargs):
        if kwargs is None or not 0 < len(kwargs) <= self.capacity:
            raise ValueError("ScienceWorld reset requires a nonempty dataset task batch within capacity")
        self.active_size = len(kwargs)
        while len(self.workers) < self.active_size:
            self.workers.append(self.worker_cls.remote(self.max_steps, self.reward_mode))
        results = ray.get([self.workers[i].reset.remote(dict(task)) for i, task in enumerate(kwargs)])
        observations, infos = zip(*results)
        return list(observations), list(infos)

    def step(self, actions):
        if len(actions) != self.active_size:
            raise ValueError("ScienceWorld action batch differs from active task batch")
        return self.step_at(list(range(self.active_size)), actions)

    def step_at(self, indices, actions):
        if len(indices) != len(actions) or len(set(indices)) != len(indices):
            raise ValueError("ScienceWorld selected action indices must be distinct and aligned")
        if not indices or any(i < 0 or i >= self.active_size for i in indices):
            raise IndexError("ScienceWorld environment index out of range")
        results = ray.get([self.workers[i].step.remote(action) for i, action in zip(indices, actions)])
        return tuple(map(list, zip(*results)))

    def close(self):
        workers, self.workers = self.workers, []
        if workers and ray.is_initialized():
            try:
                ray.get([worker.close.remote() for worker in workers], timeout=30)
            finally:
                for worker in workers:
                    ray.kill(worker, no_restart=True)
