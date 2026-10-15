"""Text observations for ScienceWorld, isolated from other benchmark prompts."""

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase


ACTION_HELP = """You are an agent in ScienceWorld. Complete the task by interacting with the environment.
Available action forms:
open OBJ; close OBJ; activate OBJ; deactivate OBJ; connect OBJ to OBJ;
disconnect OBJ; use TOOL on OBJ; look around; look at OBJ; look in OBJ;
read OBJ; move OBJ to CONTAINER; pick up OBJ; put down OBJ;
pour OBJ into CONTAINER; dunk CONTAINER into LIQUID; mix CONTAINER;
go to ROOM; eat OBJ; flush OBJ; focus on OBJ; wait; wait1; task; inventory.
Use object names from observations. If the environment asks you to disambiguate,
answer with the corresponding option number as your next action.
Return exactly one action in <action>...</action>, with no reasoning or extra text.
"""


class ScienceWorldEnvironmentManager(EnvironmentManagerBase):
    def reset(self, kwargs):
        observations, infos = self.envs.reset(kwargs)
        self.tasks = [info["task_description"] for info in infos]
        self.history = [[] for _ in infos]
        self.observations = observations
        return self._observations(list(range(len(infos))), observations), infos

    def _observations(self, indices, observations):
        texts = []
        history_length = int(self.config.env.history_length)
        for i, observation in zip(indices, observations):
            records = self.history[i][-history_length:] if history_length > 0 else []
            history = "\n\n".join(
                f"Observation: {previous}\nAction: {action}" for previous, action in records
            )
            texts.append(
                f"{ACTION_HELP}\nTask: {self.tasks[i]}\n\n"
                f"Recent interaction history:\n{history or '(none)'}\n\n"
                f"Current observation:\n{observation}\n\nYour next action:"
            )
        return {"text": texts, "image": None, "anchor": list(observations)}

    def step(self, actions):
        return self.step_at(list(range(len(self.tasks))), actions)

    def step_at(self, indices, actions):
        observations, rewards, dones, infos = self.envs.step_at(indices, actions)
        for i, action, observation in zip(indices, actions, observations):
            self.history[i].append((self.observations[i], action))
            self.observations[i] = observation
        return (self._observations(indices, observations), np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=bool), infos)

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for step, info in zip(reversed(total_batch_list[batch_idx]), reversed(total_infos[batch_idx])):
            if step["active_masks"]:
                won = float(info["won"])
                success["success_rate"].append(won)
                success[f"scienceworld/{info['category']}_success_rate"].append(won)
                return
