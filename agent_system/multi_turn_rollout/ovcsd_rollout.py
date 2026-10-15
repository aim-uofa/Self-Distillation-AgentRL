"""Outcome-verified intervention rollouts for OVCSD.

Student groups are always sampled skill-free.  Extra environment replicas are
kept at the same initial task and are touched only for all-failure groups.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence

import numpy as np

from verl import DataProto

from verl.trainer.ppo.ovcsd_utils import (
    PrefixCandidate,
    build_prefix_candidates,
    canonicalize_text,
    find_first_aligned_divergence,
    local_comparative_advantages,
    select_deepest_candidate,
    select_nearest_ancestor,
    same_prompt_tokens,
    scienceworld_observations_equal,
    state_fingerprint,
)

from .privileged_rollout import (
    build_no_skill_prefix_turn,
    IndexedEnvironmentView,
    SkillTeacherTrajectoryCollector,
    _assign_observations,
    _select_observations,
)
from .rollout_loop import TrajectoryCollector
from .utils import to_list_of_dict


def _scienceworld_task_identity(info):
    return tuple(info[key] for key in ("item_id", "task_name", "variation_id"))


def _scienceworld_prompt_matches(obs, index, reference):
    return (
        scienceworld_observations_equal(obs["text"][index], reference["ovcsd_plain_obs_text"])
        and scienceworld_observations_equal(obs["anchor"][index], reference["anchor_obs"])
    )


class OVCSDScienceWorldStudentCollector(TrajectoryCollector):
    """Record the unmodified prompt locally; generation inputs stay unchanged."""

    def preprocess_batch(self, gen_batch, obs):
        output = super().preprocess_batch(gen_batch, obs)
        output.non_tensor_batch["ovcsd_plain_obs_text"] = np.asarray(list(obs["text"]), dtype=object)
        return output


class OVCSDScienceWorldTeacherView(IndexedEnvironmentView):
    """Reuse a student's rendering only while replayed states remain aligned.

    Never mutate the underlying manager's observations/history. After the first
    literal action divergence, the teacher continues on its own observations.
    A state mismatch on an otherwise identical prefix invalidates the attempt.
    """

    def __init__(self, *args, references, reference_actions, reference_infos, **kwargs):
        super().__init__(*args, **kwargs)
        self.references = references
        self.reference_actions = reference_actions
        self.reference_infos = reference_infos
        self.aligned = np.ones(len(self.indices), dtype=bool)
        self.rejected = np.zeros(len(self.indices), dtype=bool)
        self.rebind_count = 0

    def _align(self, obs, infos):
        result = dict(obs)
        result["text"] = list(obs["text"])
        for index, reference in enumerate(self.references):
            depth = int(self.total_steps[index])
            if self.done[index] or depth >= len(reference):
                self.aligned[index] = False
            if not self.aligned[index]:
                continue
            expected_score = (
                reference[depth - 1]["scienceworld_score"] if depth
                else self.reference_infos[index]["task_score"]
            )
            if (
                _scienceworld_task_identity(infos[index])
                != _scienceworld_task_identity(self.reference_infos[index])
                or float(infos[index]["task_score"]) != float(expected_score)
                or not _scienceworld_prompt_matches(obs, index, reference[depth])
            ):
                self.aligned[index] = False
                self.rejected[index] = True
                continue
            plain_text = reference[depth]["ovcsd_plain_obs_text"]
            self.rebind_count += int(result["text"][index] != plain_text)
            result["text"][index] = plain_text
        result["ovcsd_prompt_aligned"] = self.aligned.copy()
        return result

    def reset(self, kwargs=None):
        obs, infos = super().reset(kwargs)
        return self._align(obs, infos), infos

    def step(self, text_actions):
        for index, action in enumerate(text_actions):
            depth = int(self.total_steps[index])
            if self.aligned[index] and (
                depth >= len(self.reference_actions[index])
                or action != self.reference_actions[index][depth]
            ):
                self.aligned[index] = False
        obs, rewards, dones, infos = super().step(text_actions)
        for index, reference in enumerate(self.references):
            if not self.aligned[index]:
                continue
            depth = int(self.total_steps[index])
            previous = reference[depth - 1]
            if (
                _scienceworld_task_identity(infos[index])
                != _scienceworld_task_identity(self.reference_infos[index])
                or float(infos[index]["task_score"]) != float(previous["scienceworld_score"])
                or bool(infos[index]["is_action_valid"]) != bool(previous["is_action_valid"])
                or bool(dones[index]) != (depth >= len(reference))
            ):
                self.aligned[index] = False
                self.rejected[index] = True
        return self._align(obs, infos), rewards, dones, infos


class OVCSDTeacherTrajectoryCollector(SkillTeacherTrajectoryCollector):
    """Skill teacher that also retains the corresponding skill-free prompt."""

    def preprocess_batch(self, gen_batch: DataProto, obs: Dict) -> DataProto:
        output = super().preprocess_batch(gen_batch, obs)
        plain_text = obs.get("text")
        if plain_text is not None:
            output.non_tensor_batch["ovcsd_plain_obs_text"] = np.asarray(
                list(plain_text), dtype=object
            )
        if "ovcsd_prompt_aligned" in obs:
            output.non_tensor_batch["ovcsd_prompt_aligned"] = np.asarray(
                obs["ovcsd_prompt_aligned"], dtype=bool
            )
        return output


class OVCSDTrajectoryCollector:
    """Collect base GRPO rollouts and sparse verified teacher continuations."""

    def __init__(self, config, tokenizer, processor=None, skill_provider=None):
        ovcsd_cfg = config.algorithm.get("ovcsd", {})
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.group_size = int(config.env.rollout.n)
        self.teacher_attempt_limit = int(ovcsd_cfg.get("teacher_attempt_limit", 2))
        self.reward_std_epsilon = float(ovcsd_cfg.get("reward_std_epsilon", 1e-6))
        if self.group_size <= 1:
            raise ValueError("OVCSD requires env.rollout.n > 1")
        if self.teacher_attempt_limit <= 0:
            raise ValueError("OVCSD teacher_attempt_limit must be positive")
        if config.algorithm.filter_groups.enable:
            raise ValueError("OVCSD cannot be combined with algorithm.filter_groups")
        env_name = str(config.env.env_name).lower()
        if not any(name in env_name for name in ("alfworld", "webshop", "search", "scienceworld")):
            raise ValueError("OVCSD supports ALFWorld, WebShop, Search-QA and ScienceWorld")
        if skill_provider is None:
            raise ValueError("OVCSD requires a SkillProvider")

        self.physical_group_size = self.group_size + self.teacher_attempt_limit
        self.scienceworld_prompt_alignment = env_name == "scienceworld"
        self.student_collector = TrajectoryCollector(config, tokenizer, processor)
        teacher_config = copy.deepcopy(config)
        teacher_config.data.truncation = "left"
        self.teacher_collector = OVCSDTeacherTrajectoryCollector(
            teacher_config,
            tokenizer,
            processor,
            skill_provider=skill_provider,
        )
        self._pending_interventions = None

    def pop_pending_interventions(self):
        """Transfer this step's non-serializable supervision to the trainer."""
        payload = self._pending_interventions
        self._pending_interventions = None
        if payload is None:
            return {"records": [], "failed_traj_uids": [], "metrics": {}}
        return payload

    def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
        if not is_train:
            return self.student_collector.multi_turn_loop(
                gen_batch,
                actor_rollout_wg,
                envs,
                is_train=False,
            )
        return self._train_loop(gen_batch, actor_rollout_wg, envs)

    def _train_loop(self, gen_batch, actor_rollout_wg, envs):
        task_count = len(gen_batch)
        physical_batch = gen_batch.repeat(
            repeat_times=self.physical_group_size,
            interleave=True,
        )
        env_kwargs = physical_batch.non_tensor_batch.pop("env_kwargs", None)
        all_obs, all_infos = envs.reset(kwargs=env_kwargs)
        self._validate_same_task_groups(all_obs, all_infos, task_count)

        student_indices = [
            task_idx * self.physical_group_size + rollout_idx
            for task_idx in range(task_count)
            for rollout_idx in range(self.group_size)
        ]
        student_batch = gen_batch.repeat(
            repeat_times=self.group_size,
            interleave=True,
        )
        student_view = IndexedEnvironmentView(
            envs,
            student_indices,
            _select_observations(all_obs, student_indices),
            [all_infos[index] for index in student_indices],
            max_total_steps=int(self.config.env.max_steps),
        )
        (
            trajectories,
            episode_rewards,
            episode_lengths,
            success,
            traj_uid,
            tool_callings,
        ) = (
            OVCSDScienceWorldStudentCollector(self.config, self.tokenizer, self.processor)
            if self.scienceworld_prompt_alignment else self.student_collector
        ).vanilla_multi_turn_loop(
            student_batch,
            actor_rollout_wg,
            student_view,
        )

        records, failed_traj_uids, metrics = self._collect_interventions(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            all_obs=all_obs,
            all_infos=all_infos,
            trajectories=trajectories,
            episode_rewards=np.asarray(episode_rewards, dtype=np.float32),
            success_values=np.asarray(success["success_rate"], dtype=np.float32),
            traj_uid=traj_uid,
            task_count=task_count,
        )

        if self.scienceworld_prompt_alignment:
            # These snapshots are only needed for intervention matching. Keep
            # the ordinary training batch schema identical to the base path.
            for trajectory in trajectories:
                for turn in trajectory:
                    turn.pop("ovcsd_plain_obs_text", None)
        output = self.student_collector.gather_rollout_data(
            total_batch_list=trajectories,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success,
            traj_uid=traj_uid,
            tool_callings=tool_callings,
        )
        self._pending_interventions = {
            "records": records,
            "failed_traj_uids": list(failed_traj_uids),
            "metrics": metrics,
        }
        return output

    def _collect_interventions(
        self,
        gen_batch,
        actor_rollout_wg,
        envs,
        all_obs,
        all_infos,
        trajectories,
        episode_rewards,
        success_values,
        traj_uid,
        task_count,
    ):
        task_states = {}
        failed_traj_uids = set()
        available_depths = []
        all_failure_groups = 0
        student_steps = sum(
            bool(turn["active_masks"])
            for trajectory in trajectories
            for turn in trajectory
        )

        for task_idx in range(task_count):
            start = task_idx * self.group_size
            stop = start + self.group_size
            group_success = success_values[start:stop]
            group_rewards = episode_rewards[start:stop]
            if group_success.size != self.group_size:
                continue
            if group_success.max(initial=0.0) > 0.5:
                continue
            if float(group_rewards.std()) > self.reward_std_epsilon:
                continue

            group_trajectories = [
                [turn for turn in trajectories[index] if bool(turn["active_masks"])]
                for index in range(start, stop)
            ]
            group_actions = [self._decode_actions(item) for item in group_trajectories]
            candidates = build_prefix_candidates(
                group_trajectories, group_actions,
                strict_prompt=self.scienceworld_prompt_alignment,
            )
            if not candidates:
                continue

            all_failure_groups += 1
            failed_traj_uids.update(str(value) for value in traj_uid[start:stop])
            uncovered = set(range(self.group_size))
            candidate = select_deepest_candidate(candidates, uncovered, set())
            if candidate is None:
                continue
            available_depths.append(candidate.depth)
            task_states[task_idx] = {
                "trajectories": group_trajectories,
                "actions": group_actions,
                "candidates": candidates,
                "uncovered": uncovered,
                "attempted": set(),
                "candidate": candidate,
                "global_start": start,
            }

        records = []
        used_depths = []
        replay_steps = 0
        teacher_steps = 0
        teacher_attempts = 0
        verified_attempts = 0
        replay_rejects = 0
        alignment_rejects = 0
        local_group_count = 0
        prompt_rebinds = 0
        prompt_state_rejects = 0

        for attempt_idx in range(self.teacher_attempt_limit):
            active_tasks = [
                task_idx
                for task_idx, state in task_states.items()
                if state["candidate"] is not None and state["uncovered"]
            ]
            if not active_tasks:
                break

            physical_indices = [
                task_idx * self.physical_group_size + self.group_size + attempt_idx
                for task_idx in active_tasks
            ]
            candidates = [task_states[task_idx]["candidate"] for task_idx in active_tasks]
            references = reference_infos = None
            if self.scienceworld_prompt_alignment:
                references = [
                    task_states[task_idx]["trajectories"][candidate.representative]
                    for task_idx, candidate in zip(active_tasks, candidates)
                ]
                reference_infos = [
                    all_infos[task_idx * self.physical_group_size + candidate.representative]
                    for task_idx, candidate in zip(active_tasks, candidates)
                ]
            teacher_attempts += len(active_tasks)
            replay_ok, consumed_replay = self._replay_candidates(
                envs,
                all_obs,
                all_infos,
                physical_indices,
                candidates,
                references=references,
                reference_infos=reference_infos,
            )
            replay_steps += consumed_replay

            valid_local_indices = [index for index, ok in enumerate(replay_ok) if ok]
            replay_rejects += len(active_tasks) - len(valid_local_indices)
            teacher_results = {}
            if valid_local_indices:
                valid_tasks = [active_tasks[index] for index in valid_local_indices]
                valid_physical = [physical_indices[index] for index in valid_local_indices]
                valid_candidates = [candidates[index] for index in valid_local_indices]
                view_type = IndexedEnvironmentView
                view_kwargs = {}
                if self.scienceworld_prompt_alignment:
                    view_type = OVCSDScienceWorldTeacherView
                    view_kwargs = {
                        "references": [references[index] for index in valid_local_indices],
                        "reference_infos": [reference_infos[index] for index in valid_local_indices],
                        "reference_actions": [
                            task_states[task_idx]["actions"][candidate.representative]
                            for task_idx, candidate in zip(valid_tasks, valid_candidates)
                        ],
                    }
                teacher_view = view_type(
                    envs,
                    valid_physical,
                    _select_observations(all_obs, valid_physical),
                    [all_infos[index] for index in valid_physical],
                    prefix_lengths=[candidate.depth for candidate in valid_candidates],
                    max_total_steps=int(self.config.env.max_steps),
                    **view_kwargs,
                )
                teacher_batch = gen_batch.select_idxs(np.asarray(valid_tasks, dtype=np.int64))
                (
                    teacher_trajectories,
                    _,
                    teacher_lengths,
                    teacher_success,
                    _,
                    _,
                ) = self.teacher_collector.vanilla_multi_turn_loop(
                    teacher_batch,
                    actor_rollout_wg,
                    teacher_view,
                )
                teacher_steps += int(np.asarray(teacher_lengths).sum())
                if self.scienceworld_prompt_alignment:
                    prompt_rebinds += teacher_view.rebind_count
                    prompt_state_rejects += int(teacher_view.rejected.sum())
                for local_idx, task_idx in enumerate(valid_tasks):
                    teacher_results[task_idx] = (
                        teacher_trajectories[local_idx],
                        float(teacher_success["success_rate"][local_idx]) > 0.5
                        and not (self.scienceworld_prompt_alignment and teacher_view.rejected[local_idx]),
                    )

            for task_idx in active_tasks:
                state = task_states[task_idx]
                candidate = state["candidate"]
                state["attempted"].add(candidate.key)
                used_depths.append(candidate.depth)
                teacher_trajectory, verified = teacher_results.get(task_idx, ([], False))

                if verified:
                    verified_attempts += 1
                    new_records, rejected, groups = self._build_supervision_records(
                        task_idx=task_idx,
                        gen_batch=gen_batch,
                        candidate=candidate,
                        teacher_trajectory=teacher_trajectory,
                        task_state=state,
                        traj_uid=traj_uid,
                    )
                    records.extend(new_records)
                    alignment_rejects += rejected
                    local_group_count += groups
                    state["uncovered"].difference_update(candidate.members)
                    state["candidate"] = select_deepest_candidate(
                        state["candidates"],
                        state["uncovered"],
                        state["attempted"],
                    )
                else:
                    ancestor = select_nearest_ancestor(
                        candidate,
                        state["candidates"],
                        state["uncovered"],
                        state["attempted"],
                    )
                    if ancestor is None:
                        ancestor = select_deepest_candidate(
                            state["candidates"],
                            state["uncovered"],
                            state["attempted"],
                        )
                    state["candidate"] = ancestor

        suffix_rows = sum(record["kind"] == "suffix" for record in records)
        metrics = {
            "ovcsd/all_failure_group_count": float(all_failure_groups),
            "ovcsd/all_failure_group_rate": self._safe_ratio(all_failure_groups, task_count),
            "ovcsd/intercepted_group_count": float(len(task_states)),
            "ovcsd/teacher_attempt_count": float(teacher_attempts),
            "ovcsd/teacher_verified_count": float(verified_attempts),
            "ovcsd/teacher_success_rate": self._safe_ratio(verified_attempts, teacher_attempts),
            "ovcsd/replay_consistency_reject_count": float(replay_rejects),
            "ovcsd/alignment_reject_count": float(alignment_rejects),
            "ovcsd/candidate_depth_available_mean": self._safe_mean(available_depths),
            "ovcsd/candidate_depth_used_mean": self._safe_mean(used_depths),
            "ovcsd/local_group_count": float(local_group_count),
            "ovcsd/suffix_row_count": float(suffix_rows),
            "ovcsd/replay_action_steps": float(replay_steps),
            "ovcsd/teacher_action_steps": float(teacher_steps),
            "ovcsd/student_action_steps": float(student_steps),
            "ovcsd/intervention_step_overhead": self._safe_ratio(
                replay_steps + teacher_steps,
                student_steps,
            ),
        }
        if self.scienceworld_prompt_alignment:
            metrics.update({
                "ovcsd/teacher_prompt_rebind_count": float(prompt_rebinds),
                "ovcsd/teacher_prompt_state_reject_count": float(prompt_state_rejects),
                "ovcsd/strict_prompt_student_pair_count": float(
                    sum(record["kind"] == "student_divergence" for record in records)
                ),
            })
        return records, failed_traj_uids, metrics

    def _replay_candidates(
        self,
        envs,
        all_obs,
        all_infos,
        physical_indices: Sequence[int],
        candidates: Sequence[PrefixCandidate],
        *,
        references=None,
        reference_infos=None,
    ):
        ok = np.ones(len(candidates), dtype=bool)
        replay_steps = 0
        if self.scienceworld_prompt_alignment:
            for index, env_idx in enumerate(physical_indices):
                ok[index] = (
                    _scienceworld_task_identity(all_infos[env_idx])
                    == _scienceworld_task_identity(reference_infos[index])
                    and float(all_infos[env_idx]["task_score"])
                    == float(reference_infos[index]["task_score"])
                    and _scienceworld_prompt_matches(all_obs, env_idx, references[index][0])
                )
        max_depth = max((candidate.depth for candidate in candidates), default=0)
        for depth in range(max_depth):
            local_indices = [
                index
                for index, candidate in enumerate(candidates)
                if ok[index] and depth < candidate.depth
            ]
            if not local_indices:
                continue
            selected_physical = [physical_indices[index] for index in local_indices]
            actions = [candidates[index].prefix_actions[depth] for index in local_indices]
            next_obs, _, dones, infos = envs.step_at(selected_physical, actions)
            replay_steps += len(selected_physical)
            _assign_observations(all_obs, selected_physical, next_obs)
            for result_idx, candidate_idx in enumerate(local_indices):
                env_idx = selected_physical[result_idx]
                all_infos[env_idx] = infos[result_idx]
                if bool(np.asarray(dones[result_idx]).squeeze()):
                    ok[candidate_idx] = False
                if self.scienceworld_prompt_alignment:
                    reference = references[candidate_idx]
                    info = infos[result_idx]
                    if (
                        _scienceworld_task_identity(info)
                        != _scienceworld_task_identity(reference_infos[candidate_idx])
                        or float(info["task_score"]) != float(reference[depth]["scienceworld_score"])
                        or bool(info["is_action_valid"]) != bool(reference[depth]["is_action_valid"])
                        or not _scienceworld_prompt_matches(all_obs, env_idx, reference[depth + 1])
                    ):
                        ok[candidate_idx] = False

        anchors = all_obs.get("anchor")
        if anchors is None:
            anchors = all_obs.get("text")
        for local_idx, (env_idx, candidate) in enumerate(zip(physical_indices, candidates)):
            if (not self.scienceworld_prompt_alignment and ok[local_idx]
                    and state_fingerprint(anchors[env_idx]) != candidate.state_fingerprint):
                ok[local_idx] = False
        return ok, replay_steps

    def _build_supervision_records(
        self,
        task_idx,
        gen_batch,
        candidate,
        teacher_trajectory,
        task_state,
        traj_uid,
    ):
        active_teacher = [
            turn for turn in teacher_trajectory if bool(turn["active_masks"])
        ]
        teacher_actions = self._decode_actions(active_teacher)
        grouped = {}
        rejected = 0
        task_batch = gen_batch.select_idxs(np.asarray([task_idx], dtype=np.int64))
        no_skill_rows = None
        if self.scienceworld_prompt_alignment:
            no_skill_rows = [self._build_no_skill_teacher_row(task_batch, turn) for turn in active_teacher]
        for member in candidate.members:
            if member not in task_state["uncovered"]:
                continue
            divergence = find_first_aligned_divergence(
                task_state["trajectories"][member],
                task_state["actions"][member],
                active_teacher,
                teacher_actions,
                candidate.depth,
                strict_prompt=self.scienceworld_prompt_alignment,
            )
            if divergence is None:
                rejected += 1
                continue
            student_turn, teacher_turn = divergence
            if self.scienceworld_prompt_alignment and not same_prompt_tokens(
                task_state["trajectories"][member][student_turn], no_skill_rows[teacher_turn]
            ):
                rejected += 1
                continue
            key = (
                teacher_turn,
                state_fingerprint(active_teacher[teacher_turn].get("anchor_obs")),
                canonicalize_text(teacher_actions[teacher_turn]),
            )
            grouped.setdefault(key, []).append((member, student_turn, teacher_turn))

        records = []
        if no_skill_rows is None:
            no_skill_rows = [
                self._build_no_skill_teacher_row(task_batch, turn)
                for turn in active_teacher
            ]
        group_count = 0
        for group_index, matches in enumerate(grouped.values()):
            group_count += 1
            teacher_turn = matches[0][2]
            teacher_advantage, student_advantage = local_comparative_advantages(len(matches))
            pair_id = f"task{task_idx}-depth{candidate.depth}-div{teacher_turn}-g{group_index}"
            for member, student_turn, _ in matches:
                global_idx = task_state["global_start"] + member
                records.append(
                    {
                        "kind": "student_divergence",
                        "traj_uid": str(traj_uid[global_idx]),
                        "turn_step": int(student_turn),
                        "advantage": float(student_advantage),
                        "pair_id": pair_id,
                    }
                )

            records.append(
                {
                    "kind": "teacher_divergence",
                    "no_skill_row": no_skill_rows[teacher_turn],
                    "privileged_row": active_teacher[teacher_turn],
                    "advantage": float(teacher_advantage),
                    "pair_id": pair_id,
                }
            )
            for suffix_turn in range(teacher_turn + 1, len(active_teacher)):
                records.append(
                    {
                        "kind": "suffix",
                        "no_skill_row": no_skill_rows[suffix_turn],
                        "privileged_row": active_teacher[suffix_turn],
                        "advantage": 0.0,
                        "pair_id": pair_id,
                    }
                )
        return records, rejected, group_count

    def _build_no_skill_teacher_row(self, task_batch, teacher_turn):
        plain_text = teacher_turn.get("ovcsd_plain_obs_text")
        if plain_text is None:
            raise RuntimeError("OVCSD teacher row is missing its skill-free observation")
        obs = {
            "text": [str(plain_text)],
            "image": None,
            "anchor": [teacher_turn.get("anchor_obs")],
        }
        gamefile = teacher_turn.get("gamefile")
        if gamefile is not None:
            obs["gamefile"] = [gamefile]
        prompt_batch = self.student_collector.preprocess_batch(task_batch, obs)
        prompt_row = to_list_of_dict(prompt_batch)[0]
        return build_no_skill_prefix_turn(
            teacher_turn,
            prompt_row,
            reward=0.0,
            info={"is_action_valid": True},
        )

    def _decode_actions(self, trajectory: Sequence[Dict]) -> List[str]:
        return [
            self.tokenizer.decode(turn["responses"], skip_special_tokens=True)
            for turn in trajectory
            if bool(turn["active_masks"])
        ]

    def _validate_same_task_groups(self, all_obs, all_infos, task_count):
        obs_texts = all_obs.get("text")
        if obs_texts is None:
            obs_texts = all_obs.get("anchor")
        expected = task_count * self.physical_group_size
        if len(obs_texts) != expected:
            raise ValueError(f"OVCSD environment size mismatch: expected={expected}, actual={len(obs_texts)}")
        for task_idx in range(task_count):
            start = task_idx * self.physical_group_size
            if self.scienceworld_prompt_alignment:
                reference_info = all_infos[start]
                anchors = all_obs.get("anchor")
                for env_idx in range(start, start + self.physical_group_size):
                    info = all_infos[env_idx]
                    if (
                        _scienceworld_task_identity(info) != _scienceworld_task_identity(reference_info)
                        or float(info["task_score"]) != float(reference_info["task_score"])
                        or bool(info["won"]) != bool(reference_info["won"])
                        or not scienceworld_observations_equal(obs_texts[env_idx], obs_texts[start])
                        or (anchors is not None and not scienceworld_observations_equal(
                            anchors[env_idx], anchors[start]
                        ))
                    ):
                        raise RuntimeError(f"OVCSD replicas disagree for task index {task_idx}")
                continue
            fingerprints = []
            for offset in range(self.physical_group_size):
                env_idx = start + offset
                gamefile = all_infos[env_idx].get("extra.gamefile")
                fingerprints.append(str(gamefile) if gamefile else state_fingerprint(obs_texts[env_idx]))
            if len(set(fingerprints)) != 1:
                raise RuntimeError(f"OVCSD replicas disagree for task index {task_idx}")

    @staticmethod
    def _safe_mean(values) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _safe_ratio(numerator, denominator) -> float:
        return float(numerator / denominator) if denominator else 0.0
