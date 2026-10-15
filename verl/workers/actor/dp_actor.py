# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import time
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        ovcsd_top_k=None,
        ovcsd_target_token_ids=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        ovcsd_distribution_mode = ovcsd_top_k is not None or ovcsd_target_token_ids is not None
        if ovcsd_distribution_mode and self.use_fused_kernels:
            raise ValueError("OVCSD top-k distribution loss does not support fused actor kernels")
        if ovcsd_distribution_mode and self.use_ulysses_sp:
            raise ValueError("OVCSD top-k distribution loss currently requires ulysses_sequence_parallel_size=1")
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy or ovcsd_distribution_mode:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                    ovcsd_values_rmpad = None
                    ovcsd_indices_rmpad = None
                    if ovcsd_top_k is not None:
                        top_k = int(ovcsd_top_k)
                        if top_k > logits_rmpad.shape[-1]:
                            raise ValueError("OVCSD top-k exceeds the model vocabulary size")
                        top_values, top_indices = torch.topk(logits_rmpad, k=top_k, dim=-1)
                        log_normalizer = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        ovcsd_values_rmpad = top_values - log_normalizer
                        ovcsd_indices_rmpad = top_indices
                    elif ovcsd_target_token_ids is not None:
                        target_ids_full = torch.zeros(
                            (batch_size, seqlen, ovcsd_target_token_ids.shape[-1]),
                            dtype=ovcsd_target_token_ids.dtype,
                            device=ovcsd_target_token_ids.device,
                        )
                        target_ids_full[:, -response_length - 1 : -1, :] = ovcsd_target_token_ids
                        target_ids_rmpad = index_first_axis(
                            rearrange(target_ids_full, "b s k -> (b s) k"),
                            indices,
                        )
                        selected_logits = torch.gather(
                            logits_rmpad,
                            dim=-1,
                            index=target_ids_rmpad.long(),
                        )
                        log_normalizer = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        ovcsd_values_rmpad = selected_logits - log_normalizer

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                ovcsd_aux = None
                if ovcsd_values_rmpad is not None:
                    full_values = pad_input(
                        hidden_states=ovcsd_values_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    ovcsd_aux = {
                        "topk_log_probs": full_values[:, -response_length - 1 : -1, :]
                    }
                    if ovcsd_indices_rmpad is not None:
                        full_indices = pad_input(
                            hidden_states=ovcsd_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        ovcsd_aux["topk_token_ids"] = full_indices[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                    ovcsd_aux = None
                    if ovcsd_top_k is not None:
                        top_k = int(ovcsd_top_k)
                        if top_k > logits.shape[-1]:
                            raise ValueError("OVCSD top-k exceeds the model vocabulary size")
                        top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
                        log_normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
                        ovcsd_aux = {
                            "topk_log_probs": top_values - log_normalizer,
                            "topk_token_ids": top_indices,
                        }
                    elif ovcsd_target_token_ids is not None:
                        selected_logits = torch.gather(
                            logits,
                            dim=-1,
                            index=ovcsd_target_token_ids.long(),
                        )
                        log_normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
                        ovcsd_aux = {"topk_log_probs": selected_logits - log_normalizer}

            if ovcsd_distribution_mode:
                topk_probs = ovcsd_aux["topk_log_probs"].float().exp().sum(dim=-1)
                tail_log_prob = torch.log1p(-topk_probs.clamp(max=1.0 - 1e-7))
                ovcsd_aux["tail_log_prob"] = tail_log_prob
                return entropy, log_probs, ovcsd_aux
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor ovcsd top-k", logger=logger)
    def compute_topk_log_prob(self, data: DataProto, top_k: int):
        """Compute detached top-k token distributions for OVCSD teacher rows."""
        self.actor_module.eval()
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)
            indices = None

        token_ids_list = []
        log_probs_list = []
        tail_list = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, _, aux = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=False,
                    ovcsd_top_k=top_k,
                )
            token_ids_list.append(aux["topk_token_ids"])
            log_probs_list.append(aux["topk_log_probs"])
            tail_list.append(aux["tail_log_prob"])

        token_ids = torch.concat(token_ids_list, dim=0)
        log_probs = torch.concat(log_probs_list, dim=0)
        tail_log_prob = torch.concat(tail_list, dim=0)
        if use_dynamic_bsz:
            flat_indices = list(itertools.chain.from_iterable(indices))
            reverse = torch.tensor(get_reverse_idx(flat_indices), dtype=torch.long)
            token_ids = token_ids[reverse]
            log_probs = log_probs[reverse]
            tail_log_prob = tail_log_prob[reverse]
        return token_ids, log_probs, tail_log_prob

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        # ADRS may change these coefficients during its teacher-active window.
        # Absent overrides, every existing method retains its configured values.
        entropy_coeff_override = data.meta_info.get("entropy_coeff_override")
        kl_loss_coef_override = data.meta_info.get("kl_loss_coef_override")

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if (
            self.config.get("use_sdl_loss", False)
            or self.config.get("use_sdar_loss", False)
            or self.config.get("use_icsd_loss", False)
            or self.config.get("use_pcsd_loss", False)
        ):
            select_keys.append("teacher_log_probs")
        use_icsd_loss = self.config.get("use_icsd_loss", False)
        if use_icsd_loss:
            select_keys.append("icsd_turn_indices")
        use_ovcsd_loss = self.config.get("use_ovcsd_loss", False)
        if use_ovcsd_loss:
            select_keys.extend(
                [
                    "ovcsd_pg_mask",
                    "ovcsd_regularization_mask",
                    "ovcsd_suffix_mask",
                    "ovcsd_suffix_weights",
                    "ovcsd_teacher_topk_token_ids",
                    "ovcsd_teacher_topk_log_probs",
                    "ovcsd_teacher_tail_log_prob",
                ]
            )
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                # ICSD calibrates across the complete actor mini-batch rather
                # than independently inside each (often size-one) micro-batch.
                # A detached, chunked pre-pass obtains current-policy ratios;
                # the normal pass below remains the only differentiable pass.
                icsd_coefficient_chunks = None
                if use_icsd_loss:
                    from verl.trainer.ppo.icsd_utils import compute_icsd_coefficients

                    micro_batches = list(micro_batches)
                    current_log_probs = []
                    old_log_probs = []
                    teacher_log_probs = []
                    advantages = []
                    response_masks = []
                    turn_indices = []
                    chunk_sizes = []

                    with torch.no_grad():
                        for micro_data in micro_batches:
                            if isinstance(micro_data, DataProto):
                                prepass_data = {
                                    **micro_data.batch.to(get_torch_device().current_device()),
                                    **micro_data.non_tensor_batch,
                                }
                            else:
                                prepass_data = micro_data.to(
                                    get_torch_device().current_device()
                                )

                            response_length = prepass_data["responses"].size(1)
                            if multi_turn:
                                prepass_mask = prepass_data["loss_mask"][
                                    :, -response_length:
                                ]
                            else:
                                prepass_mask = prepass_data["attention_mask"][
                                    :, -response_length:
                                ]
                            _, prepass_log_prob = self._forward_micro_batch(
                                micro_batch=prepass_data,
                                temperature=temperature,
                                calculate_entropy=False,
                            )
                            current_log_probs.append(prepass_log_prob.detach())
                            old_log_probs.append(prepass_data["old_log_probs"].detach())
                            teacher_log_probs.append(
                                prepass_data["teacher_log_probs"].detach()
                            )
                            advantages.append(prepass_data["advantages"].detach())
                            response_masks.append(prepass_mask.detach())
                            turn_indices.append(
                                prepass_data["icsd_turn_indices"].detach()
                            )
                            chunk_sizes.append(prepass_log_prob.shape[0])

                    icsd_coefficients, icsd_metrics = compute_icsd_coefficients(
                        student_log_probs=torch.cat(current_log_probs, dim=0),
                        teacher_log_probs=torch.cat(teacher_log_probs, dim=0),
                        old_log_probs=torch.cat(old_log_probs, dim=0),
                        advantages=torch.cat(advantages, dim=0),
                        response_mask=torch.cat(response_masks, dim=0),
                        turn_indices=torch.cat(turn_indices, dim=0),
                        trust_split=self.config.get("icsd_trust_split", 0.4),
                        influence_split=self.config.get(
                            "icsd_influence_split", 0.5
                        ),
                        scale_floor=self.config.get("icsd_scale_floor", 1e-4),
                        min_group_tokens=self.config.get(
                            "icsd_min_group_tokens", 8
                        ),
                    )
                    icsd_coefficient_chunks = list(
                        torch.split(icsd_coefficients, chunk_sizes, dim=0)
                    )
                    metrics.update(icsd_metrics)

                self.actor_optimizer.zero_grad()

                for micro_batch_index, data in enumerate(micro_batches):
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if use_ovcsd_loss:
                        regularization_mask = data["ovcsd_regularization_mask"]
                    elif multi_turn:
                        regularization_mask = data["loss_mask"][:, -response_length:]
                    else:
                        regularization_mask = attention_mask[:, -response_length:]
                    response_mask = (
                        data["ovcsd_pg_mask"]
                        if use_ovcsd_loss
                        else regularization_mask
                    )

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = (
                        self.config.entropy_coeff
                        if entropy_coeff_override is None else entropy_coeff_override
                    )
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    has_ovcsd_suffix = use_ovcsd_loss and bool(
                        data["ovcsd_suffix_mask"].any().item()
                    )
                    if has_ovcsd_suffix:
                        entropy, log_prob, ovcsd_aux = self._forward_micro_batch(
                            micro_batch=data,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            ovcsd_target_token_ids=data["ovcsd_teacher_topk_token_ids"],
                        )
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                        )
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    pg_loss_coef = self.config.get("pg_loss_coef", 1.0)
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=regularization_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss * pg_loss_coef - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss * pg_loss_coef

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=regularization_mask, loss_agg_mode=loss_agg_mode)

                        effective_kl_coef = (
                            self.config.kl_loss_coef
                            if kl_loss_coef_override is None else kl_loss_coef_override
                        )
                        policy_loss = policy_loss + kl_loss * effective_kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = effective_kl_coef

                    if has_ovcsd_suffix:
                        from verl.trainer.ppo.ovcsd_utils import compute_topk_tail_forward_kl

                        suffix_loss, suffix_metrics = compute_topk_tail_forward_kl(
                            teacher_topk_log_probs=data["ovcsd_teacher_topk_log_probs"],
                            teacher_tail_log_prob=data["ovcsd_teacher_tail_log_prob"],
                            student_topk_log_probs=ovcsd_aux["topk_log_probs"],
                            student_tail_log_prob=ovcsd_aux["tail_log_prob"],
                            suffix_mask=data["ovcsd_suffix_mask"],
                            suffix_weights=data["ovcsd_suffix_weights"],
                        )
                        suffix_coef = self.config.get("ovcsd_suffix_kl_coef", 0.1)
                        policy_loss = policy_loss + suffix_coef * suffix_loss
                        suffix_metrics["ovcsd/suffix_kl_coef"] = float(suffix_coef)
                        append_to_dict(metrics, suffix_metrics)

                    if self.config.get("use_sdl_loss", False):
                        from verl.trainer.ppo.skillsd_utils import compute_sdl_loss
                        teacher_log_probs = data["teacher_log_probs"]
                        sdl_loss = compute_sdl_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            old_log_probs=old_log_prob,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdl_coef = self.config.get("sdl_loss_coef", 0.1)
                        policy_loss = policy_loss + sdl_loss * sdl_coef
                        metrics["actor/sdl_loss"] = sdl_loss.detach().item()
                        metrics["actor/sdl_coef"] = sdl_coef

                    if self.config.get("use_sdar_loss", False):
                        from verl.trainer.ppo.sdar_utils import compute_sdar_loss
                        teacher_log_probs = data["teacher_log_probs"]
                        sdar_loss, sdar_metrics = compute_sdar_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            response_mask=response_mask,
                            gate_beta=self.config.get("sdar_gate_beta", 5.0),
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdar_coef = self.config.get("sdar_loss_coef", 0.1)
                        policy_loss = policy_loss + sdar_loss * sdar_coef
                        metrics.update(sdar_metrics)
                        metrics["sdar/coef"] = sdar_coef

                    if use_icsd_loss:
                        from verl.trainer.ppo.icsd_utils import compute_icsd_loss

                        icsd_loss, icsd_metrics = compute_icsd_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=data["teacher_log_probs"],
                            response_mask=response_mask,
                            coefficients=icsd_coefficient_chunks[
                                micro_batch_index
                            ].to(log_prob.device),
                            loss_agg_mode=loss_agg_mode,
                        )
                        icsd_coef = self.config.get("icsd_loss_coef", 0.01)
                        policy_loss = policy_loss + icsd_loss * icsd_coef
                        metrics.update(icsd_metrics)
                        metrics["icsd/coef"] = icsd_coef

                    if self.config.get("use_pcsd_loss", False):
                        from verl.trainer.ppo.pcsd_utils import compute_pcsd_loss

                        pcsd_loss, pcsd_metrics = compute_pcsd_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=data["teacher_log_probs"],
                            response_mask=response_mask,
                            n_min=self.config.get("pcsd_n_min", 1),
                            n_max=self.config.get("pcsd_n_max", 8),
                            decay=self.config.get("pcsd_decay", 0.8),
                            variance_low=self.config.get("pcsd_variance_low", 0.05),
                            variance_high=self.config.get("pcsd_variance_high", 0.5),
                            trend_gamma=self.config.get("pcsd_trend_gamma", 0.3),
                            gate_beta=self.config.get("pcsd_gate_beta", 5.0),
                            eps=self.config.get("pcsd_eps", 1e-8),
                            loss_agg_mode=loss_agg_mode,
                        )
                        pcsd_coef = self.config.get("pcsd_loss_coef", 0.01)
                        policy_loss = policy_loss + pcsd_loss * pcsd_coef
                        metrics.update(pcsd_metrics)
                        metrics["pcsd/coef"] = pcsd_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
