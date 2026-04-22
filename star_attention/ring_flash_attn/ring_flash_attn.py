# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward
from .utils import RingComm, update_out_and_lse, get_default_args


def _ring_flash_attn_forward(
    process_group,
    q,
    k,
    v,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    return_softmax,
    num_ring_steps,
):
    # Single-GPU / non-distributed: skip RingComm (which requires a live
    # process group). With world_size==1, num_ring_steps==0 and the loop
    # executes exactly once with no send/recv, so `comm` is never used.
    single_gpu = not dist.is_initialized() or dist.get_world_size() == 1
    if single_gpu:
        comm = None
        comm_rank = 0
        if num_ring_steps < 0:
            num_ring_steps = 0
    else:
        comm = RingComm(process_group)
        comm_rank = comm.rank
        if num_ring_steps < 0:
            num_ring_steps = comm.world_size - 1
        assert num_ring_steps < comm.world_size

    out = None
    lse = None

    next_k, next_v = None, None

    for step in range(num_ring_steps + 1):
        if step != num_ring_steps:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if not causal or step <= comm_rank:
            params = get_default_args(_flash_attn_forward).copy()
            params.update(
                {
                    'q': q,
                    'k': k,
                    'v': v,
                    'dropout_p': dropout_p,
                    'softmax_scale': softmax_scale,
                    'causal': causal and step == 0,
                    'window_size': window_size,
                    'softcap': softcap,
                    'alibi_slopes': alibi_slopes,
                    'return_softmax': return_softmax,
                }
            )
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(**params)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step != num_ring_steps:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


class RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        return_softmax,
        num_ring_steps,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = _ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            num_ring_steps=num_ring_steps,
        )
        # this should be out_padded
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)


def ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    return_attn_probs=False,
    num_ring_steps=-1,
    group=None,
):
    return RingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        return_attn_probs,
        num_ring_steps,
        group,
    )
