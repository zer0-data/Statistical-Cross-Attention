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

import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer


def compute_block_summaries(
    all_block_tokens: List[torch.Tensor],
    all_block_positions: List[torch.Tensor],
    k: int = 32,
    method: str = "tfidf",
) -> List[Dict[str, torch.Tensor]]:
    """Compute lightweight statistical summaries for each block.

    For each block, selects the top-k most informative tokens using
    TF-IDF scoring (or random fallback). Each summary retains original
    position IDs for RoPE compatibility.

    Args:
        all_block_tokens: list of 1D tensors, one per block (token IDs).
        all_block_positions: list of 1D tensors, one per block (position IDs).
        k: number of summary tokens per block.
        method: "tfidf" or "random".

    Returns:
        List of dicts with keys 'token_ids' and 'pos_ids', each a 1D tensor.
    """
    num_blocks = len(all_block_tokens)

    if method == "random":
        summaries = []
        for blk_tok, blk_pos in zip(all_block_tokens, all_block_positions):
            actual_k = min(k, blk_tok.shape[0])
            idx = torch.randperm(blk_tok.shape[0])[:actual_k].sort().values
            summaries.append({
                'token_ids': blk_tok[idx],
                'pos_ids': blk_pos[idx],
            })
        return summaries

    # ---------- TF-IDF scoring ----------
    # Document frequency: how many blocks contain each token
    df: Counter = Counter()
    for blk_tok in all_block_tokens:
        unique_tokens = blk_tok.unique().tolist()
        df.update(unique_tokens)

    summaries = []
    for blk_tok, blk_pos in zip(all_block_tokens, all_block_positions):
        tokens_list = blk_tok.tolist()
        block_len = len(tokens_list)
        # Term frequency within this block
        tf: Counter = Counter(tokens_list)

        scores = torch.zeros(block_len, dtype=torch.float32)
        for i, t in enumerate(tokens_list):
            tf_val = tf[t] / block_len
            idf_val = math.log(num_blocks / max(df[t], 1))
            scores[i] = tf_val * idf_val

        # --- Diminishing-returns penalty for repeated tokens ----------
        # The n-th occurrence of the same token has its score divided by
        # log2(1 + n), so: 1st → full, 2nd → /log2(3)≈0.63×,
        # 3rd → /log2(4)=0.50×, etc.  This prevents blind duplication
        # from dominating topk while still allowing positionally-
        # important repeats (e.g. entity mentions in multihop chains)
        # to survive when their base TF-IDF score is high enough.
        occurrence_count: Counter = Counter()
        for i, t in enumerate(tokens_list):
            occurrence_count[t] += 1
            n = occurrence_count[t]
            if n > 1:
                scores[i] = scores[i] / math.log2(1 + n)

        actual_k = min(k, block_len)
        _, top_indices = torch.topk(scores, actual_k)
        top_indices = top_indices.sort().values  # maintain positional order

        summaries.append({
            'token_ids': blk_tok[top_indices],
            'pos_ids': blk_pos[top_indices],
        })

    return summaries


class DistributedInferenceBaseModel:
    def __init__(
        self,
        path: str,
        max_new_tokens: int,
        stop_words: Optional[List[str]] = None,
        block_size: int = -1,
        anchor_block_size: int = -1,
    ):
        from star_attention import LlamaForCausalLM

        self._init_distributed()

        # Setup the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # Define the model
        self.model = LlamaForCausalLM.from_pretrained(
            path,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            max_memory=self.max_memory,
            attn_implementation='flash_attention_2',
        )
        self.block_size = block_size if block_size > 0 else None
        self.anchor_block_size = anchor_block_size if anchor_block_size > 0 else None

        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words if stop_words else []

    def _init_distributed(self):
        """Initialize the distributed environment"""

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))

            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

            # Assign each rank, its own set of GPUs
            # This is done so that the sharded model for each rank can be loaded on separate GPUs
            num_devices_per_rank = torch.cuda.device_count() // self.local_world_size
            device_id_start = self.local_rank * num_devices_per_rank
            self.max_memory = {
                x: f'{round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))}GB'
                for x in range(device_id_start, device_id_start + num_devices_per_rank)
            }
            print(
                '[model._init_distributed] '
                f'World size: {self.world_size}, Rank: {self.rank}, '
                f'Local World Size: {self.local_world_size}, Local rank: {self.local_rank}, '
                f'GPUs Assigned: {", ".join([str(x) for x in self.max_memory.keys()])}'
            )
        else:
            raise RuntimeError('Distributed environment is not initialized!')

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize the input text and return the token ids

        Args:
            text: input text

        Returns:
            token ids of shape (1, seq_len)
        """
        return self.tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(self.model.device)

    def _tokenize_and_partition_context(self, ctx: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Split the input context into blocks. The last block is padded to keep each block the same size.

        Args:
            ctx: input context

        Returns:
            token ids, position ids, context length (before padding)
        """
        raise NotImplementedError

    def _process_blockwise_context(
        self, ctx_ids_blocks: Tuple[torch.Tensor, ...], position_ids_blocks: Tuple[torch.Tensor, ...]
    ):
        """Generate the KV cache for the context assigned to the current rank.

        Args:
            ctx_ids_blocks: context blocks grouped by rank
            position_ids_blocks: position ids blocks grouped by rank

        Returns:
            KV cache for the context assigned to the current rank
        """
        raise NotImplementedError

    def _generate_output(self, input_ids, position_ids, past_key_values):
        """Phase 2 of Star Attention: Process input tokens followed by autoregressive token generation.

        Args:
            input_ids: input token ids
            position_ids: position ids of the input tokens
            past_key_values: KV cache

        Returns:
            generated token ids
        """
        output_seq = None
        for _ in range(self.max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    enable_star_attn=True,
                )  # type: ignore

            # Assign the new updated KV-cache to the last rank
            if self.rank == self.world_size - 1:
                past_key_values = outputs.past_key_values

            # Get the next token
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            output_seq = next_tokens if output_seq is None else torch.cat([output_seq, next_tokens])

            # Update the input_ids and position_ids for the next iteration
            input_ids = next_tokens.unsqueeze(0)
            position_ids = torch.tensor([[position_ids[-1, -1] + 1]]).to(position_ids)

        return output_seq.unsqueeze(0)

    def _get_output_text(self, output, truncate_texts=[]):
        """Convert the generated token ids to text"""
        generated_text = self.tokenizer.decode(output[0].detach().cpu().numpy().tolist())

        # Remove the input from the generated text
        for t in truncate_texts:
            t = t.strip()
            if t and generated_text.startswith(t):
                generated_text = generated_text[len(t) :].strip()

        for s in self.stop_words:
            generated_text = generated_text.split(s)[0]

        return generated_text.strip()

    def __call__(self, prompt_context: str, prompt_query: str):
        raise NotImplementedError


class StarAttentionModel(DistributedInferenceBaseModel):
    """Star Attention - Phase 1 and Phase 2

    Extended with Statistical Block Summaries for cross-block semantic
    awareness during parallel KV cache construction.
    """

    def __init__(
        self,
        path: str,
        max_new_tokens: int,
        stop_words: Optional[List[str]] = None,
        block_size: int = -1,
        anchor_block_size: int = -1,
        summary_k: int = 32,
        summary_method: str = "tfidf",
        discard_summary_kv: bool = True,
    ):
        super().__init__(path, max_new_tokens, stop_words=stop_words,
                         block_size=block_size, anchor_block_size=anchor_block_size)
        self.summary_k = summary_k
        self.summary_method = summary_method
        self.discard_summary_kv = discard_summary_kv

    def _tokenize_and_partition_context(self, ctx):
        # Tokenize the context
        ctx_ids = self._tokenize(ctx)
        ctx_len = ctx_ids.shape[-1]

        # Split the context into chunks of size `block_size`
        if self.block_size is None:
            self.block_size = ctx_ids.shape[-1] // self.world_size

        # Pad the context to be a multiple of block_size
        if ctx_ids.shape[-1] % self.block_size != 0:
            padding = self.block_size - (ctx_ids.shape[-1] % self.block_size)
            ctx_ids = torch.cat((ctx_ids, torch.zeros_like(ctx_ids)[:, :padding]), dim=-1)

        position_ids = torch.arange(0, ctx_ids.shape[-1]).unsqueeze(0).to(self.model.device)

        return ctx_ids, position_ids, ctx_len

    def _process_blockwise_context(self, ctx_ids_blocks, position_ids_blocks, block_summaries=None):
        """Phase 1 of Star Attention: Blockwise Context Encoding with
        Statistical Block Summaries.

        Each block's input context is assembled as:
            [S_1, ..., S_(i-1), B_i]
        Only summaries from earlier blocks are prepended (S_post is
        omitted because the causal mask prevents B_i from attending to
        tokens at higher sequence indices).  The sequence is sorted by
        position_id for flash-attention monotonicity, and after the
        forward pass only B_i's KV states are retained.
        """

        # Flatten all block indices so we can figure out which *global* block
        # index the current rank/idx pair corresponds to.
        blocks_before_rank = sum(len(ctx_ids_blocks[r]) for r in range(self.rank))

        kv_rank = []
        for idx in range(len(ctx_ids_blocks[self.rank])):
            # Select the current block
            ctx_block = ctx_ids_blocks[self.rank][idx]          # (1, block_size)
            position_block = position_ids_blocks[self.rank][idx]  # (1, block_size)
            block_size_cur = ctx_block.shape[-1]

            summary_len = 0
            if block_summaries is not None:
                global_block_idx = blocks_before_rank + idx

                # Collect only pre-block summaries (j < i)
                pre_token_parts, pre_pos_parts = [], []
                for j in range(global_block_idx):
                    s = block_summaries[j]
                    pre_token_parts.append(s['token_ids'])
                    pre_pos_parts.append(s['pos_ids'])

                # Assemble: [S_pre | B_i]
                all_tok = []
                all_pos = []
                pre_len = 0
                if pre_token_parts:
                    pre_tok = torch.cat(pre_token_parts).unsqueeze(0).to(ctx_block.device)
                    pre_pos = torch.cat(pre_pos_parts).unsqueeze(0).to(position_block.device)
                    pre_len = pre_tok.shape[-1]
                    all_tok.append(pre_tok)
                    all_pos.append(pre_pos)

                all_tok.append(ctx_block)
                all_pos.append(position_block)

                ctx_block = torch.cat(all_tok, dim=-1)
                position_block = torch.cat(all_pos, dim=-1)
                summary_len = ctx_block.shape[-1] - block_size_cur

            # ---------- Sort by position_id for flash-attn compatibility ----------
            # Flash attention checks for monotonically non-decreasing position_ids;
            # non-monotonic ids trigger varlen multi-sequence segmentation which
            # would incorrectly split our single-sequence context.
            if summary_len > 0:
                # Mark B_i tokens before sorting (B_i sits at [pre_len .. pre_len+block_size_cur))
                is_bi = torch.zeros(ctx_block.shape[-1], dtype=torch.bool, device=ctx_block.device)
                is_bi[pre_len:pre_len + block_size_cur] = True

                sort_idx = position_block.squeeze(0).argsort(stable=True)
                ctx_block = ctx_block[:, sort_idx]
                position_block = position_block[:, sort_idx]
                is_bi_sorted = is_bi[sort_idx]
            else:
                is_bi_sorted = None

            # ---------- Forward ----------
            with torch.no_grad():
                kv_block = self.model(
                    ctx_block,
                    position_ids=position_block,
                    use_cache=True,
                    num_ring_steps=0,  # disable ring attention (local blockwise attention)
                    enable_star_attn=False,
                ).past_key_values  # type: ignore

            # ---------- Extract only B_i's KV entries ----------
            if is_bi_sorted is not None and self.discard_summary_kv:
                kv_block = [
                    [x[0][:, :, is_bi_sorted], x[1][:, :, is_bi_sorted]] for x in kv_block
                ]

            kv_rank = (
                kv_block
                if not kv_rank
                else [
                    [torch.cat((kv_rank[i][j], kv_block[i][j]), dim=-2) for j in range(2)] for i in range(len(kv_rank))
                ]
            )

        return kv_rank

    def __call__(self, prompt_context: str, prompt_query: str) -> Dict[str, List[str]]:
        # Prepare the context
        ctx_ids, position_ids, ctx_len = self._tokenize_and_partition_context(prompt_context)

        # Split the context into blocks and divide the blocks among the ranks
        ctx_ids_blocks = torch.tensor_split(torch.stack(ctx_ids.split(self.block_size, dim=-1)), self.world_size)
        position_ids_blocks = torch.tensor_split(
            torch.stack(position_ids.split(self.block_size, dim=-1)), self.world_size
        )

        # ---------- Compute statistical block summaries ----------
        # Flatten all blocks into a simple list for summary computation.
        # Trim padding from the last block so that pad tokens (ID 0) are
        # never fed into TF-IDF scoring.
        padding = ctx_ids.shape[-1] - ctx_len  # number of trailing pad tokens
        flat_block_tokens = []   # list of 1-D CPU tensors
        flat_block_positions = []
        for rank_blocks_t, rank_blocks_p in zip(ctx_ids_blocks, position_ids_blocks):
            for blk_t, blk_p in zip(rank_blocks_t, rank_blocks_p):
                flat_block_tokens.append(blk_t.squeeze(0).cpu())
                flat_block_positions.append(blk_p.squeeze(0).cpu())

        # Strip padding from the last block (padding sits at the tail)
        if padding > 0 and flat_block_tokens:
            flat_block_tokens[-1] = flat_block_tokens[-1][:-padding]
            flat_block_positions[-1] = flat_block_positions[-1][:-padding]

        block_summaries = compute_block_summaries(
            flat_block_tokens,
            flat_block_positions,
            k=self.summary_k,
            method=self.summary_method,
        )

        # Phase 1: Generate the KV cache for the local context (with summaries)
        kv_rank = self._process_blockwise_context(ctx_ids_blocks, position_ids_blocks,
                                                   block_summaries=block_summaries)
        if self.rank == self.world_size - 1:  # discard padding from the last rank
            padding = ctx_ids.shape[-1] - ctx_len
            if padding > 0:
                kv_rank = [
                    [kv_rank[i][0][:, :, :-padding], kv_rank[i][1][:, :, :-padding]] for i in range(len(kv_rank))
                ]

        # Phase 2: Process query with global attention (unchanged)
        qry_ids = self._tokenize(prompt_query)
        qry_position_ids = torch.arange(ctx_len, ctx_len + qry_ids.shape[-1]).unsqueeze(0).to(self.model.device)
        output = self._generate_output(qry_ids, qry_position_ids, kv_rank)

        # Get the generated text
        generated_text = self._get_output_text(output)
        return {'text': [generated_text]}


class RingAttentionModel(DistributedInferenceBaseModel):
    """Ring Attention augmented with Phase 2 of Star Attention for Fast Token Generation"""

    def __init__(self, path, max_new_tokens, stop_words=None):
        super().__init__(path, max_new_tokens, stop_words=stop_words)

    def _tokenize_and_partition_context(self, ctx):
        # Tokenize the context
        ctx_ids = self._tokenize(ctx)
        ctx_len = ctx_ids.shape[-1]

        # Pad the context to be a multiple of world_size
        if ctx_ids.shape[-1] % self.world_size != 0:
            padding = self.world_size - (ctx_ids.shape[-1] % self.world_size)
            ctx_ids = torch.cat((ctx_ids, torch.zeros_like(ctx_ids)[:, :padding]), dim=-1)

        # Split the context into blocks
        self.block_size = ctx_ids.shape[-1] // self.world_size

        position_ids = torch.arange(0, ctx_ids.shape[-1]).unsqueeze(0).to(self.model.device)

        return ctx_ids, position_ids, ctx_len

    def _process_blockwise_context(self, ctx_ids_blocks, position_ids_blocks):
        assert len(ctx_ids_blocks[self.rank]) == 1, 'Ring Attention expects only one block per rank'

        ctx_block = ctx_ids_blocks[self.rank][0]
        position_block = position_ids_blocks[self.rank][0]
        with torch.no_grad():
            kv_rank = self.model(
                ctx_block,
                position_ids=position_block,
                use_cache=True,
                num_ring_steps=-1,  # enable ring attention
                enable_star_attn=False,
            ).past_key_values  # type: ignore

        return kv_rank

    def __call__(self, prompt_context: str, prompt_query: str) -> Dict[str, List[str]]:
        # Prepare the context
        ctx_ids, position_ids, ctx_len = self._tokenize_and_partition_context(prompt_context)

        # Divide the context blocks among the ranks
        ctx_ids_blocks = torch.tensor_split(torch.stack(ctx_ids.split(self.block_size, dim=-1)), self.world_size)
        position_ids_blocks = torch.tensor_split(
            torch.stack(position_ids.split(self.block_size, dim=-1)), self.world_size
        )

        # Generate the KV cache for the local context
        kv_rank = self._process_blockwise_context(ctx_ids_blocks, position_ids_blocks)
        if self.rank == self.world_size - 1:  # discard padding from the last rank
            padding = ctx_ids.shape[-1] - ctx_len
            if padding > 0:
                kv_rank = [
                    [kv_rank[i][0][:, :, :-padding], kv_rank[i][1][:, :, :-padding]] for i in range(len(kv_rank))
                ]

        # Phase 2 from Star Attention: Global attention with online softmax
        qry_ids = self._tokenize(prompt_query)
        qry_position_ids = torch.arange(ctx_len, ctx_len + qry_ids.shape[-1]).unsqueeze(0).to(self.model.device)
        output = self._generate_output(qry_ids, qry_position_ids, kv_rank)

        # Get the generated text
        generated_text = self._get_output_text(output)
        return {'text': [generated_text]}


class DenseAttentionModel:

    def __init__(self, path: str, max_new_tokens: int, stop_words):
        from transformers import AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
        )

        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words if stop_words else []

    def _generate_output(self, input_ids, position_ids):
        output_seq, past_key_values = None, None
        for _ in range(self.max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids, position_ids=position_ids, past_key_values=past_key_values, use_cache=True
                )  # type: ignore

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            output_seq = next_tokens if output_seq is None else torch.cat([output_seq, next_tokens])

            # Update the input_ids and position_ids for the next iteration
            input_ids = next_tokens.unsqueeze(0)
            position_ids = torch.tensor([[position_ids[-1, -1] + 1]]).to(position_ids)

        return output_seq.unsqueeze(0)

    def _get_output_text(self, output, truncate_texts=[]):
        # Remove the input from the generated text
        generated_text = self.tokenizer.decode(output[0].detach().cpu().numpy().tolist())

        for t in truncate_texts:
            t = t.strip()
            if t and generated_text.startswith(t):
                generated_text = generated_text[len(t) :].strip()

        for s in self.stop_words:
            generated_text = generated_text.split(s)[0]

        return generated_text.strip()

    def __call__(self, prompt_context: str, prompt_query: str):
        prompt = prompt_context + prompt_query
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(self.model.device)
        position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).to(self.model.device)

        output = self._generate_output(input_ids, position_ids)

        return {'text': [self._get_output_text(output)]}
