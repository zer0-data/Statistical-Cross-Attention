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

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache


def compute_block_summaries(
    all_block_tokens: List[torch.Tensor],
    all_block_positions: List[torch.Tensor],
    num_chunks: int = 4,
    chunk_size: int = 32,
    method: str = "tfidf",
    sink_size: int = 64,
    embed_fn=None,
) -> List[Dict[str, torch.Tensor]]:
    """Compute block summaries by selecting the most informative contiguous chunks.

    Each block is divided into non-overlapping fixed-size chunks and each
    chunk is scored as a unit using the chosen heuristic.  The top
    ``num_chunks`` chunks are selected per block and returned in positional
    order, preserving natural language structure so the Transformer
    produces healthy K/V states.

    Scoring is fully vectorized in PyTorch — no Python loops over tokens.

    TF-IDF equivalence
    ------------------
    The naive formula ``score = mean_t(TF(t) * IDF(t))`` expands to::

        (1 / chunk_len) * sum_{each token occurrence} IDF(token)

    because every occurrence of token *t* contributes ``(1/chunk_len) * IDF(t)``,
    and summing over unique types with their counts equals summing over all
    positions.  This lets us replace the Counter + loop with a single
    ``idf_table[chunk_tokens].sum() / chunk_len`` lookup.

    Args:
        all_block_tokens: list of 1D tensors, one per block (token IDs).
        all_block_positions: list of 1D tensors, one per block (position IDs).
        num_chunks: number of contiguous chunks / summary slots to select per block.
        chunk_size: number of tokens per chunk (used by chunk-based methods).
        method: scoring / selection heuristic — one of:

            * ``tfidf``        – mean TF-IDF score per chunk (global rarity).
            * ``bm25``         – BM25-scored chunks (length-normalised rarity).
            * ``entropy``      – fraction of unique tokens per chunk (local
              diversity / information density).  A chunk full of padding
              or stop-word repetition scores near 0; a dense entity-rich
              chunk scores near 1.
            * ``max_idf``      – max IDF value of any token in the chunk
              (keyword / needle detection).
            * ``evenly_spaced``– select exactly ``num_chunks * chunk_size``
              tokens at uniform intervals across the block.  Guarantees full
              block coverage regardless of token scores.  Returns original
              position IDs (sparse).
            * ``mean_pool``    – divide block into ``num_chunks`` equal groups,
              mean-pool the token embeddings in each group, and return the
              pooled float vectors as ``'embeds'``.  Requires ``embed_fn``
              (the model\'s ``embed_tokens`` layer).  Position IDs are the
              median position of each group.
            * ``anchor``       – NVIDIA Star Attention's original anchor-block
              baseline.  Block 0 contributes the first
              ``num_chunks * chunk_size`` tokens (after sink) as the anchor;
              all other blocks contribute nothing.  Downstream this yields
              ``[sink | anchor | B_i]`` for every i>0 — exactly the vanilla
              Star Attention formulation, expressed as a summary method so
              it can be swept alongside the statistical methods at matched
              token budget.

        sink_size: number of leading tokens in the first block to exclude from
            scoring (default 64).  These tokens are always injected by the
            caller as attention sinks and must not consume summary slots.
            Set to 0 to disable.
        embed_fn: callable ``(token_ids: LongTensor[N]) -> FloatTensor[N, D]``.
            Required when ``method='mean_pool'``, ignored otherwise.

    Returns:
        List of dicts, one per block.  Each dict contains:

        * ``'token_ids'`` (1D LongTensor) and ``'pos_ids'`` (1D LongTensor) –
          for all methods **except** ``mean_pool``.
        * ``'embeds'`` (2D FloatTensor ``[k, D]``) and ``'pos_ids'`` –
          for ``mean_pool``.
    """
    num_blocks = len(all_block_tokens)
    device = all_block_tokens[0].device

    # ── Corpus-level document frequency as a dense tensor ──────────────────
    # df_table[token_id] = number of blocks in which that token appears.
    # We use the maximum token id across the entire corpus as the table size.
    vocab_size = max(blk.max().item() for blk in all_block_tokens) + 1
    df_table = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    for blk_tok in all_block_tokens:
        # Mark each unique token in this block (binary, then accumulate)
        unique_ids = blk_tok.unique()
        df_table[unique_ids] += 1.0

    # ── IDF table ──────────────────────────────────────────────────────────
    # idf_table[t] = log(num_blocks / max(df(t), 1))
    df_clamped = df_table.clamp(min=1.0)
    idf_table = torch.log(torch.tensor(num_blocks, dtype=torch.float32, device=device) / df_clamped)
    # Tokens that never appear get idf = log(num_blocks/1) which is the max; that's fine.

    # ── Average chunk length (for BM25 length normalisation) ───────────────
    # Computed once on CPU to avoid a device sync inside the loop.
    if method == "bm25":
        total_len = sum(blk.shape[0] for blk in all_block_tokens)
        # Exclude sink tokens from block 0 — they are skipped during chunk scoring
        if sink_size > 0:
            total_len -= min(sink_size, all_block_tokens[0].shape[0])
        num_total_chunks = sum(
            (blk.shape[0] + chunk_size - 1) // chunk_size for blk in all_block_tokens
        )
        avg_chunk_len = float(total_len) / max(num_total_chunks, 1)
        k1, b = 1.2, 0.75

    # ── Score & select ──────────────────────────────────────────────────────
    summaries: List[Dict[str, torch.Tensor]] = []
    for block_idx, (blk_tok, blk_pos) in enumerate(zip(all_block_tokens, all_block_positions)):
        # Skip the first sink_size tokens of block 0 so they never compete
        # for summary slots — they are always injected by the caller.
        if block_idx == 0 and sink_size > 0:
            skip = min(sink_size, blk_tok.shape[0])
            blk_tok = blk_tok[skip:]
            blk_pos = blk_pos[skip:]

        block_len = blk_tok.shape[0]

        # ── Vanilla Star Attention Anchor-Block Baseline ────────────────────
        # Block 0 contributes its first (num_chunks * chunk_size) tokens as
        # the "anchor" visible to every later block. All other blocks
        # contribute an empty summary, so block i>0 sees [sink | anchor | B_i]
        # — exactly NVIDIA Star Attention's original formulation, at matched
        # token budget.
        if method == "anchor":
            if block_idx == 0:
                budget = num_chunks * chunk_size
                take = min(budget, block_len)
                summaries.append({
                    'token_ids': blk_tok[:take],
                    'pos_ids':   blk_pos[:take],
                })
            else:
                summaries.append({
                    'token_ids': blk_tok.new_empty(0),
                    'pos_ids':   blk_pos.new_empty(0),
                })
            continue  # skip chunk-split / IDF scoring entirely

        # ── Evenly-Spaced Token Selection ────────────────────────────────────
        # Select exactly `num_chunks * chunk_size` token positions at uniform
        # intervals across the block.  Guarantees full block coverage without
        # any IDF scoring.  Original (sparse) position IDs are preserved.
        if method == "evenly_spaced":
            budget = num_chunks * chunk_size
            if block_len <= budget:
                # Block shorter than budget — take everything
                summaries.append({'token_ids': blk_tok, 'pos_ids': blk_pos})
            else:
                # linspace gives `budget` evenly-spaced indices in [0, block_len-1]
                idx = torch.linspace(0, block_len - 1, budget).long()
                summaries.append({
                    'token_ids': blk_tok[idx],
                    'pos_ids':   blk_pos[idx],
                })
            continue  # skip chunk-split / IDF scoring entirely

        # ── Mean-Pool Embedding Groups ────────────────────────────────────────
        # Divide the block into `num_chunks` equal-ish groups; mean-pool the
        # token embeddings in each group to produce a soft summary vector.
        # Returns 'embeds' (FloatTensor [k, D]) instead of 'token_ids'.
        # Position of each pooled vector = median position within its group.
        # Requires embed_fn (the model's embed_tokens layer).
        if method == "mean_pool":
            if embed_fn is None:
                raise ValueError("method='mean_pool' requires embed_fn to be provided.")
            group_indices = torch.chunk(torch.arange(block_len, device=blk_tok.device), num_chunks)
            pooled_embeds, pooled_pos = [], []
            for g_idx in group_indices:
                if g_idx.numel() == 0:
                    continue
                g_tok = blk_tok[g_idx].to(next(embed_fn.parameters()).device)
                with torch.no_grad():
                    g_emb = embed_fn(g_tok)                          # (g_len, D)
                pooled_embeds.append(g_emb.mean(dim=0))              # (D,)
                pooled_pos.append(blk_pos[g_idx[g_idx.numel() // 2]])  # median pos
            summaries.append({
                'embeds':  torch.stack(pooled_embeds),               # (k, D)
                'pos_ids': torch.stack(pooled_pos).to(blk_pos),     # (k,)
            })
            continue  # skip chunk-split / IDF scoring entirely

        # ── Chunk-split (shared by all IDF-based methods) ────────────────────
        tok_chunks = blk_tok.split(chunk_size)   # tuple of 1-D tensors
        pos_chunks = blk_pos.split(chunk_size)
        n_chunks = len(tok_chunks)

        if method == "bm25":
            # ── Vectorized BM25 ──────────────────────────────────────────
            # For each chunk: score = Σ_t IDF(t) * tf_norm(t)
            # where tf_norm(t) = raw_tf*(k1+1) / (raw_tf + k1*(1-b+b*c_len/avg_len))
            scores = torch.zeros(n_chunks, dtype=torch.float32, device=device)
            for ci, c_tok in enumerate(tok_chunks):
                c_len = c_tok.shape[0]
                # raw term frequencies via bincount (GPU-native)
                tf_raw = torch.bincount(c_tok, minlength=vocab_size).float()  # (vocab_size,)
                mask = tf_raw > 0
                tf_r = tf_raw[mask]
                # BM25 uses its own IDF formula (different from TF-IDF's log(N/df))
                df_r = df_table[mask]
                idf_bm25 = torch.log(
                    (num_blocks - df_r + 0.5) / (df_r + 0.5) + 1.0
                )
                numerator = tf_r * (k1 + 1)
                denominator = tf_r + k1 * (1 - b + b * c_len / avg_chunk_len)
                scores[ci] = (idf_bm25 * numerator / denominator).sum()
        elif method == "entropy":
            # ── Vectorized Token Entropy (Information Density) ───────────
            # score(chunk) = unique_token_count / chunk_len
            #
            # Intuition: unique_count / L  ≈  type-token ratio, a fast proxy
            # for lexical diversity.  A chunk of padding / repeated stop-words
            # has unique_count → 1, score → 0.  A chunk packed with distinct
            # entities scores near 1.  torch.bincount gives unique_count as
            # (bin_counts > 0).sum() — no Python loop, no .tolist().
            scores = torch.stack([
                (torch.bincount(c_tok, minlength=vocab_size) > 0).sum().float()
                / c_tok.shape[0]
                for c_tok in tok_chunks
            ])
        elif method == "max_idf":
            # ── Vectorized Max-IDF (Keyword / Needle Detection) ──────────
            # score(chunk) = max IDF across all token positions in the chunk.
            # One hyper-rare token (UUID, proper name, NIAH needle) is enough
            # to bring the whole chunk to the top, regardless of surrounding
            # filler.  Pure gather + reduce — no Python loop.
            scores = torch.stack([
                idf_table[c_tok].max()
                for c_tok in tok_chunks
            ])
        else:
            # ── Vectorized TF-IDF (default) ──────────────────────────────
            # score(chunk) = sum_{pos} IDF(token[pos]) / chunk_len
            # This is mathematically identical to mean(TF(t)*IDF(t)) over types.
            scores = torch.stack([
                idf_table[c_tok].sum() / c_tok.shape[0]
                for c_tok in tok_chunks
            ])

        # Select top-k chunks, restore positional order
        actual_k = min(num_chunks, n_chunks)
        _, top_idx = torch.topk(scores, actual_k)
        top_idx = top_idx.sort().values

        idx_list = top_idx.tolist()
        sel_tok = torch.cat([tok_chunks[i] for i in idx_list])
        sel_pos = torch.cat([pos_chunks[i] for i in idx_list])
        summaries.append({'token_ids': sel_tok, 'pos_ids': sel_pos})

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
            # Single-GPU fallback — no distributed environment required
            self.world_size = 1
            self.local_world_size = 1
            self.rank = 0
            self.local_rank = 0
            if torch.cuda.is_available():
                mem_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
                self.max_memory = {0: f'{mem_gb}GB'}
            else:
                self.max_memory = {}
            print('[model._init_distributed] Single-GPU mode (no distributed environment)')

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

            if self.tokenizer.eos_token_id is not None and next_tokens.item() == self.tokenizer.eos_token_id:
                break

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
        summary_chunks: int = 4,
        chunk_size: int = 32,
        summary_method: str = "tfidf",
        discard_summary_kv: bool = True,
        sink_size: int = 64,
    ):
        super().__init__(path, max_new_tokens, stop_words=stop_words,
                         block_size=block_size, anchor_block_size=anchor_block_size)
        self.summary_chunks = summary_chunks
        self.chunk_size = chunk_size
        self.summary_method = summary_method
        self.discard_summary_kv = discard_summary_kv
        self.sink_size = sink_size

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

    def _process_blockwise_context(
        self,
        ctx_ids_blocks,
        position_ids_blocks,
        block_summaries=None,
        sink_ids: Optional[torch.Tensor] = None,
        sink_pos: Optional[torch.Tensor] = None,
    ):
        """Phase 1 of Star Attention: Blockwise Context Encoding with
        Statistical Block Summaries.

        Each block's input context is assembled as:
            [SINK | S_1, ..., S_(i-1), B_i]  (for i > 0)
            [B_0]                             (for i = 0, sinks already inside)
        SINK = the first ``sink_size`` tokens of the sequence.  They are
        always visible so every block can form stable attention-sink keys.
        Summaries from later blocks are omitted (causal mask).  The full
        concatenation is monotonically sorted by position_id and after the
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
                pre_parts = []   # each element: dict with 'token_ids'|'embeds' + 'pos_ids'
                for j in range(global_block_idx):
                    pre_parts.append(block_summaries[j])

                # Check whether any summary uses soft embeddings (mean_pool)
                has_embeds = any('embeds' in s for s in pre_parts)

                # Assemble: [SINK | S_pre | B_i]
                all_pos = []

                if has_embeds:
                    # ── Soft-embedding path (mean_pool summaries) ────────────
                    # Everything must be in embedding space so we pass
                    # inputs_embeds instead of input_ids.
                    embed_layer = self.model.model.embed_tokens
                    all_emb = []

                    # 1. Attention sinks
                    if global_block_idx > 0 and sink_ids is not None:
                        s_emb = embed_layer(sink_ids.to(ctx_block.device))  # (sink_len, D)
                        all_emb.append(s_emb.unsqueeze(0))                  # (1, sink_len, D)
                        all_pos.append(sink_pos.unsqueeze(0).to(position_block.device))

                    # 2. Pre-block summaries
                    for s in pre_parts:
                        if 'embeds' in s:
                            e = s['embeds'].to(ctx_block.device)            # (k, D)
                        else:
                            e = embed_layer(s['token_ids'].to(ctx_block.device))  # (n, D)
                        all_emb.append(e.unsqueeze(0))                      # (1, n/k, D)
                        all_pos.append(s['pos_ids'].unsqueeze(0).to(position_block.device))

                    # 3. Current block
                    b_emb = embed_layer(ctx_block)                          # (1, block_size, D)
                    all_emb.append(b_emb)
                    all_pos.append(position_block)

                    inputs_embeds_block = torch.cat(all_emb, dim=1)         # (1, total_len, D)
                    position_block = torch.cat(all_pos, dim=-1)
                    summary_len = inputs_embeds_block.shape[1] - block_size_cur
                    ctx_block = None  # signal to use inputs_embeds below
                else:
                    # ── Token-ID path (all other methods) ───────────────────
                    all_tok = []
                    inputs_embeds_block = None

                    # 1. Attention sinks
                    if global_block_idx > 0 and sink_ids is not None:
                        all_tok.append(sink_ids.unsqueeze(0).to(ctx_block.device))
                        all_pos.append(sink_pos.unsqueeze(0).to(position_block.device))

                    # 2. Pre-block summaries
                    if pre_parts:
                        pre_tok = torch.cat([s['token_ids'] for s in pre_parts]).unsqueeze(0).to(ctx_block.device)
                        pre_pos = torch.cat([s['pos_ids']   for s in pre_parts]).unsqueeze(0).to(position_block.device)
                        all_tok.append(pre_tok)
                        all_pos.append(pre_pos)

                    # 3. Current block
                    all_tok.append(ctx_block)
                    all_pos.append(position_block)

                    ctx_block = torch.cat(all_tok, dim=-1)
                    position_block = torch.cat(all_pos, dim=-1)
                    summary_len = ctx_block.shape[-1] - block_size_cur
            else:
                inputs_embeds_block = None

            # ---------- B_i mask for KV extraction ----------
            if summary_len > 0:
                total_len = (
                    inputs_embeds_block.shape[1] if inputs_embeds_block is not None
                    else ctx_block.shape[-1]
                )
                is_bi_sorted = torch.zeros(total_len, dtype=torch.bool,
                                           device=(inputs_embeds_block.device
                                                   if inputs_embeds_block is not None
                                                   else ctx_block.device))
                is_bi_sorted[-block_size_cur:] = True
            else:
                is_bi_sorted = None

            # ---------- Forward ----------
            with torch.no_grad():
                if inputs_embeds_block is not None:
                    kv_block = self.model(
                        inputs_embeds=inputs_embeds_block,
                        position_ids=position_block,
                        use_cache=True,
                        num_ring_steps=0,
                        enable_star_attn=False,
                    ).past_key_values  # type: ignore
                else:
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

        # ---------- Wrap in DynamicCache ----------
        # kv_rank is a List[List[Tensor]] (or DynamicCache for single-block
        # no-discard path).  Always return a proper DynamicCache so that
        # LlamaModel.forward never falls back to the deprecated
        # DynamicCache.from_legacy_cache() conversion.
        if not isinstance(kv_rank, DynamicCache):
            cache = DynamicCache()
            for layer_idx in range(len(kv_rank)):
                cache.update(kv_rank[layer_idx][0], kv_rank[layer_idx][1], layer_idx)
            kv_rank = cache

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

        # Extract attention sink tokens from the very start of the sequence.
        # These are always re-injected into every block's context (except block 0
        # which already contains them) so that attention sinks remain visible.
        sink_size = min(self.sink_size, flat_block_tokens[0].shape[0]) if self.sink_size > 0 else 0
        if sink_size > 0:
            sink_ids = flat_block_tokens[0][:sink_size].to(self.model.device)
            sink_pos = flat_block_positions[0][:sink_size].to(self.model.device)
        else:
            sink_ids = sink_pos = None

        block_summaries = compute_block_summaries(
            flat_block_tokens,
            flat_block_positions,
            num_chunks=self.summary_chunks,
            chunk_size=self.chunk_size,
            method=self.summary_method,
            sink_size=sink_size,
            embed_fn=self.model.model.embed_tokens if self.summary_method == "mean_pool" else None,
        )

        # Phase 1: Generate the KV cache for the local context (with summaries)
        kv_rank = self._process_blockwise_context(
            ctx_ids_blocks,
            position_ids_blocks,
            block_summaries=block_summaries,
            sink_ids=sink_ids,
            sink_pos=sink_pos,
        )
        if self.rank == self.world_size - 1:  # discard padding from the last rank
            padding = ctx_ids.shape[-1] - ctx_len
            if padding > 0:
                for i in range(len(kv_rank)):
                    kv_rank.key_cache[i] = kv_rank.key_cache[i][:, :, :-padding]
                    kv_rank.value_cache[i] = kv_rank.value_cache[i][:, :, :-padding]

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
                for i in range(len(kv_rank)):
                    kv_rank.key_cache[i] = kv_rank.key_cache[i][:, :, :-padding]
                    kv_rank.value_cache[i] = kv_rank.value_cache[i][:, :, :-padding]

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

            if self.tokenizer.eos_token_id is not None and next_tokens.item() == self.tokenizer.eos_token_id:
                break

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
