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
    """Star Attention - Phase 1 and Phase 2"""

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

    def _process_blockwise_context(self, ctx_ids_blocks, position_ids_blocks):
        """Phase 1 of Star Attention: Blockwise Context Encoding with Anchor Blocks"""

        # If the anchor block size is not provided, use the entire first block
        if self.anchor_block_size is None:
            self.anchor_block_size = ctx_ids_blocks[0][0].shape[-1]

        kv_rank = []
        for idx in range(len(ctx_ids_blocks[self.rank])):
            # Select the current block
            ctx_block = ctx_ids_blocks[self.rank][idx]
            position_block = position_ids_blocks[self.rank][idx]

            # From 2nd block onwards, prepend the anchor block to the current block
            if self.rank != 0 or idx > 0:
                ctx_block = torch.cat((ctx_ids_blocks[0][0][:, : self.anchor_block_size], ctx_block), dim=-1)
                position_block = torch.cat(
                    (position_ids_blocks[0][0][:, : self.anchor_block_size], position_block), dim=-1
                )

            with torch.no_grad():
                kv_block = self.model(
                    ctx_block,
                    position_ids=position_block,
                    use_cache=True,
                    num_ring_steps=0,  # disable ring attention (local blockwise attention)
                    enable_star_attn=False,
                ).past_key_values  # type: ignore

            # Discard the anchor block KV cache
            if self.rank != 0 or idx > 0:
                kv_block = [
                    [x[0][:, :, self.anchor_block_size :], x[1][:, :, self.anchor_block_size :]] for x in kv_block
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

        # Phase 1: Generate the KV cache for the local context
        kv_rank = self._process_blockwise_context(ctx_ids_blocks, position_ids_blocks)
        if self.rank == self.world_size - 1:  # discard padding from the last rank
            padding = ctx_ids.shape[-1] - ctx_len
            if padding > 0:
                kv_rank = [
                    [kv_rank[i][0][:, :, :-padding], kv_rank[i][1][:, :, :-padding]] for i in range(len(kv_rank))
                ]

        # Phase 2: Process query with global attention
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

# model.py

# ... (keep all existing code from the original file, including imports and all other classes)
# V V V V REPLACE THE PREVIOUS CustomAccuracyModel WITH THIS FINAL VERSION V V V V

class CustomAccuracyModel:
    """
    Implements the custom two-pass mechanism for improved accuracy on a single GPU.
    - Pass 1: Uses a standard 'eager' attention model to find top-k summary tokens.
    - Pass 2: Uses a 'flash_attention_2' model to build the final KV cache sequentially.
    - Phase 2: Uses the 'flash_attention_2' model for fast autoregressive generation.
    """

    def __init__(self, path: str, max_new_tokens: int, block_size: int, k_summary_size: int, stop_words: Optional[List[str]] = None):
        if not torch.cuda.is_available():
            raise RuntimeError("This implementation requires a CUDA-enabled GPU.")
        
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        print("\n[SETUP] Loading model with Flash Attention 2 for generation (Pass 2)...")
        self.model_flash = LlamaForCausalLM.from_pretrained(
            path,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
        )
        self.model_flash.eval()

        print("[SETUP] Loading model with Eager Attention for summary finding (Pass 1)...")
        # For Pass 1, we need the standard 'eager' implementation to get attention scores
        self.model_eager = LlamaForCausalLM.from_pretrained(
            path,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            attn_implementation='eager',
        )
        self.model_eager.eval()

        # Generation and model parameters
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words if stop_words else []
        self.block_size = block_size
        self.k = k_summary_size
        print("[SETUP] Initialization complete.")

    def _get_top_k_summary(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Pass 1: Processes a block to find the top-k summary tokens based on attention.
        This uses the 'eager' model to access attention weights.
        """
        with torch.no_grad():
            outputs = self.model_eager(input_ids, output_attentions=True)
        
        all_attentions = torch.stack(outputs.attentions)
        token_scores = all_attentions.sum(dim=(0, 1, 2, -1))
        
        _, top_k_indices = torch.topk(token_scores, min(self.k, token_scores.shape[0]))
        
        top_k_indices_sorted, _ = torch.sort(top_k_indices)
        summary_ids = input_ids[:, top_k_indices_sorted]
        
        return summary_ids

    def _get_output_text(self, output_ids: torch.Tensor) -> str:
        """Decodes the generated token IDs and cleans up the text."""
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        assistant_marker = "assistant\n\n"
        if assistant_marker in generated_text:
            generated_text = generated_text.split(assistant_marker)[-1]

        for s in self.stop_words:
            generated_text = generated_text.split(s)[0]

        return generated_text.strip()

    def __call__(self, prompt_context: str, prompt_query: str) -> Dict[str, List[str]]:
        context_ids = self.tokenizer.encode(prompt_context, return_tensors='pt').to(self.device)
        context_blocks = list(context_ids.split(self.block_size, dim=1))
        # --- DEBUG PRINT ---
        print(f"\n[DEBUG] Total context tokens: {context_ids.shape[1]}")
        print(f"[DEBUG] Context split into {len(context_blocks)} blocks of max size {self.block_size}.")

        with torch.no_grad():
            # === PHASE 1, PASS 1: GENERATE SUMMARIES ===
            print("\n--- Phase 1, Pass 1: Generating summaries... ---")
            summaries = []
            for i, block in enumerate(context_blocks):
                print(f"  [Pass 1] Processing block {i+1}/{len(context_blocks)} (shape: {block.shape}) to find summary...")
                summary_ids = self._get_top_k_summary(block)
                summaries.append(summary_ids)
                # --- DEBUG PRINT ---
                print(f"    -> Generated summary shape: {summary_ids.shape}")
                
                # CRUCIAL FIX: Clear the cache to release memory from the eager forward pass
                torch.cuda.empty_cache()
            print("--- [Pass 1] All summaries generated. ---")

            # === PHASE 1, PASS 2: BUILD FINAL KV CACHE ===
            print("\n--- Phase 1, Pass 2: Building final KV cache incrementally... ---")
            full_kv_cache = None
            for i, block in enumerate(context_blocks):
                print(f"  [Pass 2] Processing block {i+1}/{len(context_blocks)} (shape: {block.shape}) to build its KV cache...")
                
                previous_summaries = summaries[:i]
                if previous_summaries:
                    augmented_input_ids = torch.cat(previous_summaries + [block], dim=1)
                    summary_len = sum(s.shape[1] for s in previous_summaries)
                else:
                    augmented_input_ids = block
                    summary_len = 0
                
                # --- DEBUG PRINT ---
                print(f"    -> Augmented input shape: {augmented_input_ids.shape} (Summary len: {summary_len})")
                
                outputs = self.model_flash(augmented_input_ids, use_cache=True)
                current_kv_cache = outputs.past_key_values
                # --- DEBUG PRINT ---
                print(f"    -> Intermediate KV cache key shape (layer 0): {current_kv_cache[0][0].shape}")
                
                # Slice the cache to keep only the part for the current block
                sliced_kv_cache = []
                for layer_cache in current_kv_cache:
                    key, value = layer_cache
                    sliced_key = key[:, :, summary_len:, :]
                    sliced_value = value[:, :, summary_len:, :]
                    sliced_kv_cache.append((sliced_key, sliced_value))
                # --- DEBUG PRINT ---
                print(f"    -> Sliced KV cache key shape (layer 0): {sliced_kv_cache[0][0].shape}")
                
                # Incrementally build the final KV cache
                if full_kv_cache is None:
                    full_kv_cache = tuple(sliced_kv_cache)
                else:
                    new_full_cache = []
                    for idx, layer_cache in enumerate(full_kv_cache):
                        full_key, full_value = layer_cache
                        slice_key, slice_value = sliced_kv_cache[idx]
                        combined_key = torch.cat([full_key, slice_key], dim=2)
                        combined_value = torch.cat([full_value, slice_value], dim=2)
                        new_full_cache.append((combined_key, combined_value))
                    full_kv_cache = tuple(new_full_cache)
                # --- DEBUG PRINT ---
                print(f"    -> Total KV cache key shape (layer 0) updated to: {full_kv_cache[0][0].shape}")

            print("--- [Pass 2] Final KV cache for the full context is built. ---")

            # === PHASE 2: GENERATE RESPONSE ===
            print("\n--- Phase 2: Generating response... ---")
            generation_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            generation_ids = self.tokenizer.encode(generation_prompt, return_tensors='pt').to(self.device)
            
            # --- DEBUG PRINT ---
            print(f"  [Phase 2] Input query shape: {generation_ids.shape}")
            print(f"  [Phase 2] Using past_key_values with key shape (layer 0): {full_kv_cache[0][0].shape}")

            outputs = self.model_flash.generate(
                generation_ids,
                max_new_tokens=self.max_new_tokens,
                past_key_values=full_kv_cache,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            generated_text = self._get_output_text(outputs)
            return {'text': [generated_text]}
