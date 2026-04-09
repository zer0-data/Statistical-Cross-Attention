# Theory: Statistical Cross-Attention for Star Attention

## Overview

This document analyses every modification made to the [original Star Attention](https://arxiv.org/abs/2411.17116) codebase, the reasoning behind each change, and the theoretical justification for the statistical heuristics used. The original Star Attention uses a fixed **anchor block** (the first block of the context, duplicated across all GPUs) as a shared context prefix during Phase 1. Our modification replaces this static anchor with **lightweight, non-neural statistical block summaries** and introduces **attention sinks** as a complementary mechanism.

---

## 1. Anchor Block Removal

### Original Behaviour

In the original Star Attention, the first block $B_0$ is designated the *anchor block*. Every GPU receives a copy of the anchor prepended to its local block:

$$\text{Input}_i = [\underbrace{A}_{\text{anchor}} \;|\; B_i]$$

The anchor ensures all GPUs share a minimum overlapping context, which helps each block's KV cache attend to the same opening tokens. However, the anchor is **fixed** — it always carries the first $N$ tokens regardless of their relevance — and its size directly increases each GPU's compute and memory cost.

### What We Changed

We **removed the anchor block entirely**. The role of shared cross-block context is now served by two separate, more targeted mechanisms:

1. **Attention sinks** — a small, fixed-size prefix (default 64 tokens) from the very start of the sequence.
2. **Statistical block summaries** — informationally-scored contiguous chunks from earlier blocks, prepended to each block.

### Why

| Problem with Anchor | Our Solution |
|---|---|
| Always uses the first $N$ tokens regardless of content | Summaries select the *most informative* chunks from each block |
| Fixed-size, does not scale with number of blocks | Summary budget is configurable per block (`num_chunks × chunk_size`) |
| Duplicates the entire first block on every GPU | Sinks are tiny (64 tokens); summaries are compact (~128 tokens/block) |
| No cross-block awareness beyond block 0 | Summaries carry scored information from *all* earlier blocks |

---

## 2. Attention Sinks

### Motivation

Research on LLM inference (StreamingLLM, Xiao et al. 2023) shows that the first few tokens of a sequence act as **attention sinks** — they accumulate disproportionately high attention weights across all layers, regardless of their semantic content. Removing them destabilises the softmax distribution and degrades downstream quality.

### Implementation

- The first `sink_size` tokens (default 64) of block $B_0$ are extracted before any summary computation.
- They are prepended to every block $B_i$ (for $i > 0$) as a fixed context prefix.
- They are **excluded** from summary chunk scoring in block $B_0$ to avoid double-counting (the caller always injects them separately).

### Per-Block Assembly

$$\text{Input}_i = \begin{cases} [B_0] & i = 0 \\[6pt] [\underbrace{\text{SINK}}_{\text{64 tokens}} \;|\; S_1, \ldots, S_{i-1} \;|\; B_i] & i > 0 \end{cases}$$

Sinks replace the anchor's stability role with a fraction of the cost. They ensure healthy softmax distributions without carrying semantically irrelevant bulk.

---

## 3. Statistical Block Summaries

### Core Idea

Instead of blindly duplicating tokens, we **score** every contiguous chunk within each block and select the top-$k$ most informative ones. This produces a compact, content-aware summary that tells later blocks what earlier blocks contain.

### Why Contiguous Chunks Instead of Individual Tokens

An earlier version selected individual top-scoring tokens (bag-of-words). This caused two problems:

1. **Destroyed syntactic structure**: The Transformer saw isolated, disconnected token IDs. Without local context, the self-attention mechanism cannot form meaningful keys/values — the summary tokens behave as noise.
2. **TF-IDF score duplication**: When selecting individual tokens by TF-IDF, every occurrence of the same token receives the same score. `torch.topk` would select $k$ copies of the same token, wasting the entire summary budget on a single word.

Moving to **contiguous chunks** solves both:
- Each chunk preserves local word order and sub-sentence structure, which Transformers rely on.
- Scoring is per-chunk, not per-token, so the top-$k$ selection picks $k$ *distinct regions* of the block.

### Chunk Scoring

Each block of $L$ tokens is divided into $\lceil L / \texttt{chunk\_size} \rceil$ non-overlapping chunks of `chunk_size` tokens (default 32). Each chunk is scored as a unit. The top `num_chunks` (default 4) are selected per block and returned **in positional order** to preserve natural reading order for the Transformer.

---

## 4. Statistical Heuristics — Theory and Reasoning

### 4.1 TF-IDF (default)

**Formula (per chunk)**:

$$\text{score}(\text{chunk}) = \frac{1}{|\text{chunk}|} \sum_{t \in \text{chunk}} \text{IDF}(t)$$

where $\text{IDF}(t) = \log\!\left(\frac{N}{\text{df}(t)}\right)$, $N$ is the number of blocks, and $\text{df}(t)$ is the number of blocks containing token $t$.

**Equivalence to mean TF-IDF over types**: The naive formula $\frac{1}{|\text{chunk}|}\sum_{\text{types}} \text{TF}(t) \cdot \text{IDF}(t)$ expands identically because each position contributes $\frac{1}{|\text{chunk}|} \cdot \text{IDF}(\text{token at that position})$, and summing over all positions equals summing over unique types weighted by their raw count. This allows a single vectorized lookup: `idf_table[chunk_tokens].sum() / chunk_len`.

**Why TF-IDF**: It measures **global rarity** — tokens that appear in few blocks get high IDF values, so chunks containing rare, block-specific vocabulary (proper nouns, technical terms, NIAH needles) are promoted. Common stop-words and repeated filler tokens receive near-zero IDF and are naturally suppressed.

**When it works best**: General-purpose contexts with a mix of boilerplate and block-specific entities.

### 4.2 BM25

**Formula (per chunk)**:

$$\text{score}(\text{chunk}) = \sum_{t \in \text{types}} \text{IDF}_{\text{BM25}}(t) \cdot \frac{\text{tf}(t) \cdot (k_1 + 1)}{\text{tf}(t) + k_1 \cdot \left(1 - b + b \cdot \frac{|\text{chunk}|}{\text{avgdl}}\right)}$$

where $k_1 = 1.2$, $b = 0.75$, and $\text{avgdl}$ is the average chunk length across the corpus.

The BM25 IDF variant is:

$$\text{IDF}_{\text{BM25}}(t) = \log\!\left(\frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} + 1\right)$$

**Why BM25**: BM25 adds **length normalisation** and **diminishing returns for repeated terms**. A chunk that contains 10 occurrences of a rare word does not get 10× the credit — the saturation parameter $k_1$ soft-caps the benefit. The $b$ parameter penalises longer chunks (which have more tokens by chance). This makes BM25 more robust than raw TF-IDF when chunk lengths vary (the last chunk in a block is often shorter).

**When it works best**: Documents with variable-length natural text where term frequency saturation matters.

### 4.3 Entropy (Lexical Diversity)

**Formula (per chunk)**:

$$\text{score}(\text{chunk}) = \frac{|\{t : t \in \text{chunk}\}|}{|\text{chunk}|}$$

This is the **type-token ratio** (TTR), a fast proxy for lexical diversity implemented as `(bincount > 0).sum() / chunk_len`.

**Why Entropy**: This heuristic is **corpus-independent** — it does not require global document frequencies. A chunk full of padding or repeated stop-words (e.g., "the the the…") has unique_count → 1, score → 0. A chunk packed with diverse entities and distinct vocabulary scores near 1. It is the fastest heuristic to compute and works well when the primary goal is to avoid selecting degenerate, repetitive chunks.

**When it works best**: Scenarios where the corpus is small (few blocks) and IDF estimates would be noisy, or when the goal is simply to select "dense" text regions.

### 4.4 Max-IDF (Keyword / Needle Detection)

**Formula (per chunk)**:

$$\text{score}(\text{chunk}) = \max_{t \in \text{chunk}} \text{IDF}(t)$$

**Why Max-IDF**: A single hyper-rare token — a UUID, unusual proper name, or Needle-in-a-Haystack (NIAH) insertion — is sufficient to make the entire chunk critical. Unlike TF-IDF which averages over all positions (diluting the effect of one rare token among 31 common ones), Max-IDF is a **pure needle detector**: if the chunk contains even one token that appears in very few blocks, it gets the maximum possible score.

**When it works best**: Retrieval tasks (NIAH, multi-key retrieval) where a single rare identifier determines the answer. It is the most aggressive heuristic — it prioritises rare-token presence over general informativeness.

### Heuristic Comparison Summary

| Heuristic | Signal | Strength | Weakness |
|---|---|---|---|
| **TF-IDF** | Global rarity (average) | Balanced, general-purpose | Diluted by common tokens in the same chunk |
| **BM25** | Length-normalised rarity with TF saturation | Robust to variable chunk lengths | Slightly more compute (per-chunk bincount) |
| **Entropy** | Local diversity (type-token ratio) | Corpus-independent, fast | Ignores global rarity entirely |
| **Max-IDF** | Maximum single-token rarity | Perfect for needle detection | Over-indexes on one token; ignores overall chunk quality |

---

## 5. Causal-Only Summaries (S_post Removal)

### The Bug

An earlier version of the code assembled each block's input as:

$$[S_{\text{pre}} \;|\; B_i \;|\; S_{\text{post}}]$$

where $S_{\text{post}}$ contained summaries from blocks *after* $B_i$. After sorting by position ID, $S_{\text{post}}$ tokens always ended up at sequence indices **after** $B_i$'s tokens because their position IDs ($> \text{end}(B_i)$) are strictly greater.

Under causal (autoregressive) attention — which operates on **tensor index, not position ID** — $B_i$'s tokens cannot attend to any token at a higher index. Therefore, $S_{\text{post}}$ tokens were **invisible** to $B_i$: pure wasted computation.

### The Fix

We removed $S_{\text{post}}$ entirely. Only pre-block summaries ($j < i$) are prepended. This:

- Eliminates wasted FLOPS (no dead tokens in the forward pass).
- Ensures compatibility with the causal mask without custom masking.
- Simplifies the sequence assembly to a monotonically increasing position ID order by construction, eliminating the need for an explicit `argsort`.

---

## 6. KV Cache Discard Strategy

### Why Discard Summary KV States

After the forward pass, each block's output includes KV states for **all** tokens in its input — both $B_i$'s own tokens and the prepended sink/summary tokens. If we keep the summary KV states, Phase 2 will attend to them during global attention, which:

1. **Corrupts the cache total**: The combined KV cache across all GPUs would contain overlapping summary tokens (since $S_j$ is prepended to every block after $j$), breaking the assumption that each token appears exactly once.
2. **Increases memory**: Summary tokens are already represented in their original block's KV cache. Keeping duplicates scales memory linearly with the number of blocks.

### Implementation

After the forward pass, a boolean mask `is_bi_sorted` identifies which KV positions belong to $B_i$ (the last `block_size` positions). Summary and sink KV states are sliced out:

```python
kv_block = [
    [x[0][:, :, is_bi_sorted], x[1][:, :, is_bi_sorted]] for x in kv_block
]
```

This can be toggled off with `--no_discard_summary_kv` for experimentation (e.g., to measure if the baked-in summary influence degrades Phase 2 quality).

### Cache Format Fix

An earlier implementation left `kv_block` as a Python `List[List[Tensor]]` after slicing, which caused `LlamaModel.forward` to fall back to the deprecated `DynamicCache.from_legacy_cache()` path. We now explicitly wrap the final KV cache in a `DynamicCache` object to ensure consistent API usage.

---

## 7. Vectorized Computation

### Previous Approach

The original `compute_block_summaries` used Python `Counter` objects and CPU-side loops to compute term frequencies and IDF scores. This introduced Python-loop overhead proportional to vocabulary size × number of blocks.

### Current Approach

All scoring is now **fully vectorized in PyTorch**:

| Operation | Implementation |
|---|---|
| Document frequency table | `blk.unique()` per block → accumulate into `df_table[unique_ids]` |
| IDF table | `torch.log(N / df_table.clamp(min=1))` — dense tensor lookup |
| TF-IDF chunk score | `idf_table[chunk_tokens].sum() / chunk_len` |
| BM25 chunk score | `torch.bincount` for raw TF → vectorized BM25 formula |
| Entropy score | `(bincount > 0).sum() / chunk_len` |
| Max-IDF score | `idf_table[chunk_tokens].max()` |

This eliminates all `Counter` objects and `.tolist()` calls. The only remaining Python loop is over chunks within a block (typically $\leq$ 128 iterations), which is negligible compared to transformer compute.

---

## 8. Overhead Analysis

With default parameters (`block_size=4096`, `num_chunks=4`, `chunk_size=32`, `sink_size=64`):

- **Summary tokens per block**: $4 \times 32 = 128$ tokens
- **Worst-case block input** (last block in a 16-block sequence): $64_{\text{sink}} + 15 \times 128_{\text{summaries}} + 4096_{\text{block}} = 6080$ tokens
- **Summary computation**: $O(\text{block\_size})$ per block — dominated by `torch.bincount` and `idf_table` lookup. Negligible relative to transformer forward pass.
- **Memory**: Summary KV states are discarded, so the final KV cache size equals the original context length.

---

## 9. Design Decisions Summary

| Decision | Rationale |
|---|---|
| Chunks over individual tokens | Preserves syntactic structure; avoids topk deduplication issue |
| Positional-order return | Transformer sees natural reading order → healthy KV states |
| Causal summaries only ($j < i$) | Matches causal mask; avoids dead computation |
| Attention sinks (64 tokens) | Stabilises softmax without anchor overhead |
| KV discard by default | Prevents duplicate tokens in Phase 2 global attention |
| 4 heuristics | Covers different retrieval profiles (general, length-normalised, diversity, needle) |
| Contiguous chunk selection | Each chunk is a self-contained text region the Transformer can meaningfully attend to |
