# Star Attention: Efficient LLM Inference over Long Sequences
## Extended with Statistical Cross-Attention Block Summaries

This repository contains code based on the paper [Star Attention: Efficient LLM Inference over Long Sequences](https://arxiv.org/abs/2411.17116), extended with **Statistical Cross-Attention** — a method that replaces the fixed anchor block with lightweight statistical block summaries to improve cross-block semantic awareness during parallel KV cache construction.

The method operates in two phases:
1. **Phase 1 - Context Encoding**: The context tokens are processed using blockwise-local attention. Each block is augmented with the top-scoring contiguous chunks from earlier blocks (selected via TF-IDF or BM25), providing global semantic hints while preserving parallelism.
2. **Phase 2 - Query Processing and Token Generation**: The query and response tokens attend to all prior cached tokens through sequence-global attention (unchanged).

Star Attention **improves the inference time by up to 11x** while **preserving 97-100% of accuracy**. The method is **compatible with most Transformer-based LLMs trained with global attention, operating seamlessly out-of-the-box without additional training/finetuning.** Furthermore, Star Attention is **orthogonal to other optimization methods**, including Flash Attention and KV cache compression techniques, allowing for potential combined enhancements.

This codebase contains the implementation of Star Attention in PyTorch using the [HuggingFace Transformers](https://github.com/huggingface/transformers) library, along with the code for launching inference with Star Attention on two benchmarks: RULER and BABILong.

<div align="center">
  <table>
      <thead>
          <tr>
              <th rowspan="2" style="text-align: center">Model</th>
              <th rowspan="2" style="text-align: center">Seq. Len.<br>(K)</th>
              <th rowspan="2" style="text-align: center">Block Size<br>(K)</th>
              <th rowspan="2" style="text-align: center">Ring-Attn<br>Acc. (%)</th>
              <th colspan="2" style="text-align: center">Star-Attn</th>
          </tr>
          <tr>
              <th style="text-align: center">Δ Acc.</th>
              <th style="text-align: center">Δ Speedup</th>
          </tr>
      </thead>
      <tbody>
          <tr>
              <td rowspan="4">meta-llama<br>Llama3.1-8B-Instruct</td>
              <td style="text-align: center">16</td>
              <td style="text-align: center">4</td>
              <td style="text-align: center">92.22</td>
              <td style="text-align: center">-0.94%</td>
              <td style="text-align: center">1.1x</td>
          </tr>
          <tr>
              <td style="text-align: center">32</td>
              <td style="text-align: center">8</td>
              <td style="text-align: center">87.53</td>
              <td style="text-align: center">+1.17%</td>
              <td style="text-align: center">1.2x</td>
          </tr>
          <tr>
              <td style="text-align: center">64</td>
              <td style="text-align: center">16</td>
              <td style="text-align: center">84.79</td>
              <td style="text-align: center">-1.42%</td>
              <td style="text-align: center">1.8x</td>
          </tr>
          <tr>
              <td style="text-align: center">128</td>
              <td style="text-align: center">32</td>
              <td style="text-align: center">76.31</td>
              <td style="text-align: center">-1.90%</td>
              <td style="text-align: center">2.7x</td>
          </tr>
          <tr>
              <td rowspan="3">meta-llama<br>Llama-3.1-70B-Instruct</td>
              <td style="text-align: center">16</td>
              <td style="text-align: center">4</td>
              <td style="text-align: center">95.09</td>
              <td style="text-align: center">-2.71%</td>
              <td style="text-align: center">1.7x</td>
          </tr>
          <tr>
              <td style="text-align: center">32</td>
              <td style="text-align: center">8</td>
              <td style="text-align: center">94.61</td>
              <td style="text-align: center">-2.55%</td>
              <td style="text-align: center">2.0x</td>
          </tr>
          <tr>
              <td style="text-align: center">64</td>
              <td style="text-align: center">16</td>
              <td style="text-align: center">88.54</td>
              <td style="text-align: center">-1.44%</td>
              <td style="text-align: center">4.7x</td>
          </tr>
      </tbody>
  </table>
  <p align="justify">
    <b>Table 1:</b> Accuracy and relative inference speedup of Star Attention compared to Ring Attention on RULER across sequence lengths from 16K to 128K. Accuracy is reported as the absolute difference from Ring Attention and speedup reflects relative improvements in inference efficiency. Star Attention significantly accelerates inference with minimal accuracy loss.
  </p>
</div>
<br>
<div align="center">
  <img
    src="images/star_attn_acc_ruler_babilong.png"
    alt="star attention accuracy on ruler and babilong"
  />
  <p align="justify">
    <b>Figure 1:</b> Accuracy comparison of Star Attention and Global Attention on RULER and BABILong from 16K to 128K sequence lengths using various models. All runs use a block and anchor block size set to one-quarter of the total sequence length. Star Attention maintains 97-100% of the accuracy of global attention, and in some cases, even outperform it.
  </p>
</div>

## Table of Contents
1. [Setup Instructions](#setup-instructions)
   - [Dependencies](#dependencies)
   - [RULER Setup](#ruler-setup)
   - [Downloading Models](#downloading-models)
2. [Statistical Cross-Attention](#statistical-cross-attention)
   - [How It Works](#how-it-works)
   - [Configuration](#configuration)
   - [Overhead](#overhead)
3. [Launching Inference with Star Attention](#launching-inference-with-star-attention)
   - [RULER](#ruler)
   - [BABILong](#babilong)
   - [Running Inference on Custom Data](#running-inference-on-custom-data)
4. [Two Phases of Star Attention](#two-phases-of-star-attention)
   - [Phase 1 - Context Encoding](#phase-1---context-encoding)
   - [Phase 2 - Query Processing and Token Generation](#phase-2---query-processing-and-token-generation)
5. [Repository Structure](#repository-structure)
6. [Citation](#citation)
7. [References](#references)
8. [Contact/Getting Help](#contactgetting-help)


## Setup Instructions

### Dependencies

Install all the project dependencies with
```
$ pip install -r requirements.txt
```

In a python shell, download the `punkt` tokenizer from the `nltk` library:
```
import nltk
nltk.download('punkt_tab')
```

### RULER Setup

To generate synthetic data for RULER, you need to download:
- Paul Graham Essays for NIAH from [NIAH Github](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main/needlehaystack/PaulGrahamEssays) and [Paul Graham Blog](https://paulgraham.com/articles.html).
- QA datasets from [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) and [HotpotQA](https://hotpotqa.github.io/).

To download these data, run:
```
$ bash ruler/download_data.sh
```

### Downloading Models

To download a model from HuggingFace, use the script: [`scripts/download_hf_model.py`](scripts/download_hf_model.py).

*NOTE: For certain models, you might need to input the huggingface hub token from your account settings via the `--token` flag.*

## Statistical Cross-Attention

Statistical Cross-Attention replaces the fixed anchor block with lightweight, non-neural summaries. During Phase 1, each block's context is augmented with the most informative contiguous chunks from earlier blocks, providing cross-block semantic awareness while preserving parallelism.

### How It Works

Each block is divided into non-overlapping **fixed-size chunks** (default: 32 tokens each). Every chunk is scored as a unit using one of four heuristics:

- **TF-IDF** (default): Computes mean per-token TF-IDF across the chunk, using corpus-level document frequencies computed over all blocks.
- **BM25**: Scores each chunk using BM25 with `k1=1.2`, `b=0.75`, and average chunk length normalisation.
- **Entropy**: Scores each chunk by `unique_token_count / chunk_len` — a vectorized proxy for lexical diversity. Chunks full of padding or repeated stop-words score near 0; entity-rich, information-dense chunks score near 1.
- **Max-IDF**: Scores each chunk by the maximum IDF value of any token it contains. A single hyper-rare token (UUID, unusual name, NIAH needle) is sufficient to push the chunk to the top, even if the surrounding tokens are generic filler.

The top `num_chunks` highest-scoring chunks per block are selected and returned **in positional order**, preserving natural language structure so the Transformer produces healthy K/V states. Each summary token retains its **original position ID** for RoPE compatibility.

Only **pre-block summaries** are used — summaries from block $j$ are only prepended to block $i$ where $j < i$. This ensures compatibility with the causal mask.

```
GPU0: [B0]                           # block 0 (no summaries)
GPU1: [S0 | B1]                      # chunks from block 0  + block 1
GPU2: [S0, S1 | B2]                  # chunks from blocks 0-1 + block 2
GPU3: [S0, S1, S2 | B3]              # chunks from blocks 0-2 + block 3
```

After the forward pass, only `B_i`'s KV states are retained — all summary KV states are discarded by default.

### Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--summary_chunks` | `4` | Number of contiguous chunks to select per block |
| `--chunk_size` | `32` | Number of tokens per summary chunk |
| `--summary_method` | `tfidf` | Chunk scoring heuristic (`tfidf`, `bm25`, `entropy`, or `max_idf`) |
| `--sink_size` | `64` | Number of leading tokens always injected as attention sinks into every block's context (set to `0` to disable) |
| `--no_discard_summary_kv` | off | If set, keeps summary KV states in the final cache |

The total number of summary tokens per block is `summary_chunks × chunk_size` (default: `4 × 32 = 128`).

### Overhead

With `b=4096`, `num_chunks=4`, `chunk_size=32`, `m=16` blocks: each GPU processes at most `b + (m-1) × num_chunks × chunk_size = 4096 + 15 × 128 = 6016` tokens (for the last block). Summary computation is `O(block_size)` per block, negligible vs transformer compute.

## Launching Inference with Star Attention

This repository contains code for launching inference with Star Attention on two benchmarks: RULER and BABILong. The instructions to run each of those benchmarks are shared in the following subsections.

### RULER
[ [Paper](https://arxiv.org/abs/2404.06654) | [GitHub](https://github.com/hsiehjackson/RULER) ]

To run inference on RULER, use the script: [`run_ruler.py`](run_ruler.py).

Usage:
```
$ python run_ruler.py \
    -n <experiment_name> \
    -p <path_to_model> \
    -pc <prompt_template_type> \
    -a star \
    -bs <context_block_size> \
    -l <list_of_sequence_lengths_to_run_inference> \
    -np <num_parallel_processes_per_node> \
    --output_dir <output_directory>
```

After running the evaluations, you can display all the results together using the script:
```
$ python ruler/gather_results_ruler.py \
    -e <path_to_results_directory>
```

#### Configuring the `-np` flag

The `-np` flag specifies the number of parallel processes (hosts) to use for running inference. For example, if your machine has 8 GPUs:
- `-np 8`: Launch 8 hosts, each host assigned a single GPU.
- `-np 4`: Launch 4 hosts, each host assigned 2 GPUs.

This is useful when you want to run star attention with bigger context block sizes or with bigger models where assigning a single GPU per host leads to out-of-memory errors.

To see an example of how to run this script with different configurations, check [`launch.sh`](launch.sh).

#### Configuring the `-nn` flag for Multi-Node Inference

If you have a multi-node setup (such as a slurm cluster), then you can add the `-nn <num_nodes>` for running multi-node inference. The script will launch a total of `nn * np` processes (hosts) for inference.

For example, in a system with 8 GPUs on each node, if `-nn 1` and `-np 4` are specified, the script will launch 4 processes (hosts) on a single node. This means that each host is allocated 2 GPUs to load the model. If the model or the block size is too large, you can scale the `nn` and the `np` parameters accordingly. If `-nn 2` and `-np 2`, then the script will launch a total of 4 processes (hosts) across 2 nodes, with each host containing 4 GPUs.

### BABILong
[ [Paper](https://arxiv.org/abs/2406.10149) | [GitHub](https://github.com/booydar/babilong) ]

To run inference on BABILong, use the script: [`run_babilong.py`](run_babilong.py).

The script takes in the same set of arguments as the RULER script described above. To see an example of how to run this script with different configurations, check [`launch.sh`](launch.sh).

After running the evaluations, you can display all the results together using the script:
```
$ python babilong/gather_results_babilong.py \
    -e <path_to_results_directory>
```

### Running Inference on Custom Data

To run inference on your custom data, use the script: [`run_star_attn_inference.py`](run_star_attn_inference.py). The scripts takes in the input data in `.jsonl` format in which each line of jsonl should look like:
```json
{
  "index": "<sample index>",
  "input_context": "<long context portion of the input sample>",
  "input_query": "<query portion of the input sample>",
  "output": "<expected output response>"
}
```

Script usage:
```
$ torchrun --nproc_per_node=4 run_star_attn_inference.py \
    --model_path <path_to_model> \
    --attn_type star \
    --block_size 4096 \
    --summary_chunks 4 \
    --chunk_size 32 \
    --summary_method tfidf \
    --tokens_to_generate <num_tokens_to_generate> \
    --stop_words <end_of_sequence_tokens> \
    --input_path <path_to_input_jsonl> \
    --output_path <path_to_output_jsonl>
```

For more details on the script arguments, run:
```
$ python run_star_attn_inference.py --help
```

## Two Phases of Star Attention

Given a system with $H$ hosts and an input sample with context $c$ followed by query $q$, Star Attention operates in two phases:

### Phase 1 - Context Encoding (with Statistical Summaries)

- The context is segmented into contiguous blocks:
  <div align="center">
    $$c = [c_1, c_2, \ldots, c_n]$$
  </div>
- For each block $c_i$, a statistical summary $S_i$ is computed by dividing $c_i$ into fixed-size chunks and selecting the top-scoring chunks using TF-IDF or BM25, preserving their original position IDs.
- Each block is augmented with summaries from earlier blocks only (future summaries are omitted because the causal mask prevents $c_i$ from attending to tokens at higher sequence indices):
  <div align="center">
    $$c'_i = [S_1, \ldots, S_{i-1}, c_i]$$
  </div>
- The augmented context blocks are distributed across the $H$ hosts, with each host attending only to its assigned blocks.
  - After processing, each host stores **only** $c_i$'s KV cache — all summary KV states are discarded (unless `--no_discard_summary_kv` is set).

### Phase 2 - Query Processing and Token Generation

<div align="center">
  <img
    src="images/star_attn_phase2.png"
    alt="star attention phase 2"
  />
</div>
<br />

- Designate one host as the *query* host $h_q$.
- Replicate the query tokens to all the hosts where each host first attends to its locally stored KV cache from phase 1.
  <div align="center">
    $$A_h = \left( \frac{\exp\left( \frac{QK_h^\top}{\sqrt{d}} \right)}{\sum_{k=1}^{l_k} \exp\left( \frac{QK_{h,k}^\top}{\sqrt{d}} \right)} \right)V_h$$
  </div>
- In addition to the local attention output $A_h$, the hosts also store the sum of exponents from local softmax (i.e. denominator from the equation above).
  <div align="center">
    $$s_h = \sum_{k=1}^{l_k} \exp\left( \frac{QK_{h,k}^\top}{\sqrt{d}} \right)$$
  </div>
- The query-host $h_q$ then gathers both the sum of exponents $s_h$ and the local attention output $A_h$ from all hosts:
  <div align="center">
    $$s = [s_1, s_2, \ldots, s_{H}]$$
    <br/>
    $$A = [A_1, A_2, \ldots, A_{H}]$$
  </div>
- To compute global attention, the query-host first calculates the global sum of exponents (i.e., the global softmax denominator) as:
  <div align="center">
    $$s_{\text{global}} = \sum_{h=1}^{H} s_h$$
  </div>
- Using this global sum, the query-host computes the final global attention output as:
  <div align="center">
    $$A_{\text{global}} = \sum_{h=1}^{H} \frac{s_h}{s_{\text{global}}} A_h$$
  </div>

This method ensures that attention scores are correctly normalized across all hosts, requiring only the communication of a single scalar (the sum of exponents, $s_h$) and a vector (the local attention output, $A_h$) per token.


## Repository Structure

```
Statistical-Cross-Attention/
├── model.py                        # Model classes and summary computation
│   ├── compute_block_summaries()   # Chunk-based TF-IDF / BM25 scoring
│   ├── DistributedInferenceBaseModel
│   ├── StarAttentionModel          # Phase 1 & 2 with summaries
│   ├── DenseAttentionModel         # Standard dense attention
│   └── RingAttentionModel          # Ring attention baseline
├── run_star_attn_inference.py      # CLI entry point for custom data
├── run_ruler.py                    # RULER benchmark launcher
├── run_babilong.py                 # BABILong benchmark launcher
├── launch.sh                       # Example launch configurations
├── star_attention/
│   ├── modeling_llama.py           # Modified LlamaForCausalLM with Star/Ring attn
│   ├── modeling_flash_attention_utils.py
│   ├── star_flash_attn/            # Star attention flash kernels
│   └── ring_flash_attn/            # Ring attention flash kernels
├── ruler/                          # RULER benchmark data & evaluation
├── babilong/                       # BABILong benchmark data & evaluation
├── scripts/
│   ├── download_hf_model.py        # Model download utility
│   └── time_inference.py           # Inference timing script
├── requirements.txt
└── README.md
```

## Citation
```
@inproceedings{
  acharya2025starattention,
  title={Star Attention: Efficient {LLM} Inference over Long Sequences},
  author={Shantanu Acharya and Fei Jia and Boris Ginsburg},
  booktitle={Forty-second International Conference on Machine Learning (ICML)},
  year={2025},
}
```

## References

- Llama Implementation: [huggingface/transformers](https://github.com/huggingface/transformers)
- Ring Attention Implementation: [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention)

## Contact/Getting Help

If you need any help or want to report a bug, feel free to raise an issue in the repo.
