#!/usr/bin/env bash
# Single-process, single-GPU BABILong 16K prelim sweep.
# Model loads ONCE; hyperparameters mutate in-place between cells.
#
# Dimensions (all defaulted in run_prelim_babilong.py):
#   summary_method : anchor, tfidf, bm25, entropy, max_idf, evenly_spaced
#                    anchor         — NVIDIA vanilla Star Attention reference
#                    evenly_spaced  — non-semantic (uniform) baseline
#                    mean_pool is intentionally excluded (different budget
#                    shape).
#   summary_chunks : 4, 16, 32   (caps max summary budget at ~3072 tokens;
#                                  the 4096 ceiling corresponds to K≈42)
#   position_modes : sparse, contiguous
#                    sparse     — global position IDs preserved (gaps at
#                                 chunk boundaries)
#                    contiguous — NVIDIA-style per-block renumbering 0..L-1
#                                 + Phase 2 query shifted past max L
#   tasks          : qa1, qa3, qa5
# Fixed:
#   block_size=4096, sink_size=64, chunk_size=32, dataset_config=16k
#
# With 16K context and block_size=4096 there are 4 blocks; the last block
# sees summaries from 3 prior blocks. Peak summary budget at the last block:
#   - scoring / evenly_spaced :  3 * K * chunk_size = 96 * K
#   - anchor                  :  block_size - sink_size = 4032 tokens
#                                (constant; num_chunks / chunk_size ignored)
#
# Total: 2 * 6 * 3 * 3 = 108 cells. Anchor cells are identical across K, so
# each (position_mode, task) group has 3 redundant anchor cells — useful as
# a determinism sanity check and to verify contiguous vs sparse under the
# reference method.
#
# Env overrides:
#   MODEL_PATH    HF id or local path (default: meta-llama/Llama-3.1-8B-Instruct)
#   NUM_SAMPLES   samples per cell (default: 100)
#   RESULTS_FILE  where to append one line per cell (default: prelim_accuracies.txt)

set -u

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
RESULTS_FILE="${RESULTS_FILE:-prelim_accuracies.txt}"

python run_prelim_babilong.py \
  --model_path "$MODEL_PATH" \
  --methods anchor,tfidf,bm25,entropy,max_idf,evenly_spaced \
  --summary_chunks 4,16,32 \
  --position_modes sparse,contiguous \
  --tasks qa1,qa3,qa5 \
  --block_size 4096 \
  --sink_size 64 \
  --chunk_size 32 \
  --dataset_config 16k \
  --max_new_tokens 64 \
  --num_samples "$NUM_SAMPLES" \
  --results_file "$RESULTS_FILE"
