#!/usr/bin/env bash
# Single-process, single-GPU BABILong 16K prelim sweep.
# Model loads ONCE; hyperparameters mutate in-place between cells.
#
# Dimensions (all defaulted in run_prelim_babilong.py):
#   summary_method : anchor, tfidf, bm25, entropy, max_idf, evenly_spaced
#                    anchor         — NVIDIA vanilla Star Attention reference
#                    evenly_spaced  — non-semantic (uniform) baseline
#                    mean_pool is intentionally excluded because it ignores
#                    chunk_size and has a different budget shape — run
#                    separately.
#   summary_chunks : 4, 16, 32   (caps max summary budget at ~3072 tokens;
#                                  the 4096 ceiling corresponds to K≈42)
#   tasks          : qa1, qa3, qa5
# Fixed:
#   block_size=4096, sink_size=64, chunk_size=32, dataset_config=16k
#
# With 16K context and block_size=4096 there are 4 blocks; the last block
# sees summaries from 3 prior blocks. Peak summary budget at the last block:
#   - scoring / evenly_spaced :  3 * K * chunk_size = 96 * K
#   - anchor                  :      K * chunk_size (block 0 only; constant
#                                    across i>0 since other blocks contribute
#                                    no summary). Max:  K=32 -> 1024 tokens
#
# Total: 6 * 3 * 3 = 54 cells, all within one Python process.
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
  --tasks qa1,qa3,qa5 \
  --block_size 4096 \
  --sink_size 64 \
  --chunk_size 32 \
  --dataset_config 16k \
  --max_new_tokens 64 \
  --num_samples "$NUM_SAMPLES" \
  --results_file "$RESULTS_FILE"
