#!/usr/bin/env bash
# Single-process, single-GPU BABILong 16K prelim sweep.
# Model loads ONCE; hyperparameters mutate in-place between cells.
#
# Dimensions (all defaulted in run_prelim_babilong.py):
#   summary_method : tfidf, bm25, entropy, max_idf, evenly_spaced
#                    (evenly_spaced is the non-semantic baseline; mean_pool
#                     is intentionally excluded because it ignores chunk_size
#                     and has a different budget shape — run separately)
#   summary_chunks : 4, 16, 32   (caps max summary budget at ~3072 tokens;
#                                  the 4096 ceiling corresponds to K≈42)
#   tasks          : qa1, qa3, qa5
# Fixed:
#   block_size=4096, sink_size=64, chunk_size=32, dataset_config=16k
#
# With 16K context and block_size=4096 there are 4 blocks; the last block
# sees summaries from 3 prior blocks. Peak summary budget at the last block:
#   max_budget = 3 * summary_chunks * chunk_size  =  96 * K
#   K=4  ->   384 tokens
#   K=16 ->  1536 tokens
#   K=32 ->  3072 tokens (~75% of 4096 cap)
#
# Total: 5 * 3 * 3 = 45 cells, all within one Python process.
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
  --methods tfidf,bm25,entropy,max_idf,evenly_spaced \
  --summary_chunks 4,16,32 \
  --tasks qa1,qa3,qa5 \
  --block_size 4096 \
  --sink_size 64 \
  --chunk_size 32 \
  --dataset_config 16k \
  --max_new_tokens 64 \
  --num_samples "$NUM_SAMPLES" \
  --results_file "$RESULTS_FILE"
