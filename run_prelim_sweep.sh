#!/usr/bin/env bash
# Single-process, single-GPU BABILong 16K prelim sweep.
# Model loads ONCE; hyperparameters mutate in-place between cells.
#
# Dimensions (all defaulted in run_prelim_babilong.py):
#   summary_method : tfidf, bm25, entropy, max_idf
#   summary_chunks : 2, 4, 8
#   tasks          : qa1, qa3, qa5
# Fixed:
#   block_size=4096, sink_size=64, chunk_size=32, dataset_config=16k
#
# Total: 4 * 3 * 3 = 36 cells, all within one Python process.
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
  --methods tfidf,bm25,entropy,max_idf \
  --summary_chunks 2,4,8 \
  --tasks qa1,qa3,qa5 \
  --block_size 4096 \
  --sink_size 64 \
  --chunk_size 32 \
  --dataset_config 16k \
  --max_new_tokens 64 \
  --num_samples "$NUM_SAMPLES" \
  --results_file "$RESULTS_FILE"
