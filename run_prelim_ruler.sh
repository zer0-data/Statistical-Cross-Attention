#!/usr/bin/env bash
# Single-process, single-GPU RULER v1 prelim sweep.
# Model loads ONCE; hyperparameters mutate in-place between cells.
#
# Dimensions:
#   summary_method : anchor, max_idf, entropy
#                    anchor    — NVIDIA vanilla Star Attention reference
#                    max_idf   — best-performing heuristic on RULER
#                    entropy   — lexical-diversity baseline
#   summary_chunks : scales with seq_len to maintain 12.5% budget
#                    (overridden per seq_len inside run_prelim_ruler.py
#                     when --summary_chunks is omitted; pass explicitly
#                     to fix a single budget)
#   seq_lengths    : 16384, 32768, 65536, 131072
#   tasks          : multihop NIAH + variable tracking
#
# Fixed (match 4-block design used in all paper experiments):
#   block_size = seq_len / 4   (set via --block_size per run)
#   chunk_size = 32, sink_size = 64, num_samples = 100
#
# Env overrides:
#   MODEL_PATH    HF id or local path
#   RESULTS_FILE  output log (default: prelim_ruler_accuracies.txt)

set -u

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
RESULTS_FILE="${RESULTS_FILE:-prelim_ruler_accuracies.txt}"

TASKS="niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt"

# 16K  (block=4K,  summary_chunks=16)
python run_prelim_ruler.py \
  --model_path "$MODEL_PATH" \
  --prompt_config meta-llama3 \
  --methods anchor,max_idf,entropy \
  --summary_chunks 16 \
  --seq_lengths 16384 \
  --tasks $TASKS \
  --block_size 4096 \
  --anchor_block_size 4096 \
  --chunk_size 32 \
  --sink_size 64 \
  --num_samples 100 \
  --results_file "$RESULTS_FILE"

# 32K  (block=8K,  summary_chunks=32)
python run_prelim_ruler.py \
  --model_path "$MODEL_PATH" \
  --prompt_config meta-llama3 \
  --methods anchor,max_idf,entropy \
  --summary_chunks 32 \
  --seq_lengths 32768 \
  --tasks $TASKS \
  --block_size 8192 \
  --anchor_block_size 8192 \
  --chunk_size 32 \
  --sink_size 64 \
  --num_samples 100 \
  --results_file "$RESULTS_FILE"

# 64K  (block=16K, summary_chunks=64)
python run_prelim_ruler.py \
  --model_path "$MODEL_PATH" \
  --prompt_config meta-llama3 \
  --methods anchor,max_idf,entropy \
  --summary_chunks 64 \
  --seq_lengths 65536 \
  --tasks $TASKS \
  --block_size 16384 \
  --anchor_block_size 16384 \
  --chunk_size 32 \
  --sink_size 64 \
  --num_samples 100 \
  --results_file "$RESULTS_FILE"

# 128K  (block=32K, summary_chunks=128)
python run_prelim_ruler.py \
  --model_path "$MODEL_PATH" \
  --prompt_config meta-llama3 \
  --methods anchor,max_idf,entropy \
  --summary_chunks 128 \
  --seq_lengths 131072 \
  --tasks $TASKS \
  --block_size 32768 \
  --anchor_block_size 32768 \
  --chunk_size 32 \
  --sink_size 64 \
  --num_samples 100 \
  --results_file "$RESULTS_FILE"
