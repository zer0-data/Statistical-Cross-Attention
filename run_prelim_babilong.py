"""Single-GPU preliminary BABILong sweep for Statistical Cross-Attention.

Loads the BABILong dataset from the HuggingFace hub (`RMT-team/babilong`) and
sweeps (summary_method x summary_chunks x task) in a single Python process:
  - model loaded ONCE
  - datasets loaded ONCE
  - between each (method, chunks, task) cell we only mutate the relevant
    attributes on the StarAttentionModel instance and clear transient CUDA
    state; the 16 GB of model weights stay resident on the GPU

This relies on the single-GPU fallbacks added in:
  - model._init_distributed (world_size=1 / rank=0)
  - star_attention/modeling_llama.py (RANK=0, WORLD_SIZE=1 module-level fallback)
  - star_attention/star_flash_attn/star_flash_attn.py (skip all_gather on WS=1)
so no torchrun / dist.init_process_group is required.

Example:
  python run_prelim_babilong.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --methods tfidf,bm25,entropy,max_idf \
    --summary_chunks 2,4,8 \
    --tasks qa1,qa3,qa5 \
    --num_samples 100
"""

import argparse
import datetime
import gc
import sys
import traceback
from typing import List

import torch
from datasets import load_dataset

from model import StarAttentionModel


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _csv(values: str) -> List[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def _csv_int(values: str) -> List[int]:
    return [int(v) for v in _csv(values)]


def _reset_cuda():
    """Release cached allocator blocks and any dangling references."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# --------------------------------------------------------------------------- #
# One cell of the sweep
# --------------------------------------------------------------------------- #

def run_cell(
    model: StarAttentionModel,
    ds,
    *,
    task: str,
    method: str,
    summary_chunks: int,
    chunk_size: int,
    block_size: int,
    sink_size: int,
    anchor_block_size: int,
    discard_summary_kv: bool,
    position_mode: str,
    start_sample_index: int,
    num_samples: int,
    results_file: str,
    model_tag: str,
    dataset_config: str,
):
    """Mutate hyperparameters on the resident model and evaluate one cell."""

    # Apply this cell's settings to the resident model.
    model.summary_method = method
    model.summary_chunks = summary_chunks
    model.chunk_size = chunk_size
    model.sink_size = sink_size
    model.discard_summary_kv = discard_summary_kv
    model.block_size = block_size
    model.position_mode = position_mode
    # anchor_block_size: None => model defaults to block_size
    model.anchor_block_size = anchor_block_size if anchor_block_size > 0 else None

    correct = 0
    total = 0
    end_idx = min(start_sample_index + num_samples, len(ds))
    total_samples = end_idx - start_sample_index

    print(
        f"\n>>> task={task} method={method} chunks={summary_chunks} "
        f"chunk_size={chunk_size} block={block_size} sink={sink_size} "
        f"pos={position_mode} -> {total_samples} samples"
    )

    for idx, i in enumerate(range(start_sample_index, end_idx)):
        sys.stdout.write(
            f"\r    [{idx + 1}/{total_samples}] "
            f"({(idx + 1) / total_samples * 100:.1f}%)"
        )
        sys.stdout.flush()

        total += 1
        sample = ds[i]
        context = sample["input"]
        query = sample["question"]
        target = sample["target"]

        try:
            result = model(prompt_context=context, prompt_query=query)
            prediction = result["text"][0]

            if (
                target.lower() in prediction.lower()
                or prediction.lower() in target.lower()
            ):
                correct += 1
                status = "PASS"
            else:
                status = "FAIL"

            acc_so_far = correct / total * 100
            pred_preview = prediction.strip().replace("\n", " ")
            if len(pred_preview) > 160:
                pred_preview = pred_preview[:160] + "..."
            print(
                f"\n    sample {i}: {status} | target={target!r} "
                f"| pred={pred_preview!r} | running={acc_so_far:.2f}% "
                f"({correct}/{total})"
            )
        except Exception as e:
            print(f"\n    ERROR on sample {i}: {e}")
            traceback.print_exc()

        # Free transient tensors (KV caches from prior forward, etc.)
        _reset_cuda()

    print()

    if total == 0:
        return

    accuracy = correct / total * 100
    print(
        f"    === {task}/{method}/chunks={summary_chunks}/pos={position_mode}: "
        f"{accuracy:.2f}% ({correct}/{total}) ==="
    )

    results_line = (
        f"{datetime.datetime.utcnow().isoformat()} | "
        f"model={model_tag} | "
        f"dataset={dataset_config}/{task} | "
        f"summary_method={method} | "
        f"summary_chunks={summary_chunks} | "
        f"chunk_size={chunk_size} | "
        f"block_size={block_size} | "
        f"anchor_block_size={anchor_block_size} | "
        f"sink_size={sink_size} | "
        f"discard_summary_kv={discard_summary_kv} | "
        f"position_mode={position_mode} | "
        f"samples=start:{start_sample_index},n:{num_samples} | "
        f"correct={correct}/{total} | "
        f"accuracy={accuracy:.2f}%\n"
    )
    try:
        with open(results_file, "a", encoding="utf-8") as rf:
            rf.write(results_line)
    except Exception as e:
        print(f"    Failed to write results: {e}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args):
    methods = _csv(args.methods)
    chunk_budgets = _csv_int(args.summary_chunks)
    tasks = _csv(args.tasks)
    position_modes = _csv(args.position_modes)
    for pm in position_modes:
        if pm not in ("sparse", "contiguous"):
            raise ValueError(
                f"position_modes entries must be 'sparse' or 'contiguous', "
                f"got {pm!r}"
            )

    print("=" * 68)
    print("  Statistical Cross-Attention — BABILong Prelim Sweep")
    print("=" * 68)
    print(f"  Model            : {args.model_path}")
    print(f"  Dataset config   : {args.dataset_config}")
    print(f"  Tasks            : {tasks}")
    print(f"  Summary methods  : {methods}")
    print(f"  Summary chunks   : {chunk_budgets}")
    print(f"  Position modes   : {position_modes}")
    print(f"  chunk_size       : {args.chunk_size}")
    print(f"  block_size       : {args.block_size}")
    print(f"  sink_size        : {args.sink_size}")
    print(f"  max_new_tokens   : {args.max_new_tokens}")
    print(f"  samples/cell     : {args.num_samples} "
          f"(start_index={args.start_sample_index})")
    print(f"  results_file     : {args.results_file}")
    print(f"  total cells      : "
          f"{len(methods) * len(chunk_budgets) * len(tasks) * len(position_modes)}")
    print("=" * 68)

    # --- 1. Load datasets (one per task, reused across all cells of that task) ---
    print("\n--- 1. Loading datasets ---")
    datasets_by_task = {}
    for task in tasks:
        print(f"    loading {args.dataset_config}/{task} ...")
        datasets_by_task[task] = load_dataset(
            "RMT-team/babilong", args.dataset_config, split=task
        )

    # --- 2. Load model ONCE with placeholder summary hyperparams ---
    print("\n--- 2. Loading StarAttentionModel (once) ---")
    stop_words = [w for w in args.stop_words.split(",") if w] if args.stop_words else None
    model = StarAttentionModel(
        path=args.model_path,
        block_size=args.block_size,
        max_new_tokens=args.max_new_tokens,
        stop_words=stop_words,
        anchor_block_size=(
            args.anchor_block_size if args.anchor_block_size > 0 else -1
        ),
        # Hyperparams below are overwritten per cell by run_cell(); values here
        # are just sensible defaults so __init__ completes cleanly.
        summary_chunks=chunk_budgets[0],
        chunk_size=args.chunk_size,
        summary_method=methods[0],
        discard_summary_kv=not args.no_discard_summary_kv,
        sink_size=args.sink_size,
        position_mode=position_modes[0],
    )

    # --- 3. Sweep ---
    print("\n--- 3. Sweeping ---")
    total_cells = len(methods) * len(chunk_budgets) * len(tasks) * len(position_modes)
    cell_idx = 0
    for pm in position_modes:
        for method in methods:
            for chunks in chunk_budgets:
                for task in tasks:
                    cell_idx += 1
                    print(f"\n--- [{cell_idx}/{total_cells}] ---")
                    run_cell(
                        model,
                        datasets_by_task[task],
                        task=task,
                        method=method,
                        summary_chunks=chunks,
                        chunk_size=args.chunk_size,
                        block_size=args.block_size,
                        sink_size=args.sink_size,
                        anchor_block_size=args.anchor_block_size,
                        discard_summary_kv=not args.no_discard_summary_kv,
                        position_mode=pm,
                        start_sample_index=args.start_sample_index,
                        num_samples=args.num_samples,
                        results_file=args.results_file,
                        model_tag=args.model_path,
                        dataset_config=args.dataset_config,
                    )
                    # Extra cleanup between cells — release any lingering CUDA
                    # allocator blocks from the previous summary_method / cache
                    # layout so the next cell starts fresh.
                    _reset_cuda()

    print("\n" + "=" * 68)
    print(f"  Sweep complete. {total_cells} cells appended to {args.results_file}")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model_path", default="meta-llama/Llama-3.1-8B-Instruct")

    # Sweep dimensions (CSV)
    parser.add_argument(
        "--methods",
        default="anchor,tfidf,bm25,entropy,max_idf,evenly_spaced",
        help=(
            "CSV list of summary methods to sweep. Supported by the codebase: "
            "anchor, tfidf, bm25, entropy, max_idf, evenly_spaced, mean_pool. "
            "`anchor` is NVIDIA's vanilla Star Attention — block 0 contributes "
            "its first num_chunks*chunk_size tokens as the anchor, visible to "
            "every later block. Included as the reference method to beat. "
            "`evenly_spaced` is the non-semantic baseline and shares the same "
            "num_chunks*chunk_size token budget as the scoring methods. "
            "`mean_pool` ignores chunk_size (returns num_chunks pooled "
            "embedding vectors per block), so it has a different budget shape "
            "— run it separately with matched-compression K values rather "
            "than bundling into this sweep."
        ),
    )
    parser.add_argument(
        "--summary_chunks",
        default="4,16,32",
        help=(
            "CSV list of summary_chunks budgets to sweep. "
            "With block_size=4096 and chunk_size=32 over a 16K context "
            "(4 blocks, accumulation factor 3), max summary budget at the "
            "last block = 3 * summary_chunks * chunk_size = 96 * K. "
            "K=32 → 3072 tokens (~75% of a 4096 cap); K=42 → 4032 (cap)."
        ),
    )
    parser.add_argument(
        "--tasks",
        default="qa1,qa3,qa5",
        help="CSV list of BABILong tasks (qa1..qa5) to sweep",
    )

    # Fixed settings
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--sink_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument(
        "--anchor_block_size",
        type=int,
        default=-1,
        help="-1 => default to block_size",
    )
    parser.add_argument("--no_discard_summary_kv", action="store_true")
    parser.add_argument(
        "--position_modes",
        default="sparse,contiguous",
        help=(
            "CSV list of position-mode values to sweep: 'sparse' and/or "
            "'contiguous'. sparse keeps global position IDs (gaps at chunk "
            "boundaries); contiguous re-numbers each Phase-1 assembled "
            "sequence 0..L-1 (NVIDIA Star Attention style), and shifts the "
            "Phase-2 query past the longest assembled length so RoPE deltas "
            "stay in-distribution. Default sweeps both as an ablation axis."
        ),
    )

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--stop_words", default="")

    # Dataset
    parser.add_argument("--dataset_config", default="16k")
    parser.add_argument("--start_sample_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=100)

    # Output
    parser.add_argument("--results_file", default="prelim_accuracies.txt")

    args = parser.parse_args()
    main(args)
