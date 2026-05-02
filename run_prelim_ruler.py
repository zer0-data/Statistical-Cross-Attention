"""Single-GPU preliminary RULER (v1) sweep for Statistical Cross-Attention.

This is the RULER counterpart to ``run_prelim_babilong.py``. It sweeps
``(summary_method x summary_chunks x seq_length x task)`` in a single Python
process against the NVIDIA RULER v1 benchmark
(https://github.com/NVIDIA/RULER), using the data-prep / scoring pipeline that
already ships in the ``ruler/`` package of this repo:

  * Data is generated (once per seq_length/task) via ``ruler/data/prepare.py``.
  * Inference uses the same ``StarAttentionModel`` interface and the same
    per-task hyperparameters that the single-sample driver
    (``run_star_attn_inference.py``) consumes — ``tokens_to_generate`` per task
    from ``ruler/synthetic_inference_config.yaml`` and ``stop_words`` from the
    chosen ``--prompt_config`` template.
  * Scoring is done inline using the per-task ``metric_fn`` from
    ``ruler/eval/synthetic/constants.py`` so we don't have to spawn
    ``ruler/eval/evaluate.py`` for every cell.

The model is loaded ONCE and only its summary-related attributes are mutated
between cells; the ~16 GB of weights stay resident on the GPU. This relies on
the same single-GPU fallbacks used by ``run_prelim_babilong.py``.

Example:
  python run_prelim_ruler.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --prompt_config meta-llama3 \
    --methods anchor,tfidf,bm25,entropy,max_idf,evenly_spaced \
    --summary_chunks 4,16,32 \
    --seq_lengths 16384 \
    --tasks niah_single_1,niah_multikey_1,qa_1 \
    --num_samples 100
"""

import argparse
import datetime
import gc
import json
import os
import subprocess
import sys
import traceback
from typing import Dict, List, Optional

import torch
import yaml

from model import StarAttentionModel
from ruler import PROMPT_TEMPLATES
from ruler.eval.synthetic.constants import TASKS as METRIC_TASKS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _csv(values: str) -> List[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def _csv_int(values: str) -> List[int]:
    return [int(v) for v in _csv(values)]


def _reset_cuda():
    """Release cached allocator blocks and dangling references."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_task_configs() -> Dict[str, dict]:
    """Merge ``synthetic_task_config.yaml`` with the per-category metric_fn."""
    with open(os.path.join(BASE_DIR, "ruler", "synthetic_task_config.yaml")) as f:
        tasks_customized = yaml.safe_load(f)
    merged = {}
    for task_name, cfg in tasks_customized.items():
        category = cfg["task"]
        merged[task_name] = {
            "category": category,
            "metric_fn": METRIC_TASKS[category]["metric_fn"],
        }
    return merged


def _load_tokens_to_generate() -> Dict[str, int]:
    with open(os.path.join(BASE_DIR, "ruler", "synthetic_inference_config.yaml")) as f:
        return yaml.safe_load(f)["tokens_to_generate"]


def _ensure_dataset(
    *,
    data_root: str,
    seq_length: int,
    task: str,
    tokenizer_path: str,
    prompt_config: str,
    num_samples: int,
    force_regen: bool,
) -> str:
    """Generate (or reuse) the per-(seq_length, task) RULER validation jsonl."""
    seq_dir = os.path.join(data_root, str(seq_length))
    out_file = os.path.join(seq_dir, task, "validation.jsonl")

    if os.path.exists(out_file) and not force_regen:
        return out_file

    os.makedirs(seq_dir, exist_ok=True)
    cmd = (
        f"python ruler/data/prepare.py "
        f"--save_dir {seq_dir} "
        f"--task {task} "
        f"--tokenizer_path {tokenizer_path} "
        f"--tokenizer_type hf "
        f"--max_seq_length {seq_length} "
        f"--model_template_type {prompt_config} "
        f"--num_samples {num_samples}"
    )
    print(f"    [data] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=BASE_DIR, check=False)
    if res.returncode != 0 or not os.path.exists(out_file):
        raise RuntimeError(
            f"Failed to generate dataset for seq_length={seq_length} task={task}; "
            f"expected {out_file}"
        )
    return out_file


# --------------------------------------------------------------------------- #
# One cell of the sweep
# --------------------------------------------------------------------------- #

def run_cell(
    model: StarAttentionModel,
    samples: List[dict],
    *,
    task: str,
    metric_fn,
    method: str,
    summary_chunks: int,
    chunk_size: int,
    block_size: int,
    sink_size: int,
    anchor_block_size: int,
    discard_summary_kv: bool,
    tokens_to_generate: int,
    seq_length: int,
    start_sample_index: int,
    num_samples: int,
    results_file: str,
    predictions_dir: str,
    model_tag: str,
):
    """Mutate hyperparameters on the resident model and evaluate one cell."""

    # Apply this cell's settings to the resident model.
    model.summary_method = method
    model.summary_chunks = summary_chunks
    model.chunk_size = chunk_size
    model.sink_size = sink_size
    model.discard_summary_kv = discard_summary_kv
    model.block_size = block_size
    model.anchor_block_size = anchor_block_size if anchor_block_size > 0 else None
    # tokens_to_generate is per-task in RULER (see synthetic_inference_config.yaml)
    model.max_new_tokens = tokens_to_generate

    end_idx = min(start_sample_index + num_samples, len(samples))
    total_samples = end_idx - start_sample_index

    print(
        f"\n>>> seq_length={seq_length} task={task} method={method} "
        f"chunks={summary_chunks} chunk_size={chunk_size} block={block_size} "
        f"sink={sink_size} max_new_tokens={tokens_to_generate} "
        f"-> {total_samples} samples"
    )

    preds: List[str] = []
    refs: List[List[str]] = []

    # Per-cell predictions file (drop-in for ruler/eval/evaluate.py if desired)
    cell_pred_path = os.path.join(
        predictions_dir,
        f"{seq_length}__{task}__{method}__k{summary_chunks}.jsonl",
    )
    os.makedirs(predictions_dir, exist_ok=True)

    with open(cell_pred_path, "w", encoding="utf-8", buffering=1) as fout:
        for idx, i in enumerate(range(start_sample_index, end_idx)):
            sys.stdout.write(
                f"\r    [{idx + 1}/{total_samples}] "
                f"({(idx + 1) / total_samples * 100:.1f}%)"
            )
            sys.stdout.flush()

            sample = samples[i]
            try:
                result = model(
                    prompt_context=sample["input_context"],
                    prompt_query=sample["input_query"],
                )
                prediction = result["text"][0]
            except Exception as e:
                print(f"\n    ERROR on sample {i}: {e}")
                traceback.print_exc()
                prediction = ""

            outputs = sample.get("outputs", [sample.get("output", "")])
            preds.append(prediction)
            refs.append(outputs)

            fout.write(
                json.dumps(
                    {
                        "index": sample.get("index", i),
                        "pred": prediction,
                        "input_context": sample["input_context"],
                        "input_query": sample["input_query"],
                        "outputs": outputs,
                        "others": sample.get("others", {}),
                        "truncation": sample.get("truncation", -1),
                        "length": sample.get("length", -1),
                    }
                )
                + "\n"
            )
            _reset_cuda()
    print()

    if not preds:
        return

    score = metric_fn(preds, refs)
    n_null = sum(1 for p in preds if len(p) == 0)
    print(
        f"    === seq={seq_length}/{task}/{method}/chunks={summary_chunks}: "
        f"score={score:.2f} ({n_null} nulls / {len(preds)}) ==="
    )

    results_line = (
        f"{datetime.datetime.utcnow().isoformat()} | "
        f"model={model_tag} | "
        f"seq_length={seq_length} | "
        f"task={task} | "
        f"summary_method={method} | "
        f"summary_chunks={summary_chunks} | "
        f"chunk_size={chunk_size} | "
        f"block_size={block_size} | "
        f"anchor_block_size={anchor_block_size} | "
        f"sink_size={sink_size} | "
        f"discard_summary_kv={discard_summary_kv} | "
        f"tokens_to_generate={tokens_to_generate} | "
        f"samples=start:{start_sample_index},n:{num_samples} | "
        f"nulls={n_null}/{len(preds)} | "
        f"score={score:.2f}\n"
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
    seq_lengths = _csv_int(args.seq_lengths)

    # Resolve per-task RULER hyperparameters (matches run_star_attn_inference.py).
    task_configs = _load_task_configs()
    tokens_to_generate_map = _load_tokens_to_generate()
    for t in tasks:
        if t not in task_configs:
            raise ValueError(
                f"Unknown RULER task {t!r}; valid options: {list(task_configs)}"
            )
        if t not in tokens_to_generate_map:
            raise ValueError(
                f"No tokens_to_generate for task {t!r} in "
                f"ruler/synthetic_inference_config.yaml"
            )

    if args.prompt_config not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown prompt_config {args.prompt_config!r}; "
            f"valid options: {list(PROMPT_TEMPLATES)}"
        )
    stop_words = PROMPT_TEMPLATES[args.prompt_config]["stop_words"] or None

    print("=" * 72)
    print("  Statistical Cross-Attention — RULER v1 Prelim Sweep")
    print("=" * 72)
    print(f"  Model            : {args.model_path}")
    print(f"  Prompt config    : {args.prompt_config}")
    print(f"  Stop words       : {stop_words}")
    print(f"  Seq lengths      : {seq_lengths}")
    print(f"  Tasks            : {tasks}")
    print(f"  Summary methods  : {methods}")
    print(f"  Summary chunks   : {chunk_budgets}")
    print(f"  chunk_size       : {args.chunk_size}")
    print(f"  block_size       : {args.block_size}")
    print(f"  anchor_block_size: {args.anchor_block_size}")
    print(f"  sink_size        : {args.sink_size}")
    print(f"  samples/cell     : {args.num_samples} "
          f"(start_index={args.start_sample_index})")
    print(f"  data_dir         : {args.data_dir}")
    print(f"  predictions_dir  : {args.predictions_dir}")
    print(f"  results_file     : {args.results_file}")
    print(f"  total cells      : "
          f"{len(methods) * len(chunk_budgets) * len(seq_lengths) * len(tasks)}")
    print("=" * 72)

    # --- 1. Generate / locate datasets (once per seq_length × task) ---
    print("\n--- 1. Preparing datasets ---")
    samples_by_key: Dict[tuple, List[dict]] = {}
    for seq_length in seq_lengths:
        for task in tasks:
            data_path = _ensure_dataset(
                data_root=args.data_dir,
                seq_length=seq_length,
                task=task,
                tokenizer_path=args.model_path,
                prompt_config=args.prompt_config,
                num_samples=args.num_samples + args.start_sample_index,
                force_regen=args.force_regen,
            )
            samples = _read_jsonl(data_path)
            print(f"    {seq_length}/{task}: {len(samples)} samples ({data_path})")
            samples_by_key[(seq_length, task)] = samples

    # --- 2. Load model ONCE ---
    print("\n--- 2. Loading StarAttentionModel (once) ---")
    # tokens_to_generate is mutated per-task; init with the max so the model's
    # generation buffer is large enough for any task.
    init_tokens_to_generate = max(tokens_to_generate_map[t] for t in tasks)
    model = StarAttentionModel(
        path=args.model_path,
        block_size=args.block_size,
        max_new_tokens=init_tokens_to_generate,
        stop_words=stop_words,
        anchor_block_size=(
            args.anchor_block_size if args.anchor_block_size > 0 else -1
        ),
        # placeholders — overwritten per cell
        summary_chunks=chunk_budgets[0],
        chunk_size=args.chunk_size,
        summary_method=methods[0],
        discard_summary_kv=not args.no_discard_summary_kv,
        sink_size=args.sink_size,
    )

    # --- 3. Sweep ---
    print("\n--- 3. Sweeping ---")
    total_cells = len(methods) * len(chunk_budgets) * len(seq_lengths) * len(tasks)
    cell_idx = 0
    for method in methods:
        for chunks in chunk_budgets:
            for seq_length in seq_lengths:
                for task in tasks:
                    cell_idx += 1
                    print(f"\n--- [{cell_idx}/{total_cells}] ---")
                    run_cell(
                        model,
                        samples_by_key[(seq_length, task)],
                        task=task,
                        metric_fn=task_configs[task]["metric_fn"],
                        method=method,
                        summary_chunks=chunks,
                        chunk_size=args.chunk_size,
                        block_size=args.block_size,
                        sink_size=args.sink_size,
                        anchor_block_size=args.anchor_block_size,
                        discard_summary_kv=not args.no_discard_summary_kv,
                        tokens_to_generate=tokens_to_generate_map[task],
                        seq_length=seq_length,
                        start_sample_index=args.start_sample_index,
                        num_samples=args.num_samples,
                        results_file=args.results_file,
                        predictions_dir=args.predictions_dir,
                        model_tag=args.model_path,
                    )
                    # Drop allocator blocks from prior summary_method/cache layout.
                    _reset_cuda()

    print("\n" + "=" * 72)
    print(f"  Sweep complete. {total_cells} cells appended to {args.results_file}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--prompt_config",
        required=True,
        choices=list(PROMPT_TEMPLATES.keys()),
        help="prompt template config name (see ruler/data/template.py). "
             "Determines the per-task `stop_words` used at generation time.",
    )

    # Sweep dimensions (CSV)
    parser.add_argument(
        "--methods",
        default="anchor,tfidf,bm25,entropy,max_idf,evenly_spaced",
        help=(
            "CSV list of summary methods. Supported: anchor, tfidf, bm25, "
            "entropy, max_idf, evenly_spaced, mean_pool. `anchor` = NVIDIA's "
            "vanilla Star Attention reference. `evenly_spaced` shares the "
            "same num_chunks*chunk_size token budget as the scoring methods. "
            "`mean_pool` has a different budget shape — run separately."
        ),
    )
    parser.add_argument(
        "--summary_chunks",
        default="4,16,32",
        help="CSV list of summary_chunks budgets to sweep.",
    )
    parser.add_argument(
        "--tasks",
        default=(
            "niah_single_1,niah_single_2,niah_single_3,"
            "niah_multikey_1,niah_multikey_2,niah_multikey_3,"
            "niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2"
        ),
        help="CSV list of RULER v1 tasks (see ruler/synthetic_task_config.yaml).",
    )
    parser.add_argument(
        "--seq_lengths",
        default="16384",
        help="CSV list of context sequence lengths (e.g. 4096,8192,16384,32768).",
    )

    # Fixed Star-Attention settings (matches run_star_attn_inference.py defaults)
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

    # Dataset
    parser.add_argument(
        "--data_dir",
        default=os.path.join(BASE_DIR, "dataset", "prelim_ruler"),
        help="Root for cached pre-generated RULER data "
             "(layout: <data_dir>/<seq_length>/<task>/validation.jsonl).",
    )
    parser.add_argument("--start_sample_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument(
        "--force_regen",
        action="store_true",
        help="Regenerate datasets even if validation.jsonl already exists.",
    )

    # Output
    parser.add_argument(
        "--predictions_dir",
        default=os.path.join(BASE_DIR, "results", "prelim_ruler", "predictions"),
        help="Where per-cell prediction jsonl files are written.",
    )
    parser.add_argument(
        "--results_file",
        default=os.path.join(BASE_DIR, "prelim_ruler_accuracies.txt"),
        help="Append-only sweep summary log (one line per cell).",
    )

    args = parser.parse_args()
    main(args)
