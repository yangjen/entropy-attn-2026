#!/usr/bin/env python3
"""
Oracle prior estimator for the entropy-attention controller.

Runs a validation set through the model (prefill-only forward pass) and records
the normalized attention entropy at the prefill tail, per layer and head, then
computes per-context-length-bin statistics.  The resulting JSON is used as
H*_attn[ctx_bin] in the controller instead of a per-sample or session-calibrated
prior.

What "oracle prior" means here
--------------------------------
  E[attention_entropy_at_prefill_tail | context_length_bin]

  Concretely: after a single forward pass with attn_impl=entropy_attn, each
  attention layer's EntropyTempController holds prompt_target_entropy [Z, H, 1]
  — the trimmed mean of normalized attention entropy over the last
  calibration_tail_k (default 256) prefill tokens.  We average this across all
  layers and heads to get one scalar per example, then aggregate per bin.

  Units: H_attn / log(kv_len)  — same units the controller compares against its
  EMA during decode.

  This replaces the old session-calibrated prompt_target which was measured from
  K=3 examples at session start and was contaminated by hard examples.

Supported validation sources
------------------------------
  --dataset_type mrcr   : uses the same parquet loader as mrcr_baseline_probe.py
  --dataset_type ruler  : uses RULER validation.jsonl layout
                          <data_root>/<task>/validation.jsonl

Output
-------
  <output_dir>/oracle_prior_samples.jsonl   per-example records (for analysis)
  <output_dir>/oracle_prior.json            aggregated prior per ctx-len bin

Prior JSON schema
------------------
{
  "meta": { "model": ..., "tasks": ..., "n_total": ...,
            "signal": "attention_entropy_prefill_tail (H_attn/log(kv_len), mean across layers+heads)" },
  "bins": {
    "16384":  { "lo": 0,     "hi": 16384,  "n": 42, "mean": 0.71, "median": 0.70,
                "p25": 0.68, "p75": 0.73, "std": 0.03 },
    "32768":  { ... },
    ...
  }
}

Example — RULER validation set (all four context-length splits in one run):

  RULER_BASE=/c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic
  CUDA_VISIBLE_DEVICES=0 python /c2/jenny/r3/entropy-attn-2026/estimate_oracle_prior.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset_type ruler \
    --data_root ${RULER_BASE}/4096/data,${RULER_BASE}/8192/data,${RULER_BASE}/16384/data,${RULER_BASE}/32768/data,${RULER_BASE}/65536/data,${RULER_BASE}/131072/data \
    --tasks qa_1,qa_2 \
    --output_dir /c2/jenny/r3/oracle_prior/llama31_8b_ruler \
    --ctx_bin_edges 4096,8192,16384,32768,65536,131072 \
    --max_per_bin 400
    --max_examples 200

Single context-length (e.g. just 32k):
  CUDA_VISIBLE_DEVICES=0 python /c2/jenny/r3/entropy-attn-2026/estimate_oracle_prior.py  \
    --model Qwen/Qwen3.5-2B \
    --dataset_type ruler \
    --data_root /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/131072/data \
    --tasks qa_1 \
    --output_dir /c2/jenny/r3/oracle_prior/qwen35_2b_ruler_131k \
    --ctx_bin_edges 131072 \
    --max_examples 200

NOTE on dataset roles
-----------------------
  Use RULER (qa_1, qa_2) as the oracle prior source — it is the calibration set.
  MRCR is the evaluation benchmark; estimating the prior from MRCR would
  contaminate calibration with the test distribution.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from attention_llama import LlamaRunner
from attention_qwen import QwenRunner
from mrcr_baseline_probe import load_mrcr_task, parse_messages


def make_runner(model_name: str, dtype: torch.dtype, deterministic: bool):
    if "llama" in model_name.lower():
        return LlamaRunner(
            model_name=model_name,
            attn_impl="entropy_attn",
            dtype=dtype,
            deterministic=deterministic,
        )
    return QwenRunner(
        model_name=model_name,
        attn_impl="entropy_attn",
        dtype=dtype,
        deterministic=deterministic,
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# RULER loader  (mirrors infinben_ruler_session_tuning.py)
# ---------------------------------------------------------------------------

def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_ruler_task(task: str, data_root: str) -> List[Dict[str, Any]]:
    path = os.path.join(data_root, task, "validation.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"RULER task file not found: {path}")
    rows = []
    for i, ex in enumerate(iter_jsonl(path)):
        ex["_idx"] = i
        rows.append(ex)
    return rows


def ruler_to_messages(ex: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a RULER example to a single-turn user message."""
    prompt_text = ex.get("input", "") or ""
    answer_prefix = ex.get("answer_prefix", "")
    if answer_prefix and not prompt_text.endswith(answer_prefix):
        prompt_text = prompt_text + answer_prefix
    return [{"role": "user", "content": prompt_text}]


def ruler_answers(ex: Dict[str, Any]) -> List[str]:
    outs = ex.get("outputs", ex.get("answer", []))
    if isinstance(outs, str):
        return [outs]
    return list(outs or [])


# ---------------------------------------------------------------------------
# Context-length bins
# ---------------------------------------------------------------------------

def make_bins(edges: List[int]) -> List[Tuple[int, int, str]]:
    """
    edges = [16384, 32768, 65536, 131072]
    Returns list of (lo_exclusive, hi_inclusive, label) sorted ascending.
    First bin covers [0, edges[0]], subsequent bins cover (edges[i-1], edges[i]].
    """
    bins = []
    edges_sorted = sorted(set(edges))
    lo = 0
    for hi in edges_sorted:
        label = str(hi)
        bins.append((lo, hi, label))
        lo = hi
    # open-ended final bin for anything above the last edge
    bins.append((lo, int(1e12), f">{edges_sorted[-1]}"))
    return bins


def assign_bin(ctx_len: int, bins: List[Tuple[int, int, str]]) -> str:
    for lo, hi, label in bins:
        if lo < ctx_len <= hi:
            return label
    return bins[-1][2]


# ---------------------------------------------------------------------------
# Attention entropy measurement (single prefill forward pass)
# ---------------------------------------------------------------------------

def reset_entropy_controllers(model: torch.nn.Module) -> None:
    """Clear per-layer controller state so each example starts fresh."""
    for module in model.modules():
        ctrl = getattr(module, "_entropy_temp_controller", None)
        if ctrl is not None:
            ctrl.temp = None
            ctrl.ema_entropy = None
            ctrl.prompt_target_entropy = None


@torch.inference_mode()
def prefill_attention_entropy(
    runner: QwenRunner,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[float, float, int]:
    """
    Single forward pass with entropy_attn kernel.

    Returns (mean, std, n_layers) of normalized attention entropy at prefill tail,
    averaged across all layers and heads.

    Units: H_attn / log(kv_len) — same normalization the controller uses during decode.
    The kernel sets controller.prompt_target_entropy during the prefill forward pass
    (attn_patch.py, N_CTX > 1 branch), so we read it back after the call.
    """
    reset_entropy_controllers(runner.model)

    runner.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    per_layer_means: List[float] = []
    for module in runner.model.modules():
        ctrl = getattr(module, "_entropy_temp_controller", None)
        if ctrl is not None and ctrl.prompt_target_entropy is not None:
            # prompt_target_entropy: [Z, H, 1], already normalized
            per_layer_means.append(float(ctrl.prompt_target_entropy.float().mean().item()))

    if not per_layer_means:
        return float("nan"), float("nan"), 0

    arr = np.array(per_layer_means, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), len(arr)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_bin(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None,
                "p25": None, "p75": None, "std": None}
    a = np.array(values, dtype=np.float64)
    return {
        "n":      int(len(a)),
        "mean":   round(float(a.mean()), 6),
        "median": round(float(np.median(a)), 6),
        "p25":    round(float(np.percentile(a, 25)), 6),
        "p75":    round(float(np.percentile(a, 75)), 6),
        "std":    round(float(a.std()), 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--dataset_type", choices=["mrcr", "ruler"], default="ruler")
    ap.add_argument("--tasks", default="qa_1",
                    help="Comma-separated task names.")

    # MRCR args
    ap.add_argument("--local_root", default=None,
                    help="[mrcr] Directory with MRCR parquet files.")

    # RULER args
    ap.add_argument("--data_root", default=None,
                    help="[ruler] Comma-separated roots, each with <root>/<task>/validation.jsonl. "
                         "Pass multiple to cover different context-length splits in one run, e.g. "
                         ".../16384/data,.../32768/data,.../65536/data,.../131072/data")

    # Context length filtering
    ap.add_argument("--min_ctx_tokens", type=int, default=0)
    ap.add_argument("--max_ctx_tokens", type=int, default=0)
    ap.add_argument("--ctx_bin_edges", default="16384,32768,65536,131072",
                    help="Comma-separated token counts defining bin upper boundaries.")

    # Prior estimation settings
    ap.add_argument("--max_examples", type=int, default=0,
                    help="Max examples per task.  0 = all.")
    ap.add_argument("--max_per_bin", type=int, default=0,
                    help="Stop collecting for a bin once it has this many samples.  0 = unlimited.")

    ap.add_argument("--output_dir", required=True,
                    help="Directory to write oracle_prior_samples.jsonl and oracle_prior.json.")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--status_every", type=int, default=10)
    args = ap.parse_args()

    # ---- setup ----
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    runner = make_runner(args.model, dtype_map[args.dtype], deterministic=True)

    os.makedirs(args.output_dir, exist_ok=True)
    samples_path = os.path.join(args.output_dir, "oracle_prior_samples.jsonl")
    prior_path   = os.path.join(args.output_dir, "oracle_prior.json")

    bin_edges = [int(x.strip()) for x in args.ctx_bin_edges.split(",") if x.strip()]
    bins      = make_bins(bin_edges)
    bin_data: Dict[str, List[float]] = {label: [] for _, _, label in bins}

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    data_roots = [r.strip() for r in (args.data_root or "").split(",") if r.strip()]
    total_written = 0

    with open(samples_path, "w", encoding="utf-8") as out_f:
        for task in tasks:
            print(f"\n=== Task: {task} ===")

            task_n = 0
            task_bin_counts: Dict[str, int] = {label: 0 for _, _, label in bins}
            roots_iter = data_roots if args.dataset_type == "ruler" else [None]

            for root in roots_iter:
                if args.max_examples > 0 and task_n >= args.max_examples:
                    break

                # ---- load examples ----
                if args.dataset_type == "mrcr":
                    examples = load_mrcr_task(task, args.local_root)
                else:
                    if root is None:
                        raise ValueError("--data_root required for dataset_type=ruler")
                    print(f"  data_root: {root}")
                    examples = load_ruler_task(task, root)

                for ex in examples:
                    if args.max_examples > 0 and task_n >= args.max_examples:
                        break

                    # ---- build prompt ----
                    if args.dataset_type == "mrcr":
                        messages = parse_messages(ex)
                        prompt   = runner.messages_to_prompt(messages)
                    else:
                        prompt = ex.get("input", "")
                        answer_prefix = ex.get("answer_prefix", "")
                        if answer_prefix and not prompt.endswith(answer_prefix):
                            prompt = prompt + answer_prefix

                    # ---- tokenize and length filter ----
                    enc = runner.tokenizer(prompt, return_tensors="pt")
                    input_ids = enc["input_ids"].to(runner.model.device)
                    attention_mask = enc.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(runner.model.device)
                    ctx_len = int(input_ids.shape[-1])

                    if args.min_ctx_tokens > 0 and ctx_len < args.min_ctx_tokens:
                        continue
                    if args.max_ctx_tokens > 0 and ctx_len > args.max_ctx_tokens:
                        continue

                    # ---- bin assignment + per-bin-per-task cap ----
                    bin_label = assign_bin(ctx_len, bins)
                    if args.max_per_bin > 0 and task_bin_counts[bin_label] >= args.max_per_bin:
                        if all(task_bin_counts[lbl] >= args.max_per_bin for _, _, lbl in bins[:-1]):
                            break  # all capped bins full — skip rest of this data root
                        continue

                    # ---- single prefill forward pass → attention entropy ----
                    attn_ent_mean, attn_ent_std, n_layers = prefill_attention_entropy(
                        runner, input_ids, attention_mask)

                    bin_data[bin_label].append(attn_ent_mean)
                    task_bin_counts[bin_label] += 1

                    # ---- answers for reference ----
                    if args.dataset_type == "mrcr":
                        gold = str(ex.get("answer", ""))
                    else:
                        gold = str(ruler_answers(ex))

                    row = {
                        "task":              task,
                        "sample_id":         ex.get("_idx", task_n),
                        "context_length":    ctx_len,
                        "bin_label":         bin_label,
                        "attn_entropy_mean": round(attn_ent_mean, 6),
                        "attn_entropy_std":  round(attn_ent_std, 6),
                        "n_layers":          n_layers,
                        "gold":              gold[:120],
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()

                    total_written += 1
                    task_n        += 1

                    if args.status_every > 0 and total_written % args.status_every == 0:
                        bin_summary = {
                            lbl: len(vs) for _, _, lbl in bins
                            if (vs := bin_data[lbl])
                        }
                        print(f"  n={total_written}  ctx={ctx_len}  bin={bin_label}"
                              f"  attn_ent={attn_ent_mean:.4f}  bins={bin_summary}")

    # ---- aggregate prior ----
    prior = {
        "meta": {
            "model":   args.model,
            "tasks":   args.tasks,
            "n_total": total_written,
            "signal":  "attention_entropy_prefill_tail (H_attn/log(kv_len), mean across layers+heads)",
        },
        "bins": {},
    }
    for lo, hi, label in bins:
        vals = bin_data[label]
        stats = aggregate_bin(vals)
        stats["lo"] = lo
        stats["hi"] = hi if hi < int(1e12) else None
        prior["bins"][label] = stats

    with open(prior_path, "w", encoding="utf-8") as f:
        json.dump(prior, f, indent=2, ensure_ascii=False)

    # ---- print summary ----
    print(f"\n{'='*60}")
    print(f"Oracle prior summary  (model={args.model})")
    print(f"{'='*60}")
    print(f"{'Bin':<12}  {'N':>5}  {'mean':>8}  {'median':>8}  {'std':>8}  {'p25':>8}  {'p75':>8}")
    print("-" * 60)
    for _, _, label in bins:
        s = prior["bins"][label]
        if s["n"] == 0:
            continue
        print(f"{label:<12}  {s['n']:>5}  {s['mean']:>8.5f}  "
              f"{s['median']:>8.5f}  {s['std']:>8.5f}  "
              f"{s['p25']:>8.5f}  {s['p75']:>8.5f}")
    print(f"\nSaved:\n  samples: {samples_path}\n  prior:   {prior_path}")


if __name__ == "__main__":
    main()
