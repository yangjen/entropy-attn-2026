#!/usr/bin/env python3
"""
RULER QA output entropy probe — gate threshold generalization check.

Generates short answers for RULER qa_1/qa_2 examples, records output token
entropy per decode step, scores with substring exact match, then reports
P(peak_entropy > θ) broken down by correct vs. incorrect examples.

Goal: verify P(peak_entropy > θ | incorrect) >> P(peak_entropy > θ | correct),
confirming the 0.05 gate threshold found on MRCR generalizes to RULER.

Usage:
  CUDA_VISIBLE_DEVICES=0 python ruler_entropy_probe.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data_root /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/32768/data \
    --tasks qa_1,qa_2 \
    --output_dir /c2/jenny/r3/RULER_outputs/entropy_probe/llama31_8b_32k \
    --max_examples 200 \
    --threshold 0.05

  # Multiple context-length splits in one run:
  RULER_BASE=/c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic
  CUDA_VISIBLE_DEVICES=0 python ruler_entropy_probe.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data_root ${RULER_BASE}/16384/data,${RULER_BASE}/32768/data,${RULER_BASE}/65536/data \
    --tasks qa_1,qa_2 \
    --output_dir /c2/jenny/r3/RULER_outputs/entropy_probe/llama31_8b \
    --max_examples 100 \
    --threshold 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from attention_llama import LlamaRunner
from attention_qwen import QwenRunner


# ---------------------------------------------------------------------------
# Runner factory
# ---------------------------------------------------------------------------

def make_runner(model_name: str, dtype: torch.dtype):
    if "llama" in model_name.lower():
        return LlamaRunner(
            model_name=model_name,
            attn_impl="sdpa",
            dtype=dtype,
            deterministic=True,
        )
    return QwenRunner(
        model_name=model_name,
        attn_impl="sdpa",
        dtype=dtype,
        deterministic=True,
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# RULER loader
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


def ruler_gold(ex: Dict[str, Any]) -> List[str]:
    outs = ex.get("outputs", ex.get("answer", []))
    if isinstance(outs, str):
        return [outs]
    return [str(o) for o in (outs or [])]


# ---------------------------------------------------------------------------
# Exact match scoring  (mirrors infinben_ruler_session_tuning.py)
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"<\|.*?\|>", " ", s)   # strip special tokens e.g. <|eot_id|>
    s = s.translate(_PUNCT_TABLE)
    s = " ".join(s.split())
    return s


def contains_any(pred: str, golds: List[str]) -> bool:
    p = normalize_text(pred)
    if not p:
        return False
    return any((ng := normalize_text(g)) and ng in p for g in golds)


# ---------------------------------------------------------------------------
# Generation + entropy trace
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_with_entropy(
    runner,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int,
) -> Tuple[List[float], str]:
    """
    Returns (norm_entropy_trace, generated_text).
    norm_entropy_trace[i] = H(vocab dist at step i) / log(vocab_size)
    """
    out = runner.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=runner.tokenizer.eos_token_id,
        eos_token_id=runner.tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

    trace: List[float] = []
    for step_logits in out.scores:
        logits = step_logits[0].float()
        probs = torch.softmax(logits, dim=-1)
        ent = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
        log_vocab = float(torch.log(torch.tensor(logits.numel(), dtype=torch.float32)).item())
        trace.append(round(ent / log_vocab if log_vocab > 0 else 0.0, 6))

    prompt_len = input_ids.shape[-1]
    gen_ids = out.sequences[0, prompt_len:]
    gen_text = runner.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return trace, gen_text


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(records: List[Dict], threshold: float) -> None:
    if not records:
        print("No records to summarize.")
        return

    correct = [r for r in records if r["correct"]]
    wrong   = [r for r in records if not r["correct"]]

    def stats(group):
        if not group:
            return {"n": 0, "fire_rate": float("nan"),
                    "mean_peak": float("nan"), "median_peak": float("nan")}
        peaks = np.array([r["peak_entropy"] for r in group])
        return {
            "n":           len(group),
            "fire_rate":   float((peaks > threshold).mean()),
            "mean_peak":   float(peaks.mean()),
            "median_peak": float(np.median(peaks)),
        }

    c = stats(correct)
    w = stats(wrong)

    print(f"\n{'='*58}")
    print(f"Gate threshold generalization — RULER QA  (θ={threshold})")
    print(f"{'='*58}")
    print(f"{'':12}  {'n':>5}  {'P(fire)':>8}  {'mean_peak':>10}  {'median_peak':>12}")
    print(f"{'-'*58}")
    print(f"{'correct':<12}  {c['n']:>5}  {c['fire_rate']:>8.3f}  {c['mean_peak']:>10.4f}  {c['median_peak']:>12.4f}")
    print(f"{'incorrect':<12}  {w['n']:>5}  {w['fire_rate']:>8.3f}  {w['mean_peak']:>10.4f}  {w['median_peak']:>12.4f}")

    # Per context-length breakdown
    ctx_bins = sorted(set(r["ctx_bin"] for r in records))
    if len(ctx_bins) > 1:
        print(f"\n--- Per context-length bin ---")
        print(f"{'bin':<10}  {'correct_fire':>12}  {'wrong_fire':>10}  {'n_correct':>10}  {'n_wrong':>8}")
        for cb in ctx_bins:
            gc = [r for r in correct if r["ctx_bin"] == cb]
            gw = [r for r in wrong   if r["ctx_bin"] == cb]
            cf = float(np.mean([r["peak_entropy"] > threshold for r in gc])) if gc else float("nan")
            wf = float(np.mean([r["peak_entropy"] > threshold for r in gw])) if gw else float("nan")
            print(f"{cb:<10}  {cf:>12.3f}  {wf:>10.3f}  {len(gc):>10}  {len(gw):>8}")

    # Per task breakdown
    tasks = sorted(set(r["task"] for r in records))
    if len(tasks) > 1:
        print(f"\n--- Per task ---")
        print(f"{'task':<10}  {'correct_fire':>12}  {'wrong_fire':>10}  {'n_correct':>10}  {'n_wrong':>8}")
        for t in tasks:
            gc = [r for r in correct if r["task"] == t]
            gw = [r for r in wrong   if r["task"] == t]
            cf = float(np.mean([r["peak_entropy"] > threshold for r in gc])) if gc else float("nan")
            wf = float(np.mean([r["peak_entropy"] > threshold for r in gw])) if gw else float("nan")
            print(f"{t:<10}  {cf:>12.3f}  {wf:>10.3f}  {len(gc):>10}  {len(gw):>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--data_root", required=True,
                    help="Comma-separated RULER data roots, each with <root>/<task>/validation.jsonl")
    ap.add_argument("--tasks", default="qa_1,qa_2",
                    help="Comma-separated RULER task names.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_examples", type=int, default=100,
                    help="Max examples per task per data root.")
    ap.add_argument("--max_new_tokens", type=int, default=64,
                    help="Max tokens to generate per example.")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="Gate firing threshold to evaluate.")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--status_every", type=int, default=10)
    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    runner = make_runner(args.model, dtype_map[args.dtype])

    os.makedirs(args.output_dir, exist_ok=True)
    samples_path = os.path.join(args.output_dir, "ruler_entropy_probe.jsonl")

    tasks      = [t.strip() for t in args.tasks.split(",") if t.strip()]
    data_roots = [r.strip() for r in args.data_root.split(",") if r.strip()]

    all_records = []
    total = 0

    with open(samples_path, "w", encoding="utf-8") as out_f:
        for task in tasks:
            print(f"\n=== Task: {task} ===")
            for root in data_roots:
                print(f"  data_root: {root}")
                ctx_bin = os.path.basename(os.path.dirname(root))  # e.g. "32768"

                examples = load_ruler_task(task, root)
                task_n = 0

                for ex in examples:
                    if args.max_examples > 0 and task_n >= args.max_examples:
                        break

                    prompt = ex.get("input", "")
                    answer_prefix = ex.get("answer_prefix", "")
                    if answer_prefix and not prompt.endswith(answer_prefix):
                        prompt = prompt + answer_prefix

                    enc = runner.tokenizer(prompt, return_tensors="pt")
                    input_ids = enc["input_ids"].to(runner.model.device)
                    attention_mask = enc.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(runner.model.device)

                    gold = ruler_gold(ex)
                    trace, gen_text = generate_with_entropy(
                        runner, input_ids, attention_mask, args.max_new_tokens)

                    correct    = contains_any(gen_text, gold)
                    peak_ent   = float(max(trace)) if trace else float("nan")
                    mean_ent   = float(np.mean(trace)) if trace else float("nan")
                    gate_fired = peak_ent > args.threshold

                    row = {
                        "task":              task,
                        "sample_id":         ex.get("_idx", task_n),
                        "ctx_bin":           ctx_bin,
                        "context_length":    int(input_ids.shape[-1]),
                        "correct":           correct,
                        "gate_fired":        gate_fired,
                        "peak_entropy":      round(peak_ent, 6),
                        "mean_entropy":      round(mean_ent, 6),
                        "n_gen_tokens":      len(trace),
                        "norm_entropy_trace": trace,
                        "gen_text":          gen_text[:200],
                        "gold":              gold[:3],
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    all_records.append(row)

                    total  += 1
                    task_n += 1

                    if args.status_every > 0 and total % args.status_every == 0:
                        print(f"  n={total}  correct={correct}  peak_ent={peak_ent:.4f}"
                              f"  gate={'Y' if gate_fired else 'N'}  gen='{gen_text[:40]}'")

    print_summary(all_records, args.threshold)
    print(f"\nSaved: {samples_path}")


if __name__ == "__main__":
    main()
