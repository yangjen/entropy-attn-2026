#!/usr/bin/env python3
"""
MRCR 8-needle baseline probe: logs per-token entropy/probability traces.

Logged fields per sample:
  sample_id, task, context_length, generated_text, gold_answer,
  prediction, mrcr_score, correct, generated_token_ids,
  entropy_trace, norm_entropy_trace, top1_prob_trace, top2_prob_trace,
  prob_margin_trace  (= top1_prob - top2_prob),
  logit_margin_trace (= top1_logit - top2_logit)

mrcr_score is the primary quality label (continuous SequenceMatcher ratio).
correct is binary (prefix_hit AND mrcr_score >= 0.5) for quick summaries only.
norm_entropy = entropy / log(vocab_size), making traces comparable across steps.
All traces are over decode tokens only (thinking block excluded when present).

Example:
  CUDA_VISIBLE_DEVICES=0 python /c2/jenny/r3/entropy-attn-2026/mrcr_baseline_probe.py \
    --model Qwen/Qwen3.5-9B \
    --local_root /c2/jenny/.hf/huggingface/hub/datasets--openai--mrcr/snapshots/f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d \
    --output /c2/jenny/r3/MRCR_outputs/probe_8needle_64-128k.jsonl \
    --tasks 8needle \
    --min_ctx_tokens 65536 \
    --max_ctx_tokens 131072
"""

from __future__ import annotations

import argparse
import json
import os
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from attention_qwen import QwenRunner
from mrcr_qwen35_session_tuning import (
    mark_last_layer_entropy_logger,
    mark_all_layers_entropy_logger,
    reset_entropy_logs,
    collect_entropy_logs,
    collect_all_layer_entropy_logs,
    set_attn_attr,
)


# ---- MRCR loading ----

def load_mrcr_task(task: str, local_root: Optional[str]) -> List[Dict[str, Any]]:
    task_name = task.replace(".parquet", "")
    if local_root:
        flat = os.path.join(local_root, f"{task_name}.parquet")
        if os.path.exists(flat):
            paths = [flat]
        else:
            paths = sorted(
                p for p in [
                    os.path.join(local_root, task_name, f"{task_name}_{i}.parquet")
                    for i in range(10)
                ]
                if os.path.exists(p)
            )
            if not paths:
                raise FileNotFoundError(f"No parquet for task '{task_name}' under {local_root}")
    else:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise RuntimeError("Install huggingface_hub or pass --local_root")
        shards = sorted(
            f for f in list_repo_files("openai/mrcr", repo_type="dataset")
            if f.startswith(f"{task_name}/") and f.endswith(".parquet")
        )
        paths = [
            hf_hub_download(repo_id="openai/mrcr", filename=fn, repo_type="dataset")
            for fn in shards
        ]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    rows = []
    for idx, row in df.iterrows():
        d = row.to_dict()
        d["_idx"] = int(idx)
        rows.append(d)
    return rows


def parse_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        try:
            obj = json.loads(prompt)
            if isinstance(obj, list):
                return obj
        except Exception:
            return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        return prompt
    raise ValueError("Unsupported prompt format")


def grade_mrcr(response: str, answer: str, prefix: str) -> Tuple[float, bool]:
    response = (response or "").strip()
    answer = (answer or "").strip()
    prefix_hit = not prefix or response.startswith(prefix)
    if not prefix_hit:
        return 0.0, False
    if prefix:
        response = response.removeprefix(prefix)
        answer = answer.removeprefix(prefix)
    score = float(SequenceMatcher(None, response.strip(), answer.strip()).ratio())
    return score, True


# ---- token-level trace generation ----

@torch.inference_mode()
def generate_with_traces(
    runner: QwenRunner,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int,
) -> Dict[str, Any]:
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

    prompt_len = input_ids.shape[-1]
    gen_ids = out.sequences[0, prompt_len:]

    entropy_trace: List[float] = []
    norm_entropy_trace: List[float] = []
    top1_prob_trace: List[float] = []
    top2_prob_trace: List[float] = []
    prob_margin_trace: List[float] = []
    logit_margin_trace: List[float] = []

    for step_logits in out.scores:
        logits = step_logits[0].float()  # fp32 for numerical stability; shape (vocab,)
        probs = torch.softmax(logits, dim=-1)

        ent = float(-torch.sum(probs * torch.log(probs + 1e-10)).item())
        log_vocab = torch.log(torch.tensor(logits.numel(), device=logits.device, dtype=torch.float32)).item()
        ent_norm = ent / log_vocab if log_vocab > 0 else 0.0

        top2_p = torch.topk(probs, 2)
        p1, p2 = float(top2_p.values[0]), float(top2_p.values[1])

        top2_l = torch.topk(logits, 2)
        l1, l2 = float(top2_l.values[0]), float(top2_l.values[1])

        entropy_trace.append(round(ent, 6))
        norm_entropy_trace.append(round(ent_norm, 6))
        top1_prob_trace.append(round(p1, 6))
        top2_prob_trace.append(round(p2, 6))
        prob_margin_trace.append(round(p1 - p2, 6))
        logit_margin_trace.append(round(l1 - l2, 4))

    # Strip thinking block at token level if present
    full_text = runner.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if runner.enable_thinking:
        think_end_id = runner.tokenizer.convert_tokens_to_ids("</think>")
        ids_list = gen_ids.tolist()
        if think_end_id in ids_list:
            cut = ids_list.index(think_end_id) + 1
            gen_ids = gen_ids[cut:]
            # Trim traces to post-think tokens
            entropy_trace = entropy_trace[cut:]
            top1_prob_trace = top1_prob_trace[cut:]
            top2_prob_trace = top2_prob_trace[cut:]
            prob_margin_trace = prob_margin_trace[cut:]
            logit_margin_trace = logit_margin_trace[cut:]

    prediction = runner.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return {
        "generated_text": full_text,
        "prediction": prediction,
        "generated_token_ids": gen_ids.tolist(),
        "entropy_trace": entropy_trace,
        "norm_entropy_trace": norm_entropy_trace,
        "top1_prob_trace": top1_prob_trace,
        "top2_prob_trace": top2_prob_trace,
        "prob_margin_trace": prob_margin_trace,
        "logit_margin_trace": logit_margin_trace,
    }


# ---- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--tasks", default="8needle")
    ap.add_argument("--local_root", default=None, help="Directory containing MRCR parquet files")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--min_ctx_tokens", type=int, default=0,
                    help="Min prompt token count, inclusive. 0 = no lower bound.")
    ap.add_argument("--max_ctx_tokens", type=int, default=0,
                    help="Max prompt token count, inclusive. 0 = no upper bound.")
    ap.add_argument("--max_examples", type=int, default=0, help="Stop after N samples total. 0 = all.")
    ap.add_argument("--attn_impl",
                    choices=["sdpa", "entropy_attn", "entropy_attn_layer", "flash_attention_2", "eager"],
                    default="sdpa",
                    help="entropy_attn: log last-layer mean attn entropy. "
                         "entropy_attn_layer: log per-head entropy for every layer.")
    ap.add_argument("--log_per_head_steps", type=int, default=50,
                    help="Max decode steps to log per layer with entropy_attn_layer. 0 = all steps.")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--status_every", type=int, default=5)
    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    runner = QwenRunner(
        model_name=args.model,
        attn_impl=args.attn_impl,
        dtype=dtype_map[args.dtype],
        deterministic=True,
        enable_thinking=False,
    )

    if args.attn_impl == "entropy_attn":
        mark_last_layer_entropy_logger(runner.model)
        set_attn_attr(runner.model, "temp_max_step", 0.0)   # logging-only: no temperature changes
    elif args.attn_impl == "entropy_attn_layer":
        mark_all_layers_entropy_logger(runner.model, max_log_steps=args.log_per_head_steps)
        set_attn_attr(runner.model, "temp_max_step", 0.0)   # logging-only: no temperature changes

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    tasks = [t.strip().replace(".parquet", "") for t in args.tasks.split(",") if t.strip()]
    total_written = 0
    total_skipped = 0
    score_sum = 0.0
    correct_count = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for task in tasks:
            examples = load_mrcr_task(task, args.local_root)
            for ex in examples:
                if args.max_examples > 0 and total_written >= args.max_examples:
                    break

                # Cheap chars-based pre-filter to avoid tokenizing clear outliers.
                # n_chars ≈ 4× token count; use 6× / 3× as safe margins.
                n_chars = ex.get("n_chars")
                if isinstance(n_chars, (int, float)):
                    if args.max_ctx_tokens > 0 and n_chars > args.max_ctx_tokens * 6:
                        total_skipped += 1
                        continue
                    if args.min_ctx_tokens > 0 and n_chars < args.min_ctx_tokens * 3:
                        total_skipped += 1
                        continue

                messages = parse_messages(ex)
                prompt = runner.messages_to_prompt(messages)

                enc = runner.tokenizer(prompt, return_tensors="pt")
                input_ids = enc["input_ids"].to(runner.model.device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(runner.model.device)
                ctx_len = int(input_ids.shape[-1])

                if args.min_ctx_tokens > 0 and ctx_len < args.min_ctx_tokens:
                    total_skipped += 1
                    continue
                if args.max_ctx_tokens > 0 and ctx_len > args.max_ctx_tokens:
                    total_skipped += 1
                    continue

                if args.attn_impl in ("entropy_attn", "entropy_attn_layer"):
                    reset_entropy_logs(runner.model)

                result = generate_with_traces(runner, input_ids, attention_mask, args.max_new_tokens)

                attn_entropy_trace = None
                if args.attn_impl == "entropy_attn":
                    logs = collect_entropy_logs(runner.model) or []
                    attn_entropy_trace = [
                        round(float(s["entropy_mean"]) /
                              max(float(torch.log(torch.tensor(float(s["kv_len"]))).item()), 1e-6), 6)
                        for s in logs
                    ]

                # Per-layer per-head traces: {layer_idx_str: [[H floats] × T steps]}, normalized by log(kv_len).
                attn_entropy_per_layer = None
                if args.attn_impl == "entropy_attn_layer":
                    all_logs = collect_all_layer_entropy_logs(runner.model) or {}
                    attn_entropy_per_layer = {}
                    for li, entries in sorted(all_logs.items()):
                        per_step = []
                        for entry in entries:
                            log_kv = max(float(torch.log(torch.tensor(float(entry["kv_len"]))).item()), 1e-6)
                            per_step.append([round(float(h) / log_kv, 6) for h in entry["entropy_per_head"]])
                        attn_entropy_per_layer[str(li)] = per_step

                gold_answer = str(ex.get("answer", ""))
                prefix = str(ex.get("random_string_to_prepend", ""))
                score, prefix_hit = grade_mrcr(result["prediction"], gold_answer, prefix)
                correct = prefix_hit and score >= 0.5

                score_sum += score
                correct_count += int(correct)
                total_written += 1

                row = {
                    "sample_id": ex.get("_idx", total_written - 1),
                    "task": task,
                    "context_length": ctx_len,
                    "generated_text": result["generated_text"],
                    "gold_answer": gold_answer,
                    "prediction": result["prediction"],
                    "mrcr_score": round(score, 4),
                    "correct": correct,
                    "generated_token_ids": result["generated_token_ids"],
                    "entropy_trace": result["entropy_trace"],
                    "norm_entropy_trace": result["norm_entropy_trace"],
                    "top1_prob_trace": result["top1_prob_trace"],
                    "top2_prob_trace": result["top2_prob_trace"],
                    "prob_margin_trace": result["prob_margin_trace"],
                    "logit_margin_trace": result["logit_margin_trace"],
                    "attn_entropy_trace": attn_entropy_trace,
                    "attn_entropy_per_layer": attn_entropy_per_layer,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()

                if args.status_every > 0 and total_written % args.status_every == 0:
                    print(
                        f"[{task}] n={total_written} | ctx={ctx_len} | "
                        f"score={100.0 * score_sum / total_written:.1f} | "
                        f"acc={100.0 * correct_count / total_written:.1f}"
                    )

    print(
        f"\nDone. written={total_written}, skipped={total_skipped}\n"
        f"mean_score={100.0 * score_sum / max(total_written, 1):.2f}  "
        f"accuracy={100.0 * correct_count / max(total_written, 1):.2f}\n"
        f"output: {args.output}"
    )


if __name__ == "__main__":
    main()
