#!/usr/bin/env python3
"""
MRCR session tuning harness for Qwen3.5-9B thinking/non-thinking runs.

This is a focused refactor of your RULER/InfiniteBench script for OpenAI MRCR:
  - dataset: openai/mrcr parquet files: 2needle, 4needle, 8needle
  - model default: Qwen/Qwen3.5-9B
  - supports entropy_attn session calibration and soft-instability controller
  - grading follows MRCR README: require random hash prefix, then SequenceMatcher

Example entropy run:
(soft gate instability controller with per-head calibrated session init)
CUDA_VISIBLE_DEVICES=0 python /c2/jenny/r3/entropy-attn-2026/mrcr_qwen35_session_tuning.py \
  --model Qwen/Qwen3.5-2B \
  --tasks 2needle,4needle \
  --session_size 50 \
  --session_init_mode calibrated_per_head \
  --session_calibration_samples 3 \
  --attn_impl entropy_attn \
  --dtype bf16 \
  --enable_thinking \
  --max_input_tokens 131000 \
  --max_new_tokens 1024 \
  --overlength_policy truncate \
  --truncate_strategy head_tail \
  --output_root /c2/jenny/r3/MRCR_outputs/qwen35-2b\
  --run_tag soft_target_instability.jsonl \
  --controller_mode soft_instability \
  --ema_beta 0.7 \
  --kp_slow 0.10 \
  --kp_fast 0.35 \
  --instability_threshold 0.015 \
  --target_deadband 0.05 \
  --compatibility_margin 0.15 \
  --max_step 0.005 \
  --target_trim_ratio 0.10 \
  --calibration_tail_k 256
  
  (legacy hard gate instability controller with calibrated session init)
  CUDA_VISIBLE_DEVICES=0 python /c2/jenny/r3/entropy-attn-2026/mrcr_qwen35_session_tuning.py \
  --model Qwen/Qwen3.5-2B \
  --tasks 8needle \
  --session_size 50 \
  --session_init_mode calibrated_per_head \
  --session_calibration_samples 3 \
  --attn_impl entropy_attn \
  --dtype bf16 \
  --disable_thinking \
  --max_input_tokens 65536 \
  --max_new_tokens 1024 \
  --min_tokens 32768 \
  --max_tokens 65536 \
  --overlength_policy skip \
  --controller_mode legacy \
  --max_step 0.0005 \
  --deterministic \
  --target_trim_ratio 0.10 \
  --calibration_tail_k 256 \
  --temp_max 1.2 \
  --output_root /c2/jenny/r3/MRCR_outputs/qwen35-2b \
  --run_tag legacy_controller_32-64k_maxtemp1p2


  
Baseline:
CUDA_VISIBLE_DEVICES=0 python /c2/jenny/r3/entropy-attn-2026/mrcr_qwen35_session_tuning.py \
  --model Qwen/Qwen3.5-2B \
  --tasks 2needle \
  --attn_impl sdpa \
  --dtype bf16 \
  --disable_thinking \
  --max_input_tokens 131000 \
  --max_new_tokens 1024 \
  --max_tokens 131072 \
  --overlength_policy skip \
  --output_root /c2/jenny/r3/MRCR_outputs/qwen35-9b-thinking \
  --run_tag baseline_sdpa.jsonl

CUDA_VISIBLE_DEVICES=3 python /c2/jenny/r3/entropy-attn-2026/mrcr_qwen35_session_tuning.py \
  --model Qwen/Qwen3.5-2B \
  --tasks 8needle \
  --attn_impl entropy_attn \
  --session_init_mode legacy \
  --dtype bf16 \
  --disable_thinking \
  --max_input_tokens 65536 \
  --max_new_tokens 1024 \
  --min_tokens 32768 \
  --max_tokens 65536 \
  --overlength_policy skip \
  --output_root /c2/jenny/r3/MRCR_outputs/qwen35-9b-thinking \
  --run_tag baseline_entropy_attn_32-64k.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Callable

import pandas as pd
import torch

from attention_qwen import QwenRunner
from models.entropy_scaling_soft_gated import EntropyDynamicTempController as EntropyTempController


def _cuda_time_call(fn: Callable, enabled: bool, state: Dict[str, int], skip: int, times: List[float]):
    """Time a single CUDA generation call with CUDA Events. Skips first `skip` calls (warmup)."""
    if not enabled or not torch.cuda.is_available():
        return fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    state["n"] = state.get("n", 0) + 1
    if state["n"] > skip:
        times.append(start.elapsed_time(end) / 1000.0)  # seconds
    return out

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


# ------------------------- model/attention helpers -------------------------


def get_attn_modules(model) -> List[torch.nn.Module]:
    mods: List[torch.nn.Module] = []
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None) if core is not None else None
    if layers is not None:
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                mods.append(attn)
        return mods
    for m in model.modules():
        if hasattr(m, "num_key_value_groups") or hasattr(m, "q_proj"):
            mods.append(m)
    return mods


def infer_module_num_heads(m: torch.nn.Module) -> int:
    # q_proj.out_features / head_dim always gives query heads, not KV heads.
    # Check this first so GQA models (e.g. Qwen3.5) don't return KV head count.
    if hasattr(m, "q_proj") and hasattr(m, "head_dim"):
        outf = getattr(m.q_proj, "out_features", None)
        hdim = getattr(m, "head_dim", None)
        if isinstance(outf, int) and isinstance(hdim, int) and hdim > 0:
            return int(outf // hdim)
    for attr in ["num_heads", "num_attention_heads", "n_heads"]:
        v = getattr(m, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return 1


def mark_last_layer_entropy_logger(model):
    logger_modules = []
    try:
        core = getattr(model, "model", None)
        layers = getattr(core, "layers", None) if core is not None else None
        if layers is not None and len(layers) > 0:
            for li, layer in enumerate(layers):
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    setattr(attn, "layer_idx", li)
            attn_last = getattr(layers[-1], "self_attn", None)
            if attn_last is not None:
                setattr(attn_last, "is_entropy_log_layer", True)
                logger_modules = [attn_last]
    except Exception:
        logger_modules = []
    setattr(model, "_entropy_logger_modules", logger_modules)
    return logger_modules


def reset_entropy_logs(model):
    mods = getattr(model, "_entropy_logger_modules", None)
    if mods:
        for m in mods:
            for attr in ["_entropy_log", "_decode_step"]:
                if hasattr(m, attr):
                    delattr(m, attr)
        return
    for m in model.modules():
        for attr in ["_entropy_log", "_decode_step"]:
            if hasattr(m, attr):
                delattr(m, attr)


def collect_entropy_logs(model):
    mods = getattr(model, "_entropy_logger_modules", None)
    if mods:
        return getattr(mods[0], "_entropy_log", None)
    for m in model.modules():
        if hasattr(m, "_entropy_log"):
            return m._entropy_log
    return None


def mark_all_layers_entropy_logger(model, max_log_steps: int = 0):
    """Mark ALL attention layers for per-head entropy logging (used with entropy_attn_layer).

    max_log_steps: cap decode steps logged per layer. 0 = unlimited.
    Returns list of marked attention modules.
    """
    logger_modules = []
    try:
        core = getattr(model, "model", None)
        layers = getattr(core, "layers", None) if core is not None else None
        if layers is not None:
            for li, layer in enumerate(layers):
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    setattr(attn, "layer_idx", li)
                    setattr(attn, "is_entropy_log_layer", True)
                    if max_log_steps > 0:
                        setattr(attn, "max_log_steps", max_log_steps)
                    logger_modules.append(attn)
    except Exception:
        logger_modules = []
    setattr(model, "_entropy_logger_modules", logger_modules)
    return logger_modules


def collect_all_layer_entropy_logs(model) -> Dict[int, List[Dict]]:
    """Collect per-layer entropy logs (used with entropy_attn_layer).

    Returns {layer_idx: [step_dicts]} where each step_dict has
    entropy_per_head, entropy_mean, entropy_std, kv_len, step.
    """
    result: Dict[int, List[Dict]] = {}
    mods = getattr(model, "_entropy_logger_modules", [])
    for m in mods:
        li = getattr(m, "layer_idx", -1)
        log = getattr(m, "_entropy_log", None)
        if log:
            result[li] = log
    return result


def collect_controller_debug(model) -> List[Dict[str, Any]]:
    rows = []
    for li, m in enumerate(get_attn_modules(model)):
        c = getattr(m, "_entropy_temp_controller", None)
        if c is None:
            continue
        stats = getattr(c, "last_stats", {}) or {}
        rows.append({"layer_like_idx": li, **stats})
    return rows


def reset_entropy_controller_state(model):
    for m in get_attn_modules(model):
        for attr in ["_entropy_temp_controller", "past_entropy", "past_temp"]:
            if hasattr(m, attr):
                delattr(m, attr)


def set_attn_attr(model, attr: str, value: Any):
    if value is None:
        return
    for m in get_attn_modules(model):
        setattr(m, attr, value)
        if attr in {"temp_max_step"} and hasattr(m, "_entropy_temp_controller"):
            delattr(m, "_entropy_temp_controller")


def collect_prompt_target_mean(model) -> Optional[float]:
    vals: List[float] = []
    for m in get_attn_modules(model):
        c = getattr(m, "_entropy_temp_controller", None)
        if c is None:
            continue
        tgt = getattr(c, "prompt_target_entropy", None)
        if tgt is not None and tgt.numel() > 0:
            v = float(tgt.mean().item())
            if v == v:
                vals.append(v)
    return None if not vals else sum(vals) / len(vals)


def collect_prompt_targets_by_module(model) -> List[Optional[torch.Tensor]]:
    out: List[Optional[torch.Tensor]] = []
    for m in get_attn_modules(model):
        c = getattr(m, "_entropy_temp_controller", None)
        tgt = getattr(c, "prompt_target_entropy", None) if c is not None else None
        out.append(None if tgt is None or tgt.numel() == 0 else tgt.detach().clone())
    return out


def aggregate_target_tensors(tensors: List[torch.Tensor], stat: str) -> torch.Tensor:
    xs = torch.stack(tensors, dim=0)
    out = xs.mean(dim=0) if stat == "mean" else xs.median(dim=0).values
    if out.ndim == 3:
        out = out.unsqueeze(0)
    if out.shape[0] != 1:
        out = out.mean(dim=0, keepdim=True)
    return out


@torch.inference_mode()
def run_prefill_probe(runner: QwenRunner, prompt: str):
    return runner.prefill_prompt(prompt)


def initialize_entropy_controller_state(
    model,
    temp_init: float,
    ema_init: float,
    target_init: float,
    per_module_targets: Optional[List[Optional[torch.Tensor]]],
    ema_init_mode: str,
    args: argparse.Namespace,
):
    device = next(model.parameters()).device
    for mi, m in enumerate(get_attn_modules(model)):
        n_heads = infer_module_num_heads(m)
        shape = (1, n_heads, 1)
        c = EntropyTempController(
            temp_init=float(temp_init),
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            ema_beta=args.ema_beta,
            kp_slow=args.kp_slow,
            kp_fast=args.kp_fast,
            max_step=args.max_step,
            instability_threshold=args.instability_threshold,
            target_deadband=args.target_deadband,
            compatibility_margin=args.compatibility_margin,
            rebound_window=args.rebound_window,
            refractory_steps=args.refractory_steps,
            mode=args.controller_mode,
        )
        c._init_state(shape, device)
        c.temp.fill_(float(temp_init))

        target_tensor = None
        if per_module_targets is not None and mi < len(per_module_targets):
            target_tensor = per_module_targets[mi]
        if target_tensor is not None:
            tt = target_tensor.to(device=device, dtype=c.temp.dtype)
            if tt.ndim == 2:
                tt = tt.unsqueeze(0)
            if tt.ndim != 3 or tt.shape[1] != n_heads:
                tt = torch.full(shape, float(tt.mean().item()), device=device, dtype=c.temp.dtype)
            if tt.shape[0] != 1:
                tt = tt.mean(dim=0, keepdim=True)
            c.prompt_target_entropy = tt
        else:
            c.prompt_target_entropy = torch.full(shape, float(target_init), device=device, dtype=c.temp.dtype)

        if ema_init_mode == "target":
            c.ema_entropy.copy_(c.prompt_target_entropy)
        else:
            c.ema_entropy.fill_(float(ema_init))
        m._entropy_temp_controller = c


# ------------------------- MRCR loading and grading -------------------------


def load_mrcr_task(task: str, local_root: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    # Strip .parquet suffix to get the bare task name (e.g. "2needle")
    task_name = task[:-len(".parquet")] if task.endswith(".parquet") else task
    if local_root:
        # Try flat file first, fall back to sharded layout
        flat_path = os.path.join(local_root, f"{task_name}.parquet")
        if os.path.exists(flat_path):
            paths = [flat_path]
        else:
            paths = sorted(
                p for p in [
                    os.path.join(local_root, task_name, f"{task_name}_{i}.parquet")
                    for i in range(10)
                ]
                if os.path.exists(p)
            )
            if not paths:
                raise FileNotFoundError(f"No parquet files found for task '{task_name}' under {local_root}")
    else:
        if hf_hub_download is None:
            raise RuntimeError("Install huggingface_hub or pass --local_root with MRCR parquet files.")
        # Dataset uses sharded layout: {task}/{task}_0.parquet, {task}/{task}_1.parquet, ...
        from huggingface_hub import list_repo_files
        shard_filenames = sorted(
            f for f in list_repo_files("openai/mrcr", repo_type="dataset")
            if f.startswith(f"{task_name}/") and f.endswith(".parquet")
        )
        if not shard_filenames:
            raise FileNotFoundError(f"No parquet shards found for task '{task_name}' in openai/mrcr dataset")
        paths = [
            hf_hub_download(repo_id="openai/mrcr", filename=fn, repo_type="dataset")
            for fn in shard_filenames
        ]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    for idx, row in df.iterrows():
        d = row.to_dict()
        d["_idx"] = int(idx)
        d["_task_file"] = f"{task_name}.parquet"
        yield d


def parse_mrcr_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
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
    raise ValueError("MRCR row has unsupported prompt format.")


def grade_mrcr(response: str, answer: str, random_string_to_prepend: str) -> float:
    response = (response or "").strip()
    answer = (answer or "").strip()
    prefix = random_string_to_prepend or ""
    if prefix and not response.startswith(prefix):
        return 0.0
    if prefix:
        response = response.removeprefix(prefix)
        answer = answer.removeprefix(prefix)
    return float(SequenceMatcher(None, response.strip(), answer.strip()).ratio())


def batched_sessions(examples: Iterable, session_size: int, max_examples: int):
    buf = []
    seen = 0
    for ex in examples:
        if max_examples > 0 and seen >= max_examples:
            break
        buf.append(ex)
        seen += 1
        if len(buf) >= session_size:
            yield buf
            buf = []
    if buf:
        yield buf


def iter_preprocessed_examples(examples: Iterable[Dict[str, Any]], runner, args, task: str = "", out_f=None):
    """Build prompt, apply overlength + token-window filters before session batching.

    Yields (ex, prompt, trunc) for valid examples.
    Skipped examples are excluded from session counts (written to out_f with session_idx=-1).
    """
    for ex in examples:
        messages = parse_mrcr_messages(ex)
        prompt = runner.messages_to_prompt(messages)
        trunc = maybe_truncate_prompt(runner, prompt, args)
        if trunc.get("skip"):
            if out_f is not None:
                out_f.write(
                    json.dumps(
                        {
                            "task": task,
                            "idx": ex.get("_idx"),
                            "session_idx": -1,
                            "prompt_tokens": trunc["prompt_tokens"],
                            "prompt_token_limit": trunc["limit"],
                            "_skipped_overlength": True,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            continue
        pt = trunc["prompt_tokens"]
        if (args.min_tokens > 0 and pt < args.min_tokens) or (args.max_tokens > 0 and pt > args.max_tokens):
            continue
        yield (ex, trunc["prompt"], trunc)


# ------------------------- token truncation -------------------------


def infer_model_input_limit(runner: QwenRunner, max_new_tokens: int) -> int:
    cfg = getattr(runner.model, "config", None)
    max_pos = getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
    tok_max = getattr(runner.tokenizer, "model_max_length", None)
    cands = []
    if isinstance(max_pos, int) and max_pos > 0:
        cands.append(max_pos)
    if isinstance(tok_max, int) and 0 < tok_max < 10_000_000:
        cands.append(tok_max)
    return max(1, min(cands) - max_new_tokens) if cands else max(1, 262144 - max_new_tokens)


def truncate_token_ids(token_ids: List[int], limit: int, strategy: str, head_keep_ratio: float) -> List[int]:
    if len(token_ids) <= limit:
        return token_ids
    if strategy == "tail":
        return token_ids[-limit:]
    ratio = max(0.0, min(1.0, head_keep_ratio))
    head_n = int(round(limit * ratio))
    head_n = max(0, min(limit, head_n))
    tail_n = limit - head_n
    if head_n == 0:
        return token_ids[-tail_n:]
    if tail_n == 0:
        return token_ids[:head_n]
    return token_ids[:head_n] + token_ids[-tail_n:]


def maybe_truncate_prompt(runner: QwenRunner, prompt: str, args: argparse.Namespace) -> Dict[str, Any]:
    ids = runner.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    limit = args.max_input_tokens if args.max_input_tokens > 0 else infer_model_input_limit(runner, args.max_new_tokens)
    if len(ids) <= limit:
        return {"prompt": prompt, "prompt_tokens": len(ids), "limit": limit, "truncated": False}
    if args.overlength_policy == "error":
        raise ValueError(f"prompt has {len(ids)} tokens, exceeds limit {limit}")
    if args.overlength_policy == "skip":
        return {"prompt": prompt, "prompt_tokens": len(ids), "limit": limit, "skip": True, "truncated": False}
    kept = truncate_token_ids(ids, limit, args.truncate_strategy, args.head_keep_ratio)
    return {
        "prompt": runner.tokenizer.decode(kept, skip_special_tokens=False),
        "prompt_tokens": limit,
        "limit": limit,
        "truncated": True,
    }


# ------------------------- main -------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--tasks", default="2needle,4needle,8needle")
    ap.add_argument("--local_root", default=None)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--run_tag", default="mrcr_predictions.jsonl")
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--session_size", type=int, default=50)
    ap.add_argument("--session_init_mode", choices=["legacy", "calibrated_scalar", "calibrated_per_head"], default="calibrated_per_head")
    ap.add_argument("--session_calibration_samples", type=int, default=3)
    ap.add_argument("--calibration_max_tokens", type=int, default=0,
                    help="Truncate calibration probes to this many tokens (tail kept). "
                         "0 = use full prompt. Speeds up calibration at long contexts "
                         "at the cost of using early-prefix entropy as a proxy target.")
    ap.add_argument("--session_target_stat", choices=["mean", "median"], default="mean")
    ap.add_argument("--session_ema_init_mode", choices=["zero", "target"], default="target")
    ap.add_argument("--session_temp_init", type=float, default=1.0)
    ap.add_argument("--session_temp_target_gain", type=float, default=0.15)

    ap.add_argument("--attn_impl", choices=["entropy_attn", "sdpa", "flash_attention_2", "eager"], default="entropy_attn")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--disable_thinking", action="store_true")
    ap.add_argument("--do_sample", action="store_true", help="For Qwen thinking, sampling may be healthier than greedy; keep off for deterministic ablations.")
    ap.add_argument("--sample_temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)

    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_input_tokens", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=0,
                    help="Skip samples whose prompt_tokens (post-truncation) exceed this value. 0=no filter.")
    ap.add_argument("--min_tokens", type=int, default=0,
                    help="Skip samples whose prompt_tokens (post-truncation) are below this value. 0=no filter.")
    ap.add_argument("--overlength_policy", choices=["error", "skip", "truncate"], default="truncate")
    ap.add_argument("--truncate_strategy", choices=["tail", "head_tail"], default="head_tail")
    ap.add_argument("--head_keep_ratio", type=float, default=0.5)
    ap.add_argument("--stop_on_newline", action="store_true")
    ap.add_argument("--status_every", type=int, default=10)
    ap.add_argument("--time", action="store_true", help="Measure CUDA generation time per example.")
    ap.add_argument("--time_skip", type=int, default=2, help="Warmup examples to skip before recording times.")

    # Old/prompt calibration patch knobs.
    ap.add_argument("--max_step", type=float, default=5e-4)
    ap.add_argument("--target_trim_ratio", type=float, default=0.10)
    ap.add_argument("--calibration_tail_k", type=int, default=256)

    # New soft-instability controller knobs.
    ap.add_argument("--controller_mode", choices=["legacy", "soft_instability", "instability_only"], default="soft_instability")
    ap.add_argument("--temp_min", type=float, default=0.7)
    ap.add_argument("--temp_max", type=float, default=1.0)
    ap.add_argument("--ema_beta", type=float, default=0.7)
    ap.add_argument("--kp_slow", type=float, default=0.10)
    ap.add_argument("--kp_fast", type=float, default=0.35)
    ap.add_argument("--instability_threshold", type=float, default=0.015)
    ap.add_argument("--target_deadband", type=float, default=0.020)
    ap.add_argument("--compatibility_margin", type=float, default=0.150)
    ap.add_argument("--rebound_window", type=int, default=4)
    ap.add_argument("--refractory_steps", type=int, default=0)

    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    thinking = bool(args.enable_thinking and not args.disable_thinking)
    runner = QwenRunner(
        model_name=args.model,
        attn_impl=args.attn_impl,
        dtype=dtype_map[args.dtype],
        deterministic=args.deterministic,
        enable_thinking=thinking,
    )

    if args.attn_impl == "entropy_attn":
        mark_last_layer_entropy_logger(runner.model)
        set_attn_attr(runner.model, "temp_max_step", float(args.max_step))
        set_attn_attr(runner.model, "target_trim_ratio", float(args.target_trim_ratio))
        set_attn_attr(runner.model, "calibration_tail_k", int(args.calibration_tail_k))

    os.makedirs(args.output_root, exist_ok=True)
    pred_time_state: Dict[str, int] = {}
    pred_times_s: List[float] = []
    tasks = [t.strip().replace(".parquet", "") for t in args.tasks.split(",") if t.strip()]
    final_summary: Dict[str, Any] = {"config": vars(args), "tasks": {}}

    interrupted = False
    for task in tasks:
        if interrupted:
            break
        task_dir = os.path.join(args.output_root, task)
        os.makedirs(task_dir, exist_ok=True)
        pred_path = os.path.join(task_dir, f"{task}_{args.attn_impl}_{args.run_tag}")
        entropy_path = os.path.join(task_dir, f"{task}_{args.attn_impl}_entropy_{args.run_tag}")
        session_path = os.path.join(task_dir, f"{task}_{args.attn_impl}_sessions_{args.run_tag}")

        total = 0
        score_sum = 0.0
        prefix_hits = 0
        session_rows = []

        examples = load_mrcr_task(task, args.local_root)
        entropy_ctx = open(entropy_path, "w", encoding="utf-8") if args.attn_impl == "entropy_attn" else nullcontext(None)
        with open(pred_path, "w", encoding="utf-8") as out_f, open(session_path, "w", encoding="utf-8") as sess_f, entropy_ctx as ent_f:
            preprocessed = iter_preprocessed_examples(examples, runner, args, task=task, out_f=out_f)
            for session_idx, session_examples in enumerate(batched_sessions(preprocessed, max(1, args.session_size), args.max_examples)):
                session_target_init = None
                session_temp_init = None
                session_ema_init = None

                if args.attn_impl == "entropy_attn":
                    reset_entropy_controller_state(runner.model)
                    reset_entropy_logs(runner.model)

                    if args.session_init_mode != "legacy":
                        K = max(0, min(args.session_calibration_samples, len(session_examples)))
                        target_vals: List[float] = []
                        n_mods = len(get_attn_modules(runner.model))
                        per_mod_samples: List[List[torch.Tensor]] = [[] for _ in range(n_mods)]

                        for (_, cal_prompt, _) in session_examples[:K]:
                            if args.calibration_max_tokens > 0:
                                cal_ids = runner.tokenizer(cal_prompt, add_special_tokens=False)["input_ids"]
                                if len(cal_ids) > args.calibration_max_tokens:
                                    cal_ids = cal_ids[-args.calibration_max_tokens:]
                                    cal_prompt = runner.tokenizer.decode(cal_ids, skip_special_tokens=False)
                            reset_entropy_controller_state(runner.model)
                            run_prefill_probe(runner, cal_prompt)
                            tmean = collect_prompt_target_mean(runner.model)
                            if tmean is not None:
                                target_vals.append(float(tmean))
                            if args.session_init_mode == "calibrated_per_head":
                                per_mod_targets = collect_prompt_targets_by_module(runner.model)
                                for mi, tgt in enumerate(per_mod_targets):
                                    if tgt is not None:
                                        per_mod_samples[mi].append(tgt.detach().clone())

                        if target_vals:
                            session_target_init = sum(target_vals) / len(target_vals) if args.session_target_stat == "mean" else sorted(target_vals)[len(target_vals) // 2]
                        else:
                            session_target_init = 0.5
                        session_temp_init = float(args.session_temp_init) - float(args.session_temp_target_gain) * (session_target_init - 0.5)
                        session_temp_init = max(args.temp_min, min(args.temp_max, session_temp_init))
                        session_ema_init = 0.0 if args.session_ema_init_mode == "zero" else float(session_target_init)

                        per_module_targets = None
                        if args.session_init_mode == "calibrated_per_head":
                            per_module_targets = []
                            for mi, m in enumerate(get_attn_modules(runner.model)):
                                samples = per_mod_samples[mi] if mi < len(per_mod_samples) else []
                                if samples:
                                    per_module_targets.append(aggregate_target_tensors(samples, args.session_target_stat))
                                else:
                                    n_heads = infer_module_num_heads(m)
                                    per_module_targets.append(torch.full((1, n_heads, 1), float(session_target_init)))

                        reset_entropy_controller_state(runner.model)
                        initialize_entropy_controller_state(
                            runner.model,
                            temp_init=float(session_temp_init),
                            ema_init=float(session_ema_init),
                            target_init=float(session_target_init),
                            per_module_targets=per_module_targets,
                            ema_init_mode=args.session_ema_init_mode,
                            args=args,
                        )

                sess_n = 0
                sess_score = 0.0
                sess_prefix = 0

                try:
                  for (ex, prompt, trunc) in session_examples:
                    if args.attn_impl == "entropy_attn":
                        reset_entropy_logs(runner.model)

                    pred = _cuda_time_call(
                        lambda: runner.generate_one(
                            prompt,
                            max_new_tokens=args.max_new_tokens,
                            stop_on_newline=args.stop_on_newline,
                            do_sample=args.do_sample,
                            temperature=args.sample_temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        ),
                        enabled=args.time,
                        state=pred_time_state,
                        skip=args.time_skip,
                        times=pred_times_s,
                    )

                    answer = str(ex.get("answer", ""))
                    prefix = str(ex.get("random_string_to_prepend", ""))
                    score = grade_mrcr(pred, answer, prefix)
                    prefix_hit = int((pred or "").strip().startswith(prefix)) if prefix else 1

                    total += 1
                    score_sum += score
                    prefix_hits += prefix_hit
                    sess_n += 1
                    sess_score += score
                    sess_prefix += prefix_hit

                    row = {
                        "task": task,
                        "idx": ex.get("_idx"),
                        "session_idx": session_idx,
                        "prompt_tokens": trunc["prompt_tokens"],
                        "prompt_token_limit": trunc["limit"],
                        "truncated": bool(trunc.get("truncated", False)),
                        "random_string_to_prepend": prefix,
                        "prediction": pred,
                        "answer": answer,
                        "mrcr_score": score,
                        "prefix_hit": bool(prefix_hit),
                        "gen_time_s": round(pred_times_s[-1], 4) if pred_times_s and args.time else None,
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()

                    if ent_f is not None:
                        ent_f.write(json.dumps({
                            "task": task,
                            "idx": ex.get("_idx"),
                            "session_idx": session_idx,
                            "prompt_target_mean": collect_prompt_target_mean(runner.model),
                            "controller_debug": collect_controller_debug(runner.model)[-2:],
                            "entropy_log": collect_entropy_logs(runner.model),
                        }, ensure_ascii=False) + "\n")
                        ent_f.flush()

                    if args.status_every > 0 and total % args.status_every == 0:
                        print(f"[{task}] {total} done | score={100.0 * score_sum / max(total,1):.2f} | prefix={100.0 * prefix_hits / max(total,1):.2f}")
                except KeyboardInterrupt:
                    print(f"\n[interrupted] session {session_idx} partial: n={sess_n} score={100.0 * sess_score / max(sess_n,1):.2f}")
                    interrupted = True
                    break

                if sess_n > 0:
                    sess_row = {
                        "task": task,
                        "session_idx": session_idx,
                        "session_init_mode": args.session_init_mode,
                        "evaluated_n": sess_n,
                        "partial": interrupted,
                        "session_score": round(100.0 * sess_score / max(sess_n, 1), 2),
                        "session_prefix_acc": round(100.0 * sess_prefix / max(sess_n, 1), 2),
                        "session_target_init": session_target_init,
                        "session_temp_init": session_temp_init,
                        "session_ema_init": session_ema_init,
                    }
                    session_rows.append(sess_row)
                    sess_f.write(json.dumps(sess_row, ensure_ascii=False) + "\n")
                    sess_f.flush()
                    print(f"[{task}] session {session_idx} | n={sess_n} | score={sess_row['session_score']:.2f} | prefix={sess_row['session_prefix_acc']:.2f}")
                if interrupted:
                    break

        if total > 0:
            timing_summary = {}
            if args.time and pred_times_s:
                timing_summary = {
                    "mean_gen_time_s": round(sum(pred_times_s) / len(pred_times_s), 4),
                    "total_gen_time_s": round(sum(pred_times_s), 2),
                    "timed_examples": len(pred_times_s),
                }
            final_summary["tasks"][task] = {
                "n": total,
                "partial": interrupted,
                "mrcr_score": round(100.0 * score_sum / max(total, 1), 2),
                "prefix_acc": round(100.0 * prefix_hits / max(total, 1), 2),
                "prediction_file": pred_path,
                "entropy_file": entropy_path if args.attn_impl == "entropy_attn" else None,
                "session_file": session_path,
                "num_sessions": len(session_rows),
                **timing_summary,
            }

    if interrupted:
        final_summary["interrupted"] = True
        print("\n[interrupted] writing partial summary...")
    summary_path = os.path.join(args.output_root, f"mrcr_summary_{args.attn_impl}_{args.run_tag}.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(final_summary, sf, indent=2, ensure_ascii=False)
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    print(f"[saved] summary: {summary_path}")


if __name__ == "__main__":
    main()
