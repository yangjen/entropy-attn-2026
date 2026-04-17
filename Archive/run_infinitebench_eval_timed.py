"""
Example usage (HF dataset):
CUDA_VISIBLE_DEVICES=1 python /c2/jenny/r3/entropy-attn-2026/run_infinitebench_eval_timed.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tasks longdialogue_qa_eng \
  --max_examples 50 \
  --output_root /c2/jenny/r3/InfiniteBench_outputs/llama3.1-8b-chat/data \
  --attn_impl entropy_attn \
  --dtype bf16 \
  --max_new_tokens 16 \
  --deterministic \
  --time \
  --time_skip 4 \
  --max_step 0.0005 \
  --target_trim_ratio 0.10 \
  --max_input_tokens 100000 \
  --overlength_policy truncate \
  --run_tag scaled_pipeline_testing_truncate.jsonl

Baseline (fixed temperature with entropy_attn path):
CUDA_VISIBLE_DEVICES=1 python /c2/jenny/r3/entropy-attn-2026/run_infinitebench_eval_timed.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tasks longdialogue_qa_eng \
  --max_examples 50 \
  --output_root /c2/jenny/r3/InfiniteBench_outputs/llama3.1-8b-chat/data \
  --attn_impl sdpa \
  --dtype bf16 \
  --max_new_tokens 16 \
  --deterministic \
  --time --time_skip 4 \
  --max_step 0.0 \
  --target_trim_ratio 0.0 \
  --max_input_tokens 32768 \
  --overlength_policy truncate \
  --run_tag baseline_pipeline_testing_truncate.jsonl
  
"""

import argparse
import json
import os
import re
import string
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional

import torch
from attention_llama import LlamaRunner

try:
    from datasets import Features, Sequence, Value, load_dataset
except Exception:
    load_dataset = None
    Features = None
    Sequence = None
    Value = None

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _summarize_times(times: List[float]) -> Dict[str, float]:
    if not times:
        return {"n": 0}
    ts = sorted(times)
    n = len(ts)

    def pct(p: float) -> float:
        if n == 1:
            return ts[0]
        k = int(round((p / 100.0) * (n - 1)))
        k = max(0, min(n - 1, k))
        return ts[k]

    return {
        "n": n,
        "mean_s": sum(ts) / n,
        "median_s": pct(50),
        "p90_s": pct(90),
        "p95_s": pct(95),
        "min_s": ts[0],
        "max_s": ts[-1],
    }


def _cuda_time_call(fn, enabled: bool, state: Dict[str, int], skip: int, times: List[float]):
    if not enabled or (not torch.cuda.is_available()):
        return fn()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    out = fn()
    end_event.record()
    torch.cuda.synchronize()
    state["n"] = state.get("n", 0) + 1
    if state["n"] > skip:
        times.append(start_event.elapsed_time(end_event) / 1000.0)
    return out


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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
            if hasattr(m, "_entropy_log"):
                delattr(m, "_entropy_log")
            if hasattr(m, "_decode_step"):
                delattr(m, "_decode_step")
        return
    for m in model.modules():
        if hasattr(m, "_entropy_log"):
            delattr(m, "_entropy_log")
        if hasattr(m, "_decode_step"):
            delattr(m, "_decode_step")


def collect_entropy_logs(model):
    mods = getattr(model, "_entropy_logger_modules", None)
    if mods:
        return getattr(mods[0], "_entropy_log", None)
    for m in model.modules():
        if hasattr(m, "_entropy_log"):
            return m._entropy_log
    return None


def set_temp_max_step(model, max_step: Optional[float]):
    if max_step is None:
        return
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None) if core is not None else None
    if layers is not None:
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                setattr(attn, "temp_max_step", float(max_step))
                if hasattr(attn, "_entropy_temp_controller"):
                    delattr(attn, "_entropy_temp_controller")
        return
    for m in model.modules():
        if hasattr(m, "num_key_value_groups"):
            setattr(m, "temp_max_step", float(max_step))
            if hasattr(m, "_entropy_temp_controller"):
                delattr(m, "_entropy_temp_controller")


def set_target_trim_ratio(model, trim_ratio: Optional[float]):
    if trim_ratio is None:
        return
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None) if core is not None else None
    if layers is not None:
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                setattr(attn, "target_trim_ratio", float(trim_ratio))
        return
    for m in model.modules():
        if hasattr(m, "num_key_value_groups"):
            setattr(m, "target_trim_ratio", float(trim_ratio))


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"<\|.*?\|>", " ", s)
    s = s.translate(_PUNCT_TABLE)
    s = " ".join(s.split())
    return s


def contains_any(pred: str, golds: List[str]) -> int:
    p = normalize_text(pred)
    if not p:
        return 0
    for g in golds:
        ng = normalize_text(g)
        if ng and ng in p:
            return 1
    return 0


def exact_any(pred: str, golds: List[str]) -> int:
    p = normalize_text(pred)
    return int(any(p == normalize_text(g) for g in golds))


def build_prompt(runner: LlamaRunner, ex: Dict[str, Any], style: str) -> str:
    context = ex.get("context", "") or ""
    question = ex.get("input", "") or ""
    if style == "minimal":
        user_text = f"Context:\n{context}\n\nQuestion:\n{question}\n\nGive only the final answer."
    else:
        user_text = (
            "Read the following long context and answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer with only the final answer."
        )

    tok = runner.tokenizer
    if getattr(tok, "chat_template", None):
        messages = [{"role": "user", "content": user_text}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_text + "\n"


def infer_model_input_limit(runner: LlamaRunner, max_new_tokens: int) -> int:
    cfg = getattr(runner.model, "config", None)
    max_pos = getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
    tok_max = getattr(runner.tokenizer, "model_max_length", None)

    cands = []
    if isinstance(max_pos, int) and max_pos > 0:
        cands.append(max_pos)
    # tokenizers often use very large sentinels when "unbounded"; ignore those.
    if isinstance(tok_max, int) and tok_max > 0 and tok_max < 10_000_000:
        cands.append(tok_max)

    if cands:
        return max(1, min(cands) - max_new_tokens)
    return max(1, 131072 - max_new_tokens)


def truncate_token_ids(
    token_ids: List[int],
    limit: int,
    strategy: str,
    head_keep_ratio: float,
) -> List[int]:
    if len(token_ids) <= limit:
        return token_ids
    if strategy == "tail":
        return token_ids[-limit:]

    # head_tail: keep prefix + suffix, drop middle.
    ratio = max(0.0, min(1.0, head_keep_ratio))
    head_n = int(round(limit * ratio))
    head_n = max(0, min(limit, head_n))
    tail_n = limit - head_n

    if head_n == 0:
        return token_ids[-tail_n:]
    if tail_n == 0:
        return token_ids[:head_n]
    return token_ids[:head_n] + token_ids[-tail_n:]


def task_examples_from_hf(task: str):
    # Preferred path: direct file download avoids fsspec glob issues seen with some
    # datasets/fsspec version combinations on repo-level load_dataset().
    if hf_hub_download is not None:
        try:
            local_file = hf_hub_download(
                repo_id="xinrongzhang2022/InfiniteBench",
                filename=f"{task}.jsonl",
                repo_type="dataset",
            )
            return iter_jsonl(local_file)
        except Exception:
            pass

    if load_dataset is None:
        raise RuntimeError(
            "Could not download InfiniteBench task file directly, and `datasets` is unavailable. "
            "Install `datasets`/`huggingface_hub` or use --source local_jsonl."
        )

    ds = None
    if Features is not None and Sequence is not None and Value is not None:
        try:
            features = Features(
                {
                    "id": Value("string"),
                    "context": Value("string"),
                    "input": Value("string"),
                    "answer": Sequence(Value("string")),
                    "options": Value("string"),
                }
            )
            ds = load_dataset("xinrongzhang2022/InfiniteBench", features=features)
        except Exception:
            ds = None
    if ds is None:
        ds = load_dataset("xinrongzhang2022/InfiniteBench")
    if task not in ds:
        raise KeyError(f"Task split '{task}' not found in InfiniteBench dataset.")
    return ds[task]


def task_examples_from_local(task: str, local_root: str):
    path = os.path.join(local_root, f"{task}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing local task file: {path}")
    return iter_jsonl(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="longbook_qa_eng,longdialogue_qa_eng")
    ap.add_argument("--source", choices=["hf", "local_jsonl"], default="hf")
    ap.add_argument("--local_root", default=None, help="Used when --source local_jsonl")
    ap.add_argument(
        "--output_root",
        default="/c2/jenny/r3/InfiniteBench_outputs/llama3.1-8b-chat/data",
        help="Root directory to store task outputs and summary.",
    )
    ap.add_argument("--max_examples", type=int, default=0, help="0 means all")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="Hard cap for prompt tokens. 0 means infer from model context window.",
    )
    ap.add_argument(
        "--overlength_policy",
        choices=["error", "skip", "truncate"],
        default="truncate",
        help="Behavior when prompt exceeds input token limit.",
    )
    ap.add_argument(
        "--truncate_strategy",
        choices=["tail", "head_tail"],
        default="tail",
        help="When truncating, keep only tail tokens or keep both ends and drop the middle.",
    )
    ap.add_argument(
        "--head_keep_ratio",
        type=float,
        default=0.5,
        help="Used when --truncate_strategy head_tail. Fraction of kept tokens allocated to prefix.",
    )
    ap.add_argument("--multiline", action="store_true")
    ap.add_argument("--prompt_style", choices=["infinitebench", "minimal"], default="infinitebench")
    ap.add_argument("--run_tag", default="infinitebench_predictions.jsonl")
    ap.add_argument("--status_every", type=int, default=10)
    ap.add_argument("--attn_impl", default="entropy_attn", choices=["entropy_attn", "sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--time_skip", type=int, default=4)
    ap.add_argument("--max_step", type=float, default=None, help="Entropy controller max_step. Use 0.0 for fixed-temp baseline.")
    ap.add_argument("--target_trim_ratio", type=float, default=0.0)
    args = ap.parse_args()

    if args.source == "local_jsonl" and not args.local_root:
        raise ValueError("--local_root is required when --source local_jsonl")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    runner = LlamaRunner(
        args.model,
        attn_impl=args.attn_impl,
        dtype=dtype_map[args.dtype],
        deterministic=args.deterministic,
    )

    if args.attn_impl == "entropy_attn":
        model_obj = getattr(runner, "model", None)
        if model_obj is not None:
            mark_last_layer_entropy_logger(model_obj)
            set_temp_max_step(model_obj, args.max_step)
            set_target_trim_ratio(model_obj, args.target_trim_ratio)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_dir = args.output_root
    os.makedirs(out_dir, exist_ok=True)

    summary: Dict[str, Any] = {}
    pred_times_s: List[float] = []
    pred_time_state: Dict[str, int] = {"n": 0}

    for task in tasks:
        task_dir = os.path.join(out_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        pred_path = os.path.join(task_dir, f"{task}_{args.attn_impl}_predictions_{args.run_tag}")
        entropy_path = os.path.join(task_dir, f"{task}_{args.attn_impl}_entropy_log_{args.run_tag}")

        total = 0
        hit_contains = 0
        hit_exact = 0

        if args.source == "hf":
            examples = task_examples_from_hf(task)
        else:
            examples = task_examples_from_local(task, args.local_root)

        entropy_ctx = open(entropy_path, "w", encoding="utf-8") if args.attn_impl == "entropy_attn" else nullcontext(None)
        with open(pred_path, "w", encoding="utf-8") as out_f, entropy_ctx as ent_f:
            for ex in examples:
                if args.max_examples > 0 and total >= args.max_examples:
                    break

                prompt = build_prompt(runner, ex, args.prompt_style)
                tokenized = runner.tokenizer(prompt, add_special_tokens=False)
                prompt_ids = tokenized["input_ids"]
                n_prompt_tokens = len(prompt_ids)
                model_input_limit = (
                    args.max_input_tokens
                    if args.max_input_tokens > 0
                    else infer_model_input_limit(runner, args.max_new_tokens)
                )

                if n_prompt_tokens > model_input_limit:
                    ex_id = ex.get("id", str(total))
                    if args.overlength_policy == "error":
                        raise ValueError(
                            f"[{task}] example {ex_id} has {n_prompt_tokens} prompt tokens, "
                            f"exceeds limit {model_input_limit}. "
                            "Use --overlength_policy skip|truncate or set --max_input_tokens."
                        )
                    if args.overlength_policy == "skip":
                        out_f.write(
                            json.dumps(
                                {
                                    "task": task,
                                    "id": ex_id,
                                    "input": ex.get("input", ""),
                                    "answers": ex.get("answer", []),
                                    "prediction": "",
                                    "_skipped_overlength": True,
                                    "_prompt_tokens": n_prompt_tokens,
                                    "_prompt_token_limit": model_input_limit,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue

                    kept = truncate_token_ids(
                        prompt_ids,
                        limit=model_input_limit,
                        strategy=args.truncate_strategy,
                        head_keep_ratio=args.head_keep_ratio,
                    )
                    prompt = runner.tokenizer.decode(kept, skip_special_tokens=False)
                    n_prompt_tokens = model_input_limit

                if args.attn_impl == "entropy_attn":
                    reset_entropy_logs(runner.model)

                pred = _cuda_time_call(
                    lambda: runner.generate_one(
                        prompt,
                        max_new_tokens=args.max_new_tokens,
                        stop_on_newline=(not args.multiline),
                    ),
                    enabled=args.time,
                    state=pred_time_state,
                    skip=args.time_skip,
                    times=pred_times_s,
                )

                golds = ex.get("answer", []) or []
                if isinstance(golds, str):
                    golds = [golds]

                c_hit = contains_any(pred, list(golds))
                e_hit = exact_any(pred, list(golds))
                hit_contains += c_hit
                hit_exact += e_hit
                total += 1

                entropy_logs = collect_entropy_logs(runner.model) if args.attn_impl == "entropy_attn" else None
                if ent_f is not None and entropy_logs is not None:
                    ex_id = ex.get("id", str(total - 1))
                    ent_f.write(
                        json.dumps(
                            {
                                "task": task,
                                "attn_impl": args.attn_impl,
                                "example_id": ex_id,
                                "entropy_log": entropy_logs,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                ex_out = {
                    "task": task,
                    "id": ex.get("id", str(total - 1)),
                    "input": ex.get("input", ""),
                    "answers": list(golds),
                    "prediction": pred,
                    "_prompt_tokens": n_prompt_tokens,
                    "_prompt_token_limit": model_input_limit,
                    "_truncate_strategy": args.truncate_strategy,
                    "_contains_hit": bool(c_hit),
                    "_exact_hit": bool(e_hit),
                }
                out_f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")

                if args.status_every > 0 and (total % args.status_every == 0):
                    c_acc = 100.0 * hit_contains / max(total, 1)
                    e_acc = 100.0 * hit_exact / max(total, 1)
                    print(
                        f"[{task}] {total} done | contains={c_acc:.2f}% ({hit_contains}/{total}) "
                        f"| exact={e_acc:.2f}% ({hit_exact}/{total})"
                    )

        task_summary: Dict[str, Any] = {
            "n": total,
            "contains_acc": round(100.0 * hit_contains / max(total, 1), 2),
            "exact_acc": round(100.0 * hit_exact / max(total, 1), 2),
            "prediction_file": pred_path,
        }
        if args.time:
            task_summary["pred_time"] = _summarize_times(pred_times_s)
        summary[task] = task_summary

    final_report = {"config": vars(args), "summary": summary}
    summary_path = os.path.join(out_dir, f"infinitebench_summary_{args.attn_impl}_{args.run_tag}.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(final_report, sf, indent=2, ensure_ascii=False)

    print(json.dumps(final_report, indent=2, ensure_ascii=False))
    print(f"[saved] summary: {summary_path}")


if __name__ == "__main__":
    main()
