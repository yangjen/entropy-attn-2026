"""
Example usage (HF dataset):
CUDA_VISIBLE_DEVICES=2 python /c2/jenny/r3/entropy-attn-2026/infinitebench_tuning.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tasks longbook_qa_eng \
  --max_examples 200 \
  --session_size 40 \
  --session_calibration_samples 4 \
  --session_target_stat median \
  --session_ema_init_mode target \
  --session_temp_init 1.0 \
  --session_temp_target_gain 0.15 \
  --output_root /c2/jenny/r3/InfiniteBench_outputs/llama3.1-8b-chat/data \
  --attn_impl entropy_attn \
  --dtype bf16 \
  --max_new_tokens 16 \
  --deterministic \
  --time --time_skip 4 \
  --max_step 0.0005 \
  --target_trim_ratio 0.10 \
  --max_input_tokens 32768 \
  --overlength_policy truncate \
  --run_tag tuning_sessions_testing_40.jsonl

Baseline (fixed temperature with entropy_attn path):
CUDA_VISIBLE_DEVICES=3 python /c2/jenny/r3/entropy-attn-2026/infinitebench_tuning.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tasks longbook_qa_eng \
  --max_examples 200 \
  --session_size 40 \
  --output_root /c2/jenny/r3/InfiniteBench_outputs/llama3.1-8b-chat/data \
  --attn_impl sdpa \
  --dtype bf16 \
  --max_new_tokens 16 \
  --deterministic \
  --time --time_skip 4 \
  --max_input_tokens 32768 \
  --overlength_policy truncate \
  --run_tag session_baseline_sdpa.jsonl

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
from models.entropy_scaling import EntropyTempController

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
        if hasattr(m, "num_key_value_groups"):
            mods.append(m)
    return mods


def infer_module_num_heads(m: torch.nn.Module) -> int:
    for attr in ["num_heads", "num_attention_heads", "n_heads"]:
        v = getattr(m, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    if hasattr(m, "q_proj") and hasattr(m, "head_dim"):
        outf = getattr(m.q_proj, "out_features", None)
        hdim = getattr(m, "head_dim", None)
        if isinstance(outf, int) and isinstance(hdim, int) and hdim > 0:
            return int(outf // hdim)
    return 1


def reset_entropy_controller_state(model):
    for m in get_attn_modules(model):
        if hasattr(m, "_entropy_temp_controller"):
            delattr(m, "_entropy_temp_controller")
        if hasattr(m, "past_entropy"):
            delattr(m, "past_entropy")
        if hasattr(m, "past_temp"):
            delattr(m, "past_temp")


def collect_prompt_target_mean(model) -> Optional[float]:
    vals: List[float] = []
    for m in get_attn_modules(model):
        c = getattr(m, "_entropy_temp_controller", None)
        if c is None:
            continue
        tgt = getattr(c, "prompt_target_entropy", None)
        if tgt is None:
            continue
        if tgt.numel() == 0:
            continue
        v = float(tgt.mean().item())
        if v == v:  # not NaN
            vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def collect_prompt_targets_by_module(model) -> List[Optional[torch.Tensor]]:
    out: List[Optional[torch.Tensor]] = []
    for m in get_attn_modules(model):
        c = getattr(m, "_entropy_temp_controller", None)
        if c is None:
            out.append(None)
            continue
        tgt = getattr(c, "prompt_target_entropy", None)
        if tgt is None or tgt.numel() == 0:
            out.append(None)
            continue
        out.append(tgt.detach().clone())
    return out


def aggregate_target_tensors(tensors: List[torch.Tensor], stat: str) -> torch.Tensor:
    xs = torch.stack(tensors, dim=0)  # [K, Z, H, 1]
    if stat == "mean":
        out = xs.mean(dim=0)
    else:
        out = xs.median(dim=0).values
    # collapse any batch variation from calibration to one batch lane
    if out.ndim == 3:
        out = out.unsqueeze(0)
    if out.shape[0] != 1:
        out = out.mean(dim=0, keepdim=True)
    return out


@torch.inference_mode()
def run_prefill_probe(runner: LlamaRunner, prompt: str):
    inputs = runner.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(runner.model.device) for k, v in inputs.items()}
    _ = runner.model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        use_cache=False,
        return_dict=True,
    )


def initialize_entropy_controller_state(
    model,
    temp_init: float,
    ema_init: float,
    target_init: float,
    per_module_targets: Optional[List[Optional[torch.Tensor]]] = None,
    ema_init_mode: str = "target",
):
    device = next(model.parameters()).device
    for mi, m in enumerate(get_attn_modules(model)):
        max_step = float(getattr(m, "temp_max_step", 0.0005))
        c = EntropyTempController(
            temp_init=float(temp_init),
            temp_min=0.7,
            temp_max=1.0,
            ema_beta=0.9,
            kp=0.35,
            max_step=max_step,
        )
        n_heads = infer_module_num_heads(m)
        shape = (1, n_heads, 1)
        c._init_state(shape, device)
        c.temp.fill_(float(temp_init))

        target_tensor = None
        if per_module_targets is not None and mi < len(per_module_targets):
            target_tensor = per_module_targets[mi]
        if target_tensor is not None:
            tt = target_tensor.to(device=device, dtype=c.temp.dtype)
            if tt.ndim == 2:
                tt = tt.unsqueeze(0)
            if tt.ndim != 3:
                tt = torch.full(shape, float(target_init), device=device, dtype=c.temp.dtype)
            if tt.shape[0] != 1:
                tt = tt.mean(dim=0, keepdim=True)
            if tt.shape[1] != n_heads:
                # Fallback to scalar mean if head count mismatches.
                tt_scalar = float(tt.mean().item())
                tt = torch.full(shape, tt_scalar, device=device, dtype=c.temp.dtype)
            c.prompt_target_entropy = tt
        else:
            c.prompt_target_entropy = torch.full(shape, float(target_init), device=device, dtype=c.temp.dtype)

        if ema_init_mode == "target":
            c.ema_entropy.copy_(c.prompt_target_entropy)
        else:
            c.ema_entropy.fill_(float(ema_init))
        m._entropy_temp_controller = c


def batched_sessions(examples: Iterable[Dict[str, Any]], session_size: int, max_examples: int):
    buf: List[Dict[str, Any]] = []
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
    ap.add_argument("--session_size", type=int, default=20, help="Number of samples per session.")
    ap.add_argument(
        "--session_init_mode",
        choices=["legacy", "calibrated_scalar", "calibrated_per_head"],
        default="calibrated_scalar",
        help=(
            "legacy: no manual init (first evaluated sample sets per-layer/head target); "
            "calibrated_scalar: K-sample scalar target broadcast; "
            "calibrated_per_head: K-sample per-layer/per-head target aggregation."
        ),
    )
    ap.add_argument(
        "--session_calibration_samples",
        type=int,
        default=3,
        help="Use first K samples in each session to estimate initialization target.",
    )
    ap.add_argument(
        "--session_target_stat",
        choices=["mean", "median"],
        default="median",
        help="Aggregation used over calibration target estimates.",
    )
    ap.add_argument(
        "--session_temp_init",
        type=float,
        default=1.0,
        help="Base initial temperature for each session.",
    )
    ap.add_argument(
        "--session_temp_target_gain",
        type=float,
        default=0.0,
        help="Adjust initial temp using calibration target: temp_init - gain*(target-0.5).",
    )
    ap.add_argument(
        "--session_ema_init_mode",
        choices=["zero", "target"],
        default="target",
        help="How to initialize EMA entropy at session start.",
    )
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
        session_summary_path = os.path.join(task_dir, f"{task}_{args.attn_impl}_session_summary_{args.run_tag}")

        total = 0
        hit_contains = 0
        hit_exact = 0
        session_summaries: List[Dict[str, Any]] = []

        if args.source == "hf":
            examples = task_examples_from_hf(task)
        else:
            examples = task_examples_from_local(task, args.local_root)

        entropy_ctx = open(entropy_path, "w", encoding="utf-8") if args.attn_impl == "entropy_attn" else nullcontext(None)
        with open(pred_path, "w", encoding="utf-8") as out_f, entropy_ctx as ent_f:
            sessions = batched_sessions(examples, max(1, args.session_size), args.max_examples)
            for session_idx, session_examples in enumerate(sessions):
                session_hit_contains = 0
                session_hit_exact = 0
                session_n = 0
                session_target_init = None
                session_temp_init = None
                session_ema_init = None

                if args.attn_impl == "entropy_attn":
                    # Hard reset at every new session boundary.
                    reset_entropy_controller_state(runner.model)
                    reset_entropy_logs(runner.model)

                    if args.session_init_mode != "legacy":
                        # Session calibration using first K prompts, to avoid anchoring to only sample 1.
                        K = max(0, min(args.session_calibration_samples, len(session_examples)))
                        target_vals: List[float] = []
                        n_mods = len(get_attn_modules(runner.model))
                        per_mod_samples: List[List[torch.Tensor]] = [[] for _ in range(n_mods)]

                        for ex_cal in session_examples[:K]:
                            prompt_cal = build_prompt(runner, ex_cal, args.prompt_style)
                            tokenized_cal = runner.tokenizer(prompt_cal, add_special_tokens=False)
                            prompt_ids_cal = tokenized_cal["input_ids"]
                            limit_cal = args.max_input_tokens if args.max_input_tokens > 0 else infer_model_input_limit(runner, args.max_new_tokens)
                            if len(prompt_ids_cal) > limit_cal:
                                if args.overlength_policy == "skip":
                                    continue
                                if args.overlength_policy == "error":
                                    continue
                                kept_cal = truncate_token_ids(
                                    prompt_ids_cal,
                                    limit=limit_cal,
                                    strategy=args.truncate_strategy,
                                    head_keep_ratio=args.head_keep_ratio,
                                )
                                prompt_cal = runner.tokenizer.decode(kept_cal, skip_special_tokens=False)

                            reset_entropy_controller_state(runner.model)
                            run_prefill_probe(runner, prompt_cal)
                            tmean = collect_prompt_target_mean(runner.model)
                            if tmean is not None:
                                target_vals.append(float(tmean))
                            if args.session_init_mode == "calibrated_per_head":
                                per_mod_targets = collect_prompt_targets_by_module(runner.model)
                                for mi, tgt in enumerate(per_mod_targets):
                                    if tgt is not None:
                                        per_mod_samples[mi].append(tgt.detach().clone())

                        if target_vals:
                            if args.session_target_stat == "mean":
                                session_target_init = sum(target_vals) / len(target_vals)
                            else:
                                sv = sorted(target_vals)
                                session_target_init = sv[len(sv) // 2]
                        else:
                            session_target_init = 0.5

                        session_temp_init = float(args.session_temp_init) - float(args.session_temp_target_gain) * (session_target_init - 0.5)
                        session_temp_init = max(0.7, min(1.0, session_temp_init))
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
                                    fallback = torch.full((1, n_heads, 1), float(session_target_init))
                                    per_module_targets.append(fallback)

                        reset_entropy_controller_state(runner.model)
                        initialize_entropy_controller_state(
                            runner.model,
                            temp_init=float(session_temp_init),
                            ema_init=float(session_ema_init),
                            target_init=float(session_target_init),
                            per_module_targets=per_module_targets,
                            ema_init_mode=args.session_ema_init_mode,
                        )

                for ex in session_examples:
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
                                        "_session_idx": session_idx,
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
                    session_hit_contains += c_hit
                    session_hit_exact += e_hit
                    total += 1
                    session_n += 1

                    entropy_logs = collect_entropy_logs(runner.model) if args.attn_impl == "entropy_attn" else None
                    if ent_f is not None and entropy_logs is not None:
                        ex_id = ex.get("id", str(total - 1))
                        ent_f.write(
                            json.dumps(
                                {
                                    "task": task,
                                    "attn_impl": args.attn_impl,
                                    "example_id": ex_id,
                                    "session_idx": session_idx,
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
                        "_session_idx": session_idx,
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

                session_summary = {
                    "task": task,
                    "session_idx": session_idx,
                    "session_init_mode": args.session_init_mode,
                    "session_size": len(session_examples),
                    "evaluated_n": session_n,
                    "contains_acc": round(100.0 * session_hit_contains / max(session_n, 1), 2),
                    "exact_acc": round(100.0 * session_hit_exact / max(session_n, 1), 2),
                }
                if args.attn_impl == "entropy_attn":
                    session_summary["session_target_init"] = float(session_target_init) if session_target_init is not None else None
                    session_summary["session_ema_init"] = float(session_ema_init) if session_ema_init is not None else None
                    session_summary["session_temp_init"] = float(session_temp_init) if session_temp_init is not None else None
                session_summaries.append(session_summary)
                print(
                    f"[{task}] session {session_idx} | n={session_n} | "
                    f"contains={session_summary['contains_acc']:.2f}% | exact={session_summary['exact_acc']:.2f}%"
                )

        with open(session_summary_path, "w", encoding="utf-8") as sf:
            for row in session_summaries:
                sf.write(json.dumps(row, ensure_ascii=False) + "\n")

        task_summary: Dict[str, Any] = {
            "n": total,
            "contains_acc": round(100.0 * hit_contains / max(total, 1), 2),
            "exact_acc": round(100.0 * hit_exact / max(total, 1), 2),
            "prediction_file": pred_path,
            "session_summary_file": session_summary_path,
            "num_sessions": len(session_summaries),
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
