"""
Quick side-by-side comparison of two models on a few RULER qa_1 examples.

Usage:
  python compare_models_qa.py \
    --model_a meta-llama/Llama-3.1-8B-Instruct \
    --model_b gradientai/Llama-3-8B-Instruct-262k \
    --data_path /c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/131072/data/qa_1/validation.jsonl \
    --n 5 \
    --max_new_tokens 32 \
    --skip 0
"""

import argparse
import json
import textwrap

import torch
from attention_llama import LlamaRunner


def load_examples(path: str, n: int, skip: int):
    examples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            line = line.strip()
            if line:
                examples.append(json.loads(line))
            if len(examples) >= n:
                break
    return examples


def build_prompt(ex: dict, append_prefix: bool) -> str:
    prompt = ex.get("input", "") or ""
    if append_prefix:
        prefix = ex.get("answer_prefix", "") or ""
        if prefix and not prompt.endswith(prefix):
            prompt = prompt + prefix
    return prompt


def run(runner: LlamaRunner, prompt: str, max_new_tokens: int) -> str:
    inputs = runner.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(runner.model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = runner.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return runner.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--model_b", default="gradientai/Llama-3-8B-Instruct-262k")
    ap.add_argument(
        "--data_path",
        default="/c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic/131072/data/qa_1/validation.jsonl",
    )
    ap.add_argument("--n", type=int, default=5, help="Number of examples to check")
    ap.add_argument("--skip", type=int, default=0, help="Skip first N examples")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--no_prefix", action="store_true", help="Don't append answer_prefix")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--prompt_tail_chars", type=int, default=300,
                    help="How many chars of the prompt tail to display")
    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    append_prefix = not args.no_prefix

    print(f"Loading model A: {args.model_a}")
    runner_a = LlamaRunner(args.model_a, attn_impl="sdpa", dtype=dtype_map[args.dtype])
    print(f"Loading model B: {args.model_b}")
    runner_b = LlamaRunner(args.model_b, attn_impl="sdpa", dtype=dtype_map[args.dtype])

    examples = load_examples(args.data_path, args.n, args.skip)
    print(f"\nLoaded {len(examples)} examples (skip={args.skip})\n")
    print("=" * 80)

    for i, ex in enumerate(examples):
        prompt = build_prompt(ex, append_prefix)
        n_tokens_a = len(runner_a.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        expected = ex.get("outputs", [])

        pred_a = run(runner_a, prompt, args.max_new_tokens)
        pred_b = run(runner_b, prompt, args.max_new_tokens)

        hit_a = any((e or "").lower() in pred_a.lower() for e in expected)
        hit_b = any((e or "").lower() in pred_b.lower() for e in expected)

        print(f"[{i}] tokens={n_tokens_a}  expected={expected}")
        print(f"  ...{repr(prompt[-args.prompt_tail_chars:])}")
        print(f"  A ({args.model_a.split('/')[-1]}): {repr(pred_a)}  {'HIT' if hit_a else 'MISS'}")
        print(f"  B ({args.model_b.split('/')[-1]}): {repr(pred_b)}  {'HIT' if hit_b else 'MISS'}")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()
