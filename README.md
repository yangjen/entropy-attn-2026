## Overall  
This repo adds an “entropy-aware” attention kernel (Triton) for Llama-style models, plus hooks to register it in Hugging Face Transformers and scripts/tests to validate correctness and run RULER evaluations.
## entropy_attn_triton.py
Implements the custom Triton attention forward that computes both attention output and per-token attention entropy, with autotuning and GPU-specific paths.
## attn_patch.py
Bridges HF attention calls to the Triton kernel, repeats KV heads, manages causal behavior, and adds an entropy‑conditioned temperature controller (per-layer/per-head) that can update temperature during decode and optionally log entropy/temperature.
## entropy_scaling.py
Defines the EntropyTempController that keeps decode-time entropy near a prompt-derived target (EMA + bounded proportional control).
## attention_llama.py
Utility to register entropy_attn into Transformers’ attention registry and run a simple Llama runner for generation.
## test_kernel.py
Numerical correctness checks for the Triton kernel vs. PyTorch reference in prefill and decode.
## test_llama.py 
End-to-end passkey retrieval test to compare attention implementations.
## run_ruler_eval_timed.py
RULER eval runner with timing/metrics aggregation and optional entropy logging.

## To run tests on the kernel

`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python test_kernel.py`

## To run tests on the end to end model with the kernel

`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python test_llama.py --attn-impl flash_attention_2`
`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python test_llama.py --attn-impl entropy_attn`

## To run on RULER with baseline flash_attention_2
`TRITON_DISABLE_AUTOTUNE=1 CUDA_VISIBLE_DEVICES=[device_num] python run_ruler_eval_timed.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data_root [data_path] \
  --tasks [qa_task] \
  --max_new_tokens 64 \
  --compact \
  --log_every 10 \
  --eval_mode ruler_part \
  --attn_impl flash_attention_2 \
  --dtype bf16 \
  --deterministic \
  --time \
  --time_skip 4`

## To run on RULER with entropy_attn kernel
`TRITON_DISABLE_AUTOTUNE=1 CUDA_VISIBLE_DEVICES=[device_num] python run_ruler_eval_timed.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data_root [data_path] \
  --tasks [qa_task] \
  --max_new_tokens 64 \
  --compact \
  --log_every 10 \
  --eval_mode ruler_part \
  --attn_impl entropy_attn \
  --dtype bf16 \
  --deterministic \
  --time \
  --time_skip 4`

