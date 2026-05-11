# attention_qwen.py
"""Generic Qwen runner for entropy-attention experiments.

This is a conservative refactor of your LlamaRunner. It keeps the same public
shape but adds Qwen/Qwen3-style thinking-mode chat templating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from models.attn_patch import entropy_attention_forward  # required for entropy_attn


def _register_entropy_attn_layer():
    """Register entropy_attn_layer: same as entropy_attn but logs per-head entropy."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from models.attn_patch_layer import entropy_attention_forward as _forward_layer

    state = {"printed": False}

    def wrapped(*args, **kwargs):
        if not state["printed"]:
            state["printed"] = True
            print("[VERIFY] entropy_attention_forward_layer CALLED")
        return _forward_layer(*args, **kwargs)

    if hasattr(ALL_ATTENTION_FUNCTIONS, "register"):
        ALL_ATTENTION_FUNCTIONS.register("entropy_attn_layer", wrapped)
    else:
        ALL_ATTENTION_FUNCTIONS["entropy_attn_layer"] = wrapped
    print("[VERIFY] Registered entropy_attn_layer attention impl")
    return wrapped


def _register_entropy_attn():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    state = {"printed": False}

    def wrapped(*args, **kwargs):
        if not state["printed"]:
            state["printed"] = True
            print("[VERIFY] entropy_attention_forward CALLED")
        return entropy_attention_forward(*args, **kwargs)

    if hasattr(ALL_ATTENTION_FUNCTIONS, "register"):
        ALL_ATTENTION_FUNCTIONS.register("entropy_attn", wrapped)
    else:
        ALL_ATTENTION_FUNCTIONS["entropy_attn"] = wrapped
    print("[VERIFY] Registered entropy_attn attention impl")
    return wrapped


@dataclass
class QwenRunner:
    model_name: str = "Qwen/Qwen3.5-9B"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    attn_impl: str = "sdpa"  # sdpa, flash_attention_2, eager, entropy_attn
    deterministic: bool = True
    enable_thinking: bool = True
    trust_remote_code: bool = True

    def __post_init__(self):
        if self.deterministic:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        if self.attn_impl == "entropy_attn":
            _register_entropy_attn()
        elif self.attn_impl == "entropy_attn_layer":
            _register_entropy_attn_layer()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        config.attn_implementation = self.attn_impl
        setattr(config, "_attn_implementation", self.attn_impl)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            torch_dtype=self.dtype,
            device_map=None,
            trust_remote_code=self.trust_remote_code,
        ).to(self.device)
        self.model.eval()

        print(
            f"[QwenRunner] model={self.model_name} attn_impl="
            f"{getattr(self.model.config, '_attn_implementation', None)} / "
            f"{getattr(self.model.config, 'attn_implementation', None)} "
            f"dtype={next(self.model.parameters()).dtype} thinking={self.enable_thinking}"
        )

    def messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                # Some templates do not accept enable_thinking. Fall back cleanly.
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        # Fallback if no chat template is available.
        return "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages]) + "\nassistant:"

    @torch.inference_mode()
    def prefill_prompt(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=True,
            return_dict=True,
        )

    @torch.inference_mode()
    def generate_one(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        stop_on_newline: bool = False,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if top_k is not None:
                gen_kwargs["top_k"] = top_k

        out = self.model.generate(**gen_kwargs)
        gen_ids = out[0, inputs["input_ids"].shape[-1] :]
        if gen_ids.numel() == 0:
            return ""
        # Strip Qwen3 thinking block at the token level before decoding.
        # <think>/<think> are special tokens so skip_special_tokens=True removes
        # them, leaving raw thinking text with no delimiter to search for.
        if self.enable_thinking:
            think_end_id = self.tokenizer.convert_tokens_to_ids("</think>")
            ids_list = gen_ids.tolist()
            if think_end_id in ids_list:
                gen_ids = gen_ids[ids_list.index(think_end_id) + 1:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # For non-thinking Qwen2.5 models, </think> is plain text and skip_special_tokens
        # doesn't remove it. Strip it and everything after (model echoing think-block pattern).
        if not self.enable_thinking and "</think>" in text:
            text = text.split("</think>")[0].strip()
        if stop_on_newline:
            text = text.splitlines()[0].strip() if text else ""
        return text
