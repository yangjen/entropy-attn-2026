# models/attn_patch_layer.py
"""
Per-layer, per-head entropy logging variant of attn_patch.py.

Identical to attn_patch.py except the decode-time log entry adds:
  "entropy_per_head": [H floats]   — raw entropy per query head at this step

and respects an optional max_log_steps attribute on the module to cap
how many decode steps are logged per layer (keeps output size manageable).

Use with mark_all_layers_entropy_logger() + collect_all_layer_entropy_logs()
from mrcr_qwen35_session_tuning.py.

To use: register as "entropy_attn_layer" via attention_qwen._register_entropy_attn_layer().
Original attn_patch.py and the "entropy_attn" impl are untouched.
"""

import torch
from typing import Optional
from models.entropy_attn_triton import attention as entropy_attention
from transformers.utils import logging
from models.entropy_scaling import EntropyTempController

logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def entropy_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`entropy` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    logger.warning_once(f"WARNING: entropy attention backward and custom attention masking across the batch is not implemented at this time")

    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    Z, H, N_CTX, D = query.size()

    # ---------- entropy-conditioned temperature (single controller) ----------
    if not hasattr(module, "_entropy_temp_controller"):
        max_step = getattr(module, "temp_max_step", 0.0005)
        dead_band = getattr(module, "dead_band", None)
        module._entropy_temp_controller = EntropyTempController(
            temp_init=1.0,
            temp_min=0.7,
            temp_max=1.0,
            ema_beta=0.9,
            kp=0.35,
            max_step=max_step,
            dead_band=dead_band,
        )

    controller = module._entropy_temp_controller

    expected_shape = (Z, H, 1)
    if controller.temp is None or tuple(controller.temp.shape) != expected_shape:
        controller._init_state((Z, H, 1), query.device)
        if (controller.prompt_target_entropy is not None
                and tuple(controller.prompt_target_entropy.shape) != expected_shape):
            controller.prompt_target_entropy = None

    temp = controller.temp.expand(Z, H, N_CTX)

    # ---------- attention ----------
    attn_output, attn_entropy = entropy_attention(
        query, key, value,
        is_causal, scaling,
        temp
    )

    # ---------- prompt entropy reference (prefill only) ----------
    if N_CTX > 1 and controller.prompt_target_entropy is None:
        kv_len = key.shape[2]

        H_norm = attn_entropy / torch.log(
            torch.tensor(float(kv_len), device=attn_entropy.device)
        ).clamp(min=1.0)

        K = min(getattr(module, "calibration_tail_k", 256), H_norm.shape[-1])
        tail = H_norm[:, :, -K:]

        trim_ratio = float(getattr(module, "target_trim_ratio", 0.0))
        trim_ratio = max(0.0, min(0.49, trim_ratio))
        trim_n = int(K * trim_ratio)
        if trim_n > 0 and (2 * trim_n) < K:
            tail_sorted = torch.sort(tail, dim=-1).values
            tail_core = tail_sorted[:, :, trim_n:(K - trim_n)]
            prompt_target = tail_core.mean(dim=-1, keepdim=True)
        else:
            prompt_target = tail.mean(dim=-1, keepdim=True)
        controller.set_prompt_target(prompt_target)

    # ---------- decode-time entropy feedback ----------
    if N_CTX == 1:
        entropy_last = attn_entropy[:, :, -1:].detach()  # [Z, H, 1]
        kv_len = key.shape[2]

        controller.update(entropy_last, kv_len)

        module.past_entropy = entropy_last
        module.past_temp = controller.temp.detach()

        # Per-layer, per-head logging (enabled when is_entropy_log_layer is set).
        if getattr(module, "is_entropy_log_layer", False):
            if not hasattr(module, "_entropy_log"):
                module._entropy_log = []
                module._decode_step = 0

            max_log_steps = getattr(module, "max_log_steps", 0)
            if max_log_steps <= 0 or module._decode_step < max_log_steps:
                # entropy_last shape: [Z=1, H, 1] — extract [H] floats
                per_head = entropy_last[0, :, 0].tolist()
                module._entropy_log.append(
                    {
                        "step": int(module._decode_step),
                        "entropy_mean": float(entropy_last.mean().item()),
                        "entropy_std": float(entropy_last.std().item()),
                        "entropy_per_head": per_head,   # [H] raw (not yet / log(kv_len))
                        "temp_mean": float(controller.temp.mean().item()),
                        "kv_len": int(kv_len),
                    }
                )
            module._decode_step += 1

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
