# models/entropy_scaling.py
"""
Drop-in entropy temperature controller with soft calibration target +
instability-gated local control.

Recommended usage:
  1. Copy this file over your repo's models/entropy_scaling.py, or import this
     EntropyTempController from your attention patch.
  2. Keep the class name EntropyTempController so your existing attention patch
     can instantiate it without changes.

Design:
  - The prompt/session target is kept as a SOFT prior.
  - When local entropy dynamics are stable, use a weak target correction.
  - When local entropy becomes unstable, ignore target error and correct using
    fast local instability signals: positive slope, rebound, and local std.

All state is per layer/head, matching your original controller shape [Z, H, 1].
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class EntropyDynamicTempController:
    """
    Soft-target + instability-gated entropy -> attention temperature controller.

    Stable regime:
      Use weak target tracking:
          delta_T = -kp_slow * (EMA(H) - H_target)

    Unstable regime:
      Ignore target error and use fast local regulation:
          delta_T = -kp_fast * instability

    where instability is winner-take-all over normalized local signals:
      - positive entropy slope
      - rebound from recent local minimum
      - local standard deviation

    Intuition:
      Target = session prior. Dynamics = real-time evidence.
      When dynamics are unstable, trust local evidence more than the prior.
    """

    def __init__(
        self,
        temp_init: float = 1.0,
        temp_min: float = 0.7,
        temp_max: float = 1.0,
        ema_beta: float = 0.7,
        kp: Optional[float] = None,
        max_step: float = 5e-4,
        dead_band: Optional[float] = None,
        # New controller knobs. Conservative defaults for RULER/MRCR start.
        kp_slow: float = 0.10,
        kp_fast: float = 0.35,
        instability_threshold: float = 0.015,
        target_deadband: float = 0.020,
        compatibility_margin: float = 0.150,
        rebound_window: int = 4,
        use_slope: bool = True,
        use_rebound: bool = True,
        use_variance: bool = True,
        refractory_steps: int = 0,
        mode: str = "soft_instability",  # "legacy", "soft_instability", "instability_only"
    ):
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)
        self.ema_beta = float(ema_beta)

        # Backward compatibility: if old code passes kp, use it as kp_fast unless
        # explicit kp_fast was provided by the caller defaults above.
        self.kp = float(kp) if kp is not None else float(kp_fast)
        self.kp_slow = float(kp_slow)
        self.kp_fast = float(kp_fast if kp is None else kp)
        self.max_step = float(max_step)
        self.dead_band = dead_band

        self.instability_threshold = float(instability_threshold)
        self.target_deadband = float(target_deadband)
        self.compatibility_margin = float(compatibility_margin)
        self.rebound_window = int(max(2, rebound_window))
        self.use_slope = bool(use_slope)
        self.use_rebound = bool(use_rebound)
        self.use_variance = bool(use_variance)
        self.refractory_steps = int(max(0, refractory_steps))
        self.mode = str(mode)

        self.temp = None                    # [Z, H, 1]
        self.ema_entropy = None             # [Z, H, 1]
        self.prompt_target_entropy = None   # [Z, H, 1]
        self.temp_init = float(temp_init)

        # New local-dynamics state.
        self.prev_entropy = None            # [Z, H, 1]
        self.entropy_history = []           # list[[Z, H, 1]], normalized entropy
        self.cooldown = 0

        # Last-step debug stats; useful for entropy logs.
        self.last_stats: Dict[str, object] = {}

    # ---------- initialization ----------

    def _init_state(self, shape, device):
        self.temp = torch.full(shape, self.temp_init, device=device)
        self.ema_entropy = torch.zeros(shape, device=device)
        self.prev_entropy = None
        self.entropy_history = []
        self.cooldown = 0

    def set_prompt_target(self, target_entropy: torch.Tensor):
        """target_entropy: [Z, H, 1], normalized."""
        self.prompt_target_entropy = target_entropy.detach()

    # ---------- helpers ----------

    def _push_history(self, h: torch.Tensor):
        self.entropy_history.append(h.detach().clone())
        if len(self.entropy_history) > self.rebound_window:
            self.entropy_history = self.entropy_history[-self.rebound_window :]

    def _compute_instability(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros_like(h)

        if self.prev_entropy is None:
            slope = zeros
        else:
            slope = torch.relu(h - self.prev_entropy)

        if len(self.entropy_history) >= 1:
            hist = torch.stack(self.entropy_history, dim=0)  # [W, Z, H, 1]
            local_min = hist.min(dim=0).values
            rebound = torch.relu(h - local_min)
            local_std = hist.std(dim=0, unbiased=False) if hist.shape[0] > 1 else zeros
        else:
            rebound = zeros
            local_std = zeros

        candidates = []
        names = []
        if self.use_slope:
            candidates.append(slope)
            names.append("slope")
        if self.use_rebound:
            candidates.append(rebound)
            names.append("rebound")
        if self.use_variance:
            candidates.append(local_std)
            names.append("variance")
        if not candidates:
            candidates = [zeros]
            names = ["none"]

        stacked = torch.stack(candidates, dim=0)  # [K, Z, H, 1]
        instability, winner_idx = stacked.max(dim=0)
        return {
            "slope": slope,
            "rebound": rebound,
            "variance": local_std,
            "instability": instability,
            "winner_idx": winner_idx,
            "winner_names": names,
        }

    # ---------- update ----------

    @torch.no_grad()
    def update(self, entropy_last: torch.Tensor, kv_len: int):
        """
        entropy_last: [Z, H, 1] raw entropy for the last query token.
        kv_len: current KV cache length.
        """
        if self.temp is None:
            self._init_state(entropy_last.shape, entropy_last.device)

        norm = torch.log(torch.tensor(float(kv_len), device=entropy_last.device)).clamp(min=1.0)
        h_norm = entropy_last / norm
        valid_entropy = torch.isfinite(h_norm)
        h_safe = torch.where(valid_entropy, h_norm, torch.zeros_like(h_norm))

        # Short EMA as local baseline, not as a slow global truth.
        ema_new = self.ema_entropy * self.ema_beta + h_safe * (1.0 - self.ema_beta)
        self.ema_entropy = torch.where(valid_entropy, ema_new, self.ema_entropy)

        # Legacy mode for exact old-controller ablation.
        if self.mode == "legacy":
            if self.prompt_target_entropy is not None:
                valid_target = torch.isfinite(self.prompt_target_entropy)
                valid = valid_entropy & valid_target
                target = torch.where(valid_target, self.prompt_target_entropy, torch.zeros_like(self.prompt_target_entropy))
                err = self.ema_entropy - target
            else:
                valid = valid_entropy
                err = self.ema_entropy
            if self.dead_band is not None:
                err = torch.where(err.abs() < self.dead_band, torch.zeros_like(err), err)
            delta = (-self.kp * err).clamp(-self.max_step, self.max_step)
            delta = torch.where(valid, delta, torch.zeros_like(delta))
            self.temp.add_(delta).clamp_(self.temp_min, self.temp_max)
            self.prev_entropy = h_safe.detach().clone()
            self._push_history(h_safe)
            self.last_stats = {"mode": "legacy", "mean_delta": float(delta.mean().item())}
            return self.temp

        sig = self._compute_instability(h_safe)
        instability = sig["instability"]
        is_unstable = instability > self.instability_threshold

        # Optional cooldown avoids repeated reactions to the same spike.
        if self.cooldown > 0:
            is_unstable = torch.zeros_like(is_unstable, dtype=torch.bool)
            self.cooldown -= 1

        # Fast branch: target is ignored. We regulate directly from local dynamics.
        fast_delta = -self.kp_fast * instability

        # Slow branch: weak soft-target correction, gated for compatibility.
        if self.prompt_target_entropy is not None and self.mode != "instability_only":
            valid_target = torch.isfinite(self.prompt_target_entropy)
            target = torch.where(valid_target, self.prompt_target_entropy, self.ema_entropy)
            target_err = self.ema_entropy - target

            # Deadband: ignore tiny target errors.
            target_err = torch.where(target_err.abs() < self.target_deadband, torch.zeros_like(target_err), target_err)

            # Compatibility gate: if target error is huge, avoid chasing a likely
            # misaligned/unreachable target. This is objective filtering, not just noise filtering.
            compatible = target_err.abs() <= self.compatibility_margin
            slow_delta = -self.kp_slow * target_err
            slow_delta = torch.where(compatible & valid_target, slow_delta, torch.zeros_like(slow_delta))
            valid = valid_entropy & valid_target
        else:
            slow_delta = torch.zeros_like(h_safe)
            target_err = torch.zeros_like(h_safe)
            compatible = torch.ones_like(h_safe, dtype=torch.bool)
            valid = valid_entropy

        # Winner-take-all switching: unstable -> fast dynamics; stable -> slow target prior.
        delta = torch.where(is_unstable, fast_delta, slow_delta)
        delta = delta.clamp(-self.max_step, self.max_step)
        delta = torch.where(valid, delta, torch.zeros_like(delta))

        self.temp.add_(delta)
        self.temp.clamp_(self.temp_min, self.temp_max)

        if bool(is_unstable.any().item()) and self.refractory_steps > 0:
            self.cooldown = self.refractory_steps

        self.prev_entropy = h_safe.detach().clone()
        self._push_history(h_safe)

        # Lightweight debug stats. Avoid storing tensors in logs.
        winner_idx_mean = int(sig["winner_idx"].float().mean().round().item()) if sig["winner_idx"].numel() else 0
        winner_names = sig["winner_names"]
        winner_name = winner_names[max(0, min(winner_idx_mean, len(winner_names) - 1))]
        self.last_stats = {
            "mode": self.mode,
            "mean_entropy_norm": float(h_safe.mean().item()),
            "mean_ema_entropy": float(self.ema_entropy.mean().item()),
            "mean_instability": float(instability.mean().item()),
            "mean_slope": float(sig["slope"].mean().item()),
            "mean_rebound": float(sig["rebound"].mean().item()),
            "mean_variance": float(sig["variance"].mean().item()),
            "unstable_frac": float(is_unstable.float().mean().item()),
            "winner_name_approx": winner_name,
            "mean_target_error": float(target_err.mean().item()) if 'target_err' in locals() else 0.0,
            "compatible_frac": float(compatible.float().mean().item()) if 'compatible' in locals() else 1.0,
            "mean_delta": float(delta.mean().item()),
            "mean_temp": float(self.temp.mean().item()),
        }
        return self.temp
