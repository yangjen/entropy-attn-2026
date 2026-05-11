# Project Status — May 6 2026
Entropy-Attention Controller for Long-Context Retrieval

---

## 1. Oracle Prior Analysis: qa_1 vs qa_2 (LLaMA 3.1-8B)

### 1.1 Per-bin results (200 samples/bin each)

| bin    | qa_1 mean | qa_2 mean | diff   | qa_1 std | qa_2 std |
|--------|-----------|-----------|--------|----------|----------|
| 4096   | ~0.342    | ~0.316    | +0.026 | ~0.016   | ~0.009   |
| 8192   | ~0.330    | ~0.305    | +0.025 | ~0.012   | ~0.008   |
| 16384  | ~0.315    | ~0.295    | +0.020 | ~0.010   | ~0.007   |
| 32768  | ~0.300    | ~0.280    | +0.020 | ~0.009   | ~0.006   |
| 65536  | ~0.283    | ~0.269    | +0.014 | ~0.007   | ~0.005   |
| 131072 | ~0.350    | ~0.341    | +0.009 | ~0.016   | ~0.009   |

Gap is statistically real (SEM ≈ 0.001; gap is 15–25× SEM). Monotonic decrease with
context length in both tasks — except the 131k anomaly (see §3).

### 1.2 Prior design decision: context-length-only prior

Per-task priors require task identity at inference time, which is unavailable on InfiniteBench
and other held-out sets. The P-controller is graded and absorbs the 0.015–0.025 inter-task
gap without breaking. Paper claim: context-length-only prior (averaged across tasks per bin).
Per-task breakdown goes in a diagnostic table motivating future online estimation.

---

## 2. Controller Asymmetry: One-Directional Sharpener

Confirmed by re-reading `entropy_scaling.py`:

```python
temp_min = 0.7,  temp_max = 1.0,  temp_init = 1.0
err   = ema_entropy - target   # decode entropy minus oracle prior target
delta = -kp * err              # positive delta would raise temp above 1.0
temp += delta, then clamp to [temp_min, temp_max]
```

`temp_init = temp_max = 1.0`. Any delta that would raise temperature above 1.0 is killed
by the clamp. **The controller can only lower temperature (sharpen). It never broadens.**

### Why qa_2 gets more gain

qa_2 has a lower oracle prior target (~0.316) than qa_1 (~0.342). During decoding, attention
entropy drifts toward a similar natural level for both tasks. Therefore:

```
err_qa2 = decode_entropy − 0.316  >  err_qa1 = decode_entropy − 0.342
```

Larger error → more aggressive sharpening → more gain. The harder task benefits more because
its tighter target demands more correction, not because the controller ever broadens attention.

---

## 3. 131k Underperformance: Prior Miscalibration Diagnosis

### Observation
Old always-fire controller underperformed at 131k on RULER QA, despite baseline accuracy
being comparable to the paper's results. Model capability was not the problem. Gain was
restored when switching to LLaMA 3.1-8B YaRN-256k.

### Diagnosis
LLaMA 3.1-8B is trained to 128k. At 131072 tokens, position encodings are past their
reliable range → attention diffuses spuriously at prefill → oracle prior measured at 131k
is inflated (~0.350 vs the expected ~0.268 from the smooth trend).

Two failure modes interact at 131k:

**Failure 1 — stalling (most decode steps):**
Session-calibrated target is inflated (~0.350). Decode attention entropy typically sits
below this inflated target → `err < 0` → delta > 0 → clamped at `temp_max = 1.0` →
controller frozen at 1.0. This matches baseline, not regression.

**Failure 2 — harmful misfires (some decode steps):**
RoPE boundary effects corrupt not just the prefill but also all decode-step attention
measurements (the KV cache still contains the 131k-token context with broken position
encodings). On steps where the broken decode attention entropy spikes above the
already-inflated target, the controller does fire — but based on two stacked artifacts
(noisy signal + wrong target). The resulting correction is miscalibrated in both
direction and magnitude, occasionally pushing performance below baseline.

Net effect: mostly stalls at baseline temp=1.0 (no gain), occasionally misfires
(slight regression). Together this explains underperformance vs baseline.

**YaRN fixes both simultaneously:** clean prefill measurement gives a sensible target,
and clean decode-step measurements give reliable signals throughout generation.

**The regression is a signal quality artifact, not a model capability ceiling.**

### Fix: targeted YaRN ablation (not a full rerun)

- Run oracle prior estimation at 131k: vanilla LLaMA 3.1-8B vs YaRN-256k
- Expected: YaRN 131k prior follows the monotonic trend (~0.270 extrapolated from 65536 = 0.283) instead of jumping to 0.350
- Re-run RULER QA at 131k only with YaRN → confirm gain is restored

Paper framing: "the 131k anomaly is a RoPE extrapolation artifact in the oracle prior; using
a context-window-extended model (YaRN) restores the monotonic trend and the controller gain."
Note: the scaling law approach (§4) offers an alternative fix without requiring a different model.

---

## 4. New Direction: Log-Linear Scaling Law for Oracle Prior

### 4.1 The finding

Attention entropy (normalized, trimmed-mean over last 256 prefill tokens) follows a
log-linear relationship with context length:

```
target ≈ a − b · log2(ctx_len)
```

From the qa_1 data (log2 space):

| log2(ctx_len) | measured | predicted |
|---|---|---|
| 12 (4k)  | 0.342 | — |
| 13 (8k)  | 0.330 | — |
| 14 (16k) | 0.315 | — |
| 15 (32k) | 0.300 | — |
| 16 (65k) | 0.283 | — |
| 17 (131k)| 0.350 | **~0.268** ← outlier, predicted from trend |

Slope b ≈ −0.015 per doubling, highly consistent from 4k to 65k.
The 131k point (0.350) is a clear outlier — detectable as > 2σ from the fit residual.
qa_2 shows the same slope, with a lower intercept a (~0.020 lower). **Slope is
task-invariant; only the intercept shifts by task type.**

### 4.2 What this enables

**Cheap calibration protocol:**
- Run ~50 samples at 4k and 8k only (cheap, fast, RoPE-safe)
- Fit (a, b) from these two points
- Predict oracle prior at 32k, 65k, 131k analytically — no expensive long-context runs
- Reduces calibration cost by ~30× vs full per-bin sampling

**Outlier detection:**
- Fit line on trusted bins (4k–65k); flag any bin with residual > 2σ
- 131k anomaly auto-detected and replaced with extrapolated value
- Cleaner than requiring a different model (YaRN) just for one bin

**Reduced prior representation:**
- Prior goes from a lookup table O(bins) to 2 scalars (a, b) per model
- If slope b is universal across architectures: 1-point calibration suffices (estimate a
  from a handful of short-context samples, borrow b)

### 4.3 LLaMA scaling law validation results (May 7)

Hold-out validation: fit on 4k+8k only, predict 16k/32k/65k.

| series | 16k error/std | 32k error/std | 65k error/std |
|---|---|---|---|
| LLaMA qa_1 | −0.89σ ✓ | −0.10σ ✓ | +1.58σ ✓ |
| LLaMA qa_2 | −0.04σ ✓ | +1.26σ ✓ | **+4.13σ ✗** |

Fitted slopes: qa_1 b=0.0154, qa_2 b=0.0121 (24% apart, R²≈0.93 both).
131k blended extrapolated target: **0.256** (vs measured anomalous 0.345 — a 0.089 correction).

**Key results:**
- 16k and 32k predictions from 4k+8k alone are reliable (all within 1.3σ)
- 65k prediction from 4k+8k alone fails for qa_2 (4.1σ) due to deceleration — the curve
  flattens faster than the linear extrapolation expects
- Slopes differ by 24% across tasks → "task-invariant slope" claim too strong; softer
  claim: "approximately parallel slopes within the same model"
- **Revised cheap calibration:** 3 bins needed (4k + 8k + one mid-range, e.g. 32k) for
  reliable extrapolation to 65k. 2-bin calibration (4k+8k) suffices only for ≤32k.

### 4.4 Qwen3.5-2B oracle prior results (May 7)

Per-bin attention entropy (200 samples/bin, Qwen-native RULER data with correct template):

| bin | Qwen qa_1 | Qwen qa_2 | LLaMA qa_1 | LLaMA qa_2 |
|---|---|---|---|---|
| 4096 | 0.502 | 0.464 | 0.342 | 0.316 |
| 8192 | 0.484 | 0.445 | 0.324 | 0.298 |
| 16384 | 0.462 | 0.440 | 0.298 | 0.281 |
| 32768 | 0.463 | 0.446 | 0.287 | 0.271 |
| 65536 | 0.465 | 0.452 | 0.283 | 0.269 |
| 131072 | 0.479 | 0.471 | 0.350* | 0.341* |

*LLaMA 131k = RoPE anomaly.

**Qwen pattern is fundamentally different from LLaMA:**

Z-score analysis per transition (positive = decreasing entropy):

| transition | Qwen qa_1 | Qwen qa_2 |
|---|---|---|
| 4k→8k | +8.2σ ✓ | +13.9σ ✓ |
| 8k→16k | +9.9σ ✓ | +4.2σ ✓ |
| 16k→32k | −0.7σ (noise) | −4.7σ ✗ reversal |
| 32k→65k | −1.4σ (noise) | −5.5σ ✗ reversal |

- Qwen qa_1: genuine **plateau** from 16k (transitions are noise)
- Qwen qa_2: **U-shaped** — decreases to 16k, then entropy *increases* significantly at 32k
  and 65k (small model struggling to maintain focused attention on multi-hop targets at
  long context — attention diffuses again)
- Both models show absolute values ~0.15–0.17 higher than LLaMA (Qwen attends more diffusely
  overall — 2B vs 8B parameter count, different architecture)

**Implication for scaling law claim:**

| Model | Behavior | Log-linear valid range |
|---|---|---|
| LLaMA 3.1-8B | Decelerating monotone decrease | 4k–65k (R²≈0.93) |
| Qwen3.5-2B qa_1 | Decrease then flat plateau | 4k–16k only |
| Qwen3.5-2B qa_2 | Decrease then U-shaped reversal | 4k–16k only |

The slope is **not universal across architectures**. Log-linear is a model-specific
approximation. Revised paper framing: "the oracle prior follows an approximately log-linear
trend within the model's reliable context range; the trend and saturation point are
model-dependent." Qwen3.5-2B saturates at 16k; LLaMA 3.1-8B continues decreasing to 65k+.

Note: invalid Qwen oracle prior file (`oracle_prior/qwen35_2b_ruler/`) was generated using
LLaMA-templated RULER data — discarded. Results above use Qwen-native RULER data
(`oracle_prior/qwen35_2b_ruler_native/`).

### 4.5 Paper framing (updated)

Sits as a supporting finding in the oracle prior section. Primary claim weakened to:
"the oracle prior decreases with context length in a predictable pattern within a model;
we fit this trend to reduce calibration cost and detect RoPE anomalies as outliers."
Universal slope claim dropped. Per-model calibration is required.

One figure: oracle prior vs log2(ctx_len), LLaMA qa_1/qa_2 with log-linear fit, 131k
outlier circled. Qwen shown in supplement as contrast case (plateau/reversal pattern).

---

## 5. Open Items

- [x] Hold-out validation: fit on 4k+8k, predict 32k/65k — done (see §4.3); 65k qa_2 fails at 4.1σ
- [x] Run Qwen3.5-2B oracle prior on Qwen-native RULER data — done (see §4.4); slope does not match LLaMA
- [ ] Run LLaMA YaRN-262k oracle prior (command ready; model local at gradientai snapshot)
- [ ] Run RULER entropy probe with Qwen3.5-2B on NIAH tasks
- [ ] Implement gated controller (§3.3): output entropy EMA gate + oracle prior JSON lookup
- [ ] MRCR ablation (§3.4): 4 conditions at 32–64k and 64–128k
- [ ] Threshold sweep (§3.5): θ ∈ {0.03, 0.04, 0.05, 0.06, 0.08}
- [ ] InfiniteBench evaluation (held-out generalization)
- [ ] Optional: targeted 131k YaRN ablation to confirm prior miscalibration diagnosis


## The two viable paper framings:

Option A — "Calibration-required method" (honest, defensible):
Frame it like temperature scaling or adapter methods. The contribution is the controller architecture and the calibration procedure. Calibration cost: 50 unlabeled prefill passes (no labels, no generation) + a small threshold sweep on a held-out set. Comparable to what most calibration-based inference papers require. Paper argues the calibration is cheap, label-free, and generalizes across tasks within a model.

Option B — "The pattern is the finding" (stronger, needs more validation):
The paper's primary claim shifts from "here is a working controller" to "attention entropy decreases predictably with context length, and controllers that exploit this outperform those that don't." The pattern is the contribution; the specific numbers are secondary. This requires showing the pattern holds across at least 3+ settings (LLaMA, Qwen, ideally one more) and that a controller using even a rough version of the pattern beats a controller that ignores it.

Option B is stronger scientifically but needs the YaRN oracle prior result and ideally one more model to be convincing. Option A is submittable now.