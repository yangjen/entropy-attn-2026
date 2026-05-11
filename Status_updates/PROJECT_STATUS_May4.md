# Entropy-Attention Controller — Project Status
*Last updated: 2026-05-04*

---

## 1. Probe Analysis → Design Justification

### 1.1 Signal Analysis (probe_analysis.ipynb)

**Key finding: output token entropy is the right detection signal.**

`norm_entropy_trace` = H(vocab distribution) / log(vocab_size), measured per decode token.

| Signal | 16-32k AUC | 32-64k AUC |
|---|---|---|
| -mean_entropy | **0.86** | 0.74 |
| -peak_entropy_early | **0.84** | 0.84 |
| -confused_score | **0.84** | 0.84 |
| -entropy_margin_corr | 0.64 | 0.67 |
| max_margin_early | 0.51 | 0.50 |

- `peak_entropy_early` = max of `norm_entropy_trace[:50]` — most stable across context lengths
- `mean_entropy` = mean over full trace — best at 16-32k, degrades slightly at 32-64k
- `confused_score = peak_entropy / (max_margin + 1e-6)` — numerically identical to peak_entropy_early at greedy decoding (max_margin ≈ 1.0 always), so not an independent signal
- `max_margin_early` = random (AUC ≈ 0.50) — commitment speed does not predict quality

**Attention entropy (attn_entropy_trace) is a weak signal.**
All AUC values near 0.35–0.59 at 16-32k. The aggregate last-layer mean attention entropy does not cleanly separate retrieval success from failure at these context lengths.

### 1.2 Failure Mode Analysis (prob_margin trace)

Both success and failure cases commit to `prob_margin ≈ 1.0` within **10–15 decode tokens** (greedy decoding).

- **Failures**: elevated output token entropy and higher variance in tokens 1–15, then commit to wrong answer confidently
- **Successes**: lower entropy from step 0, commit to correct answer quickly
- **Implication**: the discriminative signal is present from token 0; the intervention window is ≤15 tokens

### 1.3 Entropy Level vs Dynamics (2D scatter plot)

X-axis: `peak_entropy_early` (level signal)
Y-axis: `max_instability` (dynamics signal = H_std_z + H_slope_z + H_rebound_z)

Four quadrants observed:

| Quadrant | Entropy | Instability | Outcome |
|---|---|---|---|
| Confused | High | High | Mostly failures |
| Confident | Low | Low | Mostly successes |
| Exploratory | Low | High | Surprisingly more successes |
| Unstable-focused | High | Low | Mixed |

**Key insight**: high instability alone (exploratory zone) does NOT predict failure. The dangerous regime requires BOTH high entropy AND high instability. This justifies an AND-gate controller design.

### 1.4 Per-Head Attention Entropy (probe_analysis_layer.ipynb)

Qwen3.5-9B uses hybrid attention: only 8/32 layers are full-attention (layers 3,7,11,15,19,23,27,31 — every 4th layer, 4k+3 pattern). Remaining 24 use sliding-window attention and do not log.

At 16-32k, Spearman correlation between per-head attention entropy and mrcr_score:
- Only 1 significant head: L3H6, ρ=0.271, p=0.006
- Signal builds over 50 steps, not present at step 0
- 15/128 heads significant at p<0.05 but with mostly negative ρ ("confident wrong" failure mode at short contexts)

**Interpretation**: attention entropy per head is not yet a reliable controller signal at 16-32k. May improve at 64-128k where the confused failure mode dominates.

---

## 2. Controller Design — Current State

### 2.1 Old Design (no soft gate)
- Custom attention kernel monitors attention entropy at every decode step
- Compares to prefill tail attention entropy target
- Adjusts attention temperature continuously, always reacting
- Single signal, single mechanism, always on
- Weakness: reacts to exploratory behavior as if it were failure; wrong detection signal

### 2.2 New Design (soft gate)

**Architecture:**
```
output token entropy (from logits)
    → fast EMA
    → compare vs prefill tail output token entropy (prompt_target)
    → instability gate (H_std + H_slope + H_rebound of EMA)
    → gate fires when: elevated entropy AND high instability
    → kernel applies attention temperature scaling
```

**Signal-mechanism split:**
- Detection signal: output token entropy (strong predictor, AUC 0.84–0.86)
- Intervention mechanism: attention temperature scaling (the lever, unchanged)
- Kernel becomes a dumb actuator; trigger logic lives in generation loop

**EMA as running mean:**
`mean_entropy` (AUC=0.86) is well-approximated by a running mean/EMA. The current EMA architecture is correct — it just needs to be pointed at output token entropy instead of attention entropy. The prefill tail output token entropy is the natural calibration prior.

**Action window constraint:**
- Commitment happens within 10-15 tokens
- EMA half-life must be short enough to fire by step 5-10
- Recommended: α ≈ 0.5–0.7 (fast EMA, effective window ~3–5 steps)
- Current EMA is likely too slow; smooths away the early spike

**Soft gate condition:**
```
trigger if: ema_entropy > prompt_target + δ_level
        AND instability_score > δ_instability
```
The AND condition prevents false triggers on exploratory high-instability states that tend to resolve correctly.

### 2.3 Session-Based Prior — Empirical Finding and Upgrade Path

**Empirical result from RULER (old controller):**
- Prior calibrated on **first 3 examples of the session** → performance gain
- Prior **reset at every sample** (per-sample prefill tail) → no performance gain
- This contradicts naive expectation that fresher per-sample calibration should be better

**Why per-sample reset fails — the contamination problem:**

For a hard/confused example, the prefill tail entropy is itself elevated — the model is already uncertain at the end of reading the context. Using this as `prompt_target` means the controller compares a confused decode state against a confused baseline: `ema_entropy - prompt_target ≈ 0`, the gate never fires. The controller is blind to exactly the examples it should intervene on.

The session prior from the first 3 examples captures "what normal looks like on this task." Hard examples then deviate upward from that normal, which is what correctly triggers the gate.

**Why a plain running prior is also risky:**

A running mean/EMA that accumulates all session examples gradually absorbs hard examples' elevated prefill entropies, slowly raising the baseline. The controller becomes progressively desensitized — the contamination problem reintroduced more slowly.

**Proposed upgrade — contamination-resistant running prior:**

Option 1: **Selective update** — only update the running prior from examples where the controller did NOT fire (i.e., examples that looked "easy" and represent the clean baseline):
```python
if not controller_fired_this_sample:
    session_prior = alpha * new_prefill_entropy + (1 - alpha) * session_prior
```

Option 2: **Running low percentile** — maintain a running 25th-percentile estimate of past prefill entropies. Robust to upward contamination from hard examples; tracks "what easy looks like" rather than "what average looks like."

Option 3: **Floor-bounded EMA** — running EMA that can only drift downward from the initial calibration, never upward. Adapts to the model becoming more confident over the session but cannot be contaminated by hard examples raising the baseline.

**Key insight for paper**: the session prior works not because it is more accurate per-sample, but because it is *immune to selection bias* — it was set before the model encountered the hard example it needs to handle.

**Ablation relationship across prior variants:**

| Variant | Prior source | What it isolates |
|---|---|---|
| Oracle prior (Option 1) | Large validation set | Upper bound — validates the mechanism itself |
| Selective session prior (Option 2) | Session first-N + selective update | Practical approximation without held-out data |
| Old design | Session first-N, no selective update | Contamination-vulnerable baseline |
| Per-sample reset | Per-sample prefill tail | Failure case — shows why prior stability matters |

These four variants form a complete ablation story: mechanism validation → practical approximation → contamination failure → per-sample failure.

---

### 2.4 Option 1 — Oracle Prior: Experiment Design

#### Concept
Estimate `prompt_target` offline from a large validation set. Compute the expected prefill tail output token entropy across many examples, fix it as a global constant, and apply it to all test examples without any per-sample or session adaptation.

#### What this proves
If Option 1 works, the controller mechanism itself is validated independently of any prior estimation strategy. It answers: *given a good prior, does attention temperature scaling improve retrieval?* This is the right first experiment before investing in Option 2's adaptive machinery.

#### Dataset / Task Candidates

**Validation set (to estimate oracle prior):**
- **RULER QA** — user already has experience, relatively controlled context lengths, short outputs. Good for estimating model's baseline entropy behavior since the user developed v1 controller here.
- **MRCR 4-needle** — easier difficulty, higher pass rate, more "clean baseline" examples. Less contamination risk when computing the mean prefill entropy.
- **NIAH (Needle In A Haystack)** — synthetic, clean, context length fully controllable. Ideal for studying the prior as a function of context length in isolation.

**Test set:**
- **MRCR 8-needle** — harder, more failures, more room for improvement. Natural test bed.
- **LongBench QA subtasks** — more realistic, less synthetic. Tests cross-distribution generalization.

**Strongest experimental design for generalizability claim:**
Estimate oracle prior on RULER → apply controller to MRCR. If it works, the prior captures model-level behavior, not task-specific statistics. This is a cross-task transfer result, which is a substantially stronger claim than within-task validation.

#### Context-Length Dependency

The oracle prior is fundamentally `E[prefill_tail_entropy | context_length]` — it varies with context length because:
1. Longer contexts change the model's state at prefill end (more tokens processed, attention patterns differ)
2. Output token entropy at the prefill tail reflects how uncertain the model is after reading N tokens — this scales with N
3. Normalization by log(vocab_size) helps but does not fully remove context-length dependence

**Options to handle this:**

1. **Stratify by context length bin** (recommended for paper clarity): compute separate oracle priors for 16-32k, 32-64k, 64-128k. Simple, interpretable, directly shows the length-dependence curve.

2. **Fit a regression**: `prior(L) = f(log(L))` — linear or log-linear fit of mean prefill entropy vs context length. Allows interpolation to unseen lengths. Requires enough examples per length.

3. **Relative ratio instead of difference**: use `ema_entropy / prompt_target` rather than `ema_entropy - prompt_target`. Ratio may be more length-invariant if entropy scales proportionally with context length.

The stratified approach is the safest starting point. If the per-bin priors are similar across lengths, pooling is justified and the context-length problem dissolves.

#### Model Choice

**Primary**: Qwen3.5-9B — already instrumented, all probes done, natural first target.

**For generalizability claims**: need at least one other model family to avoid architecture-specific conclusions. Candidates:
- **Llama-3.1-8B** — different architecture, no hybrid attention (all full-attention), widely used baseline
- **Qwen2.5-7B** — same family, different training, tests within-family transferability of the prior

Note: Qwen3.5-9B's hybrid attention (8/32 full-attention layers) makes it architecturally unusual. Results on a fully dense-attention model (Llama) would strengthen the claim that the mechanism is general.

#### Generalizability Concerns and How to Justify

**Concern 1 — Task specificity**: Is "high output entropy early → failure" a property of retrieval tasks only, or of long-context tasks generally?
*Justification*: test on at least two structurally different tasks (needle retrieval + open-domain QA). If the oracle prior from one transfers to the other, the signal is task-agnostic.

**Concern 2 — Model specificity**: The oracle prior is computed on one model's entropy distribution. It may not transfer to other architectures.
*Justification*: repeat the experiment on Llama-3.1-8B with its own oracle prior estimated on the same validation set. If both models show improvement, the mechanism generalizes. The prior itself is model-specific (expected), but the controller architecture is general.

**Concern 3 — Distribution shift between validation and test**: If validation examples are easier than test examples, the oracle prior will be set too low (easy baseline), making the controller overly aggressive.
*Mitigation*: match difficulty distributions between validation and test (same task, same context length bin, random split). Report pass rate on validation set to confirm difficulty match.

**Concern 4 — The "oracle" is not truly oracle**: A prior estimated from 200-300 validation examples is a sample estimate with variance. Report confidence intervals on the prior estimate and show sensitivity: how much does controller performance change if the prior shifts by ±1 std?

#### Recommended Experiment Protocol

```
1. Probe RULER QA: 300 examples, context lengths 16k / 32k / 64k / 128k
   → compute mean prefill_tail_entropy per length bin
   → this is the oracle prior table

2. Apply oracle prior to MRCR 8-needle test set (separate from probe data)
   → run: no controller | per-sample | session (option 2) | oracle (option 1)
   → metrics: accuracy, mean mrcr_score, score distribution

3. Cross-architecture: repeat step 2 with Llama-3.1-8B
   → estimate its own oracle prior on RULER
   → apply to MRCR (same test set, different model)

4. Cross-task transfer ablation:
   → use RULER oracle prior on MRCR test without re-estimating
   → if performance holds: prior is task-agnostic at model level
```

---

## 3. Experiment Plan — Next Steps

### Immediate (waiting on data)
- [ ] 64-128k probe is running (`probe_attn_8needle_64-128k.jsonl`)
- [ ] When done: update `PROBE_PATH` in `probe_analysis.ipynb`, run all cells
- [ ] Key things to check at 64-128k:
  - Does `peak_entropy_early` AUC stay ~0.84 or improve?
  - Does prob_margin commitment window widen? (longer confusion period = more intervention time)
  - Does the confused zone in the 2D scatter grow relative to confident/exploratory?

### Controller experiments (after 64-128k analysis)
1. **Baseline**: plain sdpa, no controller — MRCR 8-needle, 32-64k and 64-128k
2. **Old controller** (no soft gate, attention entropy trigger): same conditions
3. **New controller** (soft gate, output token entropy trigger): same conditions
4. Compare: accuracy, mrcr_score mean, score distribution

### Per-head analysis follow-up
- Run `probe_layer_8needle_64-128k.jsonl` with `--attn_impl entropy_attn_layer`
- Check if more heads become significant at longer contexts
- If yes: apply temperature scaling selectively to retrieval heads only (more surgical)

---

## 4. What to Expect / Risks / Alternatives

### What to expect
- Signal should strengthen at 64-128k (failure mode shifts more toward "confused")
- The soft gate controller should improve over baseline more at longer contexts
- Old controller may show modest improvement or noise at short contexts, stronger at long

### Risks

**1. Wrong failure mode (most critical)**
If 64-128k still shows "confident wrong" failures (prob_margin commits fast regardless), the controller has no valid intervention window. The prob_margin trace for 64-128k is the diagnostic.

**2. Causal gap**
Probe establishes correlation: high output entropy predicts failure. It does NOT prove that adjusting attention temperature reduces output entropy or improves retrieval. This causal link is assumed and must be validated empirically.

**3. Gate fires too late**
The instability gate needs a few steps of history (window=16 for H_std/slope). If commitment is at token 10-15, the gate may fire after the model has already locked onto the wrong answer. Mitigation: use step-0 entropy directly as a fast pre-gate, with instability as a confirmation.

**4. Prefill target mismatch**
At very long contexts, output token entropy at prefill tail may differ structurally from decode entropy. May need context-length-stratified calibration of `prompt_target`.

**5. Regression on easy cases**
Temperature scaling on examples the model would have gotten right could hurt performance. The soft gate is the defense; threshold tuning matters.

### Alternatives if controller does not improve over baseline
- Apply temperature scaling only to identified retrieval heads (L3H6 and others from layer probe) rather than all attention heads
- Use output entropy as a reranking/sampling signal rather than an in-generation controller
- Explore prompt-side interventions (position encoding adjustments) for long-context retrieval

---

## 5. Insights for Paper Writing

**1. Output token entropy as retrieval difficulty proxy**
The model's uncertainty about what to generate next (vocab distribution entropy) in the first 50 decode tokens is a strong proxy for whether long-context retrieval will succeed (AUC 0.84–0.86). This is more predictive than attention entropy (AUC ~0.50), and is available without custom kernels.

**2. Confused vs exploratory — a 2D failure taxonomy**
High instability alone (the exploratory quadrant) correlates with success, not failure. The failure-predictive regime requires high entropy AND high instability simultaneously. Single-signal controllers that react to instability alone will generate false positives. The 2D scatter provides the visual justification for the AND-gate design.

**3. Context-length dependent failure mode transition**
At 16-32k: failures are "confident wrong" — model commits fast to wrong answer, low entropy on failures, max_margin_early ≈ 1.0. At 32-64k: failures become "confused" — sustained high output entropy throughout generation. This transition makes entropy-based intervention more effective at longer contexts, which is where it matters most.

**4. Signal-mechanism decoupling principle**
The detection signal (output token entropy) and the intervention mechanism (attention temperature) need not be the same modality. A controller that detects confusion via output distribution uncertainty and corrects it via attention sharpening is principled because attention determines what information the model uses to generate output.

**5. Prefill tail as a calibration-free prior**
Using the prefill tail output token entropy as the reference baseline (`prompt_target`) requires no task-specific calibration — it's computed on the fly from the model's own state at the moment generation begins. The relative signal `ema_entropy - prompt_target` is context-length agnostic to a first approximation.

**6. EMA ≈ running mean ≈ mean_entropy signal**
`mean_entropy` (AUC=0.86) is the best post-hoc signal and EMA is its real-time approximation. This means the controller architecture (EMA vs prefill target) is correctly matched to the most predictive signal, not just a heuristic choice. The key hyperparameter is α — it controls how much of the "mean" vs "recent peak" behavior is captured.

**8. Session prior vs per-sample prior — selection bias immunity**
A per-sample prior (prefill tail of the current example) fails because hard examples have elevated prefill entropy, making the relative signal `ema_entropy - prompt_target` near zero precisely when intervention is most needed. A session prior calibrated on easy examples is immune to this selection bias. This is an instance of a general principle: the reference baseline must be drawn from the *unconfused* distribution, not the distribution of all examples including the ones you are trying to detect.

**7. Hybrid attention architecture note**
Qwen3.5-9B uses hybrid attention (8/32 full-attention layers, 24 sliding-window). Analysis of per-head entropy must account for this — 75% of layers produce no usable signal for global retrieval. Any attention-based signal that aggregates across all layers will be diluted. Layer-selective analysis is necessary.

---

## File Index

| File | Purpose |
|---|---|
| `mrcr_baseline_probe.py` | Generates probe JSONL with entropy/token traces |
| `models/attn_patch.py` | Attention kernel: last-layer entropy logging + temp scaling |
| `models/attn_patch_layer.py` | Attention kernel: per-head per-layer entropy logging |
| `attention_qwen.py` | QwenRunner: model loading + attn impl registration |
| `mrcr_qwen35_session_tuning.py` | Controller + helper functions (mark/reset/collect logs) |
| `logs/probe_analysis.ipynb` | Output token entropy signal analysis (main analysis notebook) |
| `logs/probe_analysis_layer.ipynb` | Per-head attention entropy Spearman heatmaps |

| Data file | Contents |
|---|---|
| `MRCR_outputs/probe_attn_8needle_16-32k.jsonl` | 100 samples, entropy_attn impl, 16-32k ctx |
| `MRCR_outputs/probe_attn_8needle_32-64k.jsonl` | ~95 samples, entropy_attn impl, 32-64k ctx |
| `MRCR_outputs/probe_attn_8needle_64-128k.jsonl` | In progress |
| `MRCR_outputs/probe_layer_8needle_16-32k.jsonl` | 100 samples, per-head layer logging, 16-32k ctx |
