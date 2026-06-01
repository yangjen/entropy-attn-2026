# Project Status — May 18, 2026

## Summary of session

This session focused on deep analysis of the CWE 32k results, interpreting the analysis
notebook plots, and a broad reassessment of the project hypothesis and research direction.
No new experiments were run; this is a synthesis and direction-setting session.

---

## 1. CWE Analysis Notebook (`Analysis/analyze_cwe_entropy.ipynb`)

### 1.1 New cells added

Two new analysis cells were added to the working notebook:

**Analysis 1b** (after cell `42686f7d`): Entropy distribution split by `ruler_all_hit` score
groups (miss: score=0, partial: 0<score<0.5, good: score≥0.5) rather than binary hit/miss.
Shows whether entropy gap scales with how many of the 10 words were retrieved, not just
whether any were found.

**Analysis 2b** (after cell `81130bd6`): Controller tracking error and temperature trajectory
by the same three score groups. Tests whether tighter entropy tracking correlates with
higher retrieval scores within the hit population.

### 1.2 Bug fix: FWE guard in Analysis 2 and 2.5

Analysis 2 (`81130bd6`) and Analysis 2.5 (`1cbd9103`) both referenced FWE variables
(`fwe_cHn`, `fwe_hit_full`, etc.) unconditionally, causing `NameError` when cells were run
in top-to-bottom order before the FWE loading cell (`69e1ec41`).

Fix: wrapped FWE blocks with `_fwe_loaded = "fwe_cHn" in dir()` guard. Both cells now run
cleanly on CWE alone and automatically add FWE rows once the FWE loading cell has been
executed. Figure sizing adjusts dynamically to `n_rows`.

---

## 2. Key Findings from CWE Analysis Plots

### 2.1 Analysis 2 — tracking error

The error `H/log(kv_len) − target` is negative for CWE hit cases (centered ≈ −0.025) and
near zero for fail cases. **Negative error means decode attention entropy is below the
prefill-calibrated target** — the controller drives attention sharper than the prompt.

The controller was designed to *maintain* prompt entropy, but what actually helps CWE is
landing *below* prompt entropy. The controller achieves this through EMA momentum overshooting
downward. This is a systematic beneficial overshoot, not random.

Fail cases show high-variance oscillation around zero error — the controller never locks into
a stable sharpened regime for those examples.

FWE error histogram: both hit and fail cluster at −0.06 to −0.08, much larger negative error.
This is because FWE generation entropy drops far below the prefill target (~0.15–0.25 vs
target ~0.326). The controller tries to raise temperature but is saturated at `temp_max=1.0`.
The controller is structurally inert for FWE.

### 2.2 Analysis 10 — matched-example comparison

Key panels and what they show:

- **Baseline entropy (same groups)**: ctrl-only hit and both-miss look essentially identical
  on baseline — the examples are not naturally different. The controller *creates* the
  difference, it does not merely select already-easier examples.

- **Entropy delta (ctrl − baseline)**: Persistently negative (≈ −0.02 to −0.06) for
  ctrl-only hit; oscillates around zero for both-miss. This is the cleanest causal evidence
  in the dataset.

- **Temperature trajectory**: ctrl-only hit maintains temp ≈ 0.91 from early in generation;
  both-miss stays at ≈ 0.94–0.97. The temperature divergence is established at step 0,
  meaning the low-temperature regime is locked in very early.

- **Step-0 entropy distribution**: ctrl-only hit is already shifted left of the target at
  the first decode token; both-miss is centered on or right of the target. Early divergence
  precedes correctness.

- **Baseline open-loop drift**: all groups show similar patterns — no systematic difference
  between groups without the controller. Confirms causality.

### 2.3 Causality vs confounding

Analysis 10 resolves the causality question raised in Analysis 2: the controller is genuinely
enabling success on ctrl-only hit examples, not merely selecting examples that would succeed
anyway. The baseline panel is the key control.

### 2.4 Both-miss ceiling

173 examples (34.6%) that neither system can answer. Their baseline and controller entropy
trajectories are similar to ctrl-only hit but they still fail. This is either a model capacity
limit (the words are too ambiguously distributed in the context) or a calibration limit (the
right target entropy for these examples is outside the controller's operating range). The
oracle target experiment (§5.3 below) would distinguish these.

---

## 3. Hypothesis Reassessment

### 3.1 Why the original hypothesis was wrong

Original: "decode entropy should match prompt entropy."

Problem: prompt processing (reading/comprehending 32k tokens) and decode (generating one
token at a time from KV cache) are different computational operations with different
attention requirements. The prompt entropy is a measure of how the model distributed
attention while *reading*. The optimal decode entropy is determined by what the *task*
requires — these are not the same.

### 3.2 The revised hypothesis

> Long-context LLMs have an attention entropy diffusion problem during decoding: as context
> length grows, attention becomes too uniformly distributed for tasks requiring selective
> retrieval. A real-time adaptive controller that prevents this diffusion by adjusting
> per-head softmax temperature outperforms both fixed temperature baselines and unconstrained
> attention, because the diffusion problem is heterogeneous across generation steps.

Three testable corollaries:
1. The optimal decode entropy is task-specific, not universal.
2. The entropy gap (natural entropy − task-optimal entropy) grows with context length.
3. Adaptive correction outperforms fixed correction because the gap is heterogeneous
   across generation steps.

### 3.3 Task-entropy taxonomy

Evidence now establishes that different tasks need different entropy levels:
- **CWE (count frequencies across 32k)**: needs entropy *below* model default — sharpening
  helps identify repetition signal
- **FWE (count frequencies, fake words)**: model's natural entropy already drops to the
  right level during generation — no intervention needed
- **MRCR 8-needle (64–128k)**: success cases have persistently *higher* entropy than
  failure (≈ 0.50 vs 0.45) throughout generation — needs broader attention to hold 8
  simultaneous anchor points; controller would hurt this task
- **QA single/multi-hop**: modest consistent improvement (+0.2 to +5.4pp), growing with
  context length; multi-hop benefits more, consistent with multi-hop needing moderate
  entropy for chaining evidence

The MRCR entropy gap (success > failure) is likely task-specific to multi-needle retrieval,
not a universal direction reversal. But it validates the general principle that optimal
entropy is task-structure-dependent.

---

## 4. New Empirical Evidence Discussed

### 4.1 Fixed lower temperature < adaptive controller (QA 32k)

Running baseline at fixed temp ≈ 0.91 (the median of the controller's operating range) is
better than fixed temp=1.0 but still worse than the adaptive controller. This rules out the
simplest explanation ("the controller just found a better temperature level") and validates:

**The value of the controller is in the timing and trajectory of adjustment, not just the
level.** Generation has heterogeneous entropy needs across steps (content tokens vs.
structural/format tokens). Adaptive correction matches these heterogeneous needs; fixed
temperature cannot.

### 4.2 temp_max=1.2 hurts performance

Raising the temperature ceiling above 1.0 worsens performance. Two complementary
explanations:

1. **Wrong correction target**: structural/format tokens naturally have low entropy (the
   model is highly confident about them). With temp_max=1.0 the controller hits the ceiling
   and leaves them alone. With temp_max=1.2, it actively softens these tokens — degrading
   coherent format generation.

2. **Distributional shift**: all standard LLMs (Llama, Qwen, DeepSeek, GPT, Mistral) are
   trained with `softmax(QK^T / sqrt(d_k))` — no additional temperature multiplier. Going
   to temp=1.2 produces attention distributions the model has never generated during
   training. The Q/K projection weights are calibrated for the standard scaling.

**Conclusion**: `temp_max=1.0` is a principled hard constraint, not a tunable hyperparameter.
The controller's operating range [temp_min, 1.0] means it can only sharpen, never soften
beyond baseline. This turns out to be correct: the relevant failure mode is excess diffusion,
not excess sharpness.

Note: **YaRN-based models** are an exception — YaRN explicitly bakes in an attention
temperature correction (`mscale` factor) as part of its long-context RoPE extension. Stacking
the adaptive controller on top of YaRN double-corrects, which likely explains why results
are worse on YaRN than on standard Llama at the same context lengths. This is a model
compatibility issue, not a design flaw.

**Gemma 2** uses attention logit soft-capping (`tanh(x/50)*50`) which limits sharpness
nonlinearly — a different kind of built-in attention regularizer. Controller behavior on
Gemma 2 would need separate characterization.

### 4.3 Session target > per-sample target > global prior

The session target (mean entropy across 50 in-context examples) outperforms using each
example's own prefill entropy as target.

Interpretation: the session target estimates the task × context-length entropy operating
point rather than the specific example's prompt properties. The relevant calibration unit
is the (task, context length) pair, not the individual example. Example-level prefill
entropy is too noisy and does not correspond to what the generate phase needs.

This also explains why a global prior (not context-length-aware) underperforms: the optimal
entropy level shifts with context length.

### 4.4 MRCR decoding-stage entropy gap

In MRCR 8-needle (64–128k), success cases maintain consistently higher attention entropy
(≈ 0.50) than failure cases (≈ 0.45) throughout the entire generation trace. The gap is
present from step 0.

In the Qwen2B 128k+ normalized entropy trace, success and failure diverge in the first
10–20 decode tokens and then converge. Early decode behavior is predictive of outcome.

Implication: the first 10–20 decode steps contain a calibration signal that predicts
whether the model has "locked on" to the relevant context. This is the basis for the
early-decode probe target estimation proposal (§5.2).

---

## 5. Proposed Design and Experiment Direction

### 5.1 What to keep in the current design

The core architecture is sound:
- Per-head EMA proportional controller
- Session-level calibration as target anchor
- `temp_max=1.0` hard constraint (now theoretically justified)
- EMA smoothing (filters content/structure oscillation)

### 5.2 Design improvement: early-decode self-calibrating target

Replace the session-level prefill target with an example-specific early-decode target:

```
Phase 1 (decode steps 0–15): run at fixed temp=1.0, observe entropy trajectory
Phase 2 (step 16+):           target = mean(entropy[0:15])
                               engage controller to maintain this early-decode level
```

Rationale: the model's entropy in the first ~15 decode steps reflects how well it locked
onto the relevant context during prefill. Using this as a target is more example-specific
than the session target and removes the session carryover confound. This does not require
any training or external calibration data.

Predicted outcome: this would help the both-miss cases (which may need a different target
than the session mean) and perform better on YaRN (no double-correction).

### 5.3 Three highest-priority experiments

**Experiment 1 — Context-length × entropy-gap curve (establishes the core claim)**

For CWE and QA at 4k / 8k / 16k / 32k / 64k:
- Measure natural decode entropy (baseline) at each context length
- Measure entropy of successful examples at each context length (task-optimal proxy)
- Compute entropy gap = natural − task-optimal
- Plot controller gain vs. entropy gap

Hypothesis: controller gain is proportional to the entropy gap. When gap ≈ 0 (CWE at 4k),
controller hurts. When gap is large (CWE at 32k), controller helps dramatically. This would
be the key figure establishing the framework.

**Experiment 2 — Oracle target experiment (separates calibration limit from capacity limit)**

For CWE: set controller target to the mean entropy of ctrl-only hit examples (≈ 0.30)
instead of the session entropy (≈ 0.327). Run on the full 500 examples.

Questions answered:
- Does performance improve above 27.18%? → calibration gap is the bottleneck
- Do any of the 173 both-miss examples get solved? → they are not a pure capacity limit
- How much headroom exists above the current result? → upper bound on controller improvement

**Experiment 3 — Adaptive > fixed, extended (strengthens the mechanism claim)**

Run adaptive vs. fixed-at-session-median comparison on:
- CWE at 32k (not yet done explicitly)
- QA at 16k, 64k

Hypothesis: the adaptive advantage grows with context length (longer contexts → more
step-heterogeneity in entropy demands → more value from per-step adaptation). If confirmed,
this is a clean mechanism story.

### 5.4 Deprioritized directions

- Bidirectional control (temp_max > 1.0): empirically ruled out, theoretically explained
- Per-head target (valid direction but adds complexity without clear gain until oracle
  target experiment establishes overall headroom)
- YaRN experiments (the double-correction problem means YaRN needs separate treatment;
  deprioritize until standard Llama story is complete)

---

## 6. Paper Framing Assessment

### 6.1 Honest scope

The result is most appropriate for a targeted venue (long-context inference, efficient LLM
serving, EMNLP/ACL systems track) rather than a top-tier ML venue. The CWE gain is dramatic
but narrow; QA gains are real but modest. The scientific contribution is the causal
characterization of attention entropy as a controllable variable, not the specific controller
design.

### 6.2 What would make a complete paper arc

1. **Diagnosis**: attention entropy drifts above task-optimal level as context length grows
   (entropy-gap curve, Experiment 1)
2. **Causal evidence**: Analysis 10 from CWE — controller creates the difference, not
   example selection
3. **Mechanism**: adaptive > fixed (Experiment 3) — timing matters, not just level
4. **Improved design**: early-decode self-calibrating target (§5.2) — closes the gap
   between session-calibrated and oracle performance
5. **Scope characterization**: task-entropy taxonomy (CWE/QA need sharpening, MRCR needs
   broader attention, FWE already calibrated); context-length boundary (helps at ≥16k,
   hurts at ≤8k)

The oracle target experiment (§5.3) is the pivotal missing piece. If oracle target
significantly outperforms session target AND moves some both-miss examples, the full arc
is achievable. If oracle target doesn't help, the story is still publishable as a
characterization paper.

### 6.3 Current best framing

> *Inference-time attention entropy is a causal and controllable factor in long-context
> retrieval performance. Adaptive per-head temperature control, calibrated from session-level
> entropy statistics, prevents the attention diffusion that accumulates at long contexts.
> The adaptive mechanism outperforms any fixed temperature, the operating range is bounded
> at temp≤1.0 for distributional reasons, and the benefit is task- and context-length-
> specific in ways that follow from the information-theoretic demands of each task.*

---

## 7. Open Items (updated from May 7)

**Carried over from May 7 (still blocking):**
- [ ] Principled session calibration: K probes + session_size=500 + exclude calibration
      samples → confirm gains match Scaled table
- [ ] Session length ablation: session_size ∈ {50, 100, 200, 500} on qa_2 32k
- [ ] InfiniteBench evaluation (generalization + longer answers for per-step dynamics)

**New from this session:**
- [ ] Experiment 1: context-length × entropy-gap curve for CWE and QA
- [ ] Experiment 2: oracle target experiment (CWE: target = 0.30 instead of ~0.327)
- [ ] Experiment 3: adaptive vs. fixed comparison on CWE 32k and QA 16k/64k
- [ ] Early-decode probe target (Phase 1/Phase 2 design in §5.2) — implement and test on
      QA 32k first
- [ ] Analysis notebook: run 1b and 2b cells to check whether entropy gap scales with
      partial score (miss/partial/good)

**Deprioritized:**
- Bidirectional control (temp_max > 1.0): ruled out empirically and theoretically
- YaRN experiments: defer until Llama story is complete
- Head-selective temperature control: valid but premature
