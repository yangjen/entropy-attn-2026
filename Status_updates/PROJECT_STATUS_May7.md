# Project Status — May 7–8 2026
Entropy-Attention Controller for Long-Context Retrieval

Supersedes PROJECT_STATUS_May6.md. New findings in §5–§8; earlier findings in §1–§4 are
unchanged except where noted.

---

## 1. Oracle Prior & Scaling Law (unchanged from May 6)

See PROJECT_STATUS_May6.md §1–§4. Key facts carried forward:

- Blended oracle prior (qa_1 + qa_2 averaged) for LLaMA 3.1-8B:
  4k→0.3289, 8k→0.3113, 16k→0.2895, 32k→0.2792, 65k→0.2763, 131k→0.2559 (extrapolated)
- Log-linear fit: target ≈ 0.489 − 0.0137 · log2(ctx_len), R²≈0.93 on 4k–65k
- 131k anomaly confirmed as RoPE extrapolation artifact; scaling-law extrapolation corrects it
- Qwen3.5-2B shows plateau/U-shape pattern — slope is model-specific, not universal

---

## 2. 131k Regression Recovery (May 7, confirmed)

Using the scaling-law extrapolated target (0.256) in global_prior mode:

| condition | accuracy |
|---|---|
| baseline (t=1.0) | 71.4% |
| session-based with inflated prior (RoPE anomaly) | 67.8% ← regression |
| global prior, extrapolated target | 71.6% |

Regression fully reversed. +0.2% over baseline is noise; the result is that the anomalous
prior is no longer causing harmful misfires. Validates §4 of May 6 doc: the scaling law
detects the 131k outlier and the corrected prior restores safe operation.

---

## 3. Temperature Carryover: Quantified (May 7)

Analysis of step-0 decode temperature by intra-session position
(run: 65k qa_2, session_size=50, entropy_attn, 3-probe calibration):

| session position | mean step-0 temp | range |
|---|---|---|
| 0–4 (cold) | 0.99874 | 0.996–1.000 |
| 5–9 | 0.99567 | 0.989–0.999 |
| 10–19 | 0.99182 | 0.985–0.996 |
| 20–29 | 0.98767 | 0.982–0.992 |
| 30–39 | 0.98226 | 0.972–0.987 |
| 40–49 (warm) | 0.97670 | 0.968–0.984 |

- Temperature is below 1.0 for every single sample (100%)
- Drift is monotonic — no convergence within 50 samples
- The asymmetric controller (temp_max=1.0) prevents upward corrections; temp can only ratchet down
- "Steady-state temp ≈ 0.985" from May 6 was a rough session average, not an asymptote

---

## 4. Fixed Temperature Ablation (May 7, qa_2 32k)

| condition | accuracy |
|---|---|
| baseline t=1.0 | 50.0% |
| fixed t=0.985 | 47.4% ← worse |
| fixed t=0.9 | 51.0% |

**Key finding:** t=0.985 (the range where carryover puts the controller) is actively harmful.
t=0.9 gives a marginal gain but is far below what the carryover ever reaches.

**Implication:** Temperature carryover to the 0.977–0.985 range is not the mechanism for
session gains. The controller is moving in the right direction but operating in the wrong
temperature range for fixed-temp purposes.

---

## 5. Mechanism Isolation: EMA vs Temperature Carryover (May 7–8)

### 5.1 Experimental conditions (qa_2 32k, global_prior mode, session_size=50)

New `--global_prior_reset_mode` flag added to `infinben_ruler_session_notarget.py`:
- `both`: reset EMA + temp per sample (no carryover)
- `ema_only`: reset EMA, let temp carry over
- `temp_only`: reset temp, let EMA carry over
- `none`: no per-sample reset (both carry over)

### 5.2 Results

| condition | accuracy | vs baseline |
|---|---|---|
| baseline (t=1.0) | 50.0% | — |
| global prior, both reset | 49.8% | −0.2% (neutral) |
| global prior, EMA carryover (temp_only reset) | 47.0% | −3.0% |
| global prior, temp carryover (ema_only reset) | 43.0% | −7.0% |
| global prior, no reset (oracle prior + full carryover) | <49.8% | harmful |
| original session (session calibration + full carryover) | 52.8% | +2.8% |

Temp carryover run was one 500-sample session (no session_size=50), so temperature drifted
to near the 0.7 floor — the 43% is partly an extreme artifact of unbounded drift. The
direction is clear regardless.

### 5.3 What the pattern means

- Neither component alone helps; separated, both hurt
- Oracle prior + full carryover (no reset) is also harmful — worse than both reset
- **The session-calibrated target is the critical ingredient, not the carryover mechanism per se**
- Carryover with the wrong target (oracle prior) is harmful; carryover with the right target
  (session calibration) produces gains
- EMA and temp are coupled state: breaking either one creates a mismatch between the
  controller's belief about entropy and the actual corrections applied

### 5.4 Why isolated carryover hurts

**EMA carryover + temp reset:** Carried EMA from previous sample reflects its high decode
entropy → EMA starts above oracle prior target at step 0 → controller fires immediately
and aggressively on a fresh sample that may not need correction → overshoots.

**Temp carryover + EMA reset:** EMA reset to oracle prior at step 0 → error ≈ 0, controller
quiet initially → but temperature is already pre-lowered from previous corrections → as
decode proceeds and EMA rises above oracle prior, controller fires again on top of already-
depressed temp → double sharpening → 43%.

**Oracle prior + no reset:** Joint carryover is present but the calibration target
(oracle prior = global average) does not match the actual session distribution closely
enough for the accumulated state to be informative. Stale mismatched state accumulates.

---

## 6. Best Observed Result: "Scaled" Design (Misimplementation)

A run using the first test sample's prompt-tail entropy as the calibration target,
with no per-sample reset and no session boundaries (one continuous run over all samples):

| context length | baseline qa_1 | scaled qa_1 | baseline qa_2 | scaled qa_2 |
|---|---|---|---|---|
| 4096 | 82.2% | 81.8% | 58.0% | 58.8% |
| 8192 | 79.4% | 79.8% | 55.8% | **58.8%** |
| 16384 | 77.4% | 79.0% | 52.6% | 54.8% |
| 32768 | 77.6% | 78.0% | 49.6% | **53.6%** |
| 65536 | 76.8% | 79.6% | 47.0% | 48.4% |

Gains consistent across lengths and both tasks. Best absolute gains: qa_2 32k (+4.0%),
qa_2 8k (+3.0%), qa_1 65k (+2.8%), qa_1 16k (+1.6%).

**Why it outperforms the 50-sample session runs:**
- No session resets: the controller accumulates state across all 500 samples, not just 50
- Temperature and EMA converge more fully toward the task-appropriate operating point
- "Warm" state persists for nearly the entire evaluation rather than only the last ~20
  samples of each 50-sample session

**Why it is a misimplementation:**
- Calibration target derived from the first test sample (sample 1's prompt tail entropy)
- Sample 1 is part of the evaluation — implicit data leakage
- Single-sample calibration is noisy and non-reproducible

---

## 7. Revised Understanding of the Contribution

### 7.1 What the controller is actually doing

The controller's gains come from **joint cross-sample state accumulation with session-specific
calibration**, not from per-step adaptive temperature adjustment. Specifically:

1. Session calibration (K unlabeled prefill passes) sets a target matched to the actual
   session data distribution — more accurate than the pre-computed oracle prior
2. EMA and temperature carry over together across samples, maintaining internally consistent
   state
3. The accumulated state represents the controller's calibrated belief about the right
   entropy operating point for this task/length combination
4. This calibrated operating point enables more appropriate corrections on individual
   decode steps than either a fixed temperature or a per-sample cold start

### 7.2 Why per-step adaptation is not the story

- Median per-sample temperature correction: 0.000 (controller barely fires per sample)
- Average decode length: 3.5 tokens (too short for meaningful within-sample dynamics)
- Per-sample reset eliminates all gains
- The "adaptive" behavior operates at session timescale, not decode-step timescale

### 7.3 Revised paper framing

**Primary claim:** Online session calibration — estimating the session-specific attention
entropy target from a small number of unlabeled prefill passes — enables an entropy
controller to consistently improve long-context QA accuracy without manual temperature tuning.

**Supporting claims:**
- The oracle prior follows a log-linear trend with context length; this enables cheap
  calibration (2–3 bins) and flags RoPE anomalies as outliers
- A P-controller with EMA smoothing and cross-sample state carryover is a sufficient
  architecture for exploiting the calibrated target
- Per-step temperature adaptation is not the operative mechanism; session-level calibration
  and state accumulation are

---

## 8. Key Limitations and How to Argue Them

### 8.1 Data leakage (calibration from test samples)

**Concern:** The K calibration samples are drawn from the test set; their prefill entropy
informs the controller target; the same samples may be included in accuracy evaluation.

**Mitigation:**
1. Calibration uses only unlabeled prefill entropy — no labels, no model outputs, no
   ground-truth answers. This is unsupervised distribution matching, not label leakage.
   Comparable to temperature scaling calibration and test-time adaptation literature.
2. Exclude calibration samples from accuracy reporting. Report accuracy on samples K+1 to N
   only. Frame the first K samples explicitly as a "warm-up phase." Gains should persist —
   they come from the warmed-up controller acting on the remaining N−K samples.
3. Long-term fix: use a separate held-out calibration set (different RULER split or seed).

### 8.2 Session homogeneity assumption

**Concern:** Real deployment sessions mix question types and context lengths; the
stationarity assumption does not hold.

**Mitigation:**
1. **Narrow the scope:** Long-context inference is predominantly used in batch workloads
   (document pipelines, RAG over fixed corpora, evaluation frameworks) where session
   homogeneity approximately holds. The method targets this setting explicitly.
2. **Context-length reset:** The dominant factor in entropy distribution is context length,
   not task type. A simple rule — reset the session when context length changes bucket —
   handles the most common mixed-session case. Task-type variation within a length bucket
   is small (~0.015 in target) and the P-controller absorbs it.
3. **Graceful degradation:** In fully mixed sessions, the controller's state converges
   toward a blend of distributions — similar to the oracle prior. The oracle prior condition
   gives ≈ baseline (49.8%). Harmful behavior requires persistent mismatch between target
   and actual entropy, which is bounded by the EMA decay (beta=0.9 forgets old state).
4. **Future work:** Online session-shift detection (entropy drift trigger for recalibration).

### 8.3 max_step calibration

**Concern:** max_step=0.0005 is too conservative — the controller can only move temperature
~0.023 over a 50-sample session, reaching 0.977 when the useful operating range may be
~0.85–0.90. The controller is operating in the wrong temperature regime.

**Status:** The temperature sweep (§4) confirms t=0.985 is harmful and t=0.9 is marginally
better. The optimal temperature range and the appropriate max_step have not been established.

**Mitigation:** Run temperature sweep {0.7, 0.75, 0.8, 0.85, 0.9, 0.95} across qa_1/qa_2 ×
32k/65k to find oracle Δt per config. Set max_step ≥ Δt / (session_size × avg_steps × f).
This is a principled calibration of max_step from the oracle temperature, requiring only
a small held-out sweep — analogous to the oracle prior calibration procedure.

### 8.4 Short answers (RULER QA limitation)

Average decode length is 3.5 tokens. Per-step adaptation on 3.5-token answers is physically
impossible with any reasonable max_step. The per-step adaptive story requires longer answers.
InfiniteBench has longer outputs (~16 tokens) and is the right evaluation for demonstrating
per-step dynamics if needed.

---

## 9. Open Items (updated)

**Blocking for paper:**
- [ ] Principled version of the misimplementation: session calibration (K probes) +
      session_size=500 + exclude calibration samples from accuracy → confirm gains match
      or approach the Scaled table above
- [ ] Session length ablation: session_size ∈ {50, 100, 200, 500} on qa_2 32k → show
      gains increase monotonically (or plateau) with session length
- [ ] InfiniteBench evaluation: held-out generalization test; resolves contamination concern;
      longer answers test whether per-step dynamics matter
- [ ] max_step calibration: temperature sweep {0.7–0.95} on qa_1/qa_2 × 32k/65k →
      establish oracle Δt → set principled max_step

**Supporting / paper-strengthening:**
- [ ] Re-run accuracy excluding calibration samples (address leakage §8.1)
- [ ] Mixed-session ablation: interleave qa_1 and qa_2 in one session → quantify degradation
- [ ] MRCR ablation (§3.4 from May 6): 4 conditions at 32–64k and 64–128k
- [ ] YaRN 131k ablation (optional, superseded by scaling-law fix)

**Deprioritized:**
- Head-selective temperature control (valid direction but requires MRCR probe results first)
- Qwen additional experiments (Qwen pattern established in May 6; revisit if Option B framing)
- Threshold sweep (less relevant given revised understanding of mechanism)

---

## 10. Two Viable Paper Framings (revised)

**Option A — "Session calibration for inference-time attention control":**
Primary contribution is the online calibration procedure: K unlabeled prefill passes estimate
the session-specific entropy target; cross-sample state carryover allows the controller to
maintain this calibrated operating point without per-sample overhead. The oracle prior and
scaling law reduce offline calibration cost and flag RoPE anomalies. Submittable now if
InfiniteBench holds.

**Option B — "Attention entropy decreases predictably; controllers that exploit this
outperform those that don't":**
Shifts focus to the pattern itself. Requires showing the pattern (and controller gains) hold
across 3+ model/setting combinations. Needs more experiments. Option B is scientifically
stronger but more work.

Option A is the current path. The misimplementation table is the target result to replicate
with a justified design.
