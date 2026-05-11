# Project Status — May 5 2026
Entropy-Attention Controller for Long-Context Retrieval

---

## 1. Probe Analysis Findings (consolidated)

### 1.1 Signal validation across context lengths

Output token entropy (`norm_entropy_trace`) is the primary detection signal.
AUC from ROC analysis:

| Signal | 16-32k | 32-64k | 64-128k | Real-time? |
|---|---|---|---|---|
| peak_entropy_early (max, tokens 0-50) | ~0.84 | 0.84 | 0.81 | No (needs full window) |
| mean_entropy (trace mean) | ~0.86 | 0.74 | 0.79 | **Yes** (running mean) |
| mean_instability (running mean of I) | — | 0.72 | 0.78 | **Yes** |
| max_instability | — | 0.79 | 0.76 | No (needs full trace) |
| attention entropy (attn_entropy_trace) | ~0.5 | ~0.5 | ~0.5 | — |

**Key finding:** attention entropy is not a useful detection signal (AUC ~0.5 consistently).
Output token entropy is the right signal for the firing gate.

### 1.2 Instability is redundant with entropy level

Spearman ρ(mean_entropy, mean_instability) = **0.964** (p=0.000).
Combined AUC = 0.788, same as mean_entropy alone (0.788).

Root cause: instability I = H_std_z + H_slope_pos_z + H_rebound_z, all derived from
windowed entropy values. The MRCR entropy trace is bimodal (near-zero or briefly spiked),
so dynamics and level activate on the same windows. There is no "stable moderate entropy"
regime where level and dynamics could disagree.

**Design consequence:** drop instability from the controller firing signal entirely.
Use entropy level alone.

### 1.3 The 0.05 threshold is empirically grounded

The vertical cut at ~0.05 in the entropy-vs-instability scatter plot cleanly separates the
green (success) cluster from the red (failure) cluster across all three context length bins.
This line is a manually observed value, not the population median (~0.06-0.07).

Entropy is approximately **scale-invariant across context lengths** — the threshold does not
drift meaningfully from 16k to 128k. This suggests a single global threshold may suffice
without per-bin calibration.

The 0.05 threshold corresponds to the empirical boundary where P(failure | entropy > 0.05)
begins to dominate. It is a starting point for hyperparameter search, not a final value.

### 1.4 Quadrant analysis update

Revised quadrant interpretation (peak entropy vs max instability scatter):

| Quadrant | Entropy | Instability | Interpretation | Outcome |
|---|---|---|---|---|
| Top-right | High | High | Confused | Mostly fail |
| Bottom-right | High | Low | Exploratory | Mostly fail |
| Top-left | Low | High | Late derailment? | Mixed |
| Bottom-left | Low | Low | Confident | Mostly succeed |

**Key observation:** the exploratory quadrant (high entropy + low instability) also mostly
fails — same outcome as confused. This confirms entropy level alone predicts failure
regardless of whether instability is present.

### 1.5 Entropy peak at steps 8-15 (all context lengths)

The output token entropy trace peaks at decode steps 8-15 across all context lengths,
not at step 0. This is due to the MRCR `random_string_to_prepend` format: the first
~7 tokens copy the deterministic prefix (near-zero entropy), then genuine retrieval
uncertainty surfaces.

This is task-format specific, not context-length dependent.
Implication for oracle prior: step0_entropy is not the right anchor. Use a window that
covers steps 5-20, or use peak/max over the early window.

### 1.6 Signal does not transfer to RULER for probe analysis

RULER QA answers are 5-15 tokens — too short to compute windowed statistics, trace
alignment, or ROC curves. The probe analysis pipeline is MRCR-specific.

RULER serves as: (a) oracle prior calibration source, (b) controller generalization test.
Signal validation (verifying output entropy separates success/failure) can only be done on
MRCR (or InfiniteBench longbook_qa_eng, which has longer answers).

---

## 2. Controller Design (updated)

### 2.1 Architecture

```
Decode step t:
  1. Compute output token entropy h_t = H(vocab dist) / log(vocab_size)
  2. Update EMA: ema_t = α * h_t + (1-α) * ema_{t-1}
  3. GATE: if ema_t > θ  →  fire intervention
  4. INTERVENTION: scale attention temperature toward H*_attn[ctx_bin]
```

Two distinct signals serving separate roles:

| Role | Signal | Units | Source |
|---|---|---|---|
| Gate (fire or not) | Output token entropy EMA | H/log(V) ∈ [0,1] | Runtime decode |
| Intervention target | Attention entropy | H/log(kv_len) | Oracle prior |

### 2.2 Gate design

- Signal: EMA of output token entropy
- EMA α: fast (≈0.5-0.7) so the EMA responds quickly to the spike at steps 8-15
- Threshold θ: pilot value **0.05**, to be tuned experimentally
- Flat global threshold (not per-bin) justified by entropy scale-invariance finding
- Optional: replace flat threshold with oracle_prior_output_entropy[ctx_bin] + margin
  (relative threshold); defer to experimental results

### 2.3 Intervention (unchanged from old design)

Attention temperature scaling kernel:
- Computes H_current = attention entropy at current decode step
- Scales temperature so that H_current → H*_attn (the oracle prior target)
- Scaling only when H_current > H*_attn (down-scaling, sharpen attention)
- Same custom kernel as old design — no changes needed here

### 2.4 Oracle prior for the intervention target

**What it is:** E[attention entropy at prefill tail | context_length_bin]

**What it replaces:** the old session-calibrated `prompt_target_entropy`, which was
measured from K=3 examples at session start. The oracle prior is estimated offline from
a large validation set, so it is:
- More stable (not contaminated by hard examples in the current session)
- Generalizable across sessions without per-session calibration cost
- Conditioned on context length, not session-specific

**Estimation method:**
- Run prefill forward pass (no generation) on many validation examples
- Read attention entropy at the last prefill position from the entropy kernel
- Bin by actual tokenized context length
- Compute mean/median/std per bin → store as JSON lookup table

**Why not output token entropy for this:**
The intervention kernel operates on attention entropy. The oracle prior must be in the
same units as the intervention target. Output token entropy oracle prior is a separate
concept (for the gate threshold), and is arguably unnecessary given the flat 0.05 suffices.

### 2.5 Prior contamination analysis

| Design | Prior source | Contamination risk |
|---|---|---|
| Oracle prior (target) | Large offline validation set | None — not session-dependent |
| Selective session prior | Session first-N + skip examples where controller fired | Low |
| Old design | Session first-N, no selective update | High — hard examples elevate prior |
| Per-sample reset | Per-sample prefill tail | Highest — prior blinds to hard examples |

Oracle prior is the upper bound; selective session prior is a practical runtime alternative
if cross-task/cross-session transfer of the oracle prior is insufficient.

---

## 3. Experiment Plan

### 3.1 Immediate: update estimate_oracle_prior.py

Current script measures output token entropy from decode steps (generate() call).
Needs to be updated to measure **attention entropy** from prefill forward pass:
- Replace generate() with model(input_ids) forward pass
- Read attention entropy from entropy kernel at last prefill position
- This is task-format independent (no prefix copying effect)
- Faster (single forward pass vs autoregressive decode)

### 3.2 Run oracle prior estimation (RULER)

```bash
RULER_BASE=/c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic
CUDA_VISIBLE_DEVICES=0 python estimate_oracle_prior.py \
  --model Qwen/Qwen3.5-9B \
  --dataset_type ruler \
  --data_root ${RULER_BASE}/16384/data,${RULER_BASE}/32768/data,\
              ${RULER_BASE}/65536/data,${RULER_BASE}/131072/data \
  --tasks qa_1,qa_2 \
  --output_dir /c2/jenny/r3/MRCR_outputs/oracle_prior/qwen35_ruler \
  --ctx_bin_edges 16384,32768,65536,131072 \
  --max_per_bin 100
```

Expected output: oracle_prior.json with mean attention entropy per bin,
used as H*_attn[ctx_bin] in the controller.

### 3.3 Implement soft gate controller

Modify the entropy attention controller:
1. Add EMA tracker for output token entropy (α ≈ 0.5-0.7)
2. Add gate condition: fire only if EMA > θ (default θ=0.05)
3. Load oracle prior JSON at startup, look up H*_attn by context length bin
4. Use oracle prior value as prompt_target_entropy instead of session calibration

### 3.4 Ablation experiments on MRCR

Run 4 conditions on MRCR 8-needle, 32-64k and 64-128k:

| Condition | Gate | Target | Notes |
|---|---|---|---|
| sdpa baseline | None | None | Upper bound with no intervention |
| Old controller | Attention entropy (no soft gate) | Session-calibrated | Existing baseline |
| New: gate only | Output entropy > 0.05 | Session-calibrated | Gate contribution |
| New: gate + oracle prior | Output entropy > 0.05 | Oracle prior | Full new design |

Primary metric: MRCR score (mean SequenceMatcher ratio).
Secondary: false positive rate (controller firing on examples it shouldn't).

### 3.5 Threshold sweep

After baseline run with θ=0.05, sweep θ ∈ {0.03, 0.04, 0.05, 0.06, 0.08} to find
the operating point balancing TPR (catching failures) vs FPR (not firing on successes).
Use ROC operating point analysis to guide the sweep.

---

## 4. Risks and Watch-outs

| Risk | Severity | Mitigation |
|---|---|---|
| Attention scaling → output entropy link is unproven | High | Core experiment 3.4 tests this directly |
| Flat 0.05 threshold too aggressive / too lenient | Medium | Threshold sweep (3.5) |
| Oracle prior (RULER) doesn't transfer to MRCR | Medium | Selective session prior as fallback |
| RULER answers too short for signal validation | Low | Accepted — MRCR is the validation benchmark |
| EMA α too slow to catch the steps 8-15 spike | Medium | Use fast α ≈ 0.5-0.7, validate empirically |

---

## 5. Dataset Roles

| Dataset | Role | Notes |
|---|---|---|
| MRCR 8-needle | Signal validation, performance benchmark | Long outputs support probe analysis |
| RULER qa_1, qa_2 | Oracle prior source, generalization test | Short answers, can't probe signal |
| InfiniteBench | Deferred | longbook_qa_eng has long outputs; add after core results |

---

## 6. Paper Insights (accumulated)

- **Detection vs intervention signal split**: output entropy detects confusion; attention
  temperature corrects it. This split is the core architectural contribution.
- **Instability redundancy**: entropy dynamics signals are redundant on MRCR because the
  entropy distribution is bimodal. This is a useful ablation result for the paper.
- **Scale invariance**: entropy threshold ~0.05 is stable across 16-128k context lengths.
  Supports a simple, hyperparameter-light controller design.
- **Prior contamination**: per-sample/session priors are blind to hard examples. Oracle
  prior decouples calibration from inference. Ties to broader LLM calibration literature.
- **EDIS connection**: entropy dynamics motivation from EDIS holds conceptually but
  empirically reduces to level signal on retrieval tasks. Good ablation story.
