# Project Status — May 31, 2026

---

## Literature Review

### Category 1 — Foundation Papers (the problem)

**Transformers need glasses! Information over-squashing in language tasks**
Barbero et al. NeurIPS 2024 (arXiv:2406.04267)
Proves a *representational collapse* phenomenon — distinct input sequences yield arbitrarily close last-token representations as length grows, exacerbated by low-precision floats. Connects decoder-only Transformers' loss of token sensitivity to GNN over-squashing. Provides the theoretical "why" for long-context degradation; the cornerstone problem paper.

**Softmax is not Enough (for Sharp Size Generalisation)**
Veličković et al. ICML 2025 (arXiv:2410.01104)
Proves any softmax circuit *must* disperse as the number of items grows at test time. Proposes adaptive temperature as an ad-hoc inference-time technique to sharpen softmax. Most important single foundation+design paper — establishes the dispersion impossibility result AND introduces inference-time attention-temperature sharpening. The project builds on this and must distinguish from it: their adaptive temperature is a closed-form sharpening, not an EMA-of-attention-entropy feedback controller with a self-calibrating target.

**Scalable-Softmax Is Superior for Attention (SSMax)**
Nakanishi 2025 (arXiv:2501.19399)
Defines denominator dilution precisely as "attention fading": the max softmax element →0 as input size grows because the denominator grows with context size n while each token's numerator remains constant. Proposes SSMax which rescales logits by a learnable factor × log n. Foundation reference for the mechanism; a (training-based) comparator for the actuator.

**Efficient Streaming Language Models with Attention Sinks (StreamingLLM)**
Xiao et al. ICLR 2024 (arXiv:2309.17453)
Identifies the *attention sink* — initial tokens absorb large attention mass even when semantically unimportant. Foundation for the format-token / structural-token gate design: sink heads implement a learned "null attention" workaround and should be excluded from the controller's intervention target.

**Lost in the Middle: How Language Models Use Long Contexts**
Liu et al. TACL 2024 (arXiv:2307.03172)
U-shaped performance curve — accuracy highest when relevant info is at start/end, degrades sharply in the middle. Canonical empirical symptom of dispersion/position bias that the controller aims to mitigate.

**Stabilizing Transformer Training by Preventing Attention Entropy Collapse (σReparam)**
Zhai et al. ICML 2023 (arXiv:2303.06296)
Tracks per-head attention entropy as a sharpness proxy during training; entropy collapse (too-low entropy) accompanies training instability. Validates attention entropy as a meaningful, controllable state variable. Note: concerns the *opposite* failure mode (too-low entropy in training) vs. the project's too-high entropy at long context — a useful contrast.

**Variance Sensitivity Induces Attention Entropy Collapse and Instability in Transformers**
Hong & Lee. EMNLP 2025
Identifies softmax variance sensitivity as the cause of entropy collapse; motivates entropy-stable attention. Supporting foundation evidence that attention entropy is a meaningful diagnostic.

---

### Category 2 — Design Choice Support

**Entropy Adaptive Decoding (EAD): Dynamic Model Switching for Efficient Inference**
Simonds (Tufa Labs). arXiv:2502.06833, Feb 2025
Monitors a rolling window of entropy of prediction logits to switch between small/large models; achieves 96.7% of 11B LLaMA performance using it only 43% of tokens. Direct precedent for an EMA/rolling-entropy control signal — but on output logits for model switching, not attention temperature. Supports the smoothing design lever; a Category-3 contrast (different object: output entropy, not attention entropy).

**Entropy-informed Decoding (EDEN): Adaptive Information-Driven Branching**
OpenReview (dzmh4xA4Pq)
Uses per-step output-distribution entropy to set the search branching factor; training-free, plug-and-play. Supports entropy-as-inference-signal; competitor-adjacent (output entropy, search, not attention). Key differentiator from the project: EDEN uses output entropy to control branching in search; this project uses attention entropy to correct internal representation quality under long-context diffusion.

**Learning to Focus: Focal Attention for Selective and Scalable Transformers**
arXiv:2511.06818, 2025
Temperature as a hyperparameter or learned value; smaller temperature sharpens attention and improves long-context reasoning. Supports per-layer/per-head temperature as a meaningful knob (training-based).

**Optimal Attention Temperature Enhances In-Context Learning under Distribution Shift**
arXiv:2511.01292, 2025
Theoretically analyzes how attention-temperature selection shapes ICL performance under distribution shift. Supports temperature as a first-class lever for inference-time control.

**Retrieval Head Mechanistically Explains Long-Context Factuality**
Wu et al. ICLR 2025 (arXiv:2404.15574)
Retrieval heads are: universal (all long-context models have them), sparse (<5% of attention heads), intrinsic (already exist in short-context pretrained models), dynamically activated, and causal (pruning them causes retrieval failure and hallucination). Full attention is crucial because retrieval heads need the full KV cache. Direct support for gating the intervention to retrieval heads.

**Query-Focused Retrieval Heads (QRHead) Improve Long-Context Reasoning and Re-ranking**
Zhang et al. EMNLP 2025 (arXiv:2506.09944)
QRHeads (query-aggregated attention) are a sharper, more downstream-useful retrieval-head set; the QRRetriever built on them yields over 10% performance gains on LongMemEval and CLIPPER. Supports the per-head specialization lever; provides a refined head-identification method the controller could target. Note: QRHead and DySCO author teams overlap (Princeton/Danqi Chen group).

**Attention Entropy is a Key Factor: An Analysis of Parallel Context Encoding**
Zhang et al. ACL 2025 (arXiv:2412.16545)
Identifies unusually high attention entropy as a key factor in performance degradation under parallel context encoding; reduces it via attention sinks and selective mechanisms. Closest "attention entropy as diagnostic" precedent — they diagnose high attention entropy as the problem but remedy via sinks/selection, NOT dynamic temperature. Strong "build on diagnosis, novel remedy" framing for the project.

**Attention Head Entropy of LLMs Predicts Answer Correctness (Head Entropy)**
OpenReview (HJ6EGxNpgb), Feb 2026
Certain attention heads exhibit distinct entropy patterns when the model generates correct vs. incorrect answers. Using sparse logistic regression on per-head entropies achieves 0.07–0.15 AUROC improvements. Shapley analysis shows middle-layer heads contribute most. Validates per-head attention entropy as causally informative about generation quality; supports per-head gating design.

**Entropy-Guided KV Caching for Efficient LLM Inference**
MDPI Mathematics, July 2025
Computes per-head attention entropy; uses average layer entropy to assess contextual importance; allocates larger KV cache budgets to higher-entropy layers (broader dispersion). Validates "high attention entropy = dispersed = needs treatment" — same diagnostic premise as the project, different actuator (cache budget vs. temperature).

---

### Category 3 — Competing / Similar Work

**DySCO: Dynamic Attention-Scaling Decoding for Long-Context LMs**
Ye, Zhang, Yin, Yen, Chen (Princeton). arXiv:2602.22175, Feb 2026. **HIGHEST THREAT.**
Training-free, decoding-time. At each step: (1) runs a partial forward pass to aggregate retrieval-head (QRHead) attention scores, (2) selects a small set of top context tokens, (3) up-weights attention logits to those tokens across all heads. ~4% extra FLOPs; relative gains up to 25% on MRCR and LongBenchV2 at 128K (Qwen3-8B, Llama-3.1-8B).
Key differentiator: DySCO's signal is retrieval-head attention-mass driving *discrete token selection* and *selective* logit up-weighting; this project's signal is an *EMA of attention entropy* driving a *continuous, global/per-head temperature* via a feedback control loop. DySCO contrasts itself with "uniformly scaling all attention logits by a constant factor" — the global-temperature axis is acknowledged but not pursued by DySCO. **Mandatory head-to-head baseline.**

**Long-Context Generalization with Sparse Attention (α-entmax / ASEntmax)**
Vasylenko et al. ICLR 2026 (arXiv:2506.16640). **MODERATE THREAT.**
Replaces softmax with α-entmax (assigns exact zeros to irrelevant tokens); ASEntmax adds learnable context-dependent per-head temperature. Requires retraining and replaces the softmax operator. The project is training-free and keeps softmax. Cite as the "retrain-the-attention-operator" alternative.

**Entropic-Time Inference: Self-Organizing LLM Decoding Beyond Attention**
Kiruluta (UC Berkeley). arXiv:2603.03310, March 2026. **MODERATE THREAT (conceptually closest control loop).**
Entropy-aware scheduling with adaptive temperature control stabilizing generation near a target entropy regime, with a contractive temperature-update map. Key differentiator: its entropy signal and temperature both operate on the **output/sampling softmax** (next-token predictive entropy), not the attention softmax. The project's actuator (attention temperature) and signal (attention entropy) are a clean separation.

**Hold Onto That Thought: Assessing KV Cache Compression on Reasoning (SnapKV-D)**
Palnitkar et al. NeurIPS 2025 (arXiv:2512.12008). **LOW THREAT.**
Benchmarks KV-cache compression on long-*generation* reasoning tasks; introduces SnapKV-D (sliding observation window every 128 generated tokens). Key differentiator: KV-cache eviction/compression for memory efficiency — orthogonal object; the project does not evict tokens and targets accuracy, not memory. Useful: establishes the SnapKV → SnapKV-D design lineage that inspired the project's session→per-sample switch design.

**SnapKV**
Li et al. 2024
Scores token importance using an observation window at the end of the prompt during prefill; pools attention scores; selects top-B positions per head; cache is frozen after prefill. Inspired the project's prompt tail calibration design.

**SEAL: Scaling to Emphasize Attention for Long-Context Retrieval**
Lee et al. ACL 2025 (arXiv:2501.15225). **LOW–MODERATE THREAT.**
Learns head-wise (SEAL-H) and channel-wise (SEAL-C) attention scales via SGD on a small generated dataset. Significant accuracy improvements with <1 hour fine-tuning for 7B models. Key differentiator: SEAL learns *static per-head scales offline* (training-light, task-format-specific); the project is fully training-free and adapts temperature *online* per decode step from entropy.

**Attrieval: Attention Reveals More Than Tokens (Training-Free Long-Context Reasoning)**
Zhang et al. arXiv:2503.09819, 2025. **LOW THREAT.**
Training-free; uses CoT-token attention weights to retrieve implicit facts and re-inject them into the reasoning process. Key differentiator: retrieves and re-inserts facts into the prompt rather than modifying attention temperature; complementary.

**EDT: Improving LLMs' Generation by Entropy-based Dynamic Temperature Sampling**
arXiv:2403.14541, 2024. **LOW THREAT.**
Entropy-based dynamic *sampling* temperature (output softmax), not attention, not long-context-specific. Supports entropy-as-inference-signal concept; clearly distinct in object (output vs. attention).

---

## Empirical Insights

### Already collected

**MRCR full trace (Image 1 — success vs. failure entropy)**
For multi-needle reasoning tasks, success cases have *higher* attention entropy (~0.50) than failure cases (~0.45) throughout the full trace. Entropy drifts upward from the beginning (confirming smooth prefill→decode transition). Supports: task-entropy taxonomy (MRCR needs broad attention, opposite of RULER); scope characterization.

**MRCR output entropy vs. instability scatter (Image 2 — 128K Qwen2B)**
Low output entropy + low instability = confident (both pass); high output entropy + high instability = confused (both fail). Supports: output entropy as an additional gating signal for long-generation tasks. Note: MRCR-specific, not directly applicable to RULER-scoped paper.

**Early decode instability (Image 3 — norm entropy trace, 32–64K)**
First 10–20 decode steps show the highest entropy instability (especially for failure cases), which then rapidly stabilizes. Supports: having a calibrated target during the early generation window helps; motivates both session target and per-sample early-decode calibration.

**Normalized attention entropy vs. context length (Image 4 — LLaMA QA)**
H/log(kv_len) follows a clean decreasing log-linear trend from 4K to 65K for both qa_1 and qa_2. At 131K the point breaks *above* the trend (hollow outlier). Interpretation: within 4K–65K the model partially compensates for denominator dilution (absolute entropy grows slower than log N); at 128K the compensation breaks down and entropy spikes above expectation. Supports: explanation for the 128K regression; model operating at/beyond comfortable range.

---

### Critical gaps (still needed)

**Gap 1 — RULER success vs. failure attention entropy trace (MUST HAVE)**
The foundational claim "attention entropy is the right signal for retrieval tasks" has no organic support yet. Need: for RULER CWE or qa_1/qa_2, separate baseline samples by outcome (sample-level success vs. failure), plot average H_attn(t)/H_ema(t) trace across decode steps for each group. If failed retrievals show systematically higher entropy ratios, this validates attention entropy as the right diagnostic signal for the project's target task regime. This is the RULER-equivalent of Image 1.

Also produce: ctrl-only hit comparison (samples flipped from wrong to correct by the controller vs. samples that stayed wrong). This shows what entropy profile is associated with cases where correction specifically helped — the strongest version of the evidence for the gate design.

**Strongest candidate: CWE at 32K.** The baseline collapses to 0.72% at 32K (complete model failure) while the controller recovers to 27.18%. This is the most dramatic before/after in the entire experiment set, making it the highest-signal example for the success vs. failure entropy trace. Failed baseline samples at 32K represent cases where attention diffusion has caused total retrieval breakdown — their entropy traces should show the most clearly elevated H_attn/H_ema ratios relative to the few success cases. Run Gap 1 on CWE 32K first.

**Gap 2 — Temperature actually changes attention entropy (MUST HAVE)**
The mechanism claim requires: a paired plot showing entropy trace of baseline vs. controller-on for the same examples. Confirms the actuator does what it claims (temperature reduction reduces attention entropy during decode). Without this, the controller could be improving performance through an unrelated mechanism.

**How to produce this plot:**

*Core plot — paired mean entropy trace.* Run the same set of examples under two conditions (baseline SDPA and controller active) and record H_attn at every decode step. Plot two lines vs. decode step: mean H_attn(t) ± 1 std shading for baseline, and the same for controller. If the controller line is consistently below the baseline line, Gap 2 is closed. Use unnormalized absolute H_attn (not H/log(kv_len)) — the claim is about absolute entropy reduction and normalization would obscure the magnitude.

*Which examples to use.* CWE 32K ctrl-only hits (samples flipped from wrong to correct by the controller) are the strongest choice — the mechanism demonstrably mattered to the final output in these cases, and the baseline entropy at 32K is most pathologically elevated. If ctrl-only hits are too few for stable averages, use the full CWE 32K example set. Run at 65K as a secondary condition to show the effect persists at longer context.

*Optional second panel — temperature trace aligned with entropy.* A two-panel figure is more compelling: left panel shows the entropy trace (baseline vs. controller), right panel shows the mean temperature T(t) applied by the controller at each decode step, both on the same x-axis. When temperature drops below 1.0, entropy should visibly drop in the left panel. This directly demonstrates the causal mechanism. Alternatively, use a dual y-axis (entropy left, temperature right) to compress into one figure.

*Optional scatter version (supplementary).* Per-sample scatter: x = mean baseline entropy, y = mean controller entropy, one point per example. Points clustering below the diagonal (y < x) show every sample's attention entropy was reduced. Useful as a sanity check or appendix figure.

*Practical note.* RULER/HELMET short-output tasks generate only 10–30 decode steps — this is fine. Even 5–10 steps consistently showing entropy depression is sufficient evidence for the mechanism claim.

**Gap 3 — Prompt tail entropy as calibration anchor (IMPORTANT)**
For the design choice of using prompt tail as the target for short generation: compare the attention entropy of the last N prompt tokens (tail of prefill) against the attention entropy of successful decode steps for RULER tasks. If they are in a similar range, prompt tail is a natural anchor. Alternatively: show that examples with lower prompt tail entropy tend to succeed more at a given context length.

**Gap 4 — Step gate calibration (NEW)**
For the relative-threshold step gate (H_attn(t) > H_ema(t) × (1+ε)): separate decode steps by token type — format tokens (punctuation, whitespace, structural) vs. content tokens (answer tokens, entity names, numbers). Plot H_attn/H_ema ratio distribution for each type. Format tokens should cluster below 1+ε; content tokens above it. This empirically calibrates ε and validates the step gate design without requiring a session-level target.

---

### RULER Aggregation Results (LLaMA3.1-8B) — May 31

#### FWE (Frequent Word Extraction)

| Context Length | Baseline (SDPA) | Scaled Session (50/session) | Delta |
|---|---|---|---|
| 4096 | 88.80% | 90.33% | +1.53% |
| 8192 | 94.87% | 95.20% | +0.33% |
| 16384 | 94.73% | 90.33% | -4.40% |
| 32768 | 86.00% | 84.80% | -1.20% |
| 65536 | 87.33% | 82.60% (82% max temp 1.2) | -4.73% |
| 131072 | 73.80% | 74.60% | +0.80% |

#### CWE (Common Word Extraction)

| Context Length | Baseline (SDPA) | Scaled Session (50/session) | Delta |
|---|---|---|---|
| 4096 | 99.18% | 97.58% | -1.60% |
| 8192 | 94.70% | 88.32% | -6.38% |
| 16384 | 59.06% | 71.52% | +12.46% |
| 32768 | 0.72% | **27.18%** (23.32% max temp 1.2) | **+26.46%** |
| 65536 | 0.08% | 1.62% | +1.54% |
| 131072 | 0.00% | 1.50% | +1.50% |

#### Key observations

**CWE 32K: dramatic rescue from near-zero baseline.** The controller recovers from essentially complete model failure (0.72%) to 27.18% — the largest absolute gain in the entire experiment set. This is the strongest existing evidence that the controller can rescue performance when attention diffusion has caused catastrophic degradation.

**CWE regressions at 4K and 8K.** Negative results at short contexts where baseline is high (99.18%, 94.70%). This is consistent with the general pattern: at short contexts, the model is not suffering from denominator dilution, the baseline is already near-ceiling, and controller intervention is unnecessary or slightly harmful. Supports the scope condition.

**FWE regressions at 16K–65K.** FWE baselines remain high at these lengths (86–94%), and the controller shows modest regressions. FWE may require a different attention profile than CWE — FWE asks for the most *frequent* word, which might require broader distributed attention to count occurrences, whereas CWE may benefit more from focused retrieval. FWE's consistently high baseline suggests the model handles it robustly, leaving little room for the controller to add value and some risk of harmful sharpening.

**CWE vs. FWE: different task structure, different entropy regime.** CWE collapses at 32K+ while FWE stays stable. This is a meaningful task-entropy difference. CWE likely requires focused attention to identify specific word patterns at long range — exactly where denominator dilution causes failure. FWE's robustness may come from a counting/frequency signal that survives diffuse attention better. The CWE collapse pattern is the clearest single demonstration of the problem the controller is designed to address.

**max temp 1.2 consistently underperforms default.** At CWE 32K, default setting (27.18%) outperforms max temp 1.2 (23.32%). At FWE 65K, 82.6% vs. 82% — marginal but consistent direction. Confirms that allowing temperature to exceed 1.0 (increasing entropy) is harmful and the upper temperature bound should be ≤ 1.0.

#### Mentor suggestion — applying CWE 32K settings to other datasets

**Suggestion:** apply the settings from the successful CWE 32K experiment to other datasets to see if similar gains appear.

**Clarification on what is and is not transferable:**

The specific calibrated temperature and entropy target values derived from the first N CWE 32K session samples are *not* portable across tasks and context lengths — each session generates its own calibration from its own initial samples. This part does not transfer and does not need to.

What *is* transferable (and already shared across all experiments) are the hyperparameters: EMA lag α, session size, tail length, temperature bounds. These are identical across all existing runs.

**What the suggestion is really asking:** find other tasks/context-length combinations where the baseline has similarly collapsed (near-zero performance), and check whether the controller achieves a comparable rescue. CWE 32K is interesting because the gain is so large precisely because the baseline had nowhere to go but up. The question is whether other such collapse cases exist in other tasks at similar context lengths.

**Candidate investigation:** run the controller on other RULER subtasks (qa_1, qa_2) at 32K–65K and examine whether the tasks that show the largest baseline degradation (rather than just moderate decline) correspond to the largest absolute controller gains. CWE's 0.72% baseline at 32K is a qualitatively different failure mode from qa_2's 50% at 32K — the CWE failure is complete model failure, not gradual degradation. The controller's rescue effect may be specific to this complete-failure regime.

---

## Controller Design

### Overview

Training-free, inference-time attention temperature controller. Uses EMA of attention entropy as a feedback signal to dynamically adjust the softmax temperature of the attention mechanism, counteracting attention dispersion/diffusion that worsens as context length grows. No retraining required; applies to any deployed full-attention transformer.

**Core novelty:** EMA of *attention* entropy → *attention-softmax* temperature, training-free, self-calibrating target. Prior work either (a) uses output entropy for other control objectives (EAD, EDEN, EDT), (b) uses static length-based attention scaling (SSMax, YaRN), (c) does token selection using retrieval-head mass (DySCO), or (d) replaces softmax via retraining (ASEntmax). The closed-loop feedback on attention entropy specifically is novel.

---

### Component 1 — EMA Controller

The core control signal is the exponential moving average of per-step attention entropy across decode steps.

```
H_ema(t) = α × H_ema(t-1) + (1-α) × H_attn(t)
T(t) = f(H_ema(t), target)
```

The EMA serves two roles simultaneously: (1) smooths token-type noise (format tokens have structurally different entropy than content tokens), (2) tracks the slow monotonic entropy drift from denominator dilution accumulating over decode steps. The EMA's temporal lag is a feature: it responds to sustained diffusion (pathological) while ignoring transient spikes (active scanning, synthesis tokens).

Key empirical finding: the controller exhibits systematic negative tracking error — entropy ends up below the session-calibrated target. This is evidence that prefill entropy is a biased proxy for optimal decode entropy, and the EMA momentum accidentally corrects for this bias. The §5.2 self-calibrating target makes the correction explicit.

---

### Component 2 — Calibration: Session Target (short generation)

**Design:** use session-level prefill tail entropy as the calibration anchor. Session EMA carries over across samples within a session (e.g., 50 samples per session), providing stable calibration that reflects the typical entropy level at that context length and task type.

**Empirical basis:** session target works on RULER; per-sample prompt tail (individual sample, no carryover) does not. Individual sample prompt tails are too noisy to give a stable estimate; session-level averaging provides the required smoothing.

**Applicable to:** RULER/HELMET-type tasks where generation ends before N steps. The session target is used throughout the entire generation.

**128K regression finding:** at exactly LLaMA3.1's native context ceiling (128K), the session-calibrated target becomes miscalibrated — prefill entropy at 128K is too high relative to what optimal decoding needs, causing the controller to over-correct. This is not a failure of the mechanism but a scope condition: the controller is most effective when operating below the model's native trained context ceiling. LLaMA3-YaRN-256k at 128K restores positive performance because 128K is within its comfortable operating range (50% of its 256K ceiling).

---

### Component 3 — Calibration: Per-Sample Early-Decode (long generation)

**Design:** if generation exceeds N decode steps (threshold), observe the entropy distribution from steps 0–N for the current example, compute a per-sample target from those observations, and override the session target for the remainder of generation.

**Applicable to:** long-generation tasks (MRCR-type) where generation exceeds N steps. Provides example-specific calibration that adapts to the task's actual entropy requirements.

**Key insight:** per-sample calibration is only reliable when enough decode steps have accumulated to estimate stable entropy. For short-output tasks, generation ends before N steps, so only the session target applies. For long-output tasks, N steps give a sufficient observation window.

**Smooth transition:** blend from session target to per-sample target over steps N to N+k to avoid discontinuity if the two targets differ significantly.

**Sliding window recalibration:** in the long-generation mode, periodically re-estimate the target using a sliding window over recent decode steps to handle ongoing entropy drift throughout generation.

**Threshold N:** the single hyperparameter that separates short-output and long-generation regimes. Should be chosen larger than the maximum expected output length for short-answer tasks (e.g., N=30 cleanly separates RULER outputs from MRCR outputs). N also functions as the minimum observation window for reliable per-sample calibration.

---

### Component 4 — Head Gate (retrieval heads only)

**Design:** apply temperature correction only to identified retrieval heads. Non-retrieval heads (syntactic, positional, copy, attention-sink) receive T=1.0.

**Motivation:** retrieval heads are <5% of all attention heads but causally responsible for long-context factuality. Applying correction uniformly modifies heads with different functions where sharpening is either neutral or harmful. The head gate concentrates the intervention where it matters.

**Offline identification (NIAH diagnostic):**
1. Generate 200–300 NIAH examples with varying context lengths (4K–128K) and randomized needle positions.
2. Record attention weight each (layer, head) assigns to the needle token at the retrieval decode step.
3. Average needle attention weight across examples to get a retrieval score per head.
4. Select top 5% as retrieval heads; create binary mask `retrieval_mask[layer, head]`.
5. Mask is fixed and loaded at inference time (run once per model).

**Note on GQA:** LLaMA3.1 uses grouped-query attention; head indexing refers to query heads consistently.

**Layer gate:** not needed for full-attention models (LLaMA3.1, Qwen3). Only applicable if moving to hybrid architectures (Qwen3.5-style with Gated DeltaNet layers).

---

### Component 5 — Step Gate (relative entropy threshold)

**Design:** at each decode step, decide whether to fire the temperature correction. Suppresses intervention on format/structural tokens where applying sharpening is harmful or meaningless.

**Signal:** attention entropy relative threshold (not absolute). Gate fires when current step's attention entropy exceeds the recent EMA:

```
gate_fires = H_attn(t) > H_ema(t) × (1 + ε)
```

**Rationale for relative vs. absolute threshold:** using the session-calibrated absolute target would duplicate the controller logic. The relative threshold asks "is this step's attention more diffuse than usual?" rather than "is attention above the session target?" — this detects spikes above the running trend without requiring external calibration. ε is a single robust hyperparameter (e.g., 0.1 = fire when current step is 10% above EMA).

**Why not output entropy as the primary gate signal:** output entropy is a downstream proxy for attention entropy via an indirect chain (high output entropy → model uncertain → hasn't retrieved well → attention diffuse). This chain breaks for format tokens with incidentally high output entropy. Attention entropy directly measures what the actuator controls, eliminating the signal gap.

**Handling the three step types:**
- Format/structural tokens: naturally low H_attn, H_attn/H_ema < 1+ε, gate does not fire. Correct.
- Active retrieval scanning: transient spike then drop. Gate fires during spike, correction applied, entropy resolves.
- Pathological diffusion: sustained H_attn > H_ema, gate keeps firing, correction sustained. Correct.

**EMA update:** happens every step regardless of whether the gate fired, to keep the estimate current.

**Optional secondary gate:** output entropy threshold as a format-token catch:
```
gate_fires = (H_attn(t) > H_ema(t) × (1+ε)) AND (H_output(t) > θ)
```
θ calibrated from empirical distribution of output entropy across format vs. content tokens on RULER tasks.

---

### Combined gate pseudocode

```python
for step t in generation:
    H_output = entropy(next_token_logits)           # fast, no overhead
    H_attn_current = compute_attention_entropy()    # current step

    # Step gate (relative threshold)
    gate_fires = H_attn_current > H_ema * (1 + epsilon)
    # Optional: add output entropy secondary gate
    # gate_fires = gate_fires AND (H_output > theta)

    if gate_fires:
        for each (layer, head):
            if retrieval_mask[layer, head]:          # head gate
                temperature[layer, head] = T_ema    # apply correction
            else:
                temperature[layer, head] = 1.0      # leave unchanged

    # Always update EMA (even on format steps)
    H_ema = alpha * H_ema + (1 - alpha) * H_attn_current
    T_ema = controller(H_ema, target)

    # Check for phase switch (long-generation mode)
    if t == N:
        per_sample_target = estimate_from_steps(0, N)
        target = blend(session_target, per_sample_target, window=k)
        enable_sliding_window_recalibration = True
```

---

## Datasets

### RULER (primary)
Synthetic long-context benchmark. Tasks used: single-hop QA (qa_1, SQuAD-augmented), multi-hop QA (qa_2, HotpotQA-augmented), CWE (common word extraction), FWE (frequent word extraction). All tasks: long prefill (4K–128K), short output (single token or short phrase), selective retrieval. Controller shows consistent positive gains 4K–65K, regression at 128K with LLaMA3.1-8B (explained by ceiling calibration breakdown), restored with LLaMA3-YaRN-256k.

### MRCR
Multi-needle retrieval with reasoning component. Shorter prefill, longer output generation. Success requires simultaneous multi-anchor attention across multiple needle positions — broad attention is correct, making the controller's sharpening intervention structurally incompatible. Used for: task-entropy taxonomy and scope characterization (success cases need *higher* entropy than failure cases, opposite of RULER). Retained as the scope boundary analysis section, not main results.

### HELMET (secondary — proposed)
ICLR 2025. Covers 7 categories: Recall (synthetic), RAG, Passage Re-ranking, LongQA, Summarization, ICL, Cite. Key property: **HELMET contains RULER subtasks natively** — ruler_cwe, ruler_fwe, ruler_qa1, ruler_qa2 are direct data splits. Also includes real-document RAG tasks (KILT-NQ, KILT-TriviaQA, KILT-HotpotQA, PopQA) with short answers.

Recommended subtasks for main results: Recall + RAG (KILT) — short-answer retrieval from long real-document context, extends RULER finding to natural data. Summarization and Cite subtasks should be excluded from main results (long output, different regime) but could mirror the MRCR scope-boundary finding within HELMET.

Supports: generalization claim beyond synthetic tasks; both Recall and RAG subtasks directly match existing RULER experiment setup.

---

## Models

### LLaMA3.1-8B (primary)
Native context: **128K** (full native training, not YaRN extension). Architecture: full attention all layers, GQA (grouped-query attention). All layers are softmax attention — no hybrid architecture concerns. Controller regression observed at exactly 128K: this is the native ceiling where calibration breaks down. Main results: 4K–65K positive gains, 128K regression.

### LLaMA3-8B-YaRN-256k (primary — 128K supplementary)
Context extended to 256K via YaRN. At 128K, this model operates at 50% of its ceiling — comfortable mid-range. Positive controller performance restored at 128K (+3.2 qa_1, +5.4 qa_2). Used to: validate the ceiling calibration hypothesis; demonstrates controller works when 128K is within the model's comfortable range.

### Qwen3-8B or Qwen3-4B (secondary — proposed)
Native context: **32K**. Extended to 128K via YaRN. Architecture: full attention all layers (all 28/36 layers are standard softmax attention — no Gated DeltaNet). Comparable to LLaMA3.1 for architecture: same full-attention structure. At 128K operates in YaRN-extended range (analogous to LLaMA3-YaRN-256k at 128K).
Recommended for: clean replication of RULER/HELMET results on a second model family to establish generalization across architectures without hybrid-architecture confounds.
Note: Qwen3 native context is 32K (confirmed via HuggingFace model card). The 128K window requires YaRN.

### LLaMA3.2-3B (not recommended for main experiments)
Created via structured pruning + knowledge distillation from LLaMA3.1-8B; optimized for on-device/edge tasks (summarization, instruction following, rewriting). Not designed for long-context retrieval. Baseline performance on RULER/HELMET at 128K likely too weak to show meaningful controller gains. Deprioritized in favor of Qwen3.

---

## Baselines

| Baseline | What it rules out | Implementation cost |
|---|---|---|
| Plain inference (SDPA) | Controller does nothing | Already have |
| Fixed temperature (e.g., T=0.8, T=0.9) | Adaptive control is unnecessary | Trivial |
| Static length scaling (log(N)-based) | EMA feedback adds nothing over open-loop | Simple |
| DySCO | Token-selection approach is better for this task type | Feasible (training-free) |

**Priority:** fixed temperature is the most critical baseline to run first — lowest implementation cost and highest likelihood of reviewer objection. If fixed temperature achieves the same gains as the adaptive EMA controller, the contribution claim requires revision.

DySCO is mandatory for ICLR submission given the same setting (training-free, 128K, overlapping models). Key differentiation to demonstrate: DySCO requires a partial forward pass per step; this controller does not. On RULER-type tasks where DySCO may not be the right tool (it was evaluated on MRCR/LongBenchV2), relative performance is informative.

---

## Ablation

### Component Ablations (Type 1 — design choice validation)

One table showing the contribution of each design component. Components to ablate:

| Variant | Component removed | What it tests |
|---|---|---|
| Full controller | — | Baseline for comparison |
| Fixed temperature | Remove EMA, use constant T | Is adaptive control necessary? |
| No target (free-running) | Remove calibration target | Does having a target help? |
| Uniform application | Remove head gate | Does selective (retrieval-head-only) application help? |
| No step gate | Remove step gate | Does per-step gating help? |

### Sensitivity Analysis (Type 2 — robustness)

One table or figure showing performance is stable within a reasonable parameter range.

**High priority parameters:**
- EMA lag α: sweep {0.7, 0.8, 0.9}. Most theoretically motivated; directly tied to spike-smoothing vs. drift-tracking tradeoff. Should be robust within this range.
- Session size: sweep {5, 20, 50}. Validates the carryover memory finding — performance should degrade with small sessions. Provides empirical support for the session design choice.

**Medium priority:**
- Tail length: sweep {16, 32, 64} tokens. Inherited from SnapKV (default w=32); validate results are stable in that range.

**Appendix (only if long-generation results included):**
- N initial turns (phase switch threshold): relevant only for long-generation mode
- Window size (sliding window recalibration interval): relevant only for long-generation mode

---

## Positioning and Venue

**Target venue:** ICLR 2026 (submission deadline ~September/October 2026)

**Contribution claim:** first training-free, inference-time, closed-loop controller using an EMA of *attention entropy* to set the *attention-softmax temperature* with a self-calibrating target. Prior work has (a) identified attention entropy as a diagnostic of long-context degradation (Zhang ACL 2025), (b) used it as an allocation signal (CAKE, PyramidKV), but no prior work has used it as a feedback signal to directly control the attention mechanism itself via temperature.

**Scope:** long-prefill selective retrieval tasks (RULER, HELMET Recall+RAG). The controller improves performance when: context is long enough to cause denominator dilution, output is short (retrieval-type), and task requires selective single-focus rather than simultaneous multi-anchor attention. Does not apply to multi-anchor tasks (MRCR) or models operating at their native context ceiling (LLaMA3.1 at 128K). Scope limitations are documented and explained mechanistically.

**Key differentiators from closest competitors:**
- vs. DySCO: signal is attention entropy (not retrieval-head mass); actuator is continuous temperature scalar (not discrete token selection); no extra partial forward pass required
- vs. ASEntmax: training-free (ASEntmax requires retraining and replaces softmax)
- vs. Entropic-Time Inference: actuator is attention softmax temperature (not output/sampling softmax temperature)
- vs. Static scaling (SSMax, YaRN): entropy-indexed closed-loop (not length-indexed open-loop)

---

## Temperature Controller Implementation Review

### Current design (stateful accumulation)

Code reviewed: `entropy_scaling.py` — `EntropyTempController` class.

```
T(t) = T(t-1) + clip(-kp × err, ±max_step)
     = T(t-1) + clip(-0.35 × (H_ema - target), ±0.005)
```

Current parameters: `temp_init=1.0`, `temp_min=0.7`, `temp_max=1.0`, `ema_beta=0.7`, `kp=0.35`, `max_step=0.005`.

**Key observations from code review:**

`kp` is effectively unused for most meaningful errors. The binding condition for max_step is `kp × err > max_step`, i.e., `err > 0.005/0.35 = 0.014`. Any meaningful entropy error exceeds 0.014, so `delta` is clamped to `±max_step` at almost every step. In practice the controller has one effective parameter (max_step), not two (kp + max_step). kp only matters near convergence when error is tiny.

`ema_beta = 0.7` convention: in the code, beta is the weight on the OLD value — `ema = 0.7 × old + 0.3 × new`. Higher beta = more smoothing / slower response. The ablation should test `beta ∈ {0.7, 0.9}` using this convention.

`dead_band` is already the step gate mechanism — just disabled by default (`None`). Enabling it with a small value directly implements the relative-threshold step gate. However, the current dead_band uses an absolute threshold on error; the preferred design uses a relative threshold `H_attn(t) > H_ema(t) × (1+ε)`. This requires a small modification.

Temperature carries over across samples: `_init_state` is only called when `self.temp is None`, so if the same controller instance is reused across samples in a session, temperature state persists. This is what makes session carryover work — the temperature starting point for each new sample is wherever the previous sample left off, not 1.0.

**Movement limits per generation:**
- 5-token CWE output: max temperature movement = 5 × 0.005 = **0.025**
- 50-token QA output: max movement = 50 × 0.005 = **0.25** (can reach T_min from 1.0)
- For very short outputs, most of the correction must come from the session starting point, not within-sample dynamics.

---

### Proposed stateless design

```
T(t) = clip(1.0 - scale × max(0, H_ema(t) - target), T_min, T_max)
```

Temperature is computed directly from the current EMA entropy and target — no accumulation, no starting temperature dependency.

**Parameters removed:** `temp_init`, `kp`, `max_step`
**Parameter added:** `scale` (single parameter replacing two; direct interpretation: each 0.1 unit of excess entropy drops T by `scale × 0.1`)

**Key advantages over stateful:**
- Responds immediately to entropy error from step 1 (no warm-up needed). For a 5-token CWE output, stateful can only move T by 0.025; stateless immediately applies full correction on step 1.
- No starting temperature question — T is always freshly computed, not accumulated.
- One fewer parameter to tune (scale replaces kp + max_step).
- More interpretable: `scale=2.0` means an entropy excess of 0.08 produces T=0.84 immediately.

**Minimal code change:**
```python
# Replace the delta/add_ block with:
correction = self.scale * torch.clamp(err, min=0.0)
new_temp = torch.clamp(1.0 - correction, self.temp_min, self.temp_max)
self.temp = torch.where(valid, new_temp, torch.ones_like(new_temp))
```

`clamp(err, min=0.0)` ensures the controller only sharpens — it never raises temperature above 1.0 even if entropy falls below target, consistent with `temp_max=1.0` being correct.

The EMA update, normalization, validity masking, and dead_band logic all stay identical.

**When to prefer stateful:** if session carryover of temperature state is intentional and beneficial (the accumulated T from previous samples is a useful prior for the next sample's starting temperature). In this case, stateful with a larger max_step (0.01–0.05) would be the variant to test, not keeping max_step at 0.005.

**Recommended next step:** test stateless (scale=2.0) vs. current stateful on CWE 32K. If stateless matches or exceeds performance, switch to it and drop two hyperparameters.

---

## Oracle Temperature Analysis via Teacher Forcing

### Concept

With teacher forcing, ground truth tokens are fed as input at each decode step regardless of what the model generated. The pre-softmax attention logits (QKᵀ/√d) are fixed at each step; only the temperature applied to them varies. Oracle temperature at step t is the T that minimizes cross-entropy loss on the correct next token.

This provides an empirical target function for the controller: what T should have been at each step, grounded in actual task performance.

### Two implementations

**Option A — Grid search (direct, interpretable):**
1. Run teacher-forced forward pass; save pre-softmax attention logits at every (layer, head, step).
2. For each step t, loop over T ∈ {0.70, 0.75, 0.80, ..., 1.00}.
3. Recompute softmax(logits/T) → attention output → propagate to vocabulary logits → P(ground_truth_token).
4. Oracle T* = argmax over T candidates.

Cost: N_steps × N_candidates forward passes through upper layers. Feasible for CWE (5–10 token outputs).

**Option B — Gradient-based oracle (single backward pass, faster):**

Make T a differentiable scalar, run teacher-forced forward, compute total cross-entropy loss, backpropagate to get ∂L/∂T at each step:

- ∂L/∂T < 0 at step t → decreasing T (sharpening) improves prediction → oracle T < 1.0, correction warranted
- ∂L/∂T ≈ 0 → temperature-insensitive → oracle T ≈ 1.0, no intervention needed (format token candidate)
- ∂L/∂T > 0 → sharpening is harmful at this step

Gradient magnitude gives sensitivity; sign gives direction. One backward pass covers the full sequence.

### What oracle analysis provides

**Scale calibration for stateless design:** plot oracle T* against (H_ema − target) across steps. The slope is the ideal `scale` parameter. If the relationship is approximately linear, `T = 1 − scale × error` fits well.

**Step gate validation:** oracle T* ≈ 1.0 (gradient ≈ 0) on format/structural token steps validates suppressing correction there. Oracle T* < 1.0 on content/retrieval steps validates firing the gate there.

**Head gate validation:** run gradient analysis per-head (separate T per head). Heads where ∂L/∂T is consistently negative across content steps at long contexts are the retrieval heads that benefit from sharpening — alternative to NIAH diagnostic, and can be cross-validated against it.

**Gap 2 evidence:** plot controller T trajectory alongside oracle T trajectory for the same CWE 32K examples. If they track each other directionally on content steps, this is mechanistic validation the controller moves temperature in the right direction.

### Best target

CWE at 32K. Baseline collapses to 0.72%, controller recovers to 27.18% — the most dramatic effect in the experiment set. The oracle analysis would show what T was actually needed at each generation step for the correct answer, and whether the controller's T trajectory approximated it.

### Scaling function design from oracle data

The oracle analysis produces a dataset of `(features_t, T*(t))` pairs across all steps and examples. The scaling function is a model fitted to this data. All features must be available at inference time — no ground truth, no future steps.

**Feature candidates (ordered by motivation):**

`err(t) = H_ema(t) − target` — primary feature. Direct predictor of whether correction is needed. The backbone of any scaling function. Current stateless formula uses this alone.

`context length N` — secondary feature. Denominator dilution grows with N; the same error magnitude at 64K warrants more aggressive correction than at 16K. If the oracle scatter shows slope varying with N, scale should be N-dependent:
```
scale(N) = base_scale × (1 + β × log(N / N_ref))
```
One extra parameter β; N_ref set to shortest evaluation context (e.g., 4K).

`rate of entropy change dH(t) = H_attn(t) − H_attn(t−1)` — tertiary feature. Detects active deterioration vs. stable entropy. Rising entropy at the same error level warrants more correction. Can enter as a second term:
```
T(t) = clip(1.0 − scale1 × max(0, err) − scale2 × max(0, dH), T_min, T_max)
```
Only add if oracle regression shows meaningful predictive power beyond err alone. For 5–10 token outputs, dH may be too noisy to be useful.

`per-head entropy` — relevant if using per-head temperature. The feature per head becomes `H_head(t) − target_head`. Retrieval heads are expected to need more aggressive scale than non-retrieval heads.

**Oracle regression procedure:**
1. Collect dataset: for each step t across all CWE 32K examples, record `(err(t), N, t, dH(t), T*(t))`.
2. Fit `T* = 1 − scale × max(0, err)` — get R² and residuals.
3. Add N as a feature — does R² improve meaningfully? If yes, scale is N-dependent.
4. Add dH — does it further improve fit? If marginal, exclude.
5. Check residual structure — if T* saturates at low values (oracle rarely goes below 0.75), a saturating function like tanh fits better than a linear clip.

Use the simplest function that explains the oracle data well. If err alone explains 90% of variance, use the one-parameter formula. If err + log(N) explains 95%, add the N term. Don't add features that explain less than a few percent of additional variance.

---

### Bridging oracle to real decoding: the discount factor

**Why oracle scale over-estimates the right real-decoding scale:**

Teacher forcing controls three things that are uncontrolled in real decoding:

*Error accumulation.* In teacher forcing, correct tokens are always fed at t+1. In real decoding, if correction at step t causes a wrong token, that wrong token enters the KV cache and shifts attention at t+1 away from the oracle-optimal state. Aggressive correction that works perfectly under teacher forcing may cause confident wrong retrievals that cascade.

*Distribution shift from the intervention.* The marginal oracle computes T* at step t assuming all other steps use T=1.0. In real deployment, T at step t−1 was already < 1.0, so the attention pattern at step t is already different from the oracle's baseline. The oracle T values were each computed in a slightly different world.

*Short generation constraint.* For 5-token CWE outputs, the oracle at step 1 was computed knowing steps 2–5 would use correct inputs. In real decoding with only 5 steps, step 1 has to do more work with no recovery available. The marginal oracle underestimates how aggressive step 1 should be under real conditions.

**The discount factor procedure:**

`scale_oracle` — the slope fitted from the oracle regression.

`best_actual_scale` — found through a separate empirical sweep: run the stateless controller with `scale ∈ {0.5, 0.75, 1.0, 1.25, 1.5} × scale_oracle` on actual generation (not teacher forcing) and pick the scale that maximizes task accuracy (CWE 32K score). This is not derived from the oracle; it is found by running the real system.

`discount_factor = best_actual_scale / scale_oracle` — quantifies how much to pull back from the oracle-optimal to account for uncontrolled real-decoding factors. If close to 1, sequential dependencies are mild and the oracle is a reliable guide. If 0.4–0.5, compounding errors in real decoding are significant and a substantially more conservative controller is needed.

For CWE at 32K specifically, expect the ratio to be moderate-to-low: very short outputs mean almost no recovery steps, so overcorrecting at step 1 is costly.

**The autocorrelation of T* advises EMA lag:** compute the step-to-step autocorrelation of T* sequences within examples. High autocorrelation (T* stays low for several content steps, then returns to 1.0 for a format step) → EMA should be slow (high ema_beta) to avoid reacting to individual format steps. Low autocorrelation (T* flips rapidly) → EMA should be faster. This gives a principled way to set ema_beta rather than treating it as a blind hyperparameter.

---

### Limitations

- Teacher forcing only works where ground truth tokens are known step by step — directly applicable to RULER/HELMET short-answer tasks.
- Oracle T at step t assumes all other steps use T=1.0 (marginal oracle, not jointly optimal). Jointly optimal T over all steps is intractable. The marginal oracle is a useful approximation.
- Gradient approach gives direction, not exact T*. Grid search needed for exact oracle values. For the paper, the gradient sign map is sufficient to validate directional correctness.
- scale_oracle will over-estimate the right real-decoding scale due to the three uncontrolled factors above. Always validate with the best_actual_scale sweep on real generation before adopting the oracle-derived scale.

---

## To-Do List (prioritized)

### Tier 1 — Critical path (needed for submission)

1. **[ANALYSIS] Run Gap 1: CWE 32K success vs. failure entropy trace**
   Separate baseline samples by outcome; plot average H_attn(t)/H_ema(t) trace for each group. Strongest candidate first (CWE 32K). This is the foundational mechanistic claim with no organic support yet. Unblocks the core narrative.

2. **[ANALYSIS] Run Gap 2: paired entropy trace (baseline vs. controller) on CWE 32K**
   Record H_attn at every decode step for both conditions on the same examples. Two-panel figure: entropy trace (left) + temperature trace (right). Confirms the actuator does what it claims.

3. **[EXPERIMENT] Run fixed temperature baseline on CWE and QA tasks**
   Test T ∈ {0.8, 0.85, 0.9} as constant temperature throughout generation. Most critical baseline — if fixed T matches controller gains, the adaptive EMA contribution needs revision. Trivial to implement.

4. **[EXPERIMENT] Test stateless controller on CWE 32K**
   Implement scale-based stateless design (`T = clip(1.0 − scale × max(0, H_ema − target), 0.7, 1.0)`). Compare against current stateful on CWE 32K and QA. If comparable or better, switch and drop two hyperparameters.

5. **[EXPERIMENT] Run static length-scaling baseline**
   Apply temperature as a deterministic function of log(N) (SSMax-style). Tests whether EMA feedback adds value over open-loop length correction.

6. **[EXPERIMENT] Run Qwen3 (4B or 8B) on HELMET ruler_cwe, ruler_qa1, ruler_qa2**
   Secondary model replication. Full-attention architecture comparable to LLaMA3.1. Establishes generalization across model families. Use HELMET rather than standalone RULER to position within a richer benchmark.

---

### Tier 2 — Strongly recommended (needed for full paper)

7. **[EXPERIMENT] Run HELMET RAG subtasks (KILT-NQ, KILT-TriviaQA) on LLaMA3.1**
   Extends RULER finding to real-document long-context QA. Short-answer format, long prefill — same regime as RULER but with naturally occurring documents. Strengthens generalization claim beyond synthetic tasks.

8. **[ANALYSIS] Oracle temperature via teacher forcing on CWE 32K**
   Gradient-based approach (single backward pass per example). Produces oracle T direction map; calibrates `scale` parameter; validates step gate and head gate designs empirically. Addresses Gap 2 more rigorously than the paired trace alone.

9. **[IMPLEMENTATION] NIAH diagnostic for retrieval head identification on LLaMA3.1-8B**
   Generate 200–300 NIAH examples (varying context length 4K–128K, randomized needle positions). Record per-head needle attention weight at retrieval step. Select top 5% as retrieval heads. Create binary mask `retrieval_mask[layer, head]`. Run once; reuse at inference.

10. **[IMPLEMENTATION] Enable and modify step gate in existing controller**
    Convert `dead_band` from absolute to relative threshold: fire when `H_attn(t) > H_ema(t) × (1+ε)`. Test with ε ∈ {0.05, 0.10, 0.15}. Calibrate from token-type entropy distribution on RULER.

11. **[EXPERIMENT] Run DySCO comparison on RULER CWE and QA tasks**
    Mandatory for ICLR submission. Run DySCO (training-free, same models) on RULER subtasks. Key differentiator to demonstrate: no partial forward pass overhead. DySCO was evaluated on MRCR/LongBenchV2; RULER is likely a better setting for the present controller.

---

### Tier 3 — Extensions and polish

12. **[IMPLEMENTATION] Implement head gate using NIAH mask**
    Add retrieval head mask to attention forward hook. Apply temperature correction only to retrieval heads; T=1.0 for all others. Run ablation: uniform vs. gated application (already in component ablation table).

13. **[ANALYSIS] Gap 3: prompt tail entropy as calibration anchor**
    Compare attention entropy of last N prompt tokens against attention entropy of successful decode steps for RULER tasks. Validates prompt tail as a natural calibration anchor.

14. **[ANALYSIS] Gap 4: step gate calibration from token-type distribution**
    Separate decode steps by token type (format vs. content). Plot H_attn/H_ema ratio distribution for each type. Calibrate ε empirically.

15. **[EXPERIMENT] Ablation: component ablations table**
    Run: full controller, fixed T, no target (free-running), uniform application (no head gate), no step gate. One table.

16. **[EXPERIMENT] Sensitivity analysis**
    Sweep: ema_beta ∈ {0.7, 0.9}, session size ∈ {5, 20, 50}, tail length ∈ {16, 32, 64}.

17. **[EXPERIMENT] Verify session carryover mechanism**
    Confirm: does resetting T to 1.0 at the start of each sample (while keeping session target) change CWE 32K performance? This isolates whether the gain comes from the session-calibrated starting temperature or from the session target alone.

18. **[EXPERIMENT] HELMET RAG subtasks on Qwen3**
    Secondary model on real-document tasks. Completes the 2×2 (two models × two benchmark types) main results table.

19. **[PAPER] Write analysis section**
    Incorporate MRCR scope characterization, 128K regression explanation, task-entropy taxonomy, and oracle temperature findings as the analysis/discussion section. These findings are already complete; needs write-up.

20. **[PAPER] Related work section**
    Use literature review categories as structure. Distinguish: foundation (problem), design support, competitors. Ensure DySCO, ASEntmax, Entropic-Time Inference are each clearly differentiated in one sentence each.
