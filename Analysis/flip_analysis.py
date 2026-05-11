"""
Flip analysis: for each example, classify as:
  both_correct    - baseline correct, controller correct
  helped          - baseline wrong,   controller correct  (controller flipped to right)
  hurt            - baseline correct, controller wrong    (controller flipped to wrong)
  both_wrong      - baseline wrong,   controller wrong

Per-category analysis:
  - step-0 normalized entropy from BASELINE log (natural attention entropy)
  - step-0 temp from CONTROLLER log (pre-activation level)
  - session position distribution (helped vs hurt)

Matching is positional (example IDs are empty in these runs).

Usage:
  python flip_analysis.py --out_dir logs/flip_analysis
"""

import argparse
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ── config: (bin, task, baseline_tag, controller_tag) ─────────────────────────

DATA_ROOT = "/c2/jenny/r3/RULER_outputs/llama3.1-8b-chat/synthetic"

CONFIGS = [
    dict(bin=32768, task="qa_1",
         baseline_tag="baseline_entropy_log_fixed_temp",
         controller_tag="50session_3cali_prefixON"),
    dict(bin=32768, task="qa_2",
         baseline_tag="baseline_entropy_log_fixed_temp",
         controller_tag="50_session_3_calibration_samples_compact"),
    dict(bin=65536, task="qa_1",
         baseline_tag="baseline_entropy_log_fixed_temp",
         controller_tag="50session_3cali_prefixON"),
    dict(bin=65536, task="qa_2",
         baseline_tag="baseline_entropy_log_fixed_temp",
         controller_tag="50session_3cali_prefixON"),
]

CATEGORIES  = ["both_correct", "helped", "hurt", "both_wrong"]
CAT_COLORS  = {"both_correct": "#4daf4a", "helped": "#377eb8",
               "hurt": "#e41a1c", "both_wrong": "#999999"}
CAT_LABELS  = {"both_correct": "Both correct", "helped": "Controller helped",
               "hurt": "Controller hurt", "both_wrong": "Both wrong"}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _pred_path(data_root, ctx_bin, task, impl, tag):
    return os.path.join(
        data_root, str(ctx_bin), "data_session_runs",
        task, f"{task}_{impl}_predictions_{tag}.jsonl"
    )

def _elog_path(data_root, ctx_bin, task, impl, tag):
    return os.path.join(
        data_root, str(ctx_bin), "data_session_runs",
        task, f"{task}_{impl}_entropy_log_{tag}.jsonl"
    )

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ── per-example feature extraction ────────────────────────────────────────────

def ruler_hit(pred: str, refs: List[str]) -> bool:
    p = (pred or "").lower()
    return any((r or "").lower() in p for r in refs)


def step0_features(elog_rows):
    """Return list of dicts with step-0 entropy + temp, one per sample (by position)."""
    feats = []
    for row in elog_rows:
        log = row.get("entropy_log", [])
        if not log:
            feats.append(None)
            continue
        s0 = log[0]
        kv  = s0.get("kv_len", 1)
        raw = s0.get("entropy_mean", float("nan"))
        norm = raw / math.log(max(kv, 2))
        feats.append({
            "norm_entropy": norm,
            "temp":          s0.get("temp_mean", float("nan")),
            "n_steps":       len(log),
            "session_idx":   row.get("session_idx", 0),
            "prompt_target": row.get("prompt_target_mean"),
        })
    return feats


# ── per-config analysis ────────────────────────────────────────────────────────

def analyze_config(cfg, data_root):
    ctx_bin        = cfg["bin"]
    task           = cfg["task"]
    baseline_tag   = cfg["baseline_tag"]
    controller_tag = cfg["controller_tag"]
    label          = f"{ctx_bin//1024}k_{task}"

    # predictions
    base_pred_path = _pred_path(data_root, ctx_bin, task, "entropy_attn", baseline_tag)
    ctrl_pred_path = _pred_path(data_root, ctx_bin, task, "entropy_attn", controller_tag)

    # entropy logs (both baseline and controller have entropy logs)
    base_elog_path = _elog_path(data_root, ctx_bin, task, "entropy_attn", baseline_tag)
    ctrl_elog_path = _elog_path(data_root, ctx_bin, task, "entropy_attn", controller_tag)

    for p in [base_pred_path, ctrl_pred_path, base_elog_path, ctrl_elog_path]:
        if not os.path.exists(p):
            print(f"  [SKIP {label}] missing: {p}")
            return None

    base_preds = load_jsonl(base_pred_path)
    ctrl_preds = load_jsonl(ctrl_pred_path)
    base_elogs = load_jsonl(base_elog_path)
    ctrl_elogs = load_jsonl(ctrl_elog_path)

    n = min(len(base_preds), len(ctrl_preds), len(base_elogs), len(ctrl_elogs))
    if n == 0:
        print(f"  [SKIP {label}] empty files")
        return None

    base_feats = step0_features(base_elogs[:n])
    ctrl_feats = step0_features(ctrl_elogs[:n])

    records = []
    for i in range(n):
        bp = base_preds[i]
        cp = ctrl_preds[i]
        refs = bp.get("outputs", bp.get("answers", []))
        if isinstance(refs, str):
            refs = [refs]

        b_hit = ruler_hit(bp.get("prediction", ""), refs)
        c_hit = ruler_hit(cp.get("prediction", ""), refs)

        if b_hit and c_hit:
            cat = "both_correct"
        elif not b_hit and c_hit:
            cat = "helped"
        elif b_hit and not c_hit:
            cat = "hurt"
        else:
            cat = "both_wrong"

        rec = {
            "idx":        i,
            "category":   cat,
            "session_idx": cp.get("_session_idx", ctrl_feats[i]["session_idx"] if ctrl_feats[i] else 0),
            "base_feat":  base_feats[i],
            "ctrl_feat":  ctrl_feats[i],
        }
        records.append(rec)

    counts = {c: sum(1 for r in records if r["category"] == c) for c in CATEGORIES}
    total  = sum(counts.values())

    print(f"\n  {label}  (n={total})")
    print(f"    {'category':<18} {'n':>5}  {'%':>6}")
    for c in CATEGORIES:
        print(f"    {CAT_LABELS[c]:<18} {counts[c]:>5}  {100*counts[c]/max(total,1):>5.1f}%")

    return {"label": label, "ctx_bin": ctx_bin, "task": task,
            "records": records, "counts": counts, "total": total}


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for res in all_results:
        if res is None:
            continue
        label   = res["label"]
        records = res["records"]
        counts  = res["counts"]
        total   = res["total"]

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f"Controller flip analysis — {label}", fontsize=14, y=0.98)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # ── panel 1: flip counts bar ──
        ax1 = fig.add_subplot(gs[0, 0])
        cats   = CATEGORIES
        ns     = [counts[c] for c in cats]
        colors = [CAT_COLORS[c] for c in cats]
        bars   = ax1.bar(range(len(cats)), ns, color=colors, edgecolor="white", lw=0.5)
        ax1.set_xticks(range(len(cats)))
        ax1.set_xticklabels([CAT_LABELS[c].replace(" ", "\n") for c in cats], fontsize=8)
        ax1.set_ylabel("Count")
        ax1.set_title(f"Flip counts (n={total})")
        for bar, n in zip(bars, ns):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{100*n/max(total,1):.1f}%", ha="center", va="bottom", fontsize=8)

        # ── panel 2: step-0 baseline entropy by category ──
        ax2 = fig.add_subplot(gs[0, 1])
        for ci, cat in enumerate(["both_correct", "helped", "hurt"]):
            vals = [r["base_feat"]["norm_entropy"]
                    for r in records
                    if r["category"] == cat and r["base_feat"] is not None
                    and not math.isnan(r["base_feat"]["norm_entropy"])]
            if vals:
                bp = ax2.boxplot(vals, positions=[ci], widths=0.5,
                                 patch_artist=True, showfliers=False,
                                 medianprops=dict(color="black", lw=2))
                for patch in bp["boxes"]:
                    patch.set_facecolor(CAT_COLORS[cat])
                    patch.set_alpha(0.7)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(
            [CAT_LABELS[c].replace(" ", "\n") for c in ["both_correct","helped","hurt"]],
            fontsize=8)
        ax2.set_ylabel("Step-0 norm entropy (baseline)")
        ax2.set_title("Natural entropy at decode step 0\n(baseline log, no controller)")

        # mark oracle prior if available
        ctrl_targets = [r["ctrl_feat"]["prompt_target"]
                        for r in records
                        if r["ctrl_feat"] and r["ctrl_feat"]["prompt_target"] is not None]
        if ctrl_targets:
            ax2.axhline(np.mean(ctrl_targets), color="black", ls="--", lw=1,
                        label=f"session target ≈{np.mean(ctrl_targets):.3f}")
            ax2.legend(fontsize=8)

        # ── panel 3: step-0 controller temp by category ──
        ax3 = fig.add_subplot(gs[0, 2])
        for ci, cat in enumerate(["both_correct", "helped", "hurt"]):
            vals = [r["ctrl_feat"]["temp"]
                    for r in records
                    if r["category"] == cat and r["ctrl_feat"] is not None
                    and not math.isnan(r["ctrl_feat"]["temp"])]
            if vals:
                bp = ax3.boxplot(vals, positions=[ci], widths=0.5,
                                 patch_artist=True, showfliers=False,
                                 medianprops=dict(color="black", lw=2))
                for patch in bp["boxes"]:
                    patch.set_facecolor(CAT_COLORS[cat])
                    patch.set_alpha(0.7)
        ax3.set_xticks(range(3))
        ax3.set_xticklabels(
            [CAT_LABELS[c].replace(" ", "\n") for c in ["both_correct","helped","hurt"]],
            fontsize=8)
        ax3.set_ylabel("Step-0 temperature (controller)")
        ax3.set_title("Controller temperature at step 0\n(pre-activation from session carryover)")
        ax3.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.6, label="temp=1.0 (no correction)")
        ax3.legend(fontsize=8)

        # ── panel 4: session position for helped vs hurt ──
        ax4 = fig.add_subplot(gs[1, 0])
        max_session = max((r["session_idx"] for r in records), default=0)
        bins = range(0, max_session + 2)
        for cat in ["helped", "hurt"]:
            pos = [r["session_idx"] for r in records if r["category"] == cat]
            if pos:
                ax4.hist(pos, bins=bins, alpha=0.6, color=CAT_COLORS[cat],
                         label=f"{CAT_LABELS[cat]} (n={len(pos)})", edgecolor="white")
        ax4.set_xlabel("Session index")
        ax4.set_ylabel("Count")
        ax4.set_title("Session position: helped vs hurt\n(early session = cold start)")
        ax4.legend(fontsize=8)

        # ── panel 5: entropy vs temp scatter for helped/hurt ──
        ax5 = fig.add_subplot(gs[1, 1])
        for cat in ["helped", "hurt", "both_correct"]:
            pts = [(r["base_feat"]["norm_entropy"], r["ctrl_feat"]["temp"])
                   for r in records
                   if r["category"] == cat
                   and r["base_feat"] and r["ctrl_feat"]
                   and not math.isnan(r["base_feat"]["norm_entropy"])
                   and not math.isnan(r["ctrl_feat"]["temp"])]
            if pts:
                xs, ys = zip(*pts)
                ax5.scatter(xs, ys, s=10, alpha=0.5, color=CAT_COLORS[cat],
                            label=f"{CAT_LABELS[cat]} (n={len(pts)})")
        ax5.set_xlabel("Step-0 baseline norm entropy")
        ax5.set_ylabel("Step-0 controller temp")
        ax5.set_title("Natural entropy vs pre-activation\n(step 0)")
        ax5.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax5.legend(fontsize=7, markerscale=2)

        # ── panel 6: within-session position (position within the 50-sample session) ──
        ax6 = fig.add_subplot(gs[1, 2])

        # compute position within session from global index and session_idx
        session_size_guess = 50
        for cat in ["helped", "hurt"]:
            pos_in_session = [r["idx"] % session_size_guess
                              for r in records if r["category"] == cat]
            if pos_in_session:
                ax6.hist(pos_in_session, bins=range(0, session_size_guess + 1, 5),
                         alpha=0.6, color=CAT_COLORS[cat],
                         label=f"{CAT_LABELS[cat]} (n={len(pos_in_session)})",
                         edgecolor="white")
        ax6.set_xlabel("Position within session (index mod 50)")
        ax6.set_ylabel("Count")
        ax6.set_title("Intra-session position: helped vs hurt\n(position 0-2 = cold start)")
        ax6.legend(fontsize=8)

        plt.savefig(os.path.join(out_dir, f"flip_analysis_{label}.png"),
                    dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  → saved flip_analysis_{label}.png")

    # ── combined summary across all configs ──
    valid = [r for r in all_results if r is not None]
    if len(valid) < 2:
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(5 * len(valid), 5), sharey=False)
    if len(valid) == 1:
        axes = [axes]

    for ax, res in zip(axes, valid):
        counts = res["counts"]
        total  = res["total"]
        cats   = CATEGORIES
        ns     = [counts[c] for c in cats]
        colors = [CAT_COLORS[c] for c in cats]
        bars   = ax.bar(range(len(cats)), [100*n/max(total,1) for n in ns],
                        color=colors, edgecolor="white")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([CAT_LABELS[c].replace(" ", "\n") for c in cats], fontsize=7)
        ax.set_ylabel("% of examples")
        ax.set_title(res["label"])
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    str(n), ha="center", va="bottom", fontsize=8)

    fig.suptitle("Flip analysis summary — all bins/tasks", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "flip_analysis_summary.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  → saved flip_analysis_summary.png")


# ── additional: step-0 entropy gap (helped vs hurt) stats ────────────────────

def print_entropy_gap_stats(all_results):
    print("\n── Step-0 entropy gap: helped vs hurt ──")
    print(f"{'config':<14}  {'helped_med':>10}  {'hurt_med':>9}  {'gap':>6}  "
          f"{'helped_temp_med':>15}  {'hurt_temp_med':>13}")
    print("-" * 80)
    for res in all_results:
        if res is None:
            continue
        records = res["records"]
        def meds(cat, key, feat_key):
            vals = [r[feat_key][key]
                    for r in records
                    if r["category"] == cat
                    and r[feat_key] and not math.isnan(r[feat_key][key])]
            return np.median(vals) if vals else float("nan")

        h_ent  = meds("helped", "norm_entropy", "base_feat")
        hu_ent = meds("hurt",   "norm_entropy", "base_feat")
        h_tmp  = meds("helped", "temp", "ctrl_feat")
        hu_tmp = meds("hurt",   "temp", "ctrl_feat")
        print(f"{res['label']:<14}  {h_ent:>10.4f}  {hu_ent:>9.4f}  "
              f"{h_ent-hu_ent:>+6.4f}  {h_tmp:>15.4f}  {hu_tmp:>13.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="logs/flip_analysis")
    ap.add_argument("--data_root", default=DATA_ROOT)
    args = ap.parse_args()

    print("Running flip analysis...")
    all_results = []
    for cfg in CONFIGS:
        print(f"\n  {cfg['bin']//1024}k {cfg['task']}")
        res = analyze_config(cfg, args.data_root)
        all_results.append(res)

    print_entropy_gap_stats(all_results)
    print("\nGenerating plots...")
    plot_results(all_results, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
