#!/usr/bin/env python3
"""
Generate all dissertation figures from existing data.

Inputs:
  - /scratch/nb01252/models/*/checkpoint-100/trainer_state.json  (9 training runs)
  - ~/dissertation/results/*_scores.jsonl                         (10 evaluation files)
  - ~/dissertation/results/*_scores_gemini.jsonl                  (7 Exp 1 Gemini files)
  - ~/dissertation/results/results_summary_full.json
  - ~/dissertation/results/judge_agreement.json

Outputs (all to ~/dissertation/results/figures/):
  - training_loss.png             — loss curves for all 9 runs
  - reward_margins.png            — std DPO vs aligned DPO saturation
  - grad_norm.png                 — std DPO vs LN-DPO vs aligned grad_norm
  - chosen_rewards.png            — LN-DPO reward inversion vs aligned positive rewards
  - rewards_chosen_rejected.png   — chosen vs rejected rewards over training (3×3 grid)
  - score_distributions.png       — per-condition score boxplots (all 10 conditions)
  - beta_ablation.png             — overall score vs beta for all 3 DPO variants
  - dimension_means.png           — bar chart of means per dimension per condition
  - aligned_vs_misaligned.png     — NEW: headline figure with significance markers
  - sonnet_vs_gemini_scatter.png  — NEW: judge validation scatter
  - word_count_distribution.png   — NEW: generation pathology visualization
  - condition_comparison_grid.png — NEW: 10-condition summary with stars
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

CONDITIONS_TRAIN = [
    "dpo_b01", "dpo_b03", "dpo_b05",
    "dpo_ln_b01", "dpo_ln_b03", "dpo_ln_b05",
    "dpo_aligned_b01", "dpo_aligned_b03", "dpo_aligned_b05",
]
CONDITIONS_EVAL = [
    "baseline",
    "dpo_b01", "dpo_b03", "dpo_b05",
    "dpo_ln_b01", "dpo_ln_b03", "dpo_ln_b05",
    "dpo_aligned_b01", "dpo_aligned_b03", "dpo_aligned_b05",
]

COLORS = {
    "baseline":         "#444444",
    "dpo_b01":          "#1f77b4",
    "dpo_b03":          "#3a8acb",
    "dpo_b05":          "#5fa5e0",
    "dpo_ln_b01":       "#d62728",
    "dpo_ln_b03":       "#e15454",
    "dpo_ln_b05":       "#ed8080",
    "dpo_aligned_b01":  "#2ca02c",
    "dpo_aligned_b03":  "#5fb95f",
    "dpo_aligned_b05":  "#8fd28f",
}

# Family grouping for legends and aggregate plots
FAMILIES = {
    "baseline":   ["baseline"],
    "misaligned": ["dpo_b01", "dpo_b03", "dpo_b05"],
    "ln_dpo":     ["dpo_ln_b01", "dpo_ln_b03", "dpo_ln_b05",
                         "dpo_aligned_b01", "dpo_aligned_b03", "dpo_aligned_b05"],
    "aligned":    ["dpo_aligned_b01", "dpo_aligned_b03", "dpo_aligned_b05"],
}
FAMILY_COLORS = {
    "baseline":   "#444444",
    "misaligned": "#1f77b4",
    "ln_dpo":     "#d62728",
    "aligned":    "#2ca02c",
}

DIMS = ["emotional_flexibility", "emotional_arc_coherence", "subtext_density"]
DIM_LABELS = {"emotional_flexibility": "EF", "emotional_arc_coherence": "EAC", "subtext_density": "SD"}


def load_trainer_state(cond, models_dir):
    path = os.path.join(models_dir, cond, "checkpoint-100", "trainer_state.json")
    if not os.path.exists(path):
        path = os.path.join(models_dir, cond, "final", "trainer_state.json")
    if not os.path.exists(path):
        print(f"  WARNING: trainer_state.json not found for {cond}")
        return None
    with open(path) as f:
        return json.load(f)


def extract_metric(ts, key):
    """Pull (steps, values) from log_history for a given metric key."""
    steps, vals = [], []
    for entry in ts.get("log_history", []):
        if key in entry and "step" in entry:
            steps.append(entry["step"])
            vals.append(entry[key])
    return steps, vals


def load_scores(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def get_viable(records):
    return [r for r in records
            if r.get("reasoning") != "empty_story"
            and all(r.get(d, -1) >= 0 for d in DIMS)]


def _get_linestyle(cond):
    if cond.startswith("dpo_ln"):
        return "--"
    if cond.startswith("dpo_aligned"):
        return "-."
    return "-"


def fig_training_loss(states, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in CONDITIONS_TRAIN:
        ts = states.get(cond)
        if ts is None: continue
        steps, vals = extract_metric(ts, "loss")
        if not steps: continue
        ax.plot(steps, vals, label=cond, color=COLORS[cond],
                linestyle=_get_linestyle(cond), linewidth=1.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("DPO training loss across conditions (β ∈ {0.1, 0.3, 0.5})")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def fig_reward_margins(states, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in CONDITIONS_TRAIN:
        if cond.startswith("dpo_ln"):
            continue  # margins not directly comparable for LN
        ts = states.get(cond)
        if ts is None: continue
        steps, vals = extract_metric(ts, "rewards/margins")
        if not steps: continue
        ax.plot(steps, vals, label=cond, color=COLORS[cond],
                linestyle=_get_linestyle(cond), linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward margin (β·(rᶜ − rʳ))")
    ax.set_title("DPO reward margins — misaligned saturates, aligned converges")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def fig_grad_norm(states, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in CONDITIONS_TRAIN:
        ts = states.get(cond)
        if ts is None: continue
        steps, vals = extract_metric(ts, "grad_norm")
        if not steps: continue
        # Clip very small values for log scale
        vals = [max(v, 1e-15) for v in vals]
        ax.plot(steps, vals, label=cond, color=COLORS[cond],
                linestyle=_get_linestyle(cond), linewidth=1.6)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient norm (log)")
    ax.set_yscale("log")
    ax.set_title("Gradient norm — std DPO saturates, LN-DPO stays healthy, aligned converges")
    ax.legend(loc="lower left", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def fig_chosen_rewards(states, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in CONDITIONS_TRAIN:
        ts = states.get(cond)
        if ts is None: continue
        steps, vals = extract_metric(ts, "rewards/chosen")
        if not steps: continue
        ax.plot(steps, vals, label=cond, color=COLORS[cond],
                linestyle=_get_linestyle(cond), linewidth=1.8)
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Chosen reward β·(log πᶜ − log π_ref^c)")
    ax.set_title("Chosen rewards — LN-DPO inverts negative (reward inversion failure mode)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def fig_rewards_chosen_rejected(states, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for i, cond in enumerate(CONDITIONS_TRAIN):
        ax = axes.flatten()[i]
        ts = states.get(cond)
        if ts is None:
            ax.set_title(f"{cond} (missing)")
            continue
        s_c, v_c = extract_metric(ts, "rewards/chosen")
        s_r, v_r = extract_metric(ts, "rewards/rejected")
        ax.plot(s_c, v_c, label="chosen", color="#2ca02c", linewidth=1.8)
        ax.plot(s_r, v_r, label="rejected", color="#d62728", linewidth=1.8)
        ax.axhline(0, color="black", linestyle=":", linewidth=0.6)
        ax.set_title(cond, fontsize=10)
        ax.set_xlabel("Step")
        if i % 3 == 0:
            ax.set_ylabel("Reward")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def fig_score_distributions(scores_by_cond, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, dim in zip(axes, DIMS):
        data = []
        labels = []
        for cond in CONDITIONS_EVAL:
            recs = scores_by_cond.get(cond, [])
            viable = get_viable(recs)
            if viable:
                data.append([r[dim] for r in viable])
                labels.append(cond.replace("dpo_", "").replace("aligned_", "ali_"))
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        cond_list = [c for c in CONDITIONS_EVAL if get_viable(scores_by_cond.get(c, []))]
        for patch, cond in zip(bp["boxes"], cond_list):
            patch.set_facecolor(COLORS.get(cond, "#888"))
            patch.set_alpha(0.7)
        ax.set_title(DIM_LABELS[dim])
        ax.set_ylim(-0.3, 3.3)
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Score (0–3)")
    fig.suptitle("Score distributions per condition (viable stories only)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def fig_dimension_means(scores_by_cond, out_path):
    fig, ax = plt.subplots(figsize=(13, 5))
    x_labels = CONDITIONS_EVAL
    n_dims = len(DIMS)
    bar_width = 0.25
    x = np.arange(len(x_labels))
    for i, dim in enumerate(DIMS):
        means = []
        stds = []
        for cond in x_labels:
            recs = scores_by_cond.get(cond, [])
            viable = get_viable(recs)
            if viable:
                vals = [r[dim] for r in viable]
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + i*bar_width - bar_width, means, bar_width, yerr=stds, label=DIM_LABELS[dim], capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("dpo_", "").replace("aligned_", "ali_") for c in x_labels],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean score (0–3)")
    ax.set_title("Mean rubric scores per condition (error bars = SEM)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def fig_beta_ablation(scores_by_cond, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    betas = [0.1, 0.3, 0.5]
    baseline_recs = get_viable(scores_by_cond.get("baseline", []))
    if baseline_recs:
        b_means = {d: np.mean([r[d] for r in baseline_recs]) for d in DIMS}
        b_overall = np.mean(list(b_means.values()))
        ax.axhline(b_overall, color="black", linestyle="--", label="Baseline", linewidth=1.2)
    for prefix, color, label in [("dpo_b", "#1f77b4", "Misaligned DPO"),
                                  ("dpo_ln_b", "#d62728", "LN-DPO (viable only)"),
                                  ("dpo_aligned_b", "#2ca02c", "Aligned DPO")]:
        overalls = []
        for b_str in ["01", "03", "05"]:
            cond = f"{prefix}{b_str}"
            recs = get_viable(scores_by_cond.get(cond, []))
            if recs:
                vals = [(r[DIMS[0]] + r[DIMS[1]] + r[DIMS[2]]) / 3 for r in recs]
                overalls.append(np.mean(vals))
            else:
                overalls.append(np.nan)
        ax.plot(betas, overalls, marker="o", color=color, label=label, linewidth=2, markersize=8)
    ax.set_xlabel("β (DPO temperature)")
    ax.set_ylabel("Overall mean score (0–3)")
    ax.set_title("β-ablation: overall emotional-depth score by training condition")
    ax.set_xticks(betas)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


# =============================================================================
# NEW FIGURES — Exp 2 additions
# =============================================================================

def _sig_stars(p):
    """Return significance stars for a p-value."""
    if p is None or p != p:  # NaN check
        return "ns"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def fig_aligned_vs_misaligned(scores_by_cond, summary_path, out_path):
    """Headline figure: baseline vs misaligned vs LN vs aligned, per dimension,
    with significance stars vs baseline."""
    summary = None
    if summary_path and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    pairwise = summary.get("pairwise_vs_baseline", {}) if summary else {}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    family_order = ["baseline", "misaligned", "ln_dpo", "aligned"]
    family_labels = {"baseline": "Baseline", "misaligned": "Misaligned\nDPO",
                     "ln_dpo": "LN-DPO", "aligned": "Aligned\nDPO"}

    for ax, dim in zip(axes, DIMS):
        means, sems, colors, labels = [], [], [], []
        for fam in family_order:
            vals = []
            for cond in FAMILIES[fam]:
                recs = get_viable(scores_by_cond.get(cond, []))
                vals.extend([r[dim] for r in recs])
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            else:
                means.append(0)
                sems.append(0)
            colors.append(FAMILY_COLORS[fam])
            labels.append(family_labels[fam])

        x = np.arange(len(family_order))
        bars = ax.bar(x, means, yerr=sems, color=colors, capsize=4, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 2.0)
        ax.set_title(DIM_LABELS[dim], fontsize=13)

        # Add significance stars for non-baseline families
        for i, fam in enumerate(family_order):
            if fam == "baseline":
                continue
            ps = []
            for cond in FAMILIES[fam]:
                if cond in pairwise and dim in pairwise[cond]:
                    p = pairwise[cond][dim].get("welch_p")
                    if p is not None:
                        ps.append(p)
            if ps:
                best_p = min(ps)
                stars = _sig_stars(best_p)
                ax.text(i, means[i] + sems[i] + 0.05, stars, ha="center", fontsize=12, fontweight="bold")

    axes[0].set_ylabel("Mean score (0–3)")
    fig.suptitle("Effect of training condition on emotional depth dimensions\n"
                 "(* p<0.05, ** p<0.01, *** p<0.001 vs baseline, Welch's t-test)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def fig_sonnet_vs_gemini_scatter(results_dir, out_path):
    """Scatter plot of Sonnet vs Gemini scores across all paired evaluations."""
    paired_conditions = ["baseline", "dpo_b01", "dpo_b03", "dpo_b05",
                         "dpo_ln_b01", "dpo_ln_b03", "dpo_ln_b05",
                         "dpo_aligned_b01", "dpo_aligned_b03", "dpo_aligned_b05"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, dim in zip(axes, DIMS):
        son_all, gem_all = [], []
        for cond in paired_conditions:
            son = load_scores(os.path.join(results_dir, f"{cond}_scores.jsonl"))
            gem = load_scores(os.path.join(results_dir, f"{cond}_scores_gemini.jsonl"))
            if not son or not gem:
                continue
            g_by_idx = {r["index"]: r for r in gem}
            for sr in son:
                idx = sr.get("index")
                if idx not in g_by_idx: continue
                gr = g_by_idx[idx]
                if any(sr.get(d, -1) < 0 or gr.get(d, -1) < 0 for d in DIMS):
                    continue
                son_all.append(sr[dim])
                gem_all.append(gr[dim])
        if not son_all:
            continue
        # Add jitter so points don't all stack on integer coordinates
        rng = np.random.RandomState(42)
        sx = np.array(son_all) + rng.normal(0, 0.06, size=len(son_all))
        gy = np.array(gem_all) + rng.normal(0, 0.06, size=len(gem_all))
        ax.scatter(sx, gy, alpha=0.4, s=18, color=FAMILY_COLORS["aligned"])
        ax.plot([-0.3, 3.3], [-0.3, 3.3], "k--", linewidth=0.8, alpha=0.6, label="y = x")
        r = np.corrcoef(son_all, gem_all)[0, 1] if len(son_all) > 1 else float("nan")
        ax.set_title(f"{DIM_LABELS[dim]}  (Pearson r={r:.2f}, n={len(son_all)})")
        ax.set_xlabel("Claude Sonnet 4.6 score")
        ax.set_ylabel("Gemini 2.5 Flash score" if ax is axes[0] else "")
        ax.set_xlim(-0.3, 3.3)
        ax.set_ylim(-0.3, 3.3)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2, 3])
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Inter-judge agreement: Sonnet vs Gemini scores per dimension", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def fig_word_count_distribution(results_dir, out_path):
    """Boxplot of word counts per condition — shows LN collapse and misaligned outliers."""
    story_files = {
        "baseline":         "baseline_stories.jsonl",
        "dpo_b01":          "dpo_b01_stories.jsonl",
        "dpo_b03":          "dpo_b03_stories.jsonl",
        "dpo_b05":          "dpo_b05_stories.jsonl",
        "dpo_ln_b01":       "dpo_ln_stories.jsonl",
        "dpo_ln_b03":       "dpo_ln_b03_stories.jsonl",
        "dpo_ln_b05":       "dpo_ln_b05_stories.jsonl",
        "dpo_aligned_b01":  "dpo_aligned_b01_stories.jsonl",
        "dpo_aligned_b03":  "dpo_aligned_b03_stories.jsonl",
        "dpo_aligned_b05":  "dpo_aligned_b05_stories.jsonl",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    data, labels, colors = [], [], []
    for cond in CONDITIONS_EVAL:
        path = os.path.join(results_dir, story_files[cond])
        recs = load_scores(path)
        if not recs:
            continue
        wcs = [len(r.get("story", "").split()) for r in recs]
        data.append(wcs)
        labels.append(cond.replace("dpo_", "").replace("aligned_", "ali_"))
        colors.append(COLORS.get(cond, "#888"))

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showfliers=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Story length (words)")
    ax.set_title("Generated story length per condition (LN-DPO conditions show collapse)")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(1000, color="black", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.text(0.5, 1020, "≈ baseline median", fontsize=8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def fig_condition_comparison_grid(scores_by_cond, summary_path, out_path):
    """All 10 conditions × 3 dimensions, with stars and CIs."""
    summary = None
    if summary_path and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    cond_stats = summary.get("condition_statistics", {}) if summary else {}

    fig, ax = plt.subplots(figsize=(14, 6))
    n_dims = len(DIMS)
    bar_width = 0.27
    x = np.arange(len(CONDITIONS_EVAL))

    for i, dim in enumerate(DIMS):
        means, cis_lo, cis_hi = [], [], []
        for cond in CONDITIONS_EVAL:
            cs = cond_stats.get(cond, {}).get(dim, {})
            m = cs.get("mean")
            ci = cs.get("ci_95", [None, None])
            if m is None:
                # Fallback: compute from raw scores
                recs = get_viable(scores_by_cond.get(cond, []))
                m = np.mean([r[dim] for r in recs]) if recs else 0
                ci = [m, m]
            means.append(m)
            cis_lo.append(m - ci[0])
            cis_hi.append(ci[1] - m)

        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, means, bar_width,
                      yerr=[cis_lo, cis_hi], capsize=2.5,
                      label=DIM_LABELS[dim], edgecolor="black", linewidth=0.4)

        # Significance stars vs baseline
        for j, cond in enumerate(CONDITIONS_EVAL):
            if cond == "baseline":
                continue
            cs = cond_stats.get(cond, {}).get(dim, {})
            p = cs.get("welch_p")
            stars = _sig_stars(p)
            if stars != "ns":
                ax.text(j + offset, means[j] + cis_hi[j] + 0.04, stars,
                        ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("dpo_", "").replace("aligned_", "ali_") for c in CONDITIONS_EVAL],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean score (0–3)")
    ax.set_title("All conditions × all dimensions (error bars = 95% bootstrap CI; "
                 "stars = Welch p vs baseline)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 2.0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="/scratch/nb01252/models")
    p.add_argument("--results_dir", default=os.path.expanduser("~/dissertation/results"))
    p.add_argument("--out_dir", default=os.path.expanduser("~/dissertation/results/figures"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading trainer states...")
    states = {}
    for cond in CONDITIONS_TRAIN:
        ts = load_trainer_state(cond, args.models_dir)
        if ts is not None:
            states[cond] = ts
            print(f"  loaded {cond}")

    print("\nLoading score files...")
    scores_by_cond = {}
    for cond in CONDITIONS_EVAL:
        path = os.path.join(args.results_dir, f"{cond}_scores.jsonl")
        recs = load_scores(path)
        if recs:
            scores_by_cond[cond] = recs
            print(f"  loaded {cond}: {len(recs)} records")

    summary_path = os.path.join(args.results_dir, "results_summary_full.json")

    print("\nGenerating figures...")
    if states:
        fig_training_loss(states, os.path.join(args.out_dir, "training_loss.png"))
        fig_reward_margins(states, os.path.join(args.out_dir, "reward_margins.png"))
        fig_grad_norm(states, os.path.join(args.out_dir, "grad_norm.png"))
        fig_chosen_rewards(states, os.path.join(args.out_dir, "chosen_rewards.png"))
        fig_rewards_chosen_rejected(states, os.path.join(args.out_dir, "rewards_chosen_rejected.png"))

    if scores_by_cond:
        fig_score_distributions(scores_by_cond, os.path.join(args.out_dir, "score_distributions.png"))
        fig_dimension_means(scores_by_cond, os.path.join(args.out_dir, "dimension_means.png"))
        fig_beta_ablation(scores_by_cond, os.path.join(args.out_dir, "beta_ablation.png"))
        fig_aligned_vs_misaligned(scores_by_cond, summary_path,
                                   os.path.join(args.out_dir, "aligned_vs_misaligned.png"))
        fig_condition_comparison_grid(scores_by_cond, summary_path,
                                       os.path.join(args.out_dir, "condition_comparison_grid.png"))

    # Judge agreement scatter — reads paired score files directly
    fig_sonnet_vs_gemini_scatter(args.results_dir,
                                  os.path.join(args.out_dir, "sonnet_vs_gemini_scatter.png"))

    # Word count distribution — reads story files directly
    fig_word_count_distribution(args.results_dir,
                                 os.path.join(args.out_dir, "word_count_distribution.png"))

    print(f"\nAll figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
