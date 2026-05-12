#!/usr/bin/env python3
"""
Generate all dissertation figures from existing data.

Inputs:
  - /scratch/nb01252/models/*/checkpoint-100/trainer_state.json  (6 training runs)
  - ~/dissertation/results/*_scores.jsonl                         (7 evaluation files)
  - ~/dissertation/results/results_summary.json

Outputs (all to ~/dissertation/results/figures/):
  - training_loss.png             — loss curves for all 6 runs
  - reward_margins.png            — std DPO saturation visualisation
  - grad_norm.png                 — std DPO vs LN-DPO grad_norm comparison
  - chosen_rewards.png            — LN-DPO negative reward inversion
  - rewards_chosen_rejected.png   — chosen vs rejected rewards over training
  - score_distributions.png       — per-condition score histograms
  - beta_ablation.png             — overall score vs beta for std DPO
  - dimension_means.png           — bar chart of means per dimension per condition
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

CONDITIONS_TRAIN = ["dpo_b01", "dpo_b03", "dpo_b05", "dpo_ln_b01", "dpo_ln_b03", "dpo_ln_b05"]
CONDITIONS_EVAL = ["baseline", "dpo_b01", "dpo_b03", "dpo_b05", "dpo_ln_b01", "dpo_ln_b03", "dpo_ln_b05"]

COLORS = {
    "baseline":    "#444444",
    "dpo_b01":     "#1f77b4",
    "dpo_b03":     "#3a8acb",
    "dpo_b05":     "#5fa5e0",
    "dpo_ln_b01":  "#d62728",
    "dpo_ln_b03":  "#e15454",
    "dpo_ln_b05":  "#ed8080",
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


def fig_training_loss(states, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in CONDITIONS_TRAIN:
        ts = states.get(cond)
        if ts is None: continue
        steps, vals = extract_metric(ts, "loss")
        if not steps: continue
        linestyle = "--" if cond.startswith("dpo_ln") else "-"
        ax.plot(steps, vals, label=cond, color=COLORS[cond], linestyle=linestyle, linewidth=1.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("DPO training loss across conditions (β ∈ {0.1, 0.3, 0.5})")
    ax.legend(loc="upper right", ncol=2)
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
        ax.plot(steps, vals, label=cond, color=COLORS[cond], linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward margin (β·(rᶜ − rʳ))")
    ax.set_title("Standard DPO reward margins — saturation across β values")
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
        linestyle = "--" if cond.startswith("dpo_ln") else "-"
        # Clip very small values for log scale
        vals = [max(v, 1e-15) for v in vals]
        ax.plot(steps, vals, label=cond, color=COLORS[cond], linestyle=linestyle, linewidth=1.6)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient norm (log)")
    ax.set_yscale("log")
    ax.set_title("Gradient norm — std DPO saturates, LN-DPO stays healthy")
    ax.legend(loc="lower left", ncol=2)
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
        linestyle = "--" if cond.startswith("dpo_ln") else "-"
        ax.plot(steps, vals, label=cond, color=COLORS[cond], linestyle=linestyle, linewidth=1.8)
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Chosen reward β·(log πᶜ − log π_ref^c)")
    ax.set_title("Chosen rewards — LN-DPO inverts negative (reward inversion failure mode)")
    ax.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def fig_rewards_chosen_rejected(states, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
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
        ax.set_title(cond)
        ax.set_xlabel("Step")
        if i == 0 or i == 3:
            ax.set_ylabel("Reward")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  saved {out_path}")


def fig_score_distributions(scores_by_cond, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, dim in zip(axes, DIMS):
        data = []
        labels = []
        for cond in CONDITIONS_EVAL:
            recs = scores_by_cond.get(cond, [])
            viable = get_viable(recs)
            if viable:
                data.append([r[dim] for r in viable])
                labels.append(cond)
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, cond in zip(bp["boxes"], labels):
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
    fig, ax = plt.subplots(figsize=(11, 5))
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
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
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
    for prefix, color, label in [("dpo_b", "#1f77b4", "Standard DPO"),
                                  ("dpo_ln_b", "#d62728", "LN-DPO (viable only)")]:
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

    print(f"\nAll figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
