#!/usr/bin/env python3
"""calculate_irr.py — Inter-Rater Reliability between Sonnet and Gemini judges.

Loads paired Sonnet/Gemini score files for the 7 Exp 1 conditions and computes:
  - Spearman rank correlation per (condition, dimension)
  - Kendall's tau-b (better for ordinal 0-3 data with ties)
  - Exact agreement rate (% identical scores)
  - Within-1 agreement rate (|sonnet - gemini| <= 1)
  - Aggregate (pooled across all conditions) per dimension
  - Overall aggregate

Conditions WITHOUT Gemini: dpo_aligned_b01/03/05 (Sonnet only).

Output: ~/dissertation/results/judge_agreement.json
"""

import json
import os
import numpy as np
from scipy import stats
from collections import OrderedDict

RESULTS_DIR = os.path.expanduser("~/dissertation/results")
DIMENSIONS = ["emotional_flexibility", "emotional_arc_coherence", "subtext_density"]

# Conditions with BOTH Sonnet and Gemini scores
PAIRED_CONDITIONS = [
    "baseline",
    "dpo_b01", "dpo_b03", "dpo_b05",
    "dpo_ln_b01", "dpo_ln_b03", "dpo_ln_b05",
]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aligned_pair(sonnet_rows, gemini_rows):
    """Match Sonnet and Gemini rows by `index` field. Drop unmatched or failed (-1) rows."""
    g_by_idx = {r["index"]: r for r in gemini_rows}
    pairs = []
    for sr in sonnet_rows:
        idx = sr.get("index")
        if idx is None or idx not in g_by_idx:
            continue
        gr = g_by_idx[idx]
        # Skip failed evaluations on either side
        if any(sr.get(d, -1) < 0 or gr.get(d, -1) < 0 for d in DIMENSIONS):
            continue
        pairs.append((sr, gr))
    return pairs


def compute_agreement(sonnet_vals, gemini_vals):
    """Compute Spearman, Kendall, exact, and within-1 agreement."""
    s = np.array(sonnet_vals, dtype=float)
    g = np.array(gemini_vals, dtype=float)
    n = len(s)

    if n < 3:
        return {"n": n, "error": "insufficient_data"}

    # Spearman (will warn if one is constant; suppress by checking variance)
    if np.var(s) == 0 or np.var(g) == 0:
        spearman_r, spearman_p = float("nan"), float("nan")
    else:
        sp = stats.spearmanr(s, g)
        spearman_r, spearman_p = float(sp.statistic), float(sp.pvalue)

    # Kendall's tau-b (handles ties better than Spearman for ordinal data)
    if np.var(s) == 0 or np.var(g) == 0:
        kendall_t, kendall_p = float("nan"), float("nan")
    else:
        kt = stats.kendalltau(s, g, variant='b')
        kendall_t, kendall_p = float(kt.statistic), float(kt.pvalue)

    diff = np.abs(s - g)
    exact = float(np.mean(diff == 0))
    within_1 = float(np.mean(diff <= 1))

    return OrderedDict([
        ("n", int(n)),
        ("spearman_r", round(spearman_r, 4) if not np.isnan(spearman_r) else None),
        ("spearman_p", float(f"{spearman_p:.6g}") if not np.isnan(spearman_p) else None),
        ("kendall_tau_b", round(kendall_t, 4) if not np.isnan(kendall_t) else None),
        ("kendall_p", float(f"{kendall_p:.6g}") if not np.isnan(kendall_p) else None),
        ("exact_agreement", round(exact, 4)),
        ("within_1_agreement", round(within_1, 4)),
        ("sonnet_mean", round(float(np.mean(s)), 4)),
        ("gemini_mean", round(float(np.mean(g)), 4)),
        ("mean_difference", round(float(np.mean(s - g)), 4)),
    ])


def main():
    print("Loading paired score files...")
    paired_data = OrderedDict()
    for cond in PAIRED_CONDITIONS:
        son_path = os.path.join(RESULTS_DIR, f"{cond}_scores.jsonl")
        gem_path = os.path.join(RESULTS_DIR, f"{cond}_scores_gemini.jsonl")
        if not (os.path.exists(son_path) and os.path.exists(gem_path)):
            print(f"  SKIP {cond}: missing {son_path if not os.path.exists(son_path) else gem_path}")
            continue
        son = load_jsonl(son_path)
        gem = load_jsonl(gem_path)
        pairs = aligned_pair(son, gem)
        print(f"  {cond}: {len(son)} sonnet, {len(gem)} gemini, {len(pairs)} matched & valid")
        paired_data[cond] = pairs

    results = OrderedDict()

    # =================================================================
    # 1. Per-condition, per-dimension agreement
    # =================================================================
    print("\nComputing per-condition agreement...")
    per_condition = OrderedDict()
    for cond, pairs in paired_data.items():
        cond_block = OrderedDict()
        for dim in DIMENSIONS:
            son_vals = [p[0][dim] for p in pairs]
            gem_vals = [p[1][dim] for p in pairs]
            cond_block[dim] = compute_agreement(son_vals, gem_vals)
        # Overall (mean of three dims, treated as continuous)
        son_overall = [np.mean([p[0][d] for d in DIMENSIONS]) for p in pairs]
        gem_overall = [np.mean([p[1][d] for d in DIMENSIONS]) for p in pairs]
        cond_block["overall"] = compute_agreement(son_overall, gem_overall)
        per_condition[cond] = cond_block
    results["per_condition"] = per_condition

    # =================================================================
    # 2. Pooled-across-conditions per dimension
    # =================================================================
    print("Computing pooled per-dimension agreement...")
    pooled_per_dim = OrderedDict()
    for dim in DIMENSIONS:
        son_all, gem_all = [], []
        for cond, pairs in paired_data.items():
            son_all.extend([p[0][dim] for p in pairs])
            gem_all.extend([p[1][dim] for p in pairs])
        pooled_per_dim[dim] = compute_agreement(son_all, gem_all)
    results["pooled_per_dimension"] = pooled_per_dim

    # =================================================================
    # 3. Grand aggregate (all dimensions, all conditions, flattened)
    # =================================================================
    print("Computing grand aggregate...")
    son_grand, gem_grand = [], []
    for cond, pairs in paired_data.items():
        for p in pairs:
            for dim in DIMENSIONS:
                son_grand.append(p[0][dim])
                gem_grand.append(p[1][dim])
    results["grand_aggregate"] = compute_agreement(son_grand, gem_grand)

    # =================================================================
    # Save
    # =================================================================
    outpath = os.path.join(RESULTS_DIR, "judge_agreement.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Saved: {outpath}")
    print(f"{'='*70}\n")

    # =================================================================
    # Terminal summary
    # =================================================================
    print("=== POOLED PER-DIMENSION (across 7 Exp 1 conditions) ===")
    for dim in DIMENSIONS:
        a = pooled_per_dim[dim]
        print(f"  {dim}:")
        print(f"    n={a['n']}  spearman_r={a['spearman_r']}  kendall_tau_b={a['kendall_tau_b']}")
        print(f"    exact={a['exact_agreement']:.2%}  within_1={a['within_1_agreement']:.2%}")
        print(f"    sonnet_mean={a['sonnet_mean']:.3f}  gemini_mean={a['gemini_mean']:.3f}  "
              f"diff={a['mean_difference']:+.3f}")

    print("\n=== GRAND AGGREGATE (all dims pooled) ===")
    ga = results["grand_aggregate"]
    print(f"  n={ga['n']}  spearman_r={ga['spearman_r']}  kendall_tau_b={ga['kendall_tau_b']}")
    print(f"  exact={ga['exact_agreement']:.2%}  within_1={ga['within_1_agreement']:.2%}")

    print("\n=== PER-CONDITION OVERALL SPEARMAN ===")
    for cond, block in per_condition.items():
        o = block["overall"]
        print(f"  {cond:18s}: r={o['spearman_r']}  tau_b={o['kendall_tau_b']}  "
              f"exact={o['exact_agreement']:.2%}  within1={o['within_1_agreement']:.2%}")


if __name__ == "__main__":
    main()
