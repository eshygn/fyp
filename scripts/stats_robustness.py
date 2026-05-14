#!/usr/bin/env python3
"""stats_robustness.py — Comprehensive statistical analysis for dissertation.

Outputs results_summary_full.json with:
- Welch t-tests (baseline vs each condition, per dimension)
- Mann-Whitney U tests (non-parametric, ordinal-appropriate)
- 95% bootstrap confidence intervals on means
- Kruskal-Wallis across aligned β values (β-insensitivity test)
- Kruskal-Wallis across misaligned β values
- Post-hoc power analysis (n=50, α=0.05)
- Word count statistics for ALL 10 conditions
"""

import json
import os
import numpy as np
from scipy import stats
from scipy.stats import nct as nct_dist
from collections import OrderedDict

RESULTS_DIR = os.path.expanduser("~/dissertation/results")
DIMENSIONS = ["emotional_flexibility", "emotional_arc_coherence", "subtext_density"]

# Condition -> score file mapping
CONDITIONS = OrderedDict([
    ("baseline",         "baseline_scores.jsonl"),
    ("dpo_b01",          "dpo_b01_scores.jsonl"),
    ("dpo_b03",          "dpo_b03_scores.jsonl"),
    ("dpo_b05",          "dpo_b05_scores.jsonl"),
    ("dpo_ln_b01",       "dpo_ln_b01_scores.jsonl"),
    ("dpo_ln_b03",       "dpo_ln_b03_scores.jsonl"),
    ("dpo_ln_b05",       "dpo_ln_b05_scores.jsonl"),
    ("dpo_aligned_b01",  "dpo_aligned_b01_scores.jsonl"),
    ("dpo_aligned_b03",  "dpo_aligned_b03_scores.jsonl"),
    ("dpo_aligned_b05",  "dpo_aligned_b05_scores.jsonl"),
])

# Story file mapping — note: dpo_ln_b01 uses "dpo_ln_stories.jsonl"
STORY_FILES = OrderedDict([
    ("baseline",         "baseline_stories.jsonl"),
    ("dpo_b01",          "dpo_b01_stories.jsonl"),
    ("dpo_b03",          "dpo_b03_stories.jsonl"),
    ("dpo_b05",          "dpo_b05_stories.jsonl"),
    ("dpo_ln_b01",       "dpo_ln_stories.jsonl"),
    ("dpo_ln_b03",       "dpo_ln_b03_stories.jsonl"),
    ("dpo_ln_b05",       "dpo_ln_b05_stories.jsonl"),
    ("dpo_aligned_b01",  "dpo_aligned_b01_stories.jsonl"),
    ("dpo_aligned_b03",  "dpo_aligned_b03_stories.jsonl"),
    ("dpo_aligned_b05",  "dpo_aligned_b05_stories.jsonl"),
])


# =============================================================================
# Helpers
# =============================================================================

def load_jsonl(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_scores(condition):
    filepath = os.path.join(RESULTS_DIR, CONDITIONS[condition])
    data = load_jsonl(filepath)
    scores = {dim: [d[dim] for d in data] for dim in DIMENSIONS}
    scores["overall"] = [
        round(np.mean([d[dim] for dim in DIMENSIONS]), 4) for d in data
    ]
    return scores


def load_stories(condition):
    filepath = os.path.join(RESULTS_DIR, STORY_FILES[condition])
    return load_jsonl(filepath)


def word_count(text):
    if not text or not text.strip():
        return 0
    return len(text.split())


def is_empty(story_text):
    """Empty = blank or fewer than 10 words."""
    if not story_text or not story_text.strip():
        return True
    return word_count(story_text) < 10


def is_abrupt(story_text):
    """Abrupt = non-empty but last char is not sentence-ending punctuation."""
    if is_empty(story_text):
        return False  # classified as empty, not abrupt
    text = story_text.rstrip()
    if not text:
        return True
    last_char = text[-1]
    return last_char not in '.!?"\'\u201d\u2019\u2014\u2026)'


def bootstrap_ci(data, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    data = np.array(data, dtype=float)
    n = len(data)
    boot_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True)) for _ in range(n_boot)
    ])
    alpha = 1 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def cohens_d(a, b):
    """Cohen's d: positive means a > b."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = len(a), len(b)
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def post_hoc_power(d, n, alpha=0.05):
    """Power for two-sample t-test given Cohen's d and n per group."""
    df = 2 * n - 2
    noncentrality = abs(d) * np.sqrt(n / 2)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    power = 1 - nct_dist.cdf(t_crit, df, noncentrality) + \
            nct_dist.cdf(-t_crit, df, noncentrality)
    return float(power)


def detectable_effect_size(n, alpha=0.05, target_power=0.80):
    """Binary search for min detectable |d| at target power."""
    lo, hi = 0.0, 3.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if post_hoc_power(mid, n, alpha) < target_power:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading scores and stories for all 10 conditions...")

    all_scores = OrderedDict()
    all_stories = OrderedDict()
    for cond in CONDITIONS:
        all_scores[cond] = load_scores(cond)
        all_stories[cond] = load_stories(cond)
        print(f"  {cond}: {len(all_scores[cond]['overall'])} scores, "
              f"{len(all_stories[cond])} stories")

    results = OrderedDict()
    baseline = all_scores["baseline"]

    # =================================================================
    # 1. Per-condition descriptive stats + pairwise vs baseline
    # =================================================================
    print("\nComputing per-condition statistics...")
    condition_stats = OrderedDict()

    for cond in CONDITIONS:
        cond_data = OrderedDict()
        scores = all_scores[cond]

        for dim in DIMENSIONS + ["overall"]:
            s = np.array(scores[dim], dtype=float)
            desc = OrderedDict()
            desc["mean"] = round(float(np.mean(s)), 4)
            desc["std"] = round(float(np.std(s, ddof=1)), 4)
            desc["median"] = round(float(np.median(s)), 4)
            desc["n"] = int(len(s))

            ci_lo, ci_hi = bootstrap_ci(s)
            desc["ci_95"] = [round(ci_lo, 4), round(ci_hi, 4)]

            if cond != "baseline":
                b = np.array(baseline[dim], dtype=float)

                # Welch's t-test
                t_stat, t_p = stats.ttest_ind(s, b, equal_var=False)
                desc["welch_t"] = round(float(t_stat), 4)
                desc["welch_p"] = float(f"{t_p:.6g}")

                # Mann-Whitney U (non-parametric)
                u_stat, u_p = stats.mannwhitneyu(s, b, alternative='two-sided')
                desc["mann_whitney_U"] = round(float(u_stat), 1)
                desc["mann_whitney_p"] = float(f"{u_p:.6g}")

                # Cohen's d (condition minus baseline)
                d = cohens_d(list(s), list(b))
                desc["cohens_d"] = round(d, 4)

                # Post-hoc power for observed d
                pwr = post_hoc_power(d, len(s))
                desc["power"] = round(pwr, 4)

            cond_data[dim] = desc
        condition_stats[cond] = cond_data

    results["condition_statistics"] = condition_stats

    # =================================================================
    # 2. Kruskal-Wallis across aligned β values
    # =================================================================
    print("Computing Kruskal-Wallis for aligned β-insensitivity...")
    aligned = ["dpo_aligned_b01", "dpo_aligned_b03", "dpo_aligned_b05"]
    kw_aligned = OrderedDict()
    for dim in DIMENSIONS + ["overall"]:
        groups = [np.array(all_scores[c][dim], dtype=float) for c in aligned]
        h, p = stats.kruskal(*groups)
        kw_aligned[dim] = OrderedDict([
            ("H", round(float(h), 4)),
            ("p", float(f"{p:.6g}")),
            ("groups", {c: round(float(np.mean(all_scores[c][dim])), 4) for c in aligned}),
            ("interpretation",
             "no significant difference across β (β-insensitive)"
             if p > 0.05 else "significant β effect"),
        ])
    results["kruskal_wallis_aligned"] = kw_aligned

    # Same for misaligned standard DPO
    print("Computing Kruskal-Wallis for misaligned β values...")
    misaligned = ["dpo_b01", "dpo_b03", "dpo_b05"]
    kw_misaligned = OrderedDict()
    for dim in DIMENSIONS + ["overall"]:
        groups = [np.array(all_scores[c][dim], dtype=float) for c in misaligned]
        h, p = stats.kruskal(*groups)
        kw_misaligned[dim] = OrderedDict([
            ("H", round(float(h), 4)),
            ("p", float(f"{p:.6g}")),
            ("groups", {c: round(float(np.mean(all_scores[c][dim])), 4) for c in misaligned}),
        ])
    results["kruskal_wallis_misaligned"] = kw_misaligned

    # Same for LN-DPO
    print("Computing Kruskal-Wallis for LN-DPO β values...")
    ln_conds = ["dpo_ln_b01", "dpo_ln_b03", "dpo_ln_b05"]
    kw_ln = OrderedDict()
    for dim in DIMENSIONS + ["overall"]:
        groups = [np.array(all_scores[c][dim], dtype=float) for c in ln_conds]
        h, p = stats.kruskal(*groups)
        kw_ln[dim] = OrderedDict([
            ("H", round(float(h), 4)),
            ("p", float(f"{p:.6g}")),
            ("groups", {c: round(float(np.mean(all_scores[c][dim])), 4) for c in ln_conds}),
        ])
    results["kruskal_wallis_ln_dpo"] = kw_ln

    # =================================================================
    # 3. Power analysis
    # =================================================================
    print("Computing power analysis...")
    n = 50
    min_d = detectable_effect_size(n)
    power_info = OrderedDict([
        ("n_per_group", n),
        ("alpha", 0.05),
        ("min_detectable_d_80pct", min_d),
        ("power_at_d_0.2_small", round(post_hoc_power(0.2, n), 4)),
        ("power_at_d_0.5_medium", round(post_hoc_power(0.5, n), 4)),
        ("power_at_d_0.57", round(post_hoc_power(0.57, n), 4)),
        ("power_at_d_0.8_large", round(post_hoc_power(0.8, n), 4)),
        ("power_at_d_1.0", round(post_hoc_power(1.0, n), 4)),
        ("n_needed_for_d_0.2_at_80pct", None),  # filled below
    ])
    # Find n needed for d=0.2 at 80% power
    for test_n in range(50, 2000):
        if post_hoc_power(0.2, test_n) >= 0.80:
            power_info["n_needed_for_d_0.2_at_80pct"] = test_n
            break
    results["power_analysis"] = power_info

    # =================================================================
    # 4. Word count statistics
    # =================================================================
    print("Computing word count statistics...")
    word_stats = OrderedDict()
    for cond in CONDITIONS:
        stories = all_stories[cond]
        texts = [s.get("story", "") for s in stories]
        wc = np.array([word_count(t) for t in texts])

        n_empty = sum(1 for t in texts if is_empty(t))
        n_abrupt = sum(1 for t in texts if is_abrupt(t))
        non_empty_wc = wc[wc >= 10]  # word counts excluding empties

        ws = OrderedDict([
            ("n", len(texts)),
            ("mean_words", round(float(np.mean(wc)), 1)),
            ("std_words", round(float(np.std(wc, ddof=1)), 1) if len(wc) > 1 else 0.0),
            ("median_words", round(float(np.median(wc)), 1)),
            ("min_words", int(np.min(wc))),
            ("max_words", int(np.max(wc))),
            ("empty_count", n_empty),
            ("abrupt_endings", n_abrupt),
        ])
        if len(non_empty_wc) > 0 and n_empty > 0:
            ws["mean_words_non_empty"] = round(float(np.mean(non_empty_wc)), 1)
        word_stats[cond] = ws

    results["word_count_statistics"] = word_stats

    # =================================================================
    # 5. Compact pairwise summary
    # =================================================================
    print("Building pairwise summary...")
    summary = OrderedDict()
    for cond in CONDITIONS:
        if cond == "baseline":
            continue
        row = OrderedDict()
        for dim in DIMENSIONS + ["overall"]:
            cs = condition_stats[cond][dim]
            bl = condition_stats["baseline"][dim]
            row[dim] = OrderedDict([
                ("delta", round(cs["mean"] - bl["mean"], 4)),
                ("d", cs.get("cohens_d")),
                ("welch_p", cs.get("welch_p")),
                ("mw_p", cs.get("mann_whitney_p")),
                ("sig_welch", cs.get("welch_p", 1) < 0.05),
                ("sig_mw", cs.get("mann_whitney_p", 1) < 0.05),
            ])
        summary[cond] = row
    results["pairwise_vs_baseline"] = summary

    # =================================================================
    # Save
    # =================================================================
    outpath = os.path.join(RESULTS_DIR, "results_summary_full.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Saved: {outpath}")
    print(f"Size:  {os.path.getsize(outpath)} bytes")
    print(f"{'='*70}")

    # =================================================================
    # Terminal summary
    # =================================================================
    print("\n=== ALIGNED vs BASELINE ===")
    for cond in aligned:
        print(f"\n  {cond}:")
        for dim in DIMENSIONS:
            d = condition_stats[cond][dim]
            sig = ("***" if d.get("welch_p", 1) < 0.001 else
                   "**"  if d.get("welch_p", 1) < 0.01  else
                   "*"   if d.get("welch_p", 1) < 0.05  else "ns")
            mw  = ("***" if d.get("mann_whitney_p", 1) < 0.001 else
                   "**"  if d.get("mann_whitney_p", 1) < 0.01  else
                   "*"   if d.get("mann_whitney_p", 1) < 0.05  else "ns")
            print(f"    {dim}: {d['mean']:.2f} (Δ={d['mean'] - condition_stats['baseline'][dim]['mean']:+.2f}, "
                  f"d={d.get('cohens_d', 0):.2f}) welch={sig} mw={mw}")

    print("\n=== MISALIGNED vs BASELINE ===")
    for cond in misaligned:
        print(f"\n  {cond}:")
        for dim in DIMENSIONS:
            d = condition_stats[cond][dim]
            sig = ("***" if d.get("welch_p", 1) < 0.001 else
                   "**"  if d.get("welch_p", 1) < 0.01  else
                   "*"   if d.get("welch_p", 1) < 0.05  else "ns")
            print(f"    {dim}: {d['mean']:.2f} (Δ={d['mean'] - condition_stats['baseline'][dim]['mean']:+.2f}, "
                  f"d={d.get('cohens_d', 0):.2f}) welch={sig}")

    print("\n=== KRUSKAL-WALLIS β-INSENSITIVITY (aligned) ===")
    for dim in DIMENSIONS + ["overall"]:
        kw = kw_aligned[dim]
        print(f"  {dim}: H={kw['H']:.2f}, p={kw['p']:.4f} → {kw['interpretation']}")

    print("\n=== POWER ANALYSIS ===")
    pa = power_info
    print(f"  Min detectable d at 80%: {pa['min_detectable_d_80pct']}")
    print(f"  Power at d=0.2: {pa['power_at_d_0.2_small']}")
    print(f"  Power at d=0.5: {pa['power_at_d_0.5_medium']}")
    print(f"  Power at d=0.8: {pa['power_at_d_0.8_large']}")
    print(f"  n needed for d=0.2 at 80%: {pa['n_needed_for_d_0.2_at_80pct']}")

    print("\n=== WORD COUNTS ===")
    for cond in CONDITIONS:
        ws = word_stats[cond]
        extra = ""
        if ws["empty_count"] > 0 and "mean_words_non_empty" in ws:
            extra = f", non-empty mean={ws['mean_words_non_empty']:.0f}"
        print(f"  {cond:20s}: mean={ws['mean_words']:7.1f} ±{ws['std_words']:6.1f}  "
              f"empty={ws['empty_count']}  abrupt={ws['abrupt_endings']}{extra}")


if __name__ == "__main__":
    main()
