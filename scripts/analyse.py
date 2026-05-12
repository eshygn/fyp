#!/usr/bin/env python3
"""
Statistical analysis of evaluation results.
Produces results_summary.json and prints formatted tables for Chapter 5.

Usage:
  python3 analyse.py
  python3 analyse.py --results_dir ~/dissertation/results --output ~/dissertation/results/results_summary.json
"""

import json
import os
import argparse
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, ttest_ind

DIMS = ["emotional_flexibility", "emotional_arc_coherence", "subtext_density"]
DIM_SHORT = ["EF", "EAC", "SD"]
SMALL_N_THRESHOLD = 15

# Score file mapping: condition name -> filename in results_dir
CONDITIONS = {
    "baseline":   "baseline_scores.jsonl",
    "dpo_b01":    "dpo_b01_scores.jsonl",
    "dpo_b03":    "dpo_b03_scores.jsonl",
    "dpo_b05":    "dpo_b05_scores.jsonl",
    "dpo_ln_b01": "dpo_ln_b01_scores.jsonl",
    "dpo_ln_b03": "dpo_ln_b03_scores.jsonl",
    "dpo_ln_b05": "dpo_ln_b05_scores.jsonl",
}

# Known training metrics (from completed runs — do not change)
TRAINING_RUNS = {
    "dpo_b01":    {"mode": "dpo",    "beta": 0.1, "steps": 100, "final_loss": 0.0478, "final_margin": 29.9,  "grad_norm_step90": 2.04e-06, "saturated": True},
    "dpo_b03":    {"mode": "dpo",    "beta": 0.3, "steps": 100, "final_loss": 0.0442, "final_margin": 62.2,  "grad_norm_step90": 1.26e-12, "saturated": True},
    "dpo_b05":    {"mode": "dpo",    "beta": 0.5, "steps": 100, "final_loss": 0.0437, "final_margin": 85.5,  "grad_norm_step90": 0.0,      "saturated": True},
    "dpo_ln_b01": {"mode": "dpo_ln", "beta": 0.1, "steps": 100, "final_loss": 0.0803, "final_margin": None,  "grad_norm_step90": 0.855,    "saturated": False, "collapsed": True},
    "dpo_ln_b03": {"mode": "dpo_ln", "beta": 0.3, "steps": 100, "final_loss": 0.1118, "final_margin": None,  "grad_norm_step90": 0.007,    "saturated": False, "collapsed": True},
    "dpo_ln_b05": {"mode": "dpo_ln", "beta": 0.5, "steps": 100, "final_loss": 0.0532, "final_margin": None,  "grad_norm_step90": 0.026,    "saturated": False, "collapsed": True},
}


def load_scores(path):
    if not os.path.exists(path):
        return None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def get_viable(records):
    """Exclude empty stories and failed evaluations."""
    return [r for r in records
            if r.get("reasoning") != "empty_story"
            and all(r.get(d, -1) >= 0 for d in DIMS)]


def cohens_d(a, b):
    """Cohen's d: positive = b > a (treatment > control)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    return float("nan") if pooled == 0 else (np.mean(b) - np.mean(a)) / pooled


def safe_mwu(a, b):
    try:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except ValueError:
        return float("nan"), float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=os.path.expanduser("~/dissertation/results"))
    parser.add_argument("--output",      default=os.path.expanduser("~/dissertation/results/results_summary.json"))
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load all score files
    # ------------------------------------------------------------------ #
    print("\nLoading score files...")
    raw_data = {}
    for cond, fname in CONDITIONS.items():
        path = os.path.join(args.results_dir, fname)
        records = load_scores(path)
        if records is None:
            print(f"  MISSING  : {fname}")
            continue
        viable = get_viable(records)
        n_empty = sum(1 for r in records if r.get("reasoning") == "empty_story")
        flag = "  *** SMALL N ***" if len(viable) < SMALL_N_THRESHOLD else ""
        print(f"  {cond:<15}: total={len(records):3d}  empty={n_empty:3d}  viable={len(viable):3d}{flag}")
        raw_data[cond] = {"records": records, "viable": viable, "n_empty": n_empty}

    # ------------------------------------------------------------------ #
    # 2. Descriptive statistics
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("DESCRIPTIVE STATISTICS  (viable stories only, scores 0–3)")
    print("=" * 72)
    header = f"{'Condition':<15} {'N':>4}   {'EF':>10} {'EAC':>10} {'SD':>10} {'Overall':>8}"
    print(header)
    print("-" * 72)

    cond_stats = {}
    for cond in CONDITIONS:
        if cond not in raw_data:
            continue
        viable = raw_data[cond]["viable"]
        n = len(viable)
        if n == 0:
            print(f"  {cond:<15} {n:>4}   {'—':>10} {'—':>10} {'—':>10} {'—':>8}")
            continue
        dim_scores = {d: [r[d] for r in viable] for d in DIMS}
        means = {d: float(np.mean(dim_scores[d])) for d in DIMS}
        stds  = {d: float(np.std(dim_scores[d], ddof=1)) if n > 1 else 0.0 for d in DIMS}
        overall = float(np.mean([means[d] for d in DIMS]))
        cond_stats[cond] = {"n": n, "means": means, "stds": stds, "overall": overall, "dim_scores": dim_scores}
        ef_str  = f"{means['emotional_flexibility']:.2f}±{stds['emotional_flexibility']:.2f}"
        eac_str = f"{means['emotional_arc_coherence']:.2f}±{stds['emotional_arc_coherence']:.2f}"
        sd_str  = f"{means['subtext_density']:.2f}±{stds['subtext_density']:.2f}"
        print(f"  {cond:<15} {n:>4}   {ef_str:>10} {eac_str:>10} {sd_str:>10} {overall:>8.2f}")

    # ------------------------------------------------------------------ #
    # 3. Pairwise tests: baseline vs each condition
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("PAIRWISE COMPARISONS vs BASELINE")
    print("Welch's t-test + Mann-Whitney U  |  * p<0.05  ** p<0.01")
    print("=" * 72)

    pairwise_results = {}

    if "baseline" not in cond_stats:
        print("  Baseline missing — skipping pairwise tests.")
    else:
        base_scores = cond_stats["baseline"]["dim_scores"]
        comparisons = [c for c in CONDITIONS if c != "baseline"]

        for cond in comparisons:
            if cond not in cond_stats:
                continue
            n_cond = cond_stats[cond]["n"]
            n_base = cond_stats["baseline"]["n"]
            print(f"\n  baseline (n={n_base}) vs {cond} (n={n_cond})")
            if n_cond < SMALL_N_THRESHOLD:
                print(f"  *** Small n={n_cond} — interpret with caution ***")

            cond_scores = cond_stats[cond]["dim_scores"]
            pairwise_results[f"baseline_vs_{cond}"] = {}

            for dim, short in zip(DIMS, DIM_SHORT):
                a = base_scores[dim]
                b = cond_scores[dim]
                delta = float(np.mean(b) - np.mean(a))
                t_stat, p_welch = ttest_ind(a, b, equal_var=False)
                u_stat, p_mwu  = safe_mwu(a, b)
                d = cohens_d(a, b)
                sig = "**" if p_welch < 0.01 else "*" if p_welch < 0.05 else "ns"
                print(f"    {short}: Δ={delta:+.3f}  t={t_stat:.2f}  p_welch={p_welch:.3f}({sig})  "
                      f"p_mwu={p_mwu:.3f}  d={d:.2f}")
                pairwise_results[f"baseline_vs_{cond}"][dim] = {
                    "delta": delta,
                    "t_stat": float(t_stat),
                    "p_welch": float(p_welch),
                    "p_mwu": float(p_mwu) if not np.isnan(p_mwu) else None,
                    "cohens_d": float(d) if not np.isnan(d) else None,
                    "significant_welch_p05": bool(p_welch < 0.05),
                }

    # ------------------------------------------------------------------ #
    # 4. Within-condition Spearman correlations
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("SPEARMAN CORRELATIONS BETWEEN DIMENSIONS (within condition)")
    print("=" * 72)

    spearman_results = {}
    for cond, stats in cond_stats.items():
        n = stats["n"]
        if n < 5:
            print(f"  {cond}: n={n} — too small for correlation")
            continue
        ef  = stats["dim_scores"]["emotional_flexibility"]
        eac = stats["dim_scores"]["emotional_arc_coherence"]
        sd  = stats["dim_scores"]["subtext_density"]
        r1, p1 = spearmanr(ef, eac)
        r2, p2 = spearmanr(ef, sd)
        r3, p3 = spearmanr(eac, sd)
        print(f"  {cond:<15} (n={n:3d}): "
              f"EF–EAC r={r1:.2f}(p={p1:.3f})  "
              f"EF–SD r={r2:.2f}(p={p2:.3f})  "
              f"EAC–SD r={r3:.2f}(p={p3:.3f})")
        spearman_results[cond] = {
            "EF_EAC": {"r": float(r1), "p": float(p1)},
            "EF_SD":  {"r": float(r2), "p": float(p2)},
            "EAC_SD": {"r": float(r3), "p": float(p3)},
        }

    # ------------------------------------------------------------------ #
    # 5. Build and save results_summary.json
    # ------------------------------------------------------------------ #
    generation = {}
    for cond in CONDITIONS:
        if cond in raw_data:
            records = raw_data[cond]["records"]
            n_empty = raw_data[cond]["n_empty"]
            entry = {"total": len(records), "empty": n_empty, "viable": len(records) - n_empty}
            if cond.startswith("dpo_ln"):
                entry["collapsed"] = True
            generation[cond] = entry

    eval_results = {}
    for cond, stats in cond_stats.items():
        eval_results[cond] = {
            "n_viable": stats["n"],
            "means": {d: stats["means"][d] for d in DIMS},
            "stds":  {d: stats["stds"][d]  for d in DIMS},
            "overall_mean": stats["overall"],
        }

    summary = {
        "training_runs": TRAINING_RUNS,
        "generation":    generation,
        "evaluation": {
            "judge_model": "claude-sonnet-4-6",
            "story_truncation_chars": 8000,
            "dimensions": DIMS,
            "scale": "0-3",
            "results": eval_results,
        },
        "pairwise_tests":        pairwise_results,
        "spearman_correlations": spearman_results,
        "qualitative_examples":  [],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {args.output}")
    print("Done. Add qualitative_examples manually before finalising.")


if __name__ == "__main__":
    main()
