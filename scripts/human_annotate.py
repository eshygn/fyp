#!/usr/bin/env python3
"""human_annotate.py — Blinded human annotation of 10 stories vs Sonnet's ratings.

Samples 1 story per condition (10 total), shuffles for blinded order,
displays each without revealing the condition, collects 3 ratings (EF/EAC/SD,
0-3), saves to human_annotations.json, then computes Spearman + Kendall
correlation vs Sonnet on the same stories.

Usage:
    python scripts/human_annotate.py             # fresh run, seed=42
    python scripts/human_annotate.py --resume    # continue from saved
"""

import json
import os
import random
import argparse
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.expanduser("~/dissertation/results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "human_annotations.json")
DIMENSIONS = ["emotional_flexibility", "emotional_arc_coherence", "subtext_density"]

CONDITIONS = [
    "baseline",
    "dpo_b01", "dpo_b03", "dpo_b05",
    "dpo_ln_b01", "dpo_ln_b03", "dpo_ln_b05",
    "dpo_aligned_b01", "dpo_aligned_b03", "dpo_aligned_b05",
]

STORY_FILES = {
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


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def get_int_rating(prompt_str, allowed=(0, 1, 2, 3)):
    while True:
        s = input(prompt_str).strip()
        if s.lower() in ("q", "quit", "exit"):
            print("Quitting — your progress is saved.")
            raise SystemExit(0)
        try:
            v = int(s)
            if v in allowed:
                return v
            print(f"  Must be one of {allowed}.")
        except ValueError:
            print("  Enter an integer 0-3 (or 'q' to quit).")


def show_story(prompt, story, idx_in_sample, n_total):
    print("\n" + "=" * 80)
    print(f"STORY {idx_in_sample + 1} of {n_total}")
    print("=" * 80)
    print("\nPROMPT (outline):")
    print("-" * 80)
    p = prompt.strip() if prompt else "[no prompt]"
    if len(p) > 800:
        p = p[:800] + " ..."
    print(p)
    print("-" * 80)
    print("\nSTORY:")
    print("-" * 80)
    s = story.strip() if story else "[EMPTY — generation collapsed]"
    print(s)


def print_rubric():
    print("\n--- RUBRIC ---")
    print("EMOTIONAL FLEXIBILITY (inner vs outer life balance):")
    print("  0=only external action   1=superficial inner life")
    print("  2=meaningful balance     3=seamless integration")
    print()
    print("EMOTIONAL ARC COHERENCE (Vonnegut arc structure):")
    print("  0=no arc                 1=underdeveloped arc")
    print("  2=clear arc with payoff  3=compelling arc with strong resolution")
    print()
    print("SUBTEXT DENSITY (meaning beneath surface):")
    print("  0=everything explicit    1=occasional subtext")
    print("  2=consistent subtext     3=rich layered subtext")
    print()


def build_sample(n, seed):
    """Build a blinded sample — one story per condition (when n=10), shuffled."""
    rng = random.Random(seed)
    if n == 10:
        conds_to_sample = list(CONDITIONS)
    else:
        conds_to_sample = rng.choices(CONDITIONS, k=n)

    sample = []
    for cond in conds_to_sample:
        story_path = os.path.join(RESULTS_DIR, STORY_FILES[cond])
        score_path = os.path.join(RESULTS_DIR, f"{cond}_scores.jsonl")
        stories = load_jsonl(story_path)
        scores = load_jsonl(score_path)
        score_by_idx = {s["index"]: s for s in scores if "index" in s}
        story_rec = rng.choice(stories)
        idx = story_rec.get("index")
        sonnet_rec = score_by_idx.get(idx, {})
        sample.append({
            "condition": cond,
            "story_index": idx,
            "prompt": story_rec.get("prompt", ""),
            "story": story_rec.get("story", ""),
            "sonnet_ef":  sonnet_rec.get("emotional_flexibility"),
            "sonnet_eac": sonnet_rec.get("emotional_arc_coherence"),
            "sonnet_sd":  sonnet_rec.get("subtext_density"),
        })

    rng.shuffle(sample)
    return sample


def annotate(sample, existing):
    """Annotation loop — blinded, condition hidden until summary."""
    annotations = list(existing)
    print_rubric()
    remaining = len(sample) - len(annotations)
    print(f"\nYou will annotate {remaining} stories (condition hidden during rating).")
    print("Type 'q' at any rating prompt to save and quit.\n")
    input("Press ENTER to start...")

    for i in range(len(annotations), len(sample)):
        item = sample[i]
        show_story(item["prompt"], item["story"], i, len(sample))
        print_rubric()
        ef  = get_int_rating(f"Emotional Flexibility (0-3): ")
        eac = get_int_rating(f"Emotional Arc Coherence (0-3): ")
        sd  = get_int_rating(f"Subtext Density (0-3): ")
        annotations.append({
            "blinded_order": i,
            "condition":     item["condition"],
            "story_index":   item["story_index"],
            "human_ef":  ef,
            "human_eac": eac,
            "human_sd":  sd,
            "sonnet_ef":  item["sonnet_ef"],
            "sonnet_eac": item["sonnet_eac"],
            "sonnet_sd":  item["sonnet_sd"],
        })
        with open(OUTPUT_PATH, "w") as f:
            json.dump({"annotations": annotations}, f, indent=2)
        print(f"  Saved ({len(annotations)}/{len(sample)})")

    return annotations


def summarise(annotations, seed):
    """Compute Spearman, Kendall, and agreement metrics vs Sonnet."""
    valid = [a for a in annotations
             if all(a.get(f"sonnet_{d}") is not None and a[f"sonnet_{d}"] >= 0
                    for d in ["ef", "eac", "sd"])]

    print("\n" + "=" * 80)
    print("HUMAN vs SONNET — AGREEMENT")
    print("=" * 80)
    print(f"\nValid annotations: {len(valid)}/{len(annotations)}\n")

    if len(valid) < 3:
        print("Too few valid annotations for correlation.")
        return {}

    summary = {}
    for dim_short, dim_long in [("ef",  "Emotional Flexibility"),
                                 ("eac", "Emotional Arc Coherence"),
                                 ("sd",  "Subtext Density")]:
        h = np.array([a[f"human_{dim_short}"]  for a in valid], dtype=float)
        s = np.array([a[f"sonnet_{dim_short}"] for a in valid], dtype=float)
        if np.var(h) == 0 or np.var(s) == 0:
            print(f"  {dim_long}: constant ratings — Spearman undefined")
            summary[dim_long] = {"n": len(h), "note": "constant_ratings"}
            continue
        sp = stats.spearmanr(h, s)
        kt = stats.kendalltau(h, s, variant='b')
        exact   = float(np.mean(h == s))
        within1 = float(np.mean(np.abs(h - s) <= 1))
        print(f"  {dim_long}:")
        print(f"    n={len(h)}  spearman_r={sp.statistic:+.3f} (p={sp.pvalue:.4f})")
        print(f"    kendall_tau_b={kt.statistic:+.3f}  exact={exact:.0%}  within_1={within1:.0%}")
        print(f"    human_mean={h.mean():.2f}  sonnet_mean={s.mean():.2f}  "
              f"diff={h.mean()-s.mean():+.2f}")
        summary[dim_long] = {
            "n": len(h),
            "spearman_r": round(float(sp.statistic), 4),
            "spearman_p": float(f"{sp.pvalue:.6g}"),
            "kendall_tau_b": round(float(kt.statistic), 4),
            "exact_agreement": round(exact, 4),
            "within_1_agreement": round(within1, 4),
            "human_mean": round(float(h.mean()), 4),
            "sonnet_mean": round(float(s.mean()), 4),
        }

    # Overall (mean of 3 dims)
    h_overall = np.array([(a["human_ef"]  + a["human_eac"]  + a["human_sd"])  / 3 for a in valid])
    s_overall = np.array([(a["sonnet_ef"] + a["sonnet_eac"] + a["sonnet_sd"]) / 3 for a in valid])
    if np.var(h_overall) > 0 and np.var(s_overall) > 0:
        sp = stats.spearmanr(h_overall, s_overall)
        kt = stats.kendalltau(h_overall, s_overall, variant='b')
        print(f"\n  OVERALL (mean of 3 dims):")
        print(f"    spearman_r={sp.statistic:+.3f} (p={sp.pvalue:.4f})  "
              f"kendall_tau_b={kt.statistic:+.3f}")
        summary["overall"] = {
            "spearman_r": round(float(sp.statistic), 4),
            "spearman_p": float(f"{sp.pvalue:.6g}"),
            "kendall_tau_b": round(float(kt.statistic), 4),
        }

    # Per-condition reveal (no longer blinded)
    print("\n=== REVEAL — condition vs ratings ===")
    print(f"{'cond':18s} {'idx':>4s}  {'EF(h/s)':>8s}  {'EAC(h/s)':>9s}  {'SD(h/s)':>8s}")
    for a in annotations:
        print(f"  {a['condition']:18s} {a['story_index']:>3d}   "
              f"{a['human_ef']}/{a['sonnet_ef']}      "
              f"{a['human_eac']}/{a['sonnet_eac']}        "
              f"{a['human_sd']}/{a['sonnet_sd']}")

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n",    type=int, default=10)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    sample = build_sample(args.n, args.seed)

    existing = []
    if args.resume and os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            existing = json.load(f).get("annotations", [])
        print(f"Resuming: {len(existing)} stories already annotated.")
        if len(existing) >= len(sample):
            print("All annotations complete. Recomputing summary only.")
            summary = summarise(existing, args.seed)
            with open(OUTPUT_PATH, "w") as f:
                json.dump({"seed": args.seed,
                          "annotations": existing,
                          "summary": summary}, f, indent=2)
            return

    annotations = annotate(sample, existing)
    summary = summarise(annotations, args.seed)

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"seed": args.seed,
                  "annotations": annotations,
                  "summary": summary}, f, indent=2)
    print(f"\nFull output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
