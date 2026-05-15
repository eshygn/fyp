#!/usr/bin/env python3
"""extract_case_studies.py — Pull best and worst story per condition for Results chapter.

For each of the 10 conditions:
  - Best: highest sum of EF+EAC+SD (ties broken by index, lowest first)
  - Worst: lowest sum (ties broken by index)
  - Stories flagged 'empty_story' counted as worst with explicit flag

Outputs:
  - ~/dissertation/results/case_studies.md  (human-readable, paste into LaTeX)
  - ~/dissertation/results/case_studies.json (machine-readable for downstream use)
"""

import json
import os
from collections import OrderedDict

RESULTS_DIR = os.path.expanduser("~/dissertation/results")
DIMENSIONS = ["emotional_flexibility", "emotional_arc_coherence", "subtext_density"]

CONDITIONS = [
    "baseline",
    "dpo_b01", "dpo_b03", "dpo_b05",
    "dpo_ln_b01", "dpo_ln_b03", "dpo_ln_b05",
    "dpo_aligned_b01", "dpo_aligned_b03", "dpo_aligned_b05",
]

# Story file mapping (handles dpo_ln_b01 → dpo_ln_stories.jsonl quirk)
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

STORY_TRUNC = 600    # chars of story to show
PROMPT_TRUNC = 250   # chars of prompt to show


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def truncate(text, n):
    if text is None:
        return ""
    text = text.strip()
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0] + "…"


def main():
    print("Loading score + story files for all 10 conditions...")
    merged = OrderedDict()
    for cond in CONDITIONS:
        score_path = os.path.join(RESULTS_DIR, f"{cond}_scores.jsonl")
        story_path = os.path.join(RESULTS_DIR, STORY_FILES[cond])
        scores = load_jsonl(score_path)
        stories = load_jsonl(story_path)
        story_by_idx = {s["index"]: s for s in stories if "index" in s}
        rows = []
        for s in scores:
            idx = s.get("index")
            story = story_by_idx.get(idx, {}).get("story", "")
            prompt = story_by_idx.get(idx, {}).get("prompt", "")
            ef = s.get("emotional_flexibility", -1)
            eac = s.get("emotional_arc_coherence", -1)
            sd = s.get("subtext_density", -1)
            valid = ef >= 0 and eac >= 0 and sd >= 0
            total = (ef + eac + sd) if valid else -999
            rows.append({
                "index": idx,
                "prompt": prompt,
                "story": story,
                "ef": ef, "eac": eac, "sd": sd,
                "total": total,
                "reasoning": s.get("reasoning", ""),
                "is_empty": s.get("reasoning") == "empty_story",
                "valid": valid,
            })
        merged[cond] = rows
        print(f"  {cond}: {len(rows)} merged records")

    # Pick best and worst per condition
    print("\nSelecting best and worst per condition...")
    case_studies = OrderedDict()
    for cond, rows in merged.items():
        viable = [r for r in rows if r["valid"]]
        if not viable:
            # all empty/failed — pick anything to show collapse
            case_studies[cond] = {"best": None, "worst": rows[0] if rows else None}
            continue
        viable.sort(key=lambda r: (-r["total"], r["index"]))
        best = viable[0]
        viable.sort(key=lambda r: (r["total"], r["index"]))
        worst = viable[0]
        case_studies[cond] = {"best": best, "worst": worst}
        print(f"  {cond}: best idx={best['index']} (total={best['total']}), "
              f"worst idx={worst['index']} (total={worst['total']})")

    # =================================================================
    # Write Markdown
    # =================================================================
    md_path = os.path.join(RESULTS_DIR, "case_studies.md")
    with open(md_path, "w") as f:
        f.write("# Qualitative Case Studies\n\n")
        f.write("Best and worst story per condition by total rubric score (EF+EAC+SD).\n\n")
        f.write("---\n\n")

        for cond, cs in case_studies.items():
            f.write(f"## {cond}\n\n")

            best = cs["best"]
            if best is not None:
                f.write(f"### Best — idx {best['index']} "
                        f"(EF={best['ef']}, EAC={best['eac']}, SD={best['sd']}, total={best['total']})\n\n")
                f.write(f"**Prompt (truncated):**\n\n> {truncate(best['prompt'], PROMPT_TRUNC)}\n\n")
                f.write(f"**Story (truncated):**\n\n> {truncate(best['story'], STORY_TRUNC)}\n\n")
                f.write(f"**Judge reasoning:** {best['reasoning']}\n\n")
            else:
                f.write("### Best — no viable stories in this condition\n\n")

            worst = cs["worst"]
            if worst is not None:
                empty_flag = " [EMPTY/COLLAPSED]" if worst.get("is_empty") else ""
                if worst.get("valid"):
                    f.write(f"### Worst — idx {worst['index']} "
                            f"(EF={worst['ef']}, EAC={worst['eac']}, SD={worst['sd']}, "
                            f"total={worst['total']}){empty_flag}\n\n")
                else:
                    f.write(f"### Worst — idx {worst['index']}{empty_flag}\n\n")
                f.write(f"**Prompt (truncated):**\n\n> {truncate(worst['prompt'], PROMPT_TRUNC)}\n\n")
                f.write(f"**Story (truncated):**\n\n> {truncate(worst['story'], STORY_TRUNC)}\n\n")
                f.write(f"**Judge reasoning:** {worst['reasoning']}\n\n")

            f.write("---\n\n")

    print(f"\nWrote {md_path}")

    # =================================================================
    # Write JSON (machine-readable)
    # =================================================================
    json_path = os.path.join(RESULTS_DIR, "case_studies.json")
    with open(json_path, "w") as f:
        json.dump(case_studies, f, indent=2)
    print(f"Wrote {json_path}")

    # Terminal summary
    print("\n=== SUMMARY ===")
    for cond, cs in case_studies.items():
        b = cs["best"]
        w = cs["worst"]
        b_str = f"best=idx{b['index']} ({b['ef']}/{b['eac']}/{b['sd']})" if b else "best=NONE"
        w_str = f"worst=idx{w['index']} ({w['ef']}/{w['eac']}/{w['sd']})" if w and w.get('valid') else \
                (f"worst=idx{w['index']} EMPTY" if w else "worst=NONE")
        print(f"  {cond:20s} {b_str:30s} {w_str}")


if __name__ == "__main__":
    main()
