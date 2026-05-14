"""Build rubric-aligned preference dataset from Gemini-scored pairs."""
import json, os, argparse
from datasets import Dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=os.path.expanduser("~/dissertation/data/scored_pairs_gemini.jsonl"))
    p.add_argument("--output", default=os.path.expanduser("~/dissertation/data/prepared_aligned/train"))
    p.add_argument("--min_gap", type=int, default=2, help="Minimum total score gap (chosen - rejected)")
    args = p.parse_args()

    pairs = [json.loads(l) for l in open(args.input)]
    print(f"Loaded {len(pairs)} scored pairs")

    aligned = []
    for pair in pairs:
        gap = pair["chosen_total"] - pair["rejected_total"]
        if gap >= args.min_gap:
            aligned.append({
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            })
        elif gap <= -args.min_gap:
            aligned.append({
                "prompt": pair["prompt"],
                "chosen": pair["rejected"],
                "rejected": pair["chosen"],
            })

    print(f"Filtered to {len(aligned)} aligned pairs (gap >= {args.min_gap})")
    if not aligned:
        print("WARNING: No pairs meet threshold! Try --min_gap 1")
        return

    ds = Dataset.from_list(aligned)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ds.save_to_disk(args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
