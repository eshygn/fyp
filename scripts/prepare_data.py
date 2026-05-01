"""
Prepare the filtered_Qwen3-4B.jsonl dataset for DPO training.
- Renames 'outline' -> 'prompt', drops 'model' field
- Samples a subset (default 3000) for training
- Creates train/test split
- Saves in HuggingFace Dataset format
"""

import json
import random
import argparse
from pathlib import Path
from datasets import Dataset

def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file and return list of dicts."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            row = json.loads(line.strip())
            data.append({
                'prompt': row['outline'],    # rename outline -> prompt
                'chosen': row['chosen'],
                'rejected': row['rejected'],
            })
    return data

def main():
    parser = argparse.ArgumentParser(description='Prepare DPO dataset')
    parser.add_argument('--input', type=str, default='../data/filtered_Qwen3-4B.jsonl',
                        help='Path to raw JSONL file')
    parser.add_argument('--output_dir', type=str, default='../data/prepared',
                        help='Output directory for prepared dataset')
    parser.add_argument('--n_train', type=int, default=2500,
                        help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)

    # Load and shuffle
    print(f"Loading data from {args.input}...")
    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} total samples")

    random.shuffle(data)

    # Split
    n_total = args.n_train + args.n_test
    if n_total > len(data):
        raise ValueError(f"Requested {n_total} samples but only {len(data)} available")

    train_data = data[:args.n_train]
    test_data = data[args.n_train:args.n_train + args.n_test]

    # Print stats
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Test samples:  {len(test_data)}")

    avg_prompt_len = sum(len(d['prompt']) for d in train_data) / len(train_data)
    avg_chosen_len = sum(len(d['chosen']) for d in train_data) / len(train_data)
    avg_rejected_len = sum(len(d['rejected']) for d in train_data) / len(train_data)
    print(f"\nAvg prompt length:   {avg_prompt_len:.0f} chars")
    print(f"Avg chosen length:   {avg_chosen_len:.0f} chars")
    print(f"Avg rejected length: {avg_rejected_len:.0f} chars")

    # Save as HuggingFace datasets
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Dataset.from_list(train_data)
    test_ds = Dataset.from_list(test_data)

    train_ds.save_to_disk(str(output_dir / 'train'))
    test_ds.save_to_disk(str(output_dir / 'test'))

    # Also save test prompts separately for generation later
    test_prompts = [d['prompt'] for d in test_data]
    with open(output_dir / 'test_prompts.json', 'w') as f:
        json.dump(test_prompts, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print("  train/  - HuggingFace Dataset")
    print("  test/   - HuggingFace Dataset")
    print("  test_prompts.json - prompts only for generation")

if __name__ == '__main__':
    main()
