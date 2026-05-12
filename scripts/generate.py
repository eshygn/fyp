"""
Generate stories from baseline and fine-tuned Qwen3-4B models.

Produces story outputs for three conditions:
  1. Baseline Qwen3-4B (no fine-tuning)
  2. Standard DPO fine-tuned
  3. Length-normalised DPO fine-tuned

Usage:
  # Generate from baseline
  python generate.py --model_name Qwen/Qwen3-4B --output ../results/baseline_stories.jsonl

  # Generate from fine-tuned model
  python generate.py --model_name Qwen/Qwen3-4B --adapter ../models/dpo_b01/final --output ../results/dpo_b01_stories.jsonl
"""

import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model(model_name: str, adapter_path: str = None):
    """Load model, optionally with a LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_story(model, tokenizer, prompt: str, max_new_tokens: int = 1024):
    """Generate a single story from a prompt."""
    messages = [
        {"role": "system", "content": "You are a creative fiction writer. Write a short story based on the given outline. Focus on emotional depth, subtext, and show-don't-tell."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,

            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    story = tokenizer.decode(generated, skip_special_tokens=True)
    if "</think>" in story:
        story = story.split("</think>", 1)[1].strip()
    elif "<think>" in story:
        story = ""  # pure thinking, no actual story
    return story


def main():
    parser = argparse.ArgumentParser(description='Generate stories for evaluation')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--adapter', type=str, default=None,
                        help='Path to LoRA adapter (None = baseline)')
    parser.add_argument('--prompts', type=str, default='../data/prepared/test_prompts.json',
                        help='Path to test prompts JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL file for generated stories')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--n_samples', type=int, default=-1,
                        help='Number of prompts to use (-1 = all)')
    args = parser.parse_args()

    # Load prompts
    with open(args.prompts, 'r') as f:
        prompts = json.load(f)

    if args.n_samples > 0:
        prompts = prompts[:args.n_samples]

    print(f"Generating {len(prompts)} stories...")
    condition = "baseline" if args.adapter is None else args.adapter

    # Load model
    model, tokenizer = load_model(args.model_name, args.adapter)

    # Generate
    results = []
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        story = generate_story(model, tokenizer, prompt, args.max_new_tokens)
        results.append({
            'prompt': prompt,
            'story': story,
            'condition': condition,
            'index': i,
        })

    # Save
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\nSaved {len(results)} stories to {args.output}")


if __name__ == '__main__':
    main()
