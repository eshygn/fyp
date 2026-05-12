"""
Evaluate generated stories using Claude Sonnet 4.6 as independent judge.
Scores each story on three dimensions (0-3 each):
  - Emotional Flexibility (inner/outer life balance)
  - Emotional Arc Coherence (Vonnegut arc structure)
  - Subtext Density (meaning beneath surface)

Judge: Claude Sonnet 4.6 (Anthropic) — independent from Qwen3-4B (Alibaba).
Originally planned Gemini Flash 2.0 but switched due to Google API quota constraints.
"""

import json
import os
import re
import time
import argparse
from tqdm import tqdm
import anthropic

SYSTEM_PROMPT = """You are a literary evaluation machine. You output ONLY valid JSON. 
No preamble, no markdown, no explanation, no backticks. Just the raw JSON object.
If you cannot evaluate the story, still return valid JSON with scores of 0."""

RUBRIC_PROMPT = """Score this story on three dimensions (0-3 each):

EMOTIONAL_FLEXIBILITY: Balance of inner emotional life vs outer actions/dialogue.
0=only external action, 1=superficial inner life, 2=meaningful balance, 3=seamless integration

EMOTIONAL_ARC_COHERENCE: Coherent emotional arc (rags-to-riches, tragedy, man-in-hole, icarus).
0=no arc, 1=underdeveloped, 2=clear arc with payoff, 3=compelling arc with strong resolution

SUBTEXT_DENSITY: Meaning communicated beneath surface through implication/symbol.
0=everything explicit, 1=occasional subtext, 2=consistent subtext, 3=rich layered subtext

STORY:
{story}

Return ONLY this JSON:
{{"emotional_flexibility": <int 0-3>, "emotional_arc_coherence": <int 0-3>, "subtext_density": <int 0-3>, "reasoning": "<one sentence>"}}"""


def extract_json(text: str) -> dict:
    """Extract JSON from text that may contain markdown or preamble."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    text_clean = re.sub(r'```(?:json)?\s*', '', text)
    text_clean = re.sub(r'```\s*$', '', text_clean).strip()
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass

    # Find first { ... } block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from: {text[:200]}")


def call_claude(story: str, client: anthropic.Anthropic, retries: int = 3) -> dict:
    """Call Claude Sonnet 4.6 to evaluate a story."""
    for attempt in range(retries):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": RUBRIC_PROMPT.format(story=story[:8000])
                }]
            )
            text = message.content[0].text
            scores = extract_json(text)

            # Validate scores are integers 0-3
            for dim in ["emotional_flexibility", "emotional_arc_coherence", "subtext_density"]:
                val = scores.get(dim)
                if not isinstance(val, int) or val < 0 or val > 3:
                    raise ValueError(f"Invalid score for {dim}: {val}")

            return scores
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))

    return {"emotional_flexibility": -1, "emotional_arc_coherence": -1,
            "subtext_density": -1, "reasoning": "evaluation_failed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL of stories")
    parser.add_argument("--output", required=True, help="Output JSONL with scores")
    parser.add_argument("--api_key", default=os.environ.get("ANTHROPIC_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic(api_key=args.api_key)

    with open(args.input) as f:
        stories = [json.loads(line) for line in f]

    print(f"Evaluating {len(stories)} stories with Claude Sonnet 4.6...")
    # Resume support: skip already-evaluated stories
    already_done = 0
    if os.path.exists(args.output):
        with open(args.output) as f:
            already_done = sum(1 for _ in f)
        if already_done:
            print(f"Resuming: skipping first {already_done} already-evaluated stories")
            stories = stories[already_done:]

    results = []
    out_f = open(args.output, "a" if already_done else "w")

    try:
        for item in tqdm(stories):
            if len(item.get("story", "").strip()) < 10:
                scores = {"emotional_flexibility": 0, "emotional_arc_coherence": 0,
                           "subtext_density": 0, "reasoning": "empty_story"}
            else:
                scores = call_claude(item["story"], client)
            result = {**item, **scores}
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            time.sleep(2)
    finally:
        out_f.close()

    # Print summary (re-read full output file to include resumed results)
    all_results = [json.loads(l) for l in open(args.output)]
    valid = [r for r in all_results if r["emotional_flexibility"] >= 0]
    failed = len(all_results) - len(valid)
    print(f"\nCompleted {len(valid)}/{len(all_results)} evaluations ({failed} failed)")
    for dim in ["emotional_flexibility", "emotional_arc_coherence", "subtext_density"]:
        if valid:
            avg = sum(r[dim] for r in valid) / len(valid)
            print(f"  {dim}: {avg:.3f}")
        else:
            print(f"  {dim}: N/A (no valid evaluations)")


if __name__ == "__main__":
    main()
