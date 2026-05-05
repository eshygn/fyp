import json, os, time, argparse
from tqdm import tqdm
import anthropic

RUBRIC_PROMPT = """You are an expert literary critic evaluating short stories for emotional depth and subtext.

Score the following story on THREE dimensions, each from 0 to 3:

1. EMOTIONAL_FLEXIBILITY (0-3): Balance of inner emotional life vs outer actions/dialogue.
   0=only external, 1=superficial inner life, 2=meaningful balance, 3=seamless integration

2. EMOTIONAL_ARC_COHERENCE (0-3): Coherent emotional arc (rags-to-riches, tragedy, man-in-hole etc).
   0=no arc, 1=underdeveloped, 2=clear arc with payoff, 3=compelling arc with strong resolution

3. SUBTEXT_DENSITY (0-3): Meaning communicated beneath surface through implication/symbol.
   0=everything explicit, 1=occasional subtext, 2=consistent subtext, 3=rich layered subtext

STORY:
{story}

Respond ONLY with valid JSON, no other text:
{{"emotional_flexibility": <0-3>, "emotional_arc_coherence": <0-3>, "subtext_density": <0-3>, "reasoning": "<one sentence>"}}"""

def call_claude(story, client, retries=3):
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": RUBRIC_PROMPT.format(story=story[:3000])}]
            )
            text = msg.content[0].text.strip().replace("```json","").replace("```","").strip()
            return json.loads(text)
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return {"emotional_flexibility": -1, "emotional_arc_coherence": -1,
            "subtext_density": -1, "reasoning": "evaluation_failed"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api_key", default=os.environ.get("ANTHROPIC_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Set ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=args.api_key)

    with open(args.input) as f:
        stories = [json.loads(line) for line in f]

    print(f"Evaluating {len(stories)} stories...")
    results = []

    for item in tqdm(stories):
        scores = call_claude(item["story"], client)
        results.append({**item, **scores})
        time.sleep(2)

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    valid = [r for r in results if r["emotional_flexibility"] >= 0]
    print(f"\nCompleted {len(valid)}/{len(results)} successful evaluations")
    for dim in ["emotional_flexibility", "emotional_arc_coherence", "subtext_density"]:
        avg = sum(r[dim] for r in valid) / len(valid)
        print(f"  {dim}: {avg:.3f}")

if __name__ == "__main__":
    main()
