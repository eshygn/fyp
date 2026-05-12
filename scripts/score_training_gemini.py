"""Score training pairs with Gemini for rubric-aligned dataset construction."""
import json, os, re, time, random, argparse, warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
import google.generativeai as genai

RUBRIC = """Score this story on three dimensions (0-3 each):
EMOTIONAL_FLEXIBILITY: 0=only external, 1=superficial inner life, 2=meaningful balance, 3=seamless integration
EMOTIONAL_ARC_COHERENCE: 0=no arc, 1=underdeveloped, 2=clear arc with payoff, 3=compelling arc with resolution
SUBTEXT_DENSITY: 0=everything explicit, 1=occasional subtext, 2=consistent subtext, 3=rich layered subtext

STORY:
{story}

Return ONLY valid JSON with keys: emotional_flexibility, emotional_arc_coherence, subtext_density, reasoning"""

def get_text(response):
    try:
        for part in reversed(response.candidates[0].content.parts):
            if hasattr(part, 'text') and part.text and part.text.strip():
                return part.text.strip()
    except Exception:
        pass
    try:
        if response.text:
            return response.text.strip()
    except Exception:
        pass
    return ""

def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    raise ValueError("Could not extract JSON")

def score_story(story, model, retries=3):
    for attempt in range(retries):
        try:
            response = model.generate_content(
                RUBRIC.format(story=story[:8000]),
                generation_config={"response_mime_type": "application/json", "temperature": 0.1}
            )
            raw = get_text(response)
            if not raw:
                raise ValueError("Empty response")
            scores = extract_json(raw)
            for dim in ["emotional_flexibility", "emotional_arc_coherence", "subtext_density"]:
                val = scores.get(dim)
                if not isinstance(val, int) or val < 0 or val > 3:
                    raise ValueError(f"Invalid {dim}: {val}")
            return scores
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=os.path.expanduser("~/dissertation/data/filtered_Qwen3-4B.jsonl"))
    p.add_argument("--output", default=os.path.expanduser("~/dissertation/data/scored_pairs_gemini.jsonl"))
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(args.model)

    print(f"Loading {args.input}...")
    with open(args.input) as f:
        all_pairs = [json.loads(line) for line in f]
    print(f"  Total pairs: {len(all_pairs)}")

    random.seed(args.seed)
    pairs = random.sample(all_pairs, min(args.n_samples, len(all_pairs)))

    # Resume support
    already_done = 0
    if os.path.exists(args.output):
        with open(args.output) as f:
            already_done = sum(1 for _ in f)
        if already_done:
            print(f"  Resuming from {already_done}")
            pairs = pairs[already_done:]

    print(f"  Scoring {len(pairs)} pairs ({len(pairs)*2} API calls)")
    out_f = open(args.output, "a" if already_done else "w")
    scored = 0
    failed = 0

    try:
        for pair in tqdm(pairs):
            chosen_text = pair.get("chosen", "")
            rejected_text = pair.get("rejected", "")

            chosen_scores = score_story(chosen_text, model)
            time.sleep(4)
            rejected_scores = score_story(rejected_text, model)
            time.sleep(4)

            if chosen_scores and rejected_scores:
                result = {
                    "prompt": pair.get("outline", ""),
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "chosen_scores": chosen_scores,
                    "rejected_scores": rejected_scores,
                    "chosen_total": sum(chosen_scores[d] for d in ["emotional_flexibility","emotional_arc_coherence","subtext_density"]),
                    "rejected_total": sum(rejected_scores[d] for d in ["emotional_flexibility","emotional_arc_coherence","subtext_density"]),
                }
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                scored += 1
            else:
                failed += 1
    finally:
        out_f.close()

    print(f"\nDone: {scored} scored, {failed} failed")
    if os.path.exists(args.output):
        total = sum(1 for _ in open(args.output))
        print(f"Total in output file: {total}")

if __name__ == "__main__":
    main()
