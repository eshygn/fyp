"""Evaluate stories with Gemini as second judge (multi-judge validation)."""
import json, os, re, time, argparse, warnings
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
    """Extract text from Gemini response, handling thinking models."""
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
    text_clean = re.sub(r'```(?:json)?\s*', '', text)
    text_clean = re.sub(r'```\s*$', '', text_clean).strip()
    try:
        return json.loads(text_clean)
    except Exception:
        pass
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    raise ValueError(f"Could not extract JSON from: {text[:200]}")

def call_gemini(story, model, retries=3):
    for attempt in range(retries):
        try:
            response = model.generate_content(
                RUBRIC.format(story=story[:8000]),
                generation_config={"response_mime_type": "application/json", "temperature": 0.1}
            )
            raw = get_text(response)
            if not raw:
                raise ValueError("Empty response from Gemini")
            scores = extract_json(raw)
            for dim in ["emotional_flexibility", "emotional_arc_coherence", "subtext_density"]:
                val = scores.get(dim)
                if not isinstance(val, int) or val < 0 or val > 3:
                    raise ValueError(f"Invalid {dim}: {val}")
            return scores
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return {"emotional_flexibility": -1, "emotional_arc_coherence": -1,
            "subtext_density": -1, "reasoning": "evaluation_failed"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="gemini-2.5-flash")
    args = p.parse_args()

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(args.model)

    with open(args.input) as f:
        stories = [json.loads(line) for line in f]

    # Resume support
    already_done = 0
    if os.path.exists(args.output):
        with open(args.output) as f:
            already_done = sum(1 for _ in f)
        if already_done:
            print(f"Resuming: skipping {already_done} already done")
            stories = stories[already_done:]

    print(f"Evaluating {len(stories)} stories with {args.model}...")
    out_f = open(args.output, "a" if already_done else "w")
    try:
        for item in tqdm(stories):
            if len(item.get("story", "").strip()) < 10:
                scores = {"emotional_flexibility": 0, "emotional_arc_coherence": 0,
                          "subtext_density": 0, "reasoning": "empty_story"}
            else:
                scores = call_gemini(item["story"], model)
            result = {**item, **scores}
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            time.sleep(4)
    finally:
        out_f.close()

    all_results = [json.loads(l) for l in open(args.output)]
    valid = [r for r in all_results if r["emotional_flexibility"] >= 0]
    print(f"\nDone: {len(valid)}/{len(all_results)} valid")
    for dim in ["emotional_flexibility", "emotional_arc_coherence", "subtext_density"]:
        if valid:
            print(f"  {dim}: {sum(r[dim] for r in valid)/len(valid):.3f}")

if __name__ == "__main__":
    main()
