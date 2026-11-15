import argparse
import json
import os

import numpy as np

from predictor import MusicPredictor

PROMPTS = [
    "calm piano melody",
    "upbeat acoustic guitar folk tune",
    "slow ambient synthesizer pad with a soft atmosphere",
    "energetic electronic dance beat with a strong bass line",
    "relaxing lo-fi hip hop beat with soft drums",
    "dramatic orchestral music with strings and drums",
    "gentle flute melody, peaceful and slow",
    "the sound of heavy rain and thunder",
]
SECONDS = 5
SEED = 0
RESULTS = "examples/sample_results.json"
RATINGS = "examples/ratings.json"

ap = argparse.ArgumentParser()
ap.add_argument("--ratings-only", action="store_true", help="skip generation; merge ratings")
args = ap.parse_args()

if args.ratings_only:
    results = json.load(open(RESULTS))
else:
    model = MusicPredictor()
    rows, first_audio = [], None
    for i, prompt in enumerate(PROMPTS, 1):
        path = f"examples/audio/prompt_{i:02d}.wav"
        audio, rec = model.generate(prompt, seconds=SECONDS, seed=SEED, output_path=path)
        if i == 1:
            first_audio = audio
        rows.append(rec)
        print(i, prompt, "->", round(rec["generation_seconds"], 1), "s")
    repeat, _ = model.generate(PROMPTS[0], seconds=SECONDS, seed=SEED)
    same_shape = repeat.shape == first_audio.shape
    results = {
        "seconds": SECONDS,
        "seed": SEED,
        "rows": rows,
        "repeat_same_shape": bool(same_shape),
        "repeat_max_abs_diff": float(np.abs(repeat - first_audio).max()) if same_shape else None,
    }

if os.path.exists(RATINGS):
    ratings = json.load(open(RATINGS))
    for i, r in enumerate(results["rows"], 1):
        r["human_rating"] = ratings.get(str(i))


def avg(key):
    return float(np.mean([r[key] for r in results["rows"]]))


gen = [r["generation_seconds"] for r in results["rows"]]
summary = {
    "clips": len(results["rows"]),
    "mean_generation_seconds": avg("generation_seconds"),
    "min_generation_seconds": min(gen),
    "max_generation_seconds": max(gen),
    "mean_duration_seconds": avg("duration_seconds"),
    "mean_real_time_factor": avg("real_time_factor"),
    "mean_peak": avg("peak"),
    "mean_rms": avg("rms"),
    "mean_silence_fraction": avg("silence_fraction"),
    "all_finite": all(r["all_finite"] for r in results["rows"]),
    "repeat_same_shape": results.get("repeat_same_shape"),
    "repeat_max_abs_diff": results.get("repeat_max_abs_diff"),
}
given = [r.get("human_rating") for r in results["rows"]]
if all(g is not None for g in given):
    summary["mean_human_rating"] = float(np.mean(given))
results["summary"] = summary

os.makedirs("examples", exist_ok=True)
json.dump(results, open(RESULTS, "w"), indent=2)
print(json.dumps(summary, indent=2))