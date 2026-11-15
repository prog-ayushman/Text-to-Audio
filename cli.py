import argparse
import json
import sys

from predictor import MusicPredictor

p = argparse.ArgumentParser()
p.add_argument("prompt")
p.add_argument("--seconds", type=float, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--output", default="outputs/clip.wav")
p.add_argument("--revision")
a = p.parse_args()
try:
    _, record = MusicPredictor(revision=a.revision).generate(
        a.prompt, seconds=a.seconds, seed=a.seed, output_path=a.output
    )
except ValueError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
print(json.dumps(record, indent=2))