import json
import random
import os 

SEED = 20261109
N_DEV = 5
INPUT_PATH = "setting1_binary/data/baseline_choices.json"
OUTPUT_PATH = "my_analysis/scenario_split.json"


with open(INPUT_PATH) as f:     # Opening and closing input file and loading as .json
    data = json.load(f)

haiku = data["baseline_choices"]["haiku"]       # Accessing correct dict. keys 


names = sorted(haiku)   # Fixed starting order

rng = random.Random(SEED)   # Create radnomizer

rng.shuffle(names)      # Randomize order

dev = names[0:N_DEV]        # Split takes a random 5
test = names[N_DEV:]

assert len(names) == 23, f"expected 23 scenarios, got {len(names)}"     # Asserts double checking the splitting works
assert len(dev) == 5, f"expected 5 scenarios, got {len(dev)}"
assert len(test) == 18, f"expected 18 scenarios, got {len(test)}"

split = {
    "seed": SEED,
    "model": "haiku",
    "source": INPUT_PATH,
    "n_dev": N_DEV,
    "dev": sorted(dev),
    "test": sorted(test),
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(split, f, indent=2)

print(f"dev: {sorted(dev)}")




