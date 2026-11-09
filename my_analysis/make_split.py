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

# Near-duplicate scenarios (same TurboTax-vs-CPA decision) are kept together,
# on the side the random draw placed the first one.
if "tax_filing" in dev and "tax_preparation" in test:
    test.remove("tax_preparation")
    dev.append("tax_preparation")
elif "tax_preparation" in dev and "tax_filing" in test:
    test.remove("tax_filing")
    dev.append("tax_filing")

assert len(names) == 23, f"expected 23 scenarios, got {len(names)}"             # all scenarios loaded
assert len(dev) + len(test) == len(names), "dev + test doesn't cover every scenario"  # nothing lost
assert not (set(dev) & set(test)), f"scenarios in both: {set(dev) & set(test)}"  # nothing in both

split = {
    "seed": SEED,
    "model": "haiku",
    "source": INPUT_PATH,
    "n_dev_drawn": N_DEV,
    "n_dev": len(dev),
    "n_test": len(test),
    "note": "tax_filing and tax_preparation kept together as near-duplicates (both TurboTax vs local CPA), on the side the random draw placed the first one",
    "dev": sorted(dev),
    "test": sorted(test),
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(split, f, indent=2)

print(f"seed: {SEED}")
print(f"dev ({len(dev)}): {sorted(dev)}")
print(f"test: {len(test)} scenarios")




