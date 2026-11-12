# my_analysis/check_size.py
# Print the structure of the two data files that determine how many
# Setting 1 samples Haiku will run. Run from the repository root.
import json

with open("setting1_binary/data/baseline_choices.json") as f:
    baselines = json.load(f)

print("=== baseline_choices.json ===")
print("type:", type(baselines).__name__)
if isinstance(baselines, dict):
    print("top-level keys:", list(baselines.keys()))
    for key, value in baselines.items():
        if "haiku" in key.lower():
            print(f"\nHaiku block under key '{key}': type {type(value).__name__}, length {len(value)}")
            if isinstance(value, dict):
                first = next(iter(value))
                print("  example entry:", first, "->", value[first])

with open("setting1_binary/data/prompts.json") as f:
    prompts = json.load(f)

print("\n=== prompts.json ===")
print("type:", type(prompts).__name__, "| length:", len(prompts))
if isinstance(prompts, dict):
    first = next(iter(prompts))
    print("example key:", first)
    value = prompts[first]
    print("example value type:", type(value).__name__)
    if isinstance(value, dict):
        print("example value keys:", list(value.keys())[:10])
elif isinstance(prompts, list):
    item = prompts[0]
    print("example item keys:" if isinstance(item, dict) else "example item:",
          list(item.keys()) if isinstance(item, dict) else item)