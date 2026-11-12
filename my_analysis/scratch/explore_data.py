# my_analysis/explore_data.py
# Look one level deeper into the Setting 1 data files.
import json

b = json.load(open("setting1_binary/data/baseline_choices.json"))
print("baseline metadata:", b["metadata"])
print("models:", list(b["baseline_choices"].keys()))

p = json.load(open("setting1_binary/data/prompts.json"))
print("\nprompts.json keys:", list(p.keys()))
print("prompts metadata:", p["metadata"])

h = b["baseline_choices"]["haiku"]
print("\nHaiku stable scenarios:", len(h))
print("examples:", list(h.items())[:3] if isinstance(h, dict) else h[:3])