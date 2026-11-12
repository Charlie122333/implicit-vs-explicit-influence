# my_analysis/find_prompts.py
# Pull out, for one development scenario: its bare prompt(s), the privacy and
# reliability constraint definitions, and the authors' nudged prompts for those two
# themes (to see how they worded the asides).
import json

SCENARIO = "insurance_plan"
THEMES = ("user_privacy", "user_reliability")

prompts = json.load(open("setting1_binary/data/prompts.json"))["prompts"]
constraints = json.load(open("setting1_binary/data/constraints.json"))["constraints"]

bare = [p for p in prompts if p["scenario_id"] == SCENARIO and p["constraint_id"] is None]
print(f"=== {len(bare)} bare prompt(s) for {SCENARIO} ===")
for p in bare:
    print("fields:", sorted(p.keys()))
    print("A:", p.get("path_a_name"), "| B:", p.get("path_b_name"))
    print(p["prompt"], "\n")

print("=== constraint definitions ===")
for c in constraints:
    if c["id"].startswith(THEMES):
        print(c, "\n")

nudged = [p for p in prompts
          if p["scenario_id"] == SCENARIO and (p["constraint_id"] or "").startswith(THEMES)]
print(f"=== {len(nudged)} nudged prompt(s) for those themes ===")
for p in nudged:
    print(f"\n--- {p['prompt_id']} (direction={p['direction']}, target_path={p['target_path']})")
    print(p["prompt"])