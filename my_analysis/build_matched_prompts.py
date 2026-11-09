# my_analysis/build_matched_prompts.py
# Build matched-hint prompts for development scenarios: one neutral prompt and two
# hinted prompts (A = privacy, B = reliability) that are identical except for the aside.
import json

# scenario_id -> text that begins the final question; the aside goes just before it.
# Each scenario's bare prompt is worded differently, so set this by reading the prompt.
SCENARIOS = {
    "insurance_plan": "Given that",
}
HINTS = {"A": "user_privacy_positive", "B": "user_reliability_positive"}
FRAME = "I read somewhere that {tool} {hint}."
MODEL_KEY = "haiku"

DATA = "setting1_binary/data/"
SPLIT_PATH = "my_analysis/scenario_split.json"
OUTPUT_PATH = "my_analysis/matched_prompts.json"

prompts = json.load(open(DATA + "prompts.json"))["prompts"]
constraints = json.load(open(DATA + "constraints.json"))["constraints"]
baselines = json.load(open(DATA + "baseline_choices.json"))["baseline_choices"][MODEL_KEY]
split = json.load(open(SPLIT_PATH))

natural_hint = {c["id"]: c["natural_hint"] for c in constraints}

entries = []
for sid, marker in SCENARIOS.items():
    assert sid in split["dev"], f"{sid} is not a development scenario"

    baseline = baselines[sid]                    # Haiku's usual choice, "A" or "B"
    target = "B" if baseline == "A" else "A"     # both hints push toward the other option
    target_path = target.lower()                 # prompts.json uses lowercase "a"/"b"

    bare = [p for p in prompts if p["scenario_id"] == sid and p["constraint_id"] is None]
    assert len(bare) == 1, f"{sid}: expected 1 bare prompt, found {len(bare)}"
    bare = bare[0]
    neutral = bare["prompt"].strip()             # remove stray spaces at the ends

    # The name the authors used for the target option with these two nudges
    tools = {p["target_tool"] for p in prompts
             if p["scenario_id"] == sid
             and p["constraint_id"] in HINTS.values()
             and p["target_path"] == target_path}
    assert len(tools) == 1, f"{sid}: expected one target_tool, found {tools}"
    tool = tools.pop()

    assert neutral.count(marker) == 1, f"{sid}: marker {marker!r} appears {neutral.count(marker)} times"
    idx = neutral.find(marker)

    asides = {cond: FRAME.format(tool=tool, hint=natural_hint[cid]) for cond, cid in HINTS.items()}
    hinted = {cond: neutral[:idx] + aside + " " + neutral[idx:] for cond, aside in asides.items()}

    # Checks: the aside is the only difference between each hinted prompt and the neutral one
    for cond, text in hinted.items():
        assert text.count(asides[cond]) == 1, f"{sid} {cond}: aside not found exactly once"
        assert text.replace(asides[cond] + " ", "", 1) == neutral, f"{sid} {cond}: differs from neutral beyond the aside"
    for aside in asides.values():
        assert aside not in neutral, f"{sid}: an aside appears in the neutral prompt"
    assert asides["A"] not in hinted["B"] and asides["B"] not in hinted["A"], f"{sid}: asides leaked across conditions"

    common = {
        "scenario_id": sid,
        "baseline_choice": baseline,
        "target_choice": target,
        "target_path": target_path,
        "target_tool": tool,
        "option_a": bare["path_a_name"],
        "option_b": bare["path_b_name"],
        "candidate_constraint_ids": dict(HINTS),
    }
    entries.append({**common, "condition": "neutral", "constraint_id": None,
                    "aside": None, "prompt": neutral})
    for cond in ("A", "B"):
        entries.append({**common, "condition": cond, "constraint_id": HINTS[cond],
                        "aside": asides[cond], "prompt": hinted[cond]})

    print(f"\n===== {sid} (baseline {baseline}, target {target}: {tool}) =====")
    for e in entries[-3:]:
        print(f"\n--- {e['condition']} ---\n{e['prompt']}")

output = {
    "model_key": MODEL_KEY,
    "frame": FRAME,
    "hints": HINTS,
    "insertion": "aside plus one space, inserted immediately before each scenario's final-question marker",
    "markers": SCENARIOS,
    "entries": entries,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(entries)} prompts to {OUTPUT_PATH}")