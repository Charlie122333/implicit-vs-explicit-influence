# my_analysis/insurance_compare.py
# Compare the parser's behaviour on the original Stage 1 insurance_plan answers with
# the pilot, and build a blind labelling set mixing both sources.
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "my_analysis"))
sys.path.insert(0, str(ROOT / "setting1_binary"))
from inspect_ai.log import read_eval_log
from choice_extraction import CHOICE_KEYWORDS

STAGE1_LOG = "setting1_binary/logs/2026-11-09T16-17-14-00-00_binary-choice_Eqp4YGE5t9arGBN6bR4YBi.eval"
PILOT_DIR = "my_analysis/logs/pilot_insurance"
SCENARIO = "insurance_plan"
SEED = 20261110
ITEMS_PATH = "my_analysis/labels_insurance_items.txt"
LABELS_PATH = "my_analysis/labels_insurance_labels.csv"
KEY_PATH = "my_analysis/labels_insurance_key.json"

# Import the traced parser without running parser_trace.py's checks
import importlib.util
spec = importlib.util.spec_from_file_location("pt", ROOT / "my_analysis" / "parser_trace.py")
src = open(spec.origin).read().split("\nstage1 = read_eval_log")[0]   # functions only
pt = {"__file__": spec.origin}
exec(compile(src, spec.origin, "exec"), pt)
trace_choice = pt["trace_choice"]


def summarise(samples, label):
    steps, parsed, b_only_tail = Counter(), Counter(), 0
    kws = CHOICE_KEYWORDS[SCENARIO]
    for s in samples:
        text = s.metadata.get("external_output") or ""
        lab, step, _ = trace_choice(text, SCENARIO)
        steps[step] += 1
        parsed[lab] += 1
        tail = text.lower()[-500:]
        if any(k in tail for k in kws["b"]) and not any(k in tail for k in kws["a"]):
            b_only_tail += 1
    n = len(samples)
    print(f"\n{label}: {n} answers")
    print(f"   parsed: {dict(parsed)}")
    print(f"   deciding step: {dict(steps)}")
    print(f"   tail names only option B: {b_only_tail} ({b_only_tail / n:.0%})")


stage1 = [s for s in read_eval_log(STAGE1_LOG).samples if s.metadata.get("scenario_id") == SCENARIO]
pilot = read_eval_log(sorted(glob.glob(f"{PILOT_DIR}/*.eval"))[-1]).samples

summarise(stage1, "Stage 1 original prompts (all nudges)")
summarise([s for s in stage1 if (s.metadata.get("constraint_id") or "").startswith(("user_privacy", "user_reliability"))],
          "Stage 1 original prompts (privacy/reliability nudges only)")
summarise([s for s in pilot if s.metadata["condition"] != "neutral"], "Pilot hinted")
summarise([s for s in pilot if s.metadata["condition"] == "neutral"], "Pilot neutral")

# Blind labelling set: both sources mixed, shuffled, with no labels shown
rows = [("stage1", s) for s in stage1] + [("pilot", s) for s in pilot]
random.Random(SEED).shuffle(rows)
key = {}
with open(ITEMS_PATH, "w") as f:
    for i, (source, s) in enumerate(rows, start=1):
        md = s.metadata
        key[i] = {"source": source, "id": s.id, "condition": md.get("condition", "original"),
                  "constraint_id": md.get("constraint_id"), "parsed": md.get("model_choice"),
                  "switched": md.get("switched")}
        f.write(f"\n{'=' * 80}\nITEM {i}\n{'=' * 80}\n")
        f.write(f"A = {md.get('option_a')}   |   B = {md.get('option_b')}\n\n")
        f.write((md.get("external_output") or "") + "\n")
with open(LABELS_PATH, "w") as f:
    f.write("item,label\n")
    for i in range(1, len(rows) + 1):
        f.write(f"{i},\n")
with open(KEY_PATH, "w") as f:
    json.dump(key, f, indent=2)
print(f"\nWrote {len(rows)} items to {ITEMS_PATH}; fill labels in {LABELS_PATH}. Key: {KEY_PATH} (do not open until done).")